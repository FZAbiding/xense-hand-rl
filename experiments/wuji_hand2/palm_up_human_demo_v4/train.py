"""Independent V4 PPO; fresh weights and explicit stage progression."""
import argparse, copy, hashlib, json, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from train import ActorCritic
from experiments.wuji_hand2.palm_up_human_demo_v4.env import Env, assets_for_grasp
from experiments.wuji_hand2.palm_up_human_demo_v4.handover_env import HandoverEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='experiments/wuji_hand2/palm_up_human_demo_v4/config.json')
    p.add_argument('--stage',required=True)
    p.add_argument('--resume')
    a=p.parse_args()
    config=json.loads((ROOT/a.config).read_text())
    args=SimpleNamespace(**(config['training']|config['stages'][a.stage]))
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    out=ROOT/config['runs']/a.stage;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Use a new stage run; no overwrite')
    cfg=vars(args)|dict(stage=a.stage,experiment_version='human_demo_v4',grasp_version='palm_up_human_demo_v4',action_mode='delta',
        park_unused_fingers=1,action_size=12,observation_size=105,control_dt=.025,physics_dt=.005,objective='continuous_directed_rotation')
    cfg['contact_schedule']='TI_IM_MT_60deg' if args.amplitude else 'static_functional_pinch'
    cfg['stage_gate']=config['stage_gate']
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    assets,bank=assets_for_grasp('palm_up_human_demo_v4')
    source_names=['train.py','experiments/wuji_hand2/pinch.py','experiments/wuji_hand2/ring_control.py',a.config]
    source_names += [str(x.relative_to(ROOT)) for x in Path(__file__).resolve().parent.glob('*.py')]
    source_names += [str((assets/x).relative_to(ROOT)) for x in ['scene.xml',bank,'seed_bank_manifest.json']]
    source_names += [config['accepted_demo']]
    if config.get('reference_annotation'):source_names.append(config['reference_annotation'])
    manifest={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in source_names}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in source_names:
        dest=out/'snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    Environment=HandoverEnv if args.amplitude else Env
    env=Environment(num_envs=args.num_envs,amplitude=args.amplitude,period=args.period,seed=args.seed,
        episode_seconds=args.episode_seconds,park_unused_fingers=1,pinch_reward_weight=2.,command_clip_penalty=getattr(args,'command_clip_penalty',3.),tracking_sigma=args.tracking_sigma)
    naction=env.naction;obs=env.obs();agent=ActorCritic(obs.shape[-1],naction).cuda()
    opt=torch.optim.Adam(agent.parameters(),lr=args.lr);start=0
    if a.resume:
        c=torch.load(ROOT/a.resume,weights_only=False)
        if c['config']['experiment_version']!='human_demo_v4':raise ValueError('No migration from old task')
        for x in [str((assets/'scene.xml').relative_to(ROOT)),str((assets/bank).relative_to(ROOT))]:
            if c['source_manifest'][x]!=manifest[x]:raise ValueError('Changed assets')
        agent.load_state_dict(c['model'])
        normalization_reset=[]
        if args.amplitude != c['config']['amplitude']:
            # Hold has constant goal/omega; its near-zero variances would saturate
            # the new target channels. Adapt explicitly at the task-stage boundary.
            with torch.no_grad():
                agent.obs_mean[54:56]=0
                agent.obs_var[54:56]=1
                agent.obs_count.fill_(1024)
            normalization_reset=[54,55]
        # Stage change keeps weights and normalization, resets optimizer and iteration.
        (out/'lineage.json').write_text(json.dumps(dict(parent=a.resume,parent_sha256=hashlib.sha256((ROOT/a.resume).read_bytes()).hexdigest(),optimizer_reset=True,normalization_reset_channels=normalization_reset),indent=2))
    with torch.no_grad():agent.log_std.clamp_(-3,float(np.log(args.max_policy_std)))
    H,N=args.horizon,args.num_envs
    ob=torch.zeros(H,N,obs.shape[-1],device='cuda');ac=torch.zeros(H,N,naction,device='cuda')
    lp=torch.zeros(H,N,device='cuda');rw=torch.zeros_like(lp);dn=torch.zeros_like(lp);val=torch.zeros_like(lp)
    def save(it,name):
        tmp=out/(name+'.tmp');torch.save(dict(model=agent.state_dict(),optimizer=opt.state_dict(),iteration=it,config=cfg,
            nobs=obs.shape[-1],naction=naction,source_manifest=manifest),tmp);tmp.replace(out/name)
    t0=time.time();print('TRAIN_START',json.dumps(cfg),flush=True)
    for it in range(start+1,start+args.iterations+1):
        completed=[];roll_metrics=[];term_values=[]
        with torch.no_grad():
            agent.update_stats(obs)
            for t in range(H):
                mean,v=agent(obs);dist=torch.distributions.Normal(mean,agent.log_std.exp());action=dist.sample()
                ob[t]=obs;ac[t]=action;lp[t]=dist.log_prob(action).sum(-1);val[t]=v
                obs,reward,done,info=env.step(action)
                tv=agent(info['terminal_obs'])[1]
                rw[t]=reward*args.reward_scale+.99*tv*info['timeout']*(~info['fail']);dn[t]=done
                roll_metrics.append(torch.stack((info['error'].abs().mean(),info['tilt'].mean(),info['command_clip'].mean(),
                    info['normal_force'].mean(),(info['other_force']>.05).float().mean(),(info['palm_force']>.05).float().mean(),info['tips'].mean())))
                term_values.append(torch.stack([v.mean() for v in info['reward_terms'].values()]))
                if done.any():completed.append(torch.stack((info['episode_return'][done],info['episode_error'][done],info['episode_length'][done],info['fail'][done].float()),-1))
            nv=agent(obs)[1];adv=torch.zeros_like(rw);last=torch.zeros(N,device='cuda')
            for t in reversed(range(H)):
                nextv=nv if t==H-1 else val[t+1]
                delta=rw[t]+.99*nextv*(1-dn[t])-val[t];last=delta+.99*.95*(1-dn[t])*last;adv[t]=last
            returns=adv+val
        fo=ob.flatten(0,1);fa=ac.flatten(0,1);flp=lp.flatten();fadv=adv.flatten();fr=returns.flatten()
        fadv=(fadv-fadv.mean())/(fadv.std()+1e-8)
        before=copy.deepcopy(agent.state_dict());before_opt=copy.deepcopy(opt.state_dict())
        losses=[];stop=False
        for epoch in range(4):
            for ids in torch.randperm(H*N,device='cuda').chunk(8):
                mean,v=agent(fo[ids]);dist=torch.distributions.Normal(mean,agent.log_std.exp())
                logp=dist.log_prob(fa[ids]).sum(-1);lr=logp-flp[ids];ratio=lr.exp()
                approx=((ratio-1)-lr).mean()
                if approx.item()>args.target_kl*2:stop=True;break
                pl=-torch.minimum(ratio*fadv[ids],ratio.clamp(.8,1.2)*fadv[ids]).mean()
                vl=(v-fr[ids]).square().mean();entropy=dist.entropy().sum(-1).mean()
                loss=pl+.5*vl-.002*entropy
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),1.);opt.step()
                with torch.no_grad():agent.log_std.clamp_(-3,float(np.log(args.max_policy_std)))
                losses.append([loss.item(),pl.item(),vl.item(),entropy.item()])
            if stop:break
        with torch.no_grad():
            mean,v=agent(fo);dist=torch.distributions.Normal(mean,agent.log_std.exp())
            lr=dist.log_prob(fa).sum(-1)-flp;ratio=lr.exp();kl=((ratio-1)-lr).mean().item()
            clip_fraction=((ratio-1).abs()>.2).float().mean().item()
            explained=(1-(fr-v).var()/fr.var().clamp(min=1e-8)).item()
        rollback=not np.isfinite(kl) or kl>args.target_kl*4
        if rollback:agent.load_state_dict(before);opt.load_state_dict(before_opt)
        old_lr=opt.param_groups[0]['lr']
        if rollback or kl>args.target_kl*1.5:new_lr=max(3e-5,old_lr*.7)
        elif kl<args.target_kl*.5:new_lr=min(2e-4,old_lr*1.05)
        else:new_lr=old_lr
        for g in opt.param_groups:g['lr']=new_lr
        if it%10==0 or it==start+1:
            metrics=torch.stack(roll_metrics).mean(0).cpu().tolist();terms=torch.stack(term_values).mean(0).cpu().tolist()
            average=np.mean(losses,axis=0).tolist() if losses else [0.]*4
            ep=torch.cat(completed).mean(0).cpu().tolist() if completed else [None]*4
            row=dict(iteration=it,steps=(it-start)*H*N,seconds=round(time.time()-t0,2),fps=round((it-start)*H*N/(time.time()-t0)),
                     simulation_capacity=env.check_capacity(),
                     total_loss=average[0],policy_loss=average[1],value_loss=average[2],entropy=average[3],kl=0. if rollback else kl,
                     kl_proposed=kl,rollback=rollback,clip_fraction=clip_fraction,explained_variance=explained,lr=new_lr,
                     optimizer_updates=len(losses),std=agent.log_std.exp().mean().item(),return_mean=ep[0],episode_error_rad=ep[1],episode_steps=ep[2],fall_fraction=ep[3],
                     error_rad=metrics[0],tilt_deg=float(np.degrees(metrics[1])),command_clip_fraction=metrics[2],normal_force_sum_n=metrics[3],
                     non_tip_force_frame_fraction=metrics[4],palm_force_frame_fraction=metrics[5],contacting_fingers=metrics[6],reward_terms=dict(zip(info['reward_terms'],terms)))
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if it%args.save_every==0:save(it,f'checkpoint_{it:06d}.pt');save(it,'latest.pt')
    save(it,'latest.pt')


if __name__=='__main__':main()
