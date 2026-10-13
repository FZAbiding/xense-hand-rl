"""Wuji Hand2 PPO under the same Revo3 V2 task and optimizer settings."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train import ActorCritic
from experiments.wuji_hand2.env import Env,assets_for_grasp


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--iterations',type=int,default=250)
    p.add_argument('--amplitude',type=float,default=.2);p.add_argument('--period',type=float,default=6.)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--target-kl',type=float,default=.015)
    p.add_argument('--reward-scale',type=float,default=.25);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--command-clip-penalty',type=float,default=.2)
    p.add_argument('--episode-seconds',type=float,default=12.)
    p.add_argument('--tracking-sigma',type=float,default=.18)
    p.add_argument('--njmax',type=int,default=512);p.add_argument('--nconmax',type=int,default=128)
    p.add_argument('--max-policy-std',type=float,default=1.)
    p.add_argument('--resume');p.add_argument('--action-mode',choices=['delta','home'],default='delta')
    p.add_argument('--reward-mode',choices=['quality','fingertip'],default='quality')
    p.add_argument('--grasp-version',choices=['v2','long12','palm_up_v1','palm_forward_v1','palm_up_tripod_v3'],default='palm_up_v1')
    p.add_argument('--experiment-config', help='Archive the independent orchestration configuration')
    p.add_argument('--ring-profile',choices=['legacy','natural_v1'],default='legacy')
    p.add_argument('--reset-ring-head',type=int,choices=[0,1],default=0)
    p.add_argument('--pinch-reward-weight',type=float,default=0.)
    p.add_argument('--pinch-gate',type=int,choices=[0,1],default=0)
    p.add_argument('--park-unused-fingers',type=int,choices=[0,1],default=0)
    p.add_argument('--warm-start',help='Explicit weights-only migration to a new certified tripod bank')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed)
    if not np.exp(-3)<=args.max_policy_std<=1:raise ValueError('max-policy-std must be in [exp(-3), 1]')
    out=ROOT/args.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Use a new run directory; do not overwrite previous experiments.')
    cfg=dict(vars(args),robot='wuji_hand2',experiment_version='2.0',gamma=.99,gae_lambda=.95,
             clip=.2,value_coefficient=.5,entropy_coefficient=.002,epochs=4,minibatches=8,
             observation_size=105,action_size=12 if args.park_unused_fingers else 20,control_dt=.025,physics_dt=.005)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    assets,bank_name=assets_for_grasp(args.grasp_version)
    asset_rel=assets.relative_to(ROOT).as_posix()
    sources=['train.py','experiments/wuji_hand2/env.py','experiments/wuji_hand2/train.py','experiments/wuji_hand2/calibrate.py',
             f'{asset_rel}/scene.xml',f'{asset_rel}/{bank_name}',f'{asset_rel}/provenance.json']
    sources.append('experiments/wuji_hand2/ring_control.py')
    sources.append('experiments/wuji_hand2/pinch.py')
    if args.grasp_version=='palm_up_tripod_v3':
        cert=json.loads((assets/'certified_bank_manifest.json').read_text())
        for name,key in [('scene.xml','scene_sha256'),(bank_name,'bank_sha256')]:
            if hashlib.sha256((assets/name).read_bytes()).hexdigest()!=cert[key]:raise ValueError('Tripod assets changed')
        sources.extend([f'{asset_rel}/certified_bank_manifest.json',f'{asset_rel}/{cert["certification_report"]}',
                        'experiments/wuji_hand2/palm_up_tripod_v3/build.py',
                        'experiments/wuji_hand2/palm_up_tripod_v3/refine.py',
                        'experiments/wuji_hand2/palm_up_tripod_v3/policy_seeds.py',
                        'experiments/wuji_hand2/palm_up_tripod_v3/certify_clearance.py',
                        f'{asset_rel}/all_link_contact_check.json'])
    if args.grasp_version == 'palm_up_v1':
        sources.extend(['experiments/wuji_hand2/certify_grasps.py',
                        f'{asset_rel}/grasp_bank_60s_certification.json'])
    if args.grasp_version == 'palm_forward_v1':
        certification = json.loads((assets/'certified_bank_manifest.json').read_text())
        if Path(certification['certification_report']).name != certification['certification_report']:
            raise ValueError('Certification report must be local')
        for name, key in [('scene.xml', 'scene_sha256'), (bank_name, 'bank_sha256')]:
            if hashlib.sha256((assets/name).read_bytes()).hexdigest() != certification[key]:
                raise ValueError('Certified assets changed: '+name)
        sources.extend(['experiments/wuji_hand2/certify_grasps.py',
                        f'{asset_rel}/certified_bank_manifest.json',
                        f'{asset_rel}/{certification["certification_report"]}'])
        sources.extend(['experiments/wuji_hand2/palm_forward_v1/config.json',
                        'experiments/wuji_hand2/palm_forward_v1/build.py',
                        'experiments/wuji_hand2/palm_forward_v1/run.py',
                        'experiments/wuji_hand2/palm_forward_v1/calibration_search_v2.json',
                        'experiments/wuji_hand2/palm_forward_v1/calibration_search_v3.json',
                        'experiments/wuji_hand2/palm_forward_v1/refine_controls.py',
                        'experiments/wuji_hand2/palm_forward_v1/control_refinement.json'])
    if args.experiment_config:
        config_path = (ROOT/args.experiment_config).resolve()
        sources.append(config_path.relative_to(ROOT).as_posix())
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in sources:
        dest=out/'snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    env=Env(args.num_envs,args.amplitude,args.period,args.seed,episode_seconds=args.episode_seconds,action_mode=args.action_mode,reward_mode=args.reward_mode,grasp_version=args.grasp_version,command_clip_penalty=args.command_clip_penalty,tracking_sigma=args.tracking_sigma,njmax=args.njmax,nconmax=args.nconmax,ring_profile=args.ring_profile,pinch_reward_weight=args.pinch_reward_weight,park_unused_fingers=args.park_unused_fingers)
    naction=env.naction
    obs=env.obs();agent=ActorCritic(obs.shape[-1],naction).cuda();opt=torch.optim.Adam(agent.parameters(),lr=args.lr)
    start=0
    if args.resume:
        c=torch.load(ROOT/args.resume,weights_only=False)
        for key in ['experiment_version','action_mode','grasp_version']:
            if c['config'][key]!=cfg[key]:raise ValueError('Incompatible resume: '+key)
        if c['config'].get('park_unused_fingers',0)!=args.park_unused_fingers:raise ValueError('Changed action space requires warm-start')
        for name in [f'{asset_rel}/scene.xml',f'{asset_rel}/{bank_name}']:
            if c['source_manifest'][name]!=manifest[name]:raise ValueError('Cannot resume after changing task assets')
        agent.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['iteration']
        for g in opt.param_groups:g['lr']=args.lr
        if c['config'].get('ring_profile','legacy') != args.ring_profile and not args.reset_ring_head:
            raise ValueError('Changed ring controller requires explicit reset-ring-head migration')
    if args.reset_ring_head:
        if not args.resume or args.ring_profile != 'natural_v1':raise ValueError('Ring migration requires resume and natural_v1')
        with torch.no_grad():
            agent.actor[-1].weight[12:16].zero_();agent.actor[-1].bias[12:16].zero_()
        opt=torch.optim.Adam(agent.parameters(),lr=args.lr)
    if args.warm_start:
        if args.resume or not args.park_unused_fingers:raise ValueError('Warm-start is only for fresh tripod migration')
        c=torch.load(ROOT/args.warm_start,weights_only=False)
        if c['config'].get('grasp_version')!='palm_up_v1' or c['nobs']!=105 or c.get('naction')!=20:
            raise ValueError('Warm-start requires a compatible 20-action palm-up policy')
        old_assets,_=assets_for_grasp('palm_up_v1')
        old_scene=str((old_assets/'scene.xml').relative_to(ROOT))
        if c['source_manifest'][old_scene]!=manifest[f'{asset_rel}/scene.xml']:raise ValueError('Warm-start changed geometry or orientation')
        state=c['model'].copy()
        for key,value in agent.state_dict().items():
            if state[key].shape!=value.shape:
                if key not in ['actor.6.weight','actor.6.bias','log_std']:raise ValueError('Unexpected migration shape: '+key)
                state[key]=state[key][:12].clone()
        agent.load_state_dict(state)
        (out/'warm_start_lineage.json').write_text(json.dumps(dict(parent=args.warm_start,parent_sha256=hashlib.sha256((ROOT/args.warm_start).read_bytes()).hexdigest(),
            source_actions=20,target_actions=12,optimizer_reset=True,bank_migration=True),indent=2))
    # Applied before fresh on-policy sampling; PPO rollout/update use the same distribution.
    with torch.no_grad():agent.log_std.clamp_(-3,float(np.log(args.max_policy_std)))
    H,N=args.horizon,args.num_envs;ob=torch.zeros(H,N,obs.shape[-1],device='cuda');ac=torch.zeros(H,N,naction,device='cuda')
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
