"""Sustained PPO with real resume, mixed curriculum and asynchronous CPU review."""
import argparse,copy,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .env import Env,ROOT
from train import ActorCritic


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());out=ROOT/config['runs'];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    source_files=list(Path(__file__).parent.glob('*.py'))+[Path(args.config).resolve(),ROOT/'train.py',
        ROOT/'experiments/wuji_hand2/palm_up_human_demo_v10/visualize.py',
        ROOT/'experiments/wuji_hand2/palm_up_human_demo_v10/PLAN.md',
        ROOT/'experiments/wuji_hand2/palm_up_human_demo_v10/overnight_launcher.sh',
        ROOT/'docs/EXPERIMENT_REVIEW_SOP.md',ROOT/'docs/DEXTEROUS_RL_TRAINING_SOP.md']
    assets=ROOT/config['assets'];source_files += [assets/'scene.xml',assets/'grasp_seed_bank.npz',assets/'provenance.json',assets/'pad_regions.json']
    initialization=ROOT/config['init_checkpoint']
    source_files.append(initialization)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    if not (out/'source_manifest.json').exists():
        atomic_json(out/'source_manifest.json',manifest)
        for p in source_files:
            dest=out/'snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    elif json.loads((out/'source_manifest.json').read_text())!=manifest:raise RuntimeError('Source/config changed: choose a new long-training run')
    env=Env(num_envs=config['num_envs'],pinch_reward_weight=1.5,command_clip_penalty=8.,tracking_sigma=.35,config=config)
    obs=env.obs();agent=ActorCritic(obs.shape[-1],12).cuda();opt=torch.optim.Adam(agent.parameters(),lr=config['lr']);start=0;active_at_start=0.
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
        if ck['source_manifest']!=manifest:raise RuntimeError('Checkpoint sources mismatch')
        agent.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['iteration'];env.curriculum_level=ck['curriculum_level']
        active_at_start=float(ck.get('active_training_seconds',0.))
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng'])
        env.reset(torch.ones(env.n,device='cuda',dtype=torch.bool));obs=env.obs()
    else:
        source=torch.load(initialization,map_location='cpu',weights_only=False)
        if source['nobs']!=obs.shape[-1] or source['naction']!=12:raise RuntimeError('Warm-start policy architecture mismatch')
        agent.load_state_dict(source['model'])
        env.curriculum_level=config['starting_level']
        with torch.no_grad():agent.log_std.fill_(np.log(config['initial_policy_std']))
        env.reset(torch.ones(env.n,device='cuda',dtype=torch.bool));obs=env.obs()
        atomic_json(out/'warm_start.json',dict(source=str(initialization.relative_to(ROOT)),sha256=manifest[str(initialization.relative_to(ROOT))],
            source_iteration=source['iteration'],source_curriculum_level=source['curriculum_level'],fresh_optimizer=True,
            initial_policy_std=config['initial_policy_std']))
    with torch.no_grad():agent.log_std.clamp_(np.log(config['min_policy_std']),np.log(config['max_policy_std']))
    lifecycle=json.loads((out/'lifecycle.json').read_text()) if (out/'lifecycle.json').exists() else dict(started_unix=time.time(),active_training_budget_seconds=config['active_training_hours']*3600,active_training_hours=config['active_training_hours'])
    atomic_json(out/'lifecycle.json',lifecycle)
    H,N=config['horizon'],config['num_envs'];ob=torch.zeros(H,N,obs.shape[-1],device='cuda');ac=torch.zeros(H,N,12,device='cuda')
    lp=torch.zeros(H,N,device='cuda');rw=torch.zeros_like(lp);dn=torch.zeros_like(lp);val=torch.zeros_like(lp)
    stopping=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def save(it,archive=True):
        payload=dict(model=agent.state_dict(),optimizer=opt.state_dict(),iteration=it,curriculum_level=env.curriculum_level,
            config=config,nobs=obs.shape[-1],naction=12,source_manifest=manifest,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),physics_resets_on_resume=True,
            active_training_seconds=active_seconds())
        tmp=out/'latest.pt.tmp';torch.save(payload,tmp);tmp.replace(checkpoint)
        if archive:
            dest=out/f'checkpoint_{it:07d}.pt';tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(dest)
            files=sorted(out.glob('checkpoint_*.pt'));recent=set(files[-config['checkpoint_keep_recent']:])
            for p in files:
                n=int(p.stem.split('_')[-1])
                if p not in recent and n%config['checkpoint_keep_stride']!=0:p.unlink()
    worker=None;worker_log=None;last_feedback=-1;regression_streak=0;early_stopped=False;started=time.time();it=start
    active_seconds=lambda:active_at_start+time.time()-started
    print('LONG_TRAIN_START',json.dumps(dict(pid=os.getpid(),resume_iteration=start,active_at_start=active_at_start,active_budget=lifecycle['active_training_budget_seconds'],config=config)),flush=True)
    def launch_evaluation(iteration,holdout=False):
        nonlocal worker,worker_log
        if worker is not None and worker.poll() is None:return
        if worker_log is not None:worker_log.close()
        ckpath=out/f'checkpoint_{iteration:07d}.pt'
        if not ckpath.exists():save(iteration)
        worker_log=(out/f'evaluate_{iteration:07d}.log').open('a')
        command=[sys.executable,'-m','experiments.wuji_hand2.palm_up_human_demo_v10.long_training.evaluate_pair','--checkpoint',str(ckpath)]
        if holdout:raise RuntimeError('V10 holdout is sealed pending human motion approval')
        worker=subprocess.Popen(command,cwd=ROOT,stdout=worker_log,stderr=subprocess.STDOUT,env=os.environ.copy())
    try:
        for it in range(start+1,config['max_iterations']+1):
            if active_seconds()>=lifecycle['active_training_budget_seconds']:
                it-=1
                break
            completed=[];diagnostics=[]
            with torch.no_grad():
                agent.update_stats(obs)
                for t in range(H):
                    mean,v=agent(obs);dist=torch.distributions.Normal(mean,agent.log_std.exp());action=dist.sample()
                    ob[t]=obs;ac[t]=action;lp[t]=dist.log_prob(action).sum(-1);val[t]=v
                    obs,reward,done,info=env.step(action)
                    terminal_v=agent(info['terminal_obs'])[1]
                    rw[t]=reward*config['reward_scale']+.99*terminal_v*info['timeout']*(~info['fail']);dn[t]=done
                    diagnostics.append(torch.stack([info['error'].abs().mean(),info['tilt'].mean(),info['drift'].mean(),info['command_clip'].mean(),
                        (info['other_force']>.05).float().mean(),(info['parked_force']>.02).float().mean(),info['posture_violation'].float().mean()]))
                    if done.any():completed.append(torch.stack([info['episode_return'][done],info['episode_error'][done],info['episode_length'][done],info['fail'][done].float()],-1))
                next_v=agent(obs)[1];adv=torch.zeros_like(rw);last=torch.zeros(N,device='cuda')
                for t in reversed(range(H)):
                    nv=next_v if t==H-1 else val[t+1];delta=rw[t]+.99*nv*(1-dn[t])-val[t]
                    last=delta+.99*.95*(1-dn[t])*last;adv[t]=last
                ret=adv+val
            fo,fa,flp,fr=ob.flatten(0,1),ac.flatten(0,1),lp.flatten(),ret.flatten();advantages=adv.flatten();advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            before=copy.deepcopy(agent.state_dict());before_opt=copy.deepcopy(opt.state_dict());losses=[];stop=False
            for epoch in range(4):
                for ids in torch.randperm(H*N,device='cuda').chunk(8):
                    mean,v=agent(fo[ids]);dist=torch.distributions.Normal(mean,agent.log_std.exp());logp=dist.log_prob(fa[ids]).sum(-1)
                    logratio=logp-flp[ids];ratio=logratio.exp();kl=((ratio-1)-logratio).mean()
                    if not torch.isfinite(kl) or kl.item()>config['target_kl']*2:stop=True;break
                    pl=-torch.minimum(ratio*advantages[ids],ratio.clamp(.8,1.2)*advantages[ids]).mean();vl=(v-fr[ids]).square().mean();entropy=dist.entropy().sum(-1).mean()
                    loss=pl+.5*vl-.003*entropy
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),1.);opt.step()
                    with torch.no_grad():agent.log_std.clamp_(np.log(config['min_policy_std']),np.log(config['max_policy_std']))
                    losses.append([loss.item(),pl.item(),vl.item(),entropy.item()])
                if stop:break
            with torch.no_grad():
                mean,v=agent(fo);dist=torch.distributions.Normal(mean,agent.log_std.exp());lr=dist.log_prob(fa).sum(-1)-flp;ratio=lr.exp();kl=((ratio-1)-lr).mean().item()
            rollback=not np.isfinite(kl) or kl>config['target_kl']*4
            if rollback:agent.load_state_dict(before);opt.load_state_dict(before_opt)
            old_lr=opt.param_groups[0]['lr'];new_lr=max(3e-5,old_lr*.7) if rollback or kl>config['target_kl']*1.5 else min(config['lr'],old_lr*1.02) if kl<config['target_kl']*.5 else old_lr
            for g in opt.param_groups:g['lr']=new_lr
            if it%10==0 or it==start+1:
                d=torch.stack(diagnostics).mean(0).cpu().tolist();ep=torch.cat(completed).mean(0).cpu().tolist() if completed else [None]*4
                row=dict(iteration=it,steps=it*H*N,elapsed_hours=active_seconds()/3600,active_training_seconds=active_seconds(),fps=round((it-start)*H*N/(time.time()-started)),
                    curriculum_level=env.curriculum_level,curriculum_name=config['levels'][env.curriculum_level]['name'],
                    return_mean=ep[0],episode_mae_rad=ep[1],episode_steps=ep[2],episode_failure_fraction=ep[3],
                    error_rad=d[0],tilt_deg=float(np.degrees(d[1])),drift_m=d[2],command_clip_fraction=d[3],non_sensor_fraction=d[4],inactive_fraction=d[5],posture_violation_fraction=d[6],
                    loss=np.mean(losses,axis=0).tolist() if losses else [],kl_proposed=kl,rollback=rollback,lr=new_lr,std=agent.log_std.exp().mean().item(),pid=os.getpid())
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(out/'training_status.json',row|dict(status='training',active_training_budget_seconds=lifecycle['active_training_budget_seconds'],heartbeat_unix=time.time()))
                print(json.dumps(row),flush=True)
            if it%config['save_every']==0:save(it)
            if it==config['first_evaluation'] or it%config['evaluate_every']==0:launch_evaluation(it)
            feedback=out/'evaluation_feedback.json'
            if it%100==0 and feedback.exists():
                f=json.loads(feedback.read_text())
                if f['iteration']>last_feedback:
                    last_feedback=f['iteration']
                    regressed=(f.get('standard_success_fraction',1)<20/24 or
                               f.get('posture_fail_episodes',0)>=2 or
                               f.get('full_duration_fail_episodes',0)>=4)
                    regression_streak=regression_streak+1 if regressed else 0
                    if regression_streak>=2:
                        print('EARLY_STOP_REPEATED_VALIDATION_REGRESSION',json.dumps(f),flush=True)
                        early_stopped=True
                        break
                    if f.get('promote',False) and f['level']==env.curriculum_level and env.curriculum_level<len(env.levels)-1:
                        env.curriculum_level+=1
                        with torch.no_grad():agent.obs_mean[[54,55,108,109,110]]=0;agent.obs_var[[54,55,108,109,110]]=1;agent.obs_count.fill_(1024)
                        print('CURRICULUM_PROMOTED',env.curriculum_level,flush=True)
                    else:print('EVALUATION_KEEP_TRAINING',json.dumps(f),flush=True)
            if stopping[0] or (out/'STOP').exists() or active_seconds()>=lifecycle['active_training_budget_seconds']:break
        save(it)
        status='stopped_by_request' if stopping[0] or (out/'STOP').exists() else 'stopped_validation_regression' if early_stopped else 'budget_finished'
        atomic_json(out/'training_status.json',dict(status=status,iteration=it,steps=it*H*N,curriculum_level=env.curriculum_level,finished_unix=time.time(),active_training_seconds=active_seconds()))
        if status=='budget_finished' and config.get('final_holdout_enabled',True):launch_evaluation(it,holdout=True)
    finally:
        # The supervisor resumes from the latest atomic checkpoint after crashes.
        if worker_log is not None:worker_log.close()


if __name__=='__main__':main()
