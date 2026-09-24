"""PPO training with GPU contact simulation, checkpointing and measured logs."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pen_env import PenEnv

class ActorCritic(nn.Module):
    def __init__(self,nobs,naction=16):
        super().__init__()
        def net(out):
            return nn.Sequential(nn.Linear(nobs,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
        self.actor=net(naction);self.critic=net(1)
        self.log_std=nn.Parameter(torch.full((naction,),-1.5))
        self.register_buffer('obs_mean',torch.zeros(nobs));self.register_buffer('obs_var',torch.ones(nobs));self.register_buffer('obs_count',torch.tensor(1e-4))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)

    @torch.no_grad()
    def update_stats(self,x):
        mean=x.mean(0);var=x.var(0,unbiased=False);count=x.shape[0]
        delta=mean-self.obs_mean;total=self.obs_count+count
        self.obs_var.copy_((self.obs_var*self.obs_count+var*count+delta.square()*self.obs_count*count/total)/total)
        self.obs_mean.add_(delta*count/total);self.obs_count.copy_(total)

    def forward(self,obs):
        x=((obs-self.obs_mean)/torch.sqrt(self.obs_var+1e-5)).clamp(-10,10)
        return self.actor(x),self.critic(x).squeeze(-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/ppo_direct');p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--iterations',type=int,default=1000);p.add_argument('--amplitude',type=float,default=.6);p.add_argument('--period',type=float,default=4.)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--resume');p.add_argument('--reference',action='store_true');p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--robot',choices=['leap','revo3'],default='leap')
    p.add_argument('--objective',choices=['legacy','fingertip'],default='legacy')
    args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.run);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    env=PenEnv(args.num_envs,args.amplitude,args.period,seed=args.seed,reference=args.reference,robot=args.robot,objective=args.objective)
    obs=env.obs();agent=ActorCritic(obs.shape[-1],env.naction).cuda();optimizer=torch.optim.Adam(agent.parameters(),lr=args.lr)
    start=0
    if args.resume:
        ckpt=torch.load(args.resume,weights_only=False)
        if ckpt['config'].get('robot','leap')!=args.robot:raise ValueError('Cannot resume a policy for a different hand')
        agent.load_state_dict(ckpt['model']);start=ckpt['iteration']
        if 'optimizer' in ckpt:optimizer.load_state_dict(ckpt['optimizer'])
        for group in optimizer.param_groups:group['lr']=args.lr
    H,N=args.horizon,args.num_envs
    ob=torch.zeros(H,N,obs.shape[-1],device='cuda');ac=torch.zeros(H,N,env.naction,device='cuda')
    lp=torch.zeros(H,N,device='cuda');rw=torch.zeros_like(lp);dn=torch.zeros_like(lp);val=torch.zeros_like(lp)
    t0=time.time();history=[]
    def save(iteration,name):
        torch.save(dict(model=agent.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,config=vars(args),nobs=obs.shape[-1],naction=env.naction),out/name)
    print('TRAIN_START',vars(args),'obs',obs.shape[-1],flush=True)
    for iteration in range(start+1,start+args.iterations+1):
        completed=[]
        with torch.no_grad():
            # Keep normalization fixed within each rollout and its PPO update.
            agent.update_stats(obs)
            for t in range(H):
                mean,value=agent(obs);dist=torch.distributions.Normal(mean,agent.log_std.exp())
                action=dist.sample();logp=dist.log_prob(action).sum(-1)
                ob[t]=obs;ac[t]=action;lp[t]=logp;val[t]=value
                obs,reward,done,info=env.step(action)
                # Time-limit truncations bootstrap from terminal state; falls do not.
                _,terminal_v=agent(info['terminal_obs'])
                rw[t]=reward+.99*terminal_v*info['timeout']*(~info['fail'])
                dn[t]=done
                if done.any():
                    mask=done
                    completed.append(torch.stack((info['episode_return'][mask],info['episode_error'][mask],info['episode_length'][mask],info['angle_span'][mask],info['fail'][mask].float()),-1))
            _,next_value=agent(obs);adv=torch.zeros_like(rw);last=torch.zeros(N,device='cuda')
            for t in reversed(range(H)):
                nextv=next_value if t==H-1 else val[t+1]
                delta=rw[t]+.99*nextv*(1-dn[t])-val[t]
                last=delta+.99*.95*(1-dn[t])*last;adv[t]=last
            returns=adv+val
        flatob=ob.flatten(0,1);flatac=ac.flatten(0,1);flatlp=lp.flatten();flatadv=adv.flatten();flatret=returns.flatten()
        flatadv=(flatadv-flatadv.mean())/(flatadv.std()+1e-8)
        losslog=[];klval=0.
        for epoch in range(4):
            order=torch.randperm(H*N,device='cuda')
            for ids in order.chunk(8):
                mean,value=agent(flatob[ids]);dist=torch.distributions.Normal(mean,agent.log_std.clamp(-3,0).exp())
                logp=dist.log_prob(flatac[ids]).sum(-1);ratio=(logp-flatlp[ids]).exp()
                policy=-torch.minimum(ratio*flatadv[ids],ratio.clamp(.8,1.2)*flatadv[ids]).mean()
                value_loss=(value-flatret[ids]).square().mean()
                entropy=dist.entropy().sum(-1).mean()
                loss=policy+.5*value_loss-.002*entropy
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(agent.parameters(),1.);optimizer.step()
                with torch.no_grad():agent.log_std.clamp_(-3,0)
                klval=((ratio-1)-(logp-flatlp[ids])).mean().item();losslog.append(loss.item())
            if klval>.025:break
        if completed:
            a=torch.cat(completed).mean(0).cpu().tolist()
            history.append(a)
        if iteration%10==0 or iteration==start+1:
            avg=np.mean(history[-30:],axis=0).tolist() if history else [0]*5
            row=dict(iteration=iteration,steps=(iteration-start)*H*N,seconds=round(time.time()-t0,1),fps=round((iteration-start)*H*N/(time.time()-t0)),
                return_mean=avg[0],error_rad=avg[1],episode_steps=avg[2],angle_span_rad=avg[3],fall_fraction=avg[4],loss=np.mean(losslog),kl=klval,std=agent.log_std.exp().mean().item())
            print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if iteration%args.save_every==0:save(iteration,f'checkpoint_{iteration:06d}.pt');save(iteration,'latest.pt')
    save(iteration,'latest.pt')

if __name__=='__main__':main()
