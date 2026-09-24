"""Wuji Hand2 task using the same V2 reward, controls and PPO observations."""
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/wuji_hand2'
FINGERS=['thumb','index_finger','middle_finger','ring_finger','pinky']


@wp.kernel
def record_capacity_peaks(nefc:wp.array(dtype=wp.int32),nacon:wp.array(dtype=wp.int32),peaks:wp.array(dtype=wp.int32)):
    world=wp.tid()
    wp.atomic_max(peaks,0,nefc[world])
    if world==0:wp.atomic_max(peaks,1,nacon[0])


class Env:
    def __init__(self,num_envs=1024,amplitude=.2,period=6.,seed=1,episode_seconds=12.,
                 action_mode='delta',reward_mode='quality',grasp_version='v2',command_clip_penalty=.2,tracking_sigma=.18,
                 njmax=512,nconmax=128):
        wp.init();torch.manual_seed(seed)
        self.n=num_envs;self.nh=self.naction=20;self.dt=.025
        self.amplitude=amplitude;self.period=period;self.horizon=round(episode_seconds/self.dt)
        self.action_mode=action_mode;self.reward_mode=reward_mode;self.grasp_version=grasp_version
        self.command_clip_penalty=command_clip_penalty
        if tracking_sigma<=0:raise ValueError('tracking_sigma must be positive')
        self.tracking_sigma=tracking_sigma
        self.cpu_model=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'))
        self.cpu_model.opt.iterations=8;self.cpu_model.opt.ls_iterations=8
        m=self.cpu_model;cd=mujoco.MjData(m)
        bank_name = 'grasp_bank_long_initial.npz' if grasp_version == 'long12' else 'grasp_bank.npz'
        bank=np.load(ASSETS/bank_name);bq=bank['qpos'];bc=bank['ctrl']
        self.bank_q=torch.tensor(bq,device='cuda',dtype=torch.float32)
        self.bank_ctrl=torch.tensor(bc,device='cuda',dtype=torch.float32)
        self.pen_id=m.body('pen').id;self.pen_geom=m.geom('pen_collision').id
        self.tip_ids=[m.site(f'r_{f}_tip').id for f in FINGERS]
        self.palm_body=m.body('r_wrist').id
        mapping=np.zeros(m.ngeom,np.int64)
        for i,f in enumerate(FINGERS):
            for suffix in ('distal','tip_sensor_frame'):
                mapping[m.geom_bodyid==m.body(f'r_{f}_{suffix}').id]=i+1
        self.geom_finger=torch.tensor(mapping,device='cuda');self.geom_body=torch.tensor(m.geom_bodyid,device='cuda',dtype=torch.long)
        self.low=torch.tensor(m.actuator_ctrlrange[:,0],device='cuda',dtype=torch.float32)
        self.high=torch.tensor(m.actuator_ctrlrange[:,1],device='cuda',dtype=torch.float32)
        self.safe_low=self.low+torch.minimum((self.high-self.low)*.05,torch.full_like(self.low,np.radians(2)))
        self.safe_high=self.high-(self.safe_low-self.low)
        self.delta_scale=((self.high-self.low)*.03).clamp(.012,.035)
        self.delta_scale[12:]*=.5
        self.model=mjw.put_model(m);cd.qpos[:]=bq[0];cd.ctrl[:]=bc[0];mujoco.mj_forward(m,cd)
        self.data=mjw.put_data(m,cd,nworld=num_envs,nconmax=nconmax,njmax=njmax)
        self.capacity_peaks=wp.zeros(2,dtype=wp.int32,device='cuda')
        self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel);self.ctrl=wp.to_torch(self.data.ctrl)
        self.sites=wp.to_torch(self.data.site_xpos)
        self.home=torch.zeros(num_envs,20,device='cuda');self.center=torch.zeros(num_envs,3,device='cuda')
        self.yaw0=torch.zeros(num_envs,device='cuda');self.time=torch.zeros_like(self.yaw0);self.steps=torch.zeros_like(self.yaw0)
        self.last_action=torch.zeros(num_envs,20,device='cuda');self.prev_yaw=torch.zeros_like(self.yaw0)
        self.forces=torch.zeros(num_envs,7,device='cuda')
        self.total_return=torch.zeros_like(self.yaw0);self.total_error=torch.zeros_like(self.yaw0)
        cap=wp.to_torch(self.data.contact.geom).shape[0]
        self.contact_ids=wp.array(np.arange(cap,dtype=np.int32),dtype=wp.int32,device='cuda')
        self.contact_wrench=wp.zeros(cap,dtype=wp.spatial_vector,device='cuda')
        self.contact_index=torch.arange(cap,device='cuda')
        mjw.step(self.model,self.data);self.record_capacity();self.check_capacity();wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(5):
                mjw.step(self.model,self.data);self.record_capacity()
        self.graph=capture.graph
        self.reset(torch.ones(num_envs,dtype=torch.bool,device='cuda'),noise=False)

    def record_capacity(self):
        wp.launch(record_capacity_peaks,dim=self.n,inputs=[self.data.nefc,self.data.nacon,self.capacity_peaks])

    def check_capacity(self):
        nefc,nacon=wp.to_torch(self.capacity_peaks).cpu().tolist()
        if nefc>=self.data.njmax or nacon>=self.data.naconmax:
            raise RuntimeError(f'Simulation capacity exhausted: constraints {nefc}/{self.data.njmax}, contacts {nacon}/{self.data.naconmax}; results invalid, increase buffers.')
        return dict(peak_constraints_per_world=nefc,constraint_capacity_per_world=self.data.njmax,
                    peak_contacts_all_worlds=nacon,contact_capacity_all_worlds=self.data.naconmax)

    def axis(self):
        w,x,y,z=self.qpos[:,23:27].unbind(-1)
        return torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),-1)

    def angle(self):
        axis=self.axis();yaw=torch.atan2(axis[:,1],axis[:,0]);angle=torch.atan2(torch.sin(yaw-self.yaw0),torch.cos(yaw-self.yaw0))
        return angle,yaw,axis

    def contact_info(self):
        mjw.contact_force(self.model,self.data,self.contact_ids,False,self.contact_wrench)
        c=self.data.contact;pair=wp.to_torch(c.geom).long().clamp(0,self.cpu_model.ngeom-1)
        world=wp.to_torch(c.worldid).long().clamp(0,self.n-1)
        valid=(self.contact_index<wp.to_torch(self.data.nacon)[0])&((pair[:,0]==self.pen_geom)|(pair[:,1]==self.pen_geom))
        other=torch.where(pair[:,0]==self.pen_geom,pair[:,1],pair[:,0]);finger=self.geom_finger[other]
        fn=torch.where(valid,wp.to_torch(self.contact_wrench)[:,0].clamp(min=0),0.)
        sums=torch.zeros(self.n*6,device='cuda');sums.scatter_add_(0,world*6+finger,fn)
        sums=sums.reshape(self.n,6)
        palm=torch.zeros(self.n,device='cuda');palm.scatter_add_(0,world,fn*(self.geom_body[other]==self.palm_body))
        self.forces[:,:5]=sums[:,1:];self.forces[:,5]=sums[:,0];self.forces[:,6]=palm
        return (sums[:,1:]>.02).float().sum(-1),sums.sum(-1),sums[:,0],palm

    def reset(self,mask,noise=True):
        if not mask.any():return
        ids=torch.randint(len(self.bank_q),(self.n,),device='cuda')
        q=self.bank_q[ids].clone();ctrl=self.bank_ctrl[ids]
        self.home[mask]=ctrl[mask];self.center[mask]=q[mask,20:23]
        w,x,y,z=q[:,23:27].unbind(-1);yaw=torch.atan2(2*(y*z-w*x),2*(x*z+w*y))
        self.yaw0[mask]=yaw[mask];self.prev_yaw[mask]=yaw[mask]
        if noise:q[:,:20]+=torch.randn(self.n,20,device='cuda')*.005;q[:,20:23]+=torch.randn(self.n,3,device='cuda')*.0004
        self.qpos[mask]=q[mask];self.qvel[mask]=0;self.ctrl[mask]=ctrl[mask]
        wp.to_torch(self.data.qacc_warmstart)[mask]=0
        self.time[mask]=0;self.steps[mask]=0;self.last_action[mask]=0
        self.total_return[mask]=0;self.total_error[mask]=0
        mjw.forward(self.model,self.data);self.record_capacity();self.check_capacity();self.contact_info()

    def obs(self):
        angle,_,axis=self.angle();phase=self.time*(2*torch.pi/self.period)
        goal=self.amplitude*torch.sin(phase);omega=self.amplitude*2*torch.pi/self.period*torch.cos(phase)
        base=torch.cat((self.qpos[:,:20]-self.home,self.qvel[:,:20]*.1,(self.qpos[:,20:23]-self.center)*20,
                        axis,self.qvel[:,20:23],self.qvel[:,23:26]*.1,
                        torch.stack((torch.sin(angle),torch.cos(angle),goal,omega,torch.sin(phase),torch.cos(phase)),-1),self.last_action),-1)
        # Extra command state makes integrated actions Markov; forces are privileged simulator observations.
        command=2*(self.ctrl-self.low)/(self.high-self.low)-1
        return torch.cat((base,command,self.forces*.2),-1).clamp(-20,20)

    def step(self,raw_action,auto_reset=True):
        action=raw_action.clamp(-1,1)
        if self.action_mode=='delta':
            desired=self.ctrl+action*self.delta_scale;command=desired.clamp(self.safe_low,self.safe_high)
        else:
            desired=self.home+.65*action;command=desired.clamp(self.low,self.high)
        clipped=(abs(command-desired)>1e-7).float().mean(-1);self.ctrl[:]=command
        wp.capture_launch(self.graph);self.check_capacity();self.time+=self.dt;self.steps+=1
        tips,force,other,palm=self.contact_info()
        angle,yaw,axis=self.angle();phase=self.time*2*torch.pi/self.period
        goal=self.amplitude*torch.sin(phase);omega=self.amplitude*2*torch.pi/self.period*torch.cos(phase)
        error=torch.atan2(torch.sin(angle-goal),torch.cos(angle-goal))
        yaw_vel=torch.atan2(torch.sin(yaw-self.prev_yaw),torch.cos(yaw-self.prev_yaw))/self.dt
        tilt=torch.asin(axis[:,2].abs().clamp(0,1));drift=torch.linalg.vector_norm(self.qpos[:,20:23]-self.center,dim=-1)
        offset=self.sites[:,self.tip_ids]-self.qpos[:,None,20:23]
        dist=torch.linalg.vector_norm(offset-(offset*axis[:,None,:]).sum(-1).clamp(-.075,.075)[:,:,None]*axis[:,None,:],dim=-1)
        fail=(drift>.045)|(self.qpos[:,22]<self.center[:,2]-.025)|(tilt>np.radians(40))|(~torch.isfinite(self.qpos).all(-1))
        timeout=self.steps>=self.horizon;done=fail|timeout
        terms={
            'tracking':4*torch.exp(-(error/self.tracking_sigma).square()),
            'velocity':.5*torch.exp(-(yaw_vel-omega).square()),
            'horizontal':1.5*torch.exp(-(tilt/.15).square())-4*(axis[:,2].abs()-.174).clamp(min=0).square(),
            'support':.6*tips.clamp(max=2)/2,
            'workspace':-20*drift,
            'non_tip':-1.5*other.clamp(max=3)-2*palm.clamp(max=3),
            'squeeze':-.03*(force-2).clamp(min=0,max=20).square(),
            'control':-.002*action.square().sum(-1)-.005*(action-self.last_action).square().sum(-1)-self.command_clip_penalty*clipped,
            'failure':-8*fail.float(),
        }
        reward=sum(terms.values())
        if self.reward_mode=='fingertip':
            near2=dist.topk(2,largest=False).values
            reward=2*torch.exp(-error.square()/.09)+.4*torch.exp(-(yaw_vel-omega).square()/2)
            reward+=1.5*torch.exp(-((near2-.009).clamp(min=0)/.015).square()).mean(-1)+.5*tips.clamp(max=3)
            reward+=-1.5*other.clamp(max=3)-25*drift-2*axis[:,2].square()-.01*(action-self.last_action).square().sum(-1)-.004*action.square().sum(-1)-8*fail.float()
        self.total_return+=reward;self.total_error+=error.abs()
        self.last_action[:]=action;self.prev_yaw[:]=yaw
        info=dict(fail=fail,timeout=timeout,error=error,angle=angle,goal=goal,tilt=tilt,drift=drift,tips=tips,
                  normal_force=force,other_force=other,palm_force=palm,command_clip=clipped,
                  episode_return=self.total_return.clone(),episode_error=self.total_error/self.steps.clamp(min=1),episode_length=self.steps.clone(),
                  terminal_obs=self.obs(),reward_terms=terms)
        if auto_reset:self.reset(done)
        return self.obs(),reward,done,info
