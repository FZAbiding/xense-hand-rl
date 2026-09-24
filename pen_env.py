"""GPU rigid-body LEAP/Revo3 pen task. Only hand joint motors are actuated.

MuJoCo Warp supplies contact dynamics; PyTorch supplies observations/rewards.
The pen has a six-DOF free joint with no imposed force, attachment or animation.
"""
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from hand_config import hand_config

ROOT=Path(__file__).resolve().parent


class PenEnv:
    def __init__(self,num_envs=1024,amplitude=.6,period=4.,episode_seconds=12.,seed=1,reference=False,robot='leap',objective='legacy'):
        wp.init()
        self.device='cuda'
        self.n=num_envs; self.dt=.025; self.amplitude=amplitude;self.period=period
        self.horizon=int(episode_seconds/self.dt);self.reference=reference
        torch.manual_seed(seed)
        self.robot=robot;self.spec=hand_config(robot);self.objective=objective
        self.cpu_model=mujoco.MjModel.from_xml_path(str(self.spec['assets']/'scene.xml'))
        self.naction=self.cpu_model.nu;self.nh=self.naction
        assert self.cpu_model.joint('pen_free').qposadr[0]==self.nh
        self.cpu_model.opt.timestep=.005
        self.cpu_model.opt.iterations=8
        self.cpu_model.opt.ls_iterations=8
        grasp=np.load(self.spec['assets']/'grasp.npz')
        cd=mujoco.MjData(self.cpu_model)
        cd.qpos[:]=grasp['qpos'];cd.ctrl[:]=grasp['ctrl'];mujoco.mj_forward(self.cpu_model,cd)
        self.model=mjw.put_model(self.cpu_model)
        self.data=mjw.put_data(self.cpu_model,cd,nworld=num_envs,nconmax=96 if robot=='revo3' else 64,njmax=240 if robot=='revo3' else 160)
        self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel)
        self.ctrl=wp.to_torch(self.data.ctrl)
        self.xmat=wp.to_torch(self.data.xmat)
        self.site_xpos=wp.to_torch(self.data.site_xpos)
        self.init_q=torch.tensor(grasp['qpos'],device='cuda',dtype=torch.float32)
        self.home=torch.tensor(grasp['ctrl'],device='cuda',dtype=torch.float32)
        self.low=torch.tensor(self.cpu_model.actuator_ctrlrange[:,0],device='cuda',dtype=torch.float32)
        self.high=torch.tensor(self.cpu_model.actuator_ctrlrange[:,1],device='cuda',dtype=torch.float32)
        self.pen_id=self.cpu_model.body('pen').id
        self.tip_ids=[self.cpu_model.site(n).id for n in self.spec['tips']]
        self.pen_geom=self.cpu_model.geom('pen_collision').id
        finger_ids=np.zeros(self.cpu_model.ngeom,dtype=np.int64)
        for i,name in enumerate(self.spec['geoms']):
            body=self.cpu_model.geom(name).bodyid[0]
            finger_ids[self.cpu_model.geom_bodyid==body]=i+1
        self.geom_finger=torch.tensor(finger_ids,device='cuda')
        axis=cd.xmat[self.pen_id].reshape(3,3)[:,2]
        self.yaw0=float(np.arctan2(axis[1],axis[0]))
        self.time=torch.zeros(num_envs,device='cuda')
        self.last_action=torch.zeros(num_envs,self.naction,device='cuda')
        self.prev_yaw=torch.full((num_envs,),self.yaw0,device='cuda')
        self.total_return=torch.zeros(num_envs,device='cuda')
        self.total_error=torch.zeros(num_envs,device='cuda')
        self.steps=torch.zeros(num_envs,device='cuda')
        self.max_angle=torch.zeros(num_envs,device='cuda');self.min_angle=torch.zeros_like(self.max_angle)
        if reference:
            ref=np.load(self.spec['assets']/'reference.npz')
            self.ref_angles=torch.tensor(ref['angles'],device='cuda',dtype=torch.float32)
            self.ref_q=torch.tensor(ref['ctrl'],device='cuda',dtype=torch.float32)
        # Compile kernels once before graph capture. Graph covers five physics steps.
        mjw.step(self.model,self.data)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(5):mjw.step(self.model,self.data)
        self.graph=capture.graph
        self.reset(torch.ones(num_envs,device='cuda',dtype=torch.bool),noise=False)

    def reset(self,mask,noise=True):
        q=self.init_q.expand(self.n,-1).clone()
        if noise:
            q[:,:self.nh]+=torch.randn(self.n,self.nh,device='cuda')*.005
            q[:,self.nh:self.nh+3]+=torch.randn(self.n,3,device='cuda')*.0004
        self.qpos[mask]=q[mask];self.qvel[mask]=0;self.ctrl[mask]=self.home
        # Clear warmstart acceleration to avoid impulses from previous episodes.
        wp.to_torch(self.data.qacc_warmstart)[mask]=0
        self.time[mask]=0;self.last_action[mask]=0;self.prev_yaw[mask]=self.yaw0
        self.total_return[mask]=0;self.total_error[mask]=0;self.steps[mask]=0
        self.max_angle[mask]=0;self.min_angle[mask]=0
        mjw.forward(self.model,self.data)

    def target(self):
        phase=self.time*(2*torch.pi/self.period)
        return self.amplitude*torch.sin(phase),self.amplitude*(2*torch.pi/self.period)*torch.cos(phase),phase

    def angle(self):
        # Quaternion convention is wxyz; pen long axis is local +z.
        w,x,y,z=self.qpos[:,self.nh+3:self.nh+7].unbind(-1)
        ax=2*(x*z+w*y);ay=2*(y*z-w*x);az=1-2*(x*x+y*y)
        yaw=torch.atan2(ay,ax)
        angle=torch.atan2(torch.sin(yaw-self.yaw0),torch.cos(yaw-self.yaw0))
        return angle,yaw,torch.stack((ax,ay,az),-1)

    def ref_action(self,target):
        if not self.reference:return self.home.expand(self.n,-1)
        idx=torch.searchsorted(self.ref_angles,target).clamp(1,len(self.ref_angles)-1)
        alpha=((target-self.ref_angles[idx-1])/(self.ref_angles[idx]-self.ref_angles[idx-1])).clamp(0,1)
        return self.ref_q[idx-1]*(1-alpha[:,None])+self.ref_q[idx]*alpha[:,None]

    def obs(self):
        goal,omega,phase=self.target();angle,_,axis=self.angle()
        h=self.nh
        return torch.cat(((self.qpos[:,:h]-self.home),self.qvel[:,:h]*.1,
            (self.qpos[:,h:h+3]-self.init_q[h:h+3])*20,axis,self.qvel[:,h:h+3],self.qvel[:,h+3:h+6]*.1,
            torch.stack((torch.sin(angle),torch.cos(angle),goal,omega,torch.sin(phase),torch.cos(phase)),-1),
            self.last_action),-1).clamp(-20,20)

    def contact_fingers(self):
        contact=self.data.contact
        pair=wp.to_torch(contact.geom).long().clamp(0,self.cpu_model.ngeom-1)
        worlds=wp.to_torch(contact.worldid).long().clamp(0,self.n-1)
        valid=torch.arange(pair.shape[0],device='cuda')<wp.to_torch(self.data.nacon)[0]
        has_pen=(pair[:,0]==self.pen_geom)|(pair[:,1]==self.pen_geom)
        valid=valid&has_pen&(wp.to_torch(contact.dist)<.001)
        other=torch.where(pair[:,0]==self.pen_geom,pair[:,1],pair[:,0])
        fingers=self.geom_finger[other];width=len(self.tip_ids)+1
        counts=torch.zeros(self.n*width,device='cuda')
        counts.scatter_add_(0,worlds*width+fingers,valid.float())
        counts=counts.reshape(self.n,width)
        return (counts[:,1:]>0).float().sum(-1),counts[:,0]

    def step(self,action,auto_reset=True):
        action=action.clamp(-1,1)
        goal,_,_=self.target()
        desired=self.ref_action(goal)+action*(.25 if self.reference else .65)
        self.ctrl[:]=desired.clamp(self.low,self.high)
        wp.capture_launch(self.graph)
        self.time+=self.dt;self.steps+=1
        goal,goal_vel,_=self.target();angle,yaw,axis=self.angle()
        yaw_vel=torch.atan2(torch.sin(yaw-self.prev_yaw),torch.cos(yaw-self.prev_yaw))/self.dt
        error=torch.atan2(torch.sin(angle-goal),torch.cos(angle-goal))
        h=self.nh
        drift=torch.linalg.vector_norm(self.qpos[:,h:h+3]-self.init_q[h:h+3],dim=-1)
        tip=self.site_xpos[:,self.tip_ids]
        offsets=tip-self.qpos[:,None,h:h+3]
        along=(offsets*axis[:,None,:]).sum(-1).clamp(-.075,.075)
        tip_dist=torch.linalg.vector_norm(offsets-along[:,:,None]*axis[:,None,:],dim=-1)
        near=(tip_dist<.024).float().sum(-1)
        fail=(self.qpos[:,h+2]<.015)|(drift>.075)|(~torch.isfinite(self.qpos).all(-1))
        if self.objective=='fingertip':
            fail=fail|(self.qpos[:,h+2]<self.init_q[h+2]-.018)|(drift>.045)
        timeout=self.steps>=self.horizon;done=fail|timeout
        reward=(2*torch.exp(-error.square()/.09)+.5*torch.exp(-(yaw_vel-goal_vel).square()/2)
                -12*drift-1.5*axis[:,2].square()-.008*(action-self.last_action).square().sum(-1)
                -.003*action.square().sum(-1)-.5*tip_dist.clamp(max=.08).mean(-1)
                -fail.float()*5)
        if self.objective=='fingertip':
            contact_count,other_contacts=self.contact_fingers()
            nearest_two=tip_dist.topk(2,largest=False).values
            proximity=torch.exp(-((nearest_two-.009).clamp(min=0)/.015).square()).mean(-1)
            reward=(2*torch.exp(-error.square()/.09)+.4*torch.exp(-(yaw_vel-goal_vel).square()/2)
                    +1.5*proximity+.5*contact_count.clamp(max=3)-.8*other_contacts.clamp(max=3)
                    -25*drift-2*axis[:,2].square()-.01*(action-self.last_action).square().sum(-1)
                    -.004*action.square().sum(-1)-fail.float()*8)
        self.last_action[:]=action;self.prev_yaw[:]=yaw
        self.total_return+=reward;self.total_error+=error.abs()
        self.max_angle=torch.maximum(self.max_angle,angle);self.min_angle=torch.minimum(self.min_angle,angle)
        terminal_obs=self.obs()
        info={'done':done,'fail':fail,'timeout':timeout,'angle':angle,'goal':goal,'error':error,
              'drift':drift,'near_tips':near,'yaw_vel':yaw_vel,'terminal_obs':terminal_obs,
              'episode_return':self.total_return.clone(),'episode_error':self.total_error/self.steps.clamp(min=1),
              'episode_length':self.steps.clone(),'angle_span':self.max_angle-self.min_angle}
        if auto_reset:self.reset(done)
        return self.obs(),reward,done,info
