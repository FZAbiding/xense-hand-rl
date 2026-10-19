"""V7 long-training GPU environment; code scaffold from V4, no V4 policy or bank."""
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from experiments.wuji_hand2.ring_control import natural_ring_command, RING_STEP, RING_REST
from experiments.wuji_hand2.pinch import opposing_strength
from .contact import sensor_body_channels, distal_regions
from .guard import check_posture
LOW_DEG=np.array([-30,-75,5,5, 5,-12,5,5, 5,-12,5,5])
HIGH_DEG=np.array([70,35,80,80, 80,12,80,80, 80,12,60,80])

ROOT=Path(__file__).resolve().parents[4]
ASSETS=ROOT/'assets/wuji_hand2'
PALM_UP_ASSETS=ROOT/'assets/wuji_hand2_palm_up'


def assets_for_grasp(grasp_version):
    if grasp_version != 'palm_up_human_demo_v10':raise ValueError('V7 assets only')
    return ROOT/'assets/wuji_hand2_palm_up_human_demo_v10', 'grasp_seed_bank.npz'
FINGERS=['thumb','index_finger','middle_finger','ring_finger','pinky']


@wp.kernel
def record_capacity_peaks(nefc:wp.array(dtype=wp.int32),nacon:wp.array(dtype=wp.int32),peaks:wp.array(dtype=wp.int32)):
    world=wp.tid()
    wp.atomic_max(peaks,0,nefc[world])
    if world==0:wp.atomic_max(peaks,1,nacon[0])


class Env:
    def __init__(self,num_envs=1024,amplitude=.2,period=6.,seed=1,episode_seconds=12.,
                 action_mode='delta',reward_mode='quality',grasp_version='palm_up_human_demo_v10',command_clip_penalty=.2,tracking_sigma=.18,
                 njmax=512,nconmax=128,ring_profile='legacy',pinch_reward_weight=0.,park_unused_fingers=1,config=None):
        import json
        self.training_config=config or json.loads(Path(__file__).with_name('config.json').read_text())
        self.levels=self.training_config['levels'];self.curriculum_level=0
        self.reward_variant=self.training_config.get('reward_variant','combined')
        self.curriculum_mode=self.training_config.get('curriculum_mode','progressive')
        if self.reward_variant not in ('course_only','contact_only','combined'):
            raise ValueError('Unknown reward variant')
        if self.curriculum_mode not in ('legacy','progressive'):
            raise ValueError('Unknown curriculum mode')
        self.contact_reward_enabled=self.reward_variant in ('contact_only','combined')
        self.grip_gap_allow_steps=int(self.training_config.get('grip_gap_allow_steps',3))
        self.grip_gap_fail_steps=int(self.training_config.get('grip_gap_fail_steps',12))
        wp.init();torch.manual_seed(seed)
        self.n=num_envs;self.nh=self.naction=20;self.dt=.025
        self.park_unused_fingers=bool(park_unused_fingers)
        if self.park_unused_fingers:
            if grasp_version!='palm_up_human_demo_v10' or action_mode!='delta' or ring_profile!='legacy':
                raise ValueError('Parked fingers require the certified tripod scene, delta control and legacy ring profile')
            self.naction=12
        self.amplitude=amplitude;self.period=period;self.horizon=round(episode_seconds/self.dt)
        self.action_mode=action_mode;self.reward_mode=reward_mode;self.grasp_version=grasp_version
        self.command_clip_penalty=command_clip_penalty
        if ring_profile not in ('legacy', 'natural_v1'):raise ValueError('Unknown ring profile')
        if ring_profile != 'legacy' and action_mode != 'delta':raise ValueError('Natural ring requires delta control')
        self.ring_profile=ring_profile
        self.pinch_reward_weight=pinch_reward_weight
        if pinch_reward_weight<0:raise ValueError('pinch weight must be nonnegative')
        self.pinch_strength=torch.zeros(num_envs,device='cuda')
        self.policy_pinch_strength=torch.zeros(num_envs,device='cuda')
        self.parked_force=torch.zeros(num_envs,device='cuda')
        if tracking_sigma<=0:raise ValueError('tracking_sigma must be positive')
        self.tracking_sigma=tracking_sigma
        assets,bank_name=assets_for_grasp(grasp_version)
        self.cpu_model=mujoco.MjModel.from_xml_path(str(assets/'scene.xml'))
        self.cpu_model.opt.iterations=10;self.cpu_model.opt.ls_iterations=8
        m=self.cpu_model;cd=mujoco.MjData(m)
        bank=np.load(assets/bank_name);bq=bank['qpos'];bc=bank['ctrl']
        self.bank_q=torch.tensor(bq,device='cuda',dtype=torch.float32)
        self.bank_ctrl=torch.tensor(bc,device='cuda',dtype=torch.float32)
        self.pen_id=m.body('pen').id;self.pen_geom=m.geom('pen_collision').id
        self.tip_ids=[m.site(f'r_{f}_tip').id for f in FINGERS]
        self.palm_body=m.body('r_wrist').id
        mapping=np.zeros(m.ngeom,np.int64)
        for body_id,channel in sensor_body_channels(m).items():
            mapping[m.geom_bodyid==body_id]=channel+1
        legacy_mapping=mapping.copy()
        for i,finger in enumerate(FINGERS):
            legacy_mapping[m.geom_bodyid==m.body(f'r_{finger}_distal').id]=i+1
        self.geom_finger=torch.tensor(mapping,device='cuda');self.geom_body=torch.tensor(m.geom_bodyid,device='cuda',dtype=torch.long)
        self.pad_distal_regions=[(body,channel+1,torch.as_tensor(centers,device='cuda',dtype=torch.float32),radius_sq)
                                 for body,(channel,centers,radius_sq) in distal_regions(m).items()]
        self.geom_legacy_finger=torch.tensor(legacy_mapping,device='cuda')
        self.geom_parked=torch.tensor([('ring_finger' in m.body(int(b)).name or 'pinky' in m.body(int(b)).name) for b in m.geom_bodyid],device='cuda')
        self.low=torch.tensor(m.actuator_ctrlrange[:,0],device='cuda',dtype=torch.float32)
        self.high=torch.tensor(m.actuator_ctrlrange[:,1],device='cuda',dtype=torch.float32)
        self.safe_low=self.low+torch.minimum((self.high-self.low)*.05,torch.full_like(self.low,np.radians(2)))
        self.safe_high=self.high-(self.safe_low-self.low)
        self.safe_low[:12]=torch.as_tensor(np.radians(LOW_DEG),device='cuda',dtype=torch.float32)
        self.safe_high[:12]=torch.as_tensor(np.radians(HIGH_DEG),device='cuda',dtype=torch.float32)
        self.delta_scale=torch.full_like(self.low,.012)
        self.delta_scale[12:]*=.5
        if self.park_unused_fingers:self.delta_scale[12:]=0
        if self.ring_profile == 'natural_v1':self.delta_scale[12:16].clamp_(max=RING_STEP)
        self.model=mjw.put_model(m);cd.qpos[:]=bq[0];cd.ctrl[:]=bc[0];mujoco.mj_forward(m,cd)
        self.data=mjw.put_data(m,cd,nworld=num_envs,nconmax=nconmax,njmax=njmax)
        self.capacity_peaks=wp.zeros(2,dtype=wp.int32,device='cuda')
        self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel);self.ctrl=wp.to_torch(self.data.ctrl)
        self.sites=wp.to_torch(self.data.site_xpos)
        self.home=torch.zeros(num_envs,20,device='cuda');self.center=torch.zeros(num_envs,3,device='cuda')
        self.yaw0=torch.zeros(num_envs,device='cuda');self.time=torch.zeros_like(self.yaw0);self.steps=torch.zeros_like(self.yaw0)
        self.last_action=torch.zeros(num_envs,20,device='cuda');self.prev_yaw=torch.zeros_like(self.yaw0)
        self.unwrapped_angle=torch.zeros_like(self.yaw0)
        self.middle_pip_min=torch.zeros_like(self.yaw0);self.middle_pip_max=torch.zeros_like(self.yaw0)
        self.forces=torch.zeros(num_envs,7,device='cuda')
        self.policy_forces=torch.zeros_like(self.forces)
        self.total_return=torch.zeros_like(self.yaw0);self.total_error=torch.zeros_like(self.yaw0)
        self.target_theta=torch.zeros_like(self.yaw0);self.rotation_duration=torch.full_like(self.yaw0,4.)
        self.episode_limit=torch.full_like(self.yaw0,480);self.continuous=torch.zeros(num_envs,device='cuda',dtype=torch.bool)
        self.sampled_level=torch.zeros(num_envs,device='cuda',dtype=torch.long)
        self.push_enabled=torch.zeros(num_envs,device='cuda',dtype=torch.bool)
        self.push_vector=torch.zeros(num_envs,2,device='cuda')
        self.missing_grip_steps=torch.zeros(num_envs,device='cuda',dtype=torch.int32)
        self.guard_bad=wp.zeros(num_envs,dtype=wp.int32,device='cuda');self.guard_gap=wp.full(num_envs,1.,device='cuda')
        names=['proximal','proximal_abd','middle','distal']
        self.guard_bodies=wp.array([m.body(f'r_{f}_{s}').id for f in ['index_finger','middle_finger'] for s in names],dtype=wp.int32,device='cuda')
        self.order_tip_ids=[m.site(f'r_{f}_tip').id for f in ['index_finger','middle_finger']]
        self.guard_tips=wp.array(self.order_tip_ids,dtype=wp.int32,device='cuda')
        self.geom_active=torch.tensor([any(f'r_{f}' in m.body(int(b)).name for f in FINGERS[:3]) for b in m.geom_bodyid],device='cuda')
        self.validated_initial_bank=True
        cap=wp.to_torch(self.data.contact.geom).shape[0]
        self.contact_ids=wp.array(np.arange(cap,dtype=np.int32),dtype=wp.int32,device='cuda')
        self.contact_wrench=wp.zeros(cap,dtype=wp.spatial_vector,device='cuda')
        self.contact_index=torch.arange(cap,device='cuda')
        mjw.step(self.model,self.data);self.record_capacity();self.check_capacity();wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(5):
                mjw.step(self.model,self.data);mjw.kinematics(self.model,self.data);self.record_capacity();self.record_guard()
        self.graph=capture.graph
        self.reset(torch.ones(num_envs,dtype=torch.bool,device='cuda'),noise=False)

    def record_guard(self):
        wp.launch(check_posture,dim=self.n,inputs=[self.data.xpos,self.data.xmat,self.data.site_xpos,self.data.qpos,
            self.guard_bodies,self.guard_tips,self.palm_body,self.guard_bad,self.guard_gap])

    def goal_state(self):
        elapsed=(self.time-.5).clamp(min=0);fraction=elapsed/self.rotation_duration
        progress=torch.where(self.continuous,fraction,fraction.clamp(max=1))
        goal=self.target_theta*progress
        moving=(self.time>.5)&(self.continuous|(fraction<1))
        omega=torch.where(moving,self.target_theta/self.rotation_duration,0.)
        return goal,omega,fraction*2*torch.pi

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
        positions=wp.to_torch(c.pos)
        body=self.geom_body[other];body_pos=wp.to_torch(self.data.xpos)[world,body]
        rotations=wp.to_torch(self.data.xmat)[world,body]
        local=torch.bmm(rotations.transpose(1,2),(positions-body_pos).unsqueeze(-1)).squeeze(-1)
        for distal_body,channel,centers,radius_sq in self.pad_distal_regions:
            in_region=((local[:,None,:]-centers[None,:,:]).square().sum(-1)<=radius_sq).any(-1)
            finger=torch.where(valid&(body==distal_body)&in_region,channel,finger)
        sums=torch.zeros(self.n*6,device='cuda');sums.scatter_add_(0,world*6+finger,fn)
        sums=sums.reshape(self.n,6)
        legacy_finger=self.geom_legacy_finger[other]
        legacy_sums=torch.zeros(self.n*6,device='cuda')
        legacy_sums.scatter_add_(0,world*6+legacy_finger,fn)
        legacy_sums=legacy_sums.reshape(self.n,6)
        palm=torch.zeros(self.n,device='cuda');palm.scatter_add_(0,world,fn*(self.geom_body[other]==self.palm_body))
        self.forces[:,:5]=sums[:,1:];self.forces[:,5]=sums[:,0];self.forces[:,6]=palm
        self.policy_forces[:,:5]=legacy_sums[:,1:];self.policy_forces[:,5]=legacy_sums[:,0];self.policy_forces[:,6]=palm
        local_y=((positions-body_pos)*rotations[:,:,1]).sum(-1)
        side=valid&(finger>=1)&(finger<=3)&(local_y<.001)&(fn>.02)
        self.side_pad_force=torch.zeros(self.n,device='cuda').scatter_add_(0,world,fn*side)
        if self.park_unused_fingers:
            self.parked_force.zero_();self.parked_force.scatter_add_(0,world,fn*self.geom_parked[other])
        if self.pinch_reward_weight:
            normals=wp.to_torch(c.frame)[:,0,:]
            sign=torch.where(pair[:,1]==self.pen_geom,1.,-1.)
            vectors=torch.zeros(self.n*6,3,device='cuda')
            vectors.index_add_(0,world*6+finger,normals*(fn*sign)[:,None])
            finger_vectors=vectors.reshape(self.n,6,3)[:,1:]
            if self.park_unused_fingers:finger_vectors[:,3:]=0
            self.pinch_strength[:]=opposing_strength(finger_vectors)
            legacy_vectors=torch.zeros(self.n*6,3,device='cuda')
            legacy_vectors.index_add_(0,world*6+legacy_finger,normals*(fn*sign)[:,None])
            legacy_finger_vectors=legacy_vectors.reshape(self.n,6,3)[:,1:]
            if self.park_unused_fingers:legacy_finger_vectors[:,3:]=0
            self.policy_pinch_strength[:]=opposing_strength(legacy_finger_vectors)
        active=sums[:,1:4] if self.park_unused_fingers else sums[:,1:]
        return (active>.02).float().sum(-1),sums.sum(-1),sums[:,0],palm

    def reset(self,mask,noise=True):
        if not mask.any():return
        ids=torch.randint(len(self.bank_q),(self.n,),device='cuda')
        q=self.bank_q[ids].clone();ctrl=self.bank_ctrl[ids]
        self.home[mask]=ctrl[mask];self.center[mask]=q[mask,20:23]
        w,x,y,z=q[:,23:27].unbind(-1);yaw=torch.atan2(2*(y*z-w*x),2*(x*z+w*y))
        self.yaw0[mask]=yaw[mask];self.prev_yaw[mask]=yaw[mask]
        if noise:q[:,:12]+=torch.randn(self.n,12,device='cuda')*.005;q[:,20:23]+=torch.randn(self.n,3,device='cuda')*.0004
        self.qpos[mask]=q[mask];self.qvel[mask]=0;self.ctrl[mask]=ctrl[mask]
        self.middle_pip_min[mask]=q[mask,10];self.middle_pip_max[mask]=q[mask,10]
        wp.to_torch(self.data.qacc_warmstart)[mask]=0
        self.time[mask]=0;self.steps[mask]=0;self.last_action[mask]=0;self.unwrapped_angle[mask]=0
        self.total_return[mask]=0;self.total_error[mask]=0
        r=torch.rand(self.n,device='cuda')
        if self.curriculum_mode=='progressive':
            lv=torch.where(r<.8,self.curriculum_level,max(0,self.curriculum_level-1)).long()
        else:
            next_level=min(len(self.levels)-2,self.curriculum_level+1)
            lv=torch.where(r<.65,self.curriculum_level,torch.where(r<.8,max(0,self.curriculum_level-1),next_level)).long()
        self.sampled_level[mask]=lv[mask]
        self.missing_grip_steps[mask]=0
        self.push_enabled[mask]=(torch.rand(self.n,device='cuda')<.2)[mask]
        self.push_vector[mask]=(torch.randn(self.n,2,device='cuda').clamp(-1,1)*float(self.training_config.get('push_force_scale_n',.02)))[mask]
        for i,level in enumerate(self.levels):
            selected=mask&(lv==i)
            self.target_theta[selected]=np.radians(level['angle_deg']);self.rotation_duration[selected]=level['duration']
            self.episode_limit[selected]=round(level['episode_seconds']/self.dt);self.continuous[selected]=bool(level.get('continuous',False))
        wp.to_torch(self.guard_bad)[mask]=0;wp.to_torch(self.guard_gap)[mask]=1.
        mjw.forward(self.model,self.data);self.record_capacity();self.check_capacity();self.contact_info()

    def obs(self):
        angle,_,axis=self.angle();goal,omega,phase=self.goal_state()
        base=torch.cat((self.qpos[:,:20]-self.home,self.qvel[:,:20]*.1,(self.qpos[:,20:23]-self.center)*20,
                        axis,self.qvel[:,20:23],self.qvel[:,23:26]*.1,
                        torch.stack((torch.sin(angle),torch.cos(angle),torch.atan2(torch.sin(goal),torch.cos(goal)),omega,torch.sin(phase),torch.cos(phase)),-1),self.last_action),-1)
        # Extra command state makes integrated actions Markov; forces are privileged simulator observations.
        command=2*(self.ctrl-self.low)/(self.high-self.low)-1
        positions=wp.to_torch(self.data.xpos);rot=wp.to_torch(self.data.xmat)[:,self.palm_body]
        tip_gap=((self.sites[:,self.order_tip_ids[0]]-self.sites[:,self.order_tip_ids[1]])*rot[:,:,0]).sum(-1)
        extra=torch.stack((tip_gap*20,self.policy_pinch_strength,(self.unwrapped_angle-goal).clamp(-6,6),
            self.target_theta/(2*torch.pi),self.rotation_duration/10,self.continuous.float()),-1)
        return torch.cat((base,command,self.policy_forces*.2,extra),-1).clamp(-20,20)

    def step(self,raw_action,auto_reset=True):
        action=raw_action.clamp(-1,1)
        if self.park_unused_fingers:
            if action.shape[-1]!=12:raise ValueError('Tripod policy must have exactly 12 actions')
            action=torch.cat((action,torch.zeros(self.n,8,device=action.device,dtype=action.dtype)),-1)
        if self.action_mode=='delta':
            desired=self.ctrl+action*self.delta_scale;command=desired.clamp(self.safe_low,self.safe_high)
        else:
            desired=self.home+.65*action;command=desired.clamp(self.low,self.high)
        if self.ring_profile == 'natural_v1':command=natural_ring_command(self.ctrl,command)
        if self.park_unused_fingers:
            desired[:,12:]=self.home[:,12:];command[:,12:]=self.home[:,12:]
        clipping=(abs(command-desired)>1e-7).float()
        clipped=clipping[:,:12].mean(-1) if self.park_unused_fingers else clipping.mean(-1)
        self.ctrl[:]=command
        wp.to_torch(self.guard_bad).zero_();wp.to_torch(self.guard_gap).fill_(1.)
        external=wp.to_torch(self.data.xfrc_applied);external.zero_()
        pulse=(self.steps.long()%80<4)&self.push_enabled&(self.sampled_level>=int(self.training_config.get('push_min_level',2)))
        external[:,self.pen_id,:2]=self.push_vector*pulse[:,None]
        wp.capture_launch(self.graph);self.check_capacity();self.time+=self.dt;self.steps+=1
        tips,force,other,palm=self.contact_info()
        good_contact=(tips>=2)&(self.pinch_strength>=.3)
        self.missing_grip_steps=torch.where(good_contact,torch.zeros_like(self.missing_grip_steps),self.missing_grip_steps+1)
        angle,yaw,axis=self.angle();goal,omega,phase=self.goal_state()
        error=torch.atan2(torch.sin(angle-goal),torch.cos(angle-goal))
        yaw_delta=torch.atan2(torch.sin(yaw-self.prev_yaw),torch.cos(yaw-self.prev_yaw))
        yaw_vel=yaw_delta/self.dt
        self.unwrapped_angle+=yaw_delta
        error=self.unwrapped_angle-goal
        tilt=torch.asin(axis[:,2].abs().clamp(0,1));drift=torch.linalg.vector_norm(self.qpos[:,20:23]-self.center,dim=-1)
        self.middle_pip_min=torch.minimum(self.middle_pip_min,self.qpos[:,10])
        self.middle_pip_max=torch.maximum(self.middle_pip_max,self.qpos[:,10])
        middle_pip_excursion=self.middle_pip_max-self.middle_pip_min
        offset=self.sites[:,self.tip_ids]-self.qpos[:,None,20:23]
        dist=torch.linalg.vector_norm(offset-(offset*axis[:,None,:]).sum(-1).clamp(-.075,.075)[:,:,None]*axis[:,None,:],dim=-1)
        fail=(drift>.045)|(self.qpos[:,22]<self.center[:,2]-.025)|(tilt>np.radians(40))|(~torch.isfinite(self.qpos).all(-1))
        reverse=(self.qpos[:,[6,7,10,11]]<0).any(-1)
        posture=wp.to_torch(self.guard_bad)>0
        contacts=self.data.contact;pair=wp.to_torch(contacts.geom).long().clamp(0,self.cpu_model.ngeom-1)
        world=wp.to_torch(contacts.worldid).long().clamp(0,self.n-1)
        valid=(self.contact_index<wp.to_torch(self.data.nacon)[0])
        is_hand=(self.geom_body[pair[:,0]]!=self.pen_id)&(self.geom_body[pair[:,1]]!=self.pen_id)
        active=(self.geom_active[pair[:,0]]|self.geom_active[pair[:,1]])
        depths=torch.where(valid&is_hand&active,(-wp.to_torch(contacts.dist)).clamp(min=0),0.)
        self_depth=torch.zeros(self.n,device='cuda').scatter_reduce_(0,world,depths,reduce='amax',include_self=True)
        # CPU acceptance is stricter: 25 mm, 0.5 mm and 35 deg. End clearly
        # invalid GPU episodes while keeping a margin for simulator mismatch.
        fail=fail|reverse|posture|(self_depth>.0007)|(drift>.032)|(middle_pip_excursion>np.radians(42))
        if self.contact_reward_enabled:fail=fail|(self.missing_grip_steps>self.grip_gap_fail_steps)
        timeout=self.steps>=self.episode_limit;done=fail|timeout
        progress_reward=3*(yaw_vel/omega.abs().clamp(min=.1)).clamp(-2,2)*(omega!=0).float()
        if self.contact_reward_enabled:
            progress_reward=progress_reward*(self.missing_grip_steps<=self.grip_gap_allow_steps).float()
        terms={
            'tracking':8*torch.exp(-(error/self.tracking_sigma).square()),
            'progress':progress_reward,
            'functional_posture':-.6*((self.qpos[:,[6,7,10,11]]-self.home[:,[6,7,10,11]])/.6).square().mean(-1),
            'middle_pip_speed':-.02*self.qvel[:,10].square(),
            'distal_coordination':-.5*((self.qpos[:,[7,11]]-self.home[:,[7,11]]-.7*(self.qpos[:,[6,10]]-self.home[:,[6,10]]))/.35).square().mean(-1),
            'velocity':.5*torch.exp(-(yaw_vel-omega).square()),
            'horizontal':1.5*torch.exp(-(tilt/.15).square())-4*(axis[:,2].abs()-.174).clamp(min=0).square(),
            'support':.6*tips.clamp(max=2)/2,
            'workspace':-20*drift,
            'strict_workspace':-3*((drift-.017).clamp(min=0)/.008).square(),
            'strict_self_depth':-2*(self_depth/.00035).square().clamp(max=16),
            'strict_middle_pip':-3*((middle_pip_excursion-np.radians(24)).clamp(min=0)/np.radians(11)).square(),
            'non_tip':-6*other.clamp(max=3)-(4+float(self.training_config.get('extra_nonpad_penalty',0)))*(other>.05).float()-4*palm.clamp(max=3),
            'squeeze':-.03*(force-2).clamp(min=0,max=20).square(),
            'control':-.002*action.square().sum(-1)-.005*(action-self.last_action).square().sum(-1)-self.command_clip_penalty*clipped,
            'failure':-30*fail.float(),
            'slip':-.5*(self.qvel[:,20:23]*10).square().sum(-1),
            'posture_violation':-30*posture.float(),
            'side_pad_contact':torch.zeros_like(self.side_pad_force),
        }
        if self.ring_profile == 'natural_v1':
            ring=self.qpos[:,12:16]
            rest=torch.as_tensor(RING_REST,device=ring.device,dtype=ring.dtype)
            terms['ring_posture']=-.4*((ring-rest)/.5).square().mean(-1)
            terms['ring_coordination']=-.8*((ring[:,3]-.65*ring[:,2])/.35).square()
            terms['ring_hyperextension']=-2*(ring[:,[0,2,3]].clamp(max=0)/.25).square().sum(-1)
        if self.pinch_reward_weight:
            terms['opposing_pinch']=self.pinch_reward_weight*self.pinch_strength
            terms['missing_pinch']=-self.pinch_reward_weight*(self.pinch_strength<.3).float()
        if self.contact_reward_enabled:
            terms['lost_grip']=-3*(self.missing_grip_steps>self.grip_gap_allow_steps).float()
            post_goal=self.time>self.rotation_duration+.5
            terms['terminal_regrip']=2*good_contact.float()*post_goal.float()
        if self.park_unused_fingers:terms['parked_contact']=-8*self.parked_force.clamp(max=3)-4*(self.parked_force>.02).float()
        reward=sum(terms.values())
        if self.reward_mode=='fingertip':
            near2=dist.topk(2,largest=False).values
            reward=2*torch.exp(-error.square()/.09)+.4*torch.exp(-(yaw_vel-omega).square()/2)
            reward+=1.5*torch.exp(-((near2-.009).clamp(min=0)/.015).square()).mean(-1)+.5*tips.clamp(max=3)
            reward+=-1.5*other.clamp(max=3)-25*drift-2*axis[:,2].square()-.01*(action-self.last_action).square().sum(-1)-.004*action.square().sum(-1)-8*fail.float()
        self.total_return+=reward;self.total_error+=error.abs()
        self.last_action[:]=action;self.prev_yaw[:]=yaw
        info=dict(fail=fail,timeout=timeout,error=error,angle=self.unwrapped_angle.clone(),goal=goal,tilt=tilt,drift=drift,tips=tips,
                  good_contact=good_contact.clone(),missing_grip_steps=self.missing_grip_steps.clone(),
                  posture_violation=posture.clone(),self_depth=self_depth,middle_pip_excursion=middle_pip_excursion,
                  level=self.sampled_level.clone(),
                  parked_force=self.parked_force.clone(),
                  normal_force=force,other_force=other,palm_force=palm,command_clip=clipped,
                  episode_return=self.total_return.clone(),episode_error=self.total_error/self.steps.clamp(min=1),episode_length=self.steps.clone(),
                  terminal_obs=self.obs(),reward_terms=terms)
        if auto_reset:self.reset(done)
        return self.obs(),reward,done,info
