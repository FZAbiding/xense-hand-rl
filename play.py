"""Run the learned policy in a live CPU MuJoCo viewer."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco.viewer
from train import ActorCritic
from hand_config import hand_config

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'artifacts/leap/policy.pt'));p.add_argument('--seconds',type=float,default=120)
    args=p.parse_args();torch.set_num_threads(1)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);net=ActorCritic(ck['nobs'],ck.get('naction',16));net.load_state_dict(ck['model']);net.eval()
    assert not ck['config']['reference'],'Use a direct PPO checkpoint with this viewer.'
    amplitude=ck['config']['amplitude'];period=ck['config']['period'];dt=.025
    spec=hand_config(ck['config'].get('robot','leap'))
    model=mujoco.MjModel.from_xml_path(str(spec['assets']/'scene.xml'));model.opt.iterations=8;data=mujoco.MjData(model);h=model.nu
    g=np.load(spec['assets']/'grasp.npz');home=g['ctrl'];center=g['qpos'][h:h+3];pen=model.body('pen').id
    data.qpos[:]=g['qpos'];data.ctrl[:]=home;mujoco.mj_forward(model,data)
    axis=data.xmat[pen].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0]);last=np.zeros(h)
    with mujoco.viewer.launch_passive(model,data) as viewer,torch.no_grad():
        viewer.cam.lookat[:]=[.125,0,.025];viewer.cam.distance=.40;viewer.cam.azimuth=130;viewer.cam.elevation=-35;viewer.opt.geomgroup[3]=0
        for t in range(round(args.seconds/dt)):
            if not viewer.is_running():break
            begin=time.monotonic();axis=data.xmat[pen].reshape(3,3)[:,2]
            delta=np.arctan2(axis[1],axis[0])-yaw0;angle=np.arctan2(np.sin(delta),np.cos(delta))
            phase=t*dt*2*np.pi/period;goal=amplitude*np.sin(phase);velocity=amplitude*2*np.pi/period*np.cos(phase)
            obs=np.r_[data.qpos[:h]-home,data.qvel[:h]*.1,(data.qpos[h:h+3]-center)*20,axis,data.qvel[h:h+3],data.qvel[h+3:h+6]*.1,
                      np.sin(angle),np.cos(angle),goal,velocity,np.sin(phase),np.cos(phase),last]
            action=net(torch.tensor(obs,dtype=torch.float32).clamp(-20,20)[None])[0][0].numpy().clip(-1,1)
            data.ctrl[:]=np.clip(home+.65*action,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            for _ in range(5):mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data);last=action;viewer.sync()
            if data.qpos[h+2]<.015 or np.linalg.norm(data.qpos[h:h+3]-center)>.075:
                print('Pen dropped: stopping without resetting.');break
            if ck['config'].get('objective','legacy')=='fingertip' and (data.qpos[h+2]<center[2]-.018 or np.linalg.norm(data.qpos[h:h+3]-center)>.045):
                print('Pen left fingertip workspace: stopping without resetting.');break
            time.sleep(max(0,dt-(time.monotonic()-begin)))

if __name__=='__main__':main()
