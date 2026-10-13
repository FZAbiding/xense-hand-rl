"""Verify functional control and CPU/GPU observation parity on the actual scene."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import json
import numpy as np
import torch
import mujoco
from experiments.wuji_hand2.palm_up_human_demo_v4.env import Env
from experiments.wuji_hand2.palm_up_human_demo_v4.evaluate import observation
from experiments.wuji_hand2.evaluate import contact_forces
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT


def main():
    torch.set_num_threads(1)
    e=Env(num_envs=4,amplitude=1,period=18,episode_seconds=3,park_unused_fingers=1,pinch_reward_weight=2.)
    m=e.cpu_model;d=mujoco.MjData(m)
    d.qpos[:]=e.qpos[0].cpu().numpy();d.ctrl[:]=e.ctrl[0].cpu().numpy()
    home=e.home[0].cpu().numpy();center=e.center[0].cpu().numpy();yaw0=e.yaw0[0].item()
    rows=[]
    for t in range(20):
        mujoco.mj_forward(m,d)
        cpu=observation(m,d,home,center,yaw0,t*.025,np.zeros(20),contact_forces(m,d),{'amplitude':1,'period':18})
        gpu=e.obs()[0].cpu().numpy()
        rows.append(dict(step=t,kinematic_observation_max_error=float(abs(cpu[:98]-gpu[:98]).max()),
                         force_observation_max_error=float(abs(cpu[98:]-gpu[98:]).max()),
                         qpos_max_error=float(abs(d.qpos-e.qpos[0].cpu().numpy()).max())))
        if t==0:
            assert abs(cpu[:98]-gpu[:98]).max()<1e-5
        for _ in range(5):mujoco.mj_step(m,d)
        obs,r,done,info=e.step(torch.zeros(4,12,device='cuda'),auto_reset=False)
        assert torch.isfinite(obs).all()
        assert (e.ctrl[:,[6,7,10,11]]>0).all()
        assert torch.equal(e.ctrl[:,12:],e.home[:,12:])
        assert torch.allclose(info['goal'],torch.full((4,), (t+1)*.025*2*np.pi/18,device='cuda'),atol=1e-6)
    result=dict(initial_observation_parity=True,functional_command_bounds=True,inactive_commands_fixed=True,
                continuous_goal_not_sinusoidal=True,capacity=e.check_capacity(),trace=rows)
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/controller_verification.json'
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
