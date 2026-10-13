"""Regression checks before the sustained GPU run."""
import json
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from .env import Env,ROOT
from experiments.wuji_hand2.palm_up_human_demo_v5.common import order_measure


def main():
    torch.set_num_threads(2)
    env=Env(num_envs=8,pinch_reward_weight=2.,tracking_sigma=.3)
    assert env.obs().shape==(8,111)
    d=mujoco.MjData(env.cpu_model)
    old=np.load(ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v4/partial_rotation/rollout.npz')['qpos_0']
    ids=np.linspace(0,len(old)-1,8,dtype=int);env.qpos[:]=torch.as_tensor(old[ids],device='cuda',dtype=torch.float32)
    mjw.forward(env.model,env.data);wp.to_torch(env.guard_bad).zero_();wp.to_torch(env.guard_gap).fill_(1.);env.record_guard()
    assert (wp.to_torch(env.guard_bad)>0).all(),'Rejected crossing must fail on GPU'
    env.reset(torch.ones(8,device='cuda',dtype=torch.bool),noise=False)
    assert not (wp.to_torch(env.guard_bad)>0).any()
    for i in range(8):
        d.qpos[:]=env.qpos[i].cpu().numpy();mujoco.mj_forward(env.cpu_model,d);assert order_measure(env.cpu_model,d)['passed']
    for _ in range(50):
        obs,r,done,info=env.step(torch.zeros(8,12,device='cuda'))
        assert torch.isfinite(obs).all() and torch.isfinite(r).all()
        assert torch.equal(env.ctrl[:,12:],env.home[:,12:])
    result=dict(obs=111,actions=12,all20_physical_actuators=True,rejected_old_postures_on_gpu=True,
        initial_cpu_order=True,finite_physics_steps=250,parked_commands_fixed=True,capacity=env.check_capacity())
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v6b/long_training_20261014';out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
