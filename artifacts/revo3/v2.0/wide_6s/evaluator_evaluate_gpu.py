"""Frozen-policy MuJoCo Warp evaluation: no automatic reset, CPU-identical gates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train import ActorCritic
from experiments.revo3_v2.env import Env
from experiments.revo3_v2.evaluate import quality_pass


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--num-envs',type=int,default=128);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--njmax',type=int,default=512);p.add_argument('--nconmax',type=int,default=128);args=p.parse_args()
    out=ROOT/args.output
    if (out/'metrics.json').exists():raise RuntimeError('Choose a new output directory')
    torch.set_num_threads(4)
    ck=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config']
    assert cfg['experiment_version']=='2.0'
    for name in ['assets/revo3_v2/scene.xml','assets/revo3_v2/grasp_bank.npz']:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==ck['source_manifest'][name],name
    env=Env(args.num_envs,cfg['amplitude'],cfg['period'],args.seed,episode_seconds=args.seconds,
            action_mode=cfg['action_mode'],reward_mode=cfg['reward_mode'],grasp_version=cfg['grasp_version'],
            command_clip_penalty=cfg.get('command_clip_penalty',.2),tracking_sigma=cfg.get('tracking_sigma',.18),njmax=args.njmax,nconmax=args.nconmax)
    env.reset(torch.ones(args.num_envs,dtype=torch.bool,device='cuda'),noise=True)
    grasp_ids=(env.home[:,None]-env.bank_ctrl[None]).square().sum(-1).argmin(-1).cpu().numpy()
    net=ActorCritic(ck['nobs'],ck['naction']).cuda();net.load_state_dict(ck['model']);net.eval()
    T=round(args.seconds/env.dt);obs=env.obs();samples=[]
    failed=torch.zeros(args.num_envs,dtype=torch.bool,device='cuda');dropped=failed.clone()
    first_failure=torch.full((args.num_envs,),float('nan'),device='cuda')
    with torch.no_grad():
        for t in range(T):
            action=net(obs)[0];obs,_,_,info=env.step(action,auto_reset=False)
            first_failure=torch.where(info['fail']&(~failed),(t+1)*env.dt,first_failure)
            failed|=info['fail'];dropped|=(info['drift']>.075)|(env.qpos[:,23]<.015)|(~torch.isfinite(env.qpos).all(-1))
            samples.append(torch.stack((info['angle'],info['goal'],info['error'],info['tilt'],info['drift'],
                info['command_clip'],info['normal_force'],info['other_force'],info['palm_force'],info['tips']),-1))
    arrays=torch.stack(samples,1).cpu().numpy();failed=failed.cpu().numpy();dropped=dropped.cpu().numpy()
    first_failure=first_failure.cpu().numpy();rows=[];amp=cfg['amplitude'];period=cfg['period']
    length=round(period/env.dt);half=length//2
    for ep,a in enumerate(arrays):
        gains=[]
        for c in range(int(args.seconds/period)):
            angle=a[c*length:(c+1)*length,0]
            gains.extend([float(angle[:half].max()/amp),float(-angle[half:].min()/amp)])
        row=dict(episode=ep,grasp_index=int(grasp_ids[ep]),survived=not bool(dropped[ep]),within_workspace=not bool(failed[ep]),
            first_workspace_failure_s=None if np.isnan(first_failure[ep]) else float(first_failure[ep]),
            mae_deg=float(np.degrees(abs(a[:,2])).mean()),mean_tilt_deg=float(np.degrees(a[:,3]).mean()),
            p95_tilt_deg=float(np.percentile(np.degrees(a[:,3]),95)),command_clip_fraction=float(a[:,5].mean()),
            normal_force_sum_n=float(a[:,6].mean()),non_tip_force_frame_fraction=float(np.mean(a[:,7]>.05)),
            palm_force_frame_fraction=float(np.mean(a[:,8]>.05)),two_tip_fraction=float(np.mean(a[:,9]>=2)),
            minimum_half_cycle_gain=min(gains) if gains else 0.,half_cycle_gains=gains)
        row['quality_success']=quality_pass(row);rows.append(row)
    result=dict(checkpoint=args.checkpoint,experiment_version='2.0',backend='GPU MuJoCo Warp frozen policy, no resets',
        seed=args.seed,episodes=args.num_envs,seconds=args.seconds,amplitude_rad=amp,period_seconds=period,
        simulation_valid=True,simulation_capacity=env.check_capacity(),
        quality_success_rate=float(np.mean([r['quality_success'] for r in rows])),
        survival_rate=float(np.mean([r['survived'] for r in rows])),workspace_survival_rate=float(np.mean([r['within_workspace'] for r in rows])))
    for key in ['mae_deg','mean_tilt_deg','p95_tilt_deg','command_clip_fraction','normal_force_sum_n',
                'non_tip_force_frame_fraction','palm_force_frame_fraction','two_tip_fraction','minimum_half_cycle_gain']:
        result[key]=float(np.mean([r[key] for r in rows]))
    out.mkdir(parents=True,exist_ok=True)
    sources=['experiments/revo3_v2/evaluate_gpu.py','experiments/revo3_v2/evaluate.py','experiments/revo3_v2/env.py','experiments/revo3_v2/motion_audit.py','train.py']
    (out/'evaluation_manifest.json').write_text(json.dumps(dict(
        checkpoint_sha256=hashlib.sha256((ROOT/args.checkpoint).read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        joint_noise_std_rad=.005,position_noise_std_m=.0004,automatic_resets=False),indent=2))
    (out/'metrics.json').write_text(json.dumps(dict(result,per_episode=rows),indent=2))
    np.savez_compressed(out/'rollout.npz',metrics=arrays,dt=env.dt,grasp_ids=grasp_ids)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
