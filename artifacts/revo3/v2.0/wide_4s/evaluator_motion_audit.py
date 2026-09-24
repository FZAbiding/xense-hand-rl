"""Joint-motion diagnostics from saved 40 Hz CPU trajectories (not a causal ablation)."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GROUPS={'thumb':(0,5),'index':(5,9),'middle':(9,13),'ring':(13,17),'little':(17,21)}


def joint_motion(qpos,dt):
    """Finite-difference proxies; cannot resolve motion above Nyquist frequency."""
    q=np.degrees(np.asarray(qpos)[...,:21]);dq=np.diff(q,axis=-2)
    velocity=dq/dt;acceleration=np.diff(velocity,axis=-2)/dt
    return {name:dict(mean_abs_velocity_deg_s=float(abs(velocity[...,lo:hi]).mean()),
                     rms_acceleration_deg_s2=float(np.sqrt(np.mean(acceleration[...,lo:hi]**2))),
                     p95_step_deg=float(np.percentile(abs(dq[...,lo:hi]),95)),
                     max_step_deg=float(abs(dq[...,lo:hi]).max()))
            for name,(lo,hi) in GROUPS.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluations',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=12.)
    args=p.parse_args();result=[]
    for name in args.evaluations:
        path=ROOT/name;r=json.loads((path/'metrics.json').read_text());a=np.load(path/'rollout.npz');dt=float(a['dt'])
        q=a['qpos'][:3,:round(args.seconds/dt)]
        result.append(dict(evaluation=name,checkpoint=r['checkpoint'],amplitude_rad=r['amplitude_rad'],period_seconds=r['period_seconds'],
                           sampled_episodes=len(q),sampled_seconds=q.shape[1]*dt,control_dt=dt,fingers=joint_motion(q,dt)))
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(note='Actual joint-angle finite differences at 40Hz. Different amplitude/period/grasp are confounders; this is not an ablation. First three stored episodes only.',evaluations=result),indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
