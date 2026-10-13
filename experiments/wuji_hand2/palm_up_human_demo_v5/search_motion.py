"""Bounded physical candidate search for 15-degree motion, not PPO training."""
import argparse,json,time
import mujoco
import numpy as np
from .common import ASSETS,RUNS,ART,LOW,HIGH,simulate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initial',default='initial');parser.add_argument('--output',default='motion');parser.add_argument('--seed',type=int,default=2026101204);args=parser.parse_args()
    out=ART/args.output;out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));bank=np.load(ART/args.initial/'candidate_pose.npz')
    q,c=bank['qpos'],bank['ctrl'];rng=np.random.default_rng(args.seed)
    mean=c[:12].copy();std=np.full(12,.07);history=[];best=None;started=time.time()
    for generation in range(6):
        population=np.clip(mean+rng.normal(size=(48,12))*std,LOW,HIGH);population[0]=mean;population[1]=c[:12]
        evaluated=[]
        for i,target in enumerate(population):
            r,_,_,metrics,_=simulate(m,q,c,4,knots=target)
            # Score posture/contacts before angle. The best reference must remain feasible.
            errors=[abs(np.degrees(x['angle'])-15*np.clip(((j+1)*.025-.5)/3,0,1)) for j,x in enumerate(metrics)]
            cost=float(np.mean(errors)+2*abs(r['final_angle_deg']-15)+300*r['max_drift_m']+80*r['non_tip_fraction']+80*r['inactive_fraction']+100*(1-r['seconds']/4))
            r.update(generation=generation,candidate=i,cost=cost)
            r['small_motion_passed']=bool(r['physical_gate_passed'] and 13.5<=r['final_angle_deg']<=18 and np.mean(errors)<=5)
            history.append(r);evaluated.append((cost,target.copy()))
            key=(r['physical_gate_passed'],-cost)
            if best is None or key>best[0]:best=(key,target.copy(),r.copy())
        elite=np.array([x[1] for x in sorted(evaluated,key=lambda x:x[0])[:8]])
        mean=.5*mean+.5*elite.mean(0);std=np.maximum(elite.std(0),.012)
        (RUNS/f'{args.output}_search.json').write_text(json.dumps(dict(best=best[2],history=history),indent=2))
        print('V5_MOTION',generation,'elapsed',round(time.time()-started),best[2],flush=True)
    target=best[1];r,qs,cs,metrics,contacts=simulate(m,q,c,4,knots=target,record=True)
    validation_rng=np.random.default_rng(args.seed+1);tests=[]
    for i in range(24):
        row=simulate(m,q,c,4,rng=validation_rng,knots=target)[0]
        row['small_motion_passed']=bool(row['physical_gate_passed'] and 13.5<=row['final_angle_deg']<=18)
        row['episode']=i;tests.append(row)
    summary=dict(controller='open-loop physical actuator interpolation; no learned policy',target_angle_deg=15,
        seed=args.seed,population=48,generations=6,nominal=r,holdout_seed=args.seed+1,holdout_episodes=24,
        physical_holdout_passed=sum(x['physical_gate_passed'] for x in tests),
        angle_holdout_passed=sum(x['small_motion_passed'] for x in tests),episodes=tests,human_review='pending',handover_achieved=False)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));(out/'frame_metrics.json').write_text(json.dumps(metrics));(out/'contacts.json').write_text(json.dumps(contacts))
    np.savez_compressed(out/'rollout.npz',qpos=qs,ctrl=cs,dt=.025)
    np.savez(out/'controller.npz',initial_qpos=q,initial_ctrl=c,target_ctrl=target)
    print('V5_MOTION_DONE',summary['angle_holdout_passed'],r,flush=True)
    # Controlled release probes: test whether initial two other pads can take over.
    release=[]
    for degrees in [-15,-8,8,15]:
        control=c[:12].copy();control[0]=np.clip(control[0]+np.radians(degrees),LOW[0],HIGH[0])
        rr,qq,cmd,mm,cc=simulate(m,q,c,4,knots=control,record=True)
        pairs=[]
        for x in mm:
            pair=tuple(np.flatnonzero(np.array(x['forces'][:3])>.02))
            pairs.append(list(pair))
        rr.update(thumb_cmc_delta_deg=degrees,exact_index_middle_fraction=float(np.mean([x==[1,2] for x in pairs])))
        release.append(rr)
        if degrees==-15:
            failed=ART/f'{args.output}_release_probe';failed.mkdir()
            np.savez_compressed(failed/'rollout.npz',qpos=qq,ctrl=cmd,dt=.025)
            (failed/'frame_metrics.json').write_text(json.dumps(mm));(failed/'contacts.json').write_text(json.dumps(cc))
    (RUNS/f'{args.output}_release_probes.json').write_text(json.dumps(release,indent=2))


if __name__=='__main__':main()
