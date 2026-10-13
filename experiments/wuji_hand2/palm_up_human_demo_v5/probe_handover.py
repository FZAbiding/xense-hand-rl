"""Try a real TI -> tripod -> TM transfer with natural finger ordering."""
import json
import mujoco
import numpy as np
from .common import ASSETS,ART,RUNS,LOW,HIGH,simulate


def handovers(metrics):
    stable=[];last=None;count=0
    for r in metrics:
        active=tuple(int(i) for i in np.flatnonzero(np.asarray(r['forces'][:3])>.02))
        pair=active if len(active)==2 and r['opposing']>=.3 else None
        count=count+1 if pair==last else 1;last=pair
        if pair is not None and count>=4 and (not stable or stable[-1]!=pair):stable.append(pair)
    return dict(stable_pairs=[list(p) for p in stable],ti_to_tm=any(a==(0,1) and b==(0,2) for a,b in zip(stable[:-1],stable[1:])))


def main():
    out=ART/'handover_probe'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Do not overwrite handover evidence')
    out.mkdir(exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));bank=np.load(ART/'initial_settled/candidate_pose.npz');q,c=bank['qpos'],bank['ctrl']
    history=[];best=None
    for middle in [6,10,14,18]:
        for index in [6,10,14,18]:
            knots=np.tile(c[:12],(3,1));knots[0,8]-=np.radians(middle);knots[2,4]-=np.radians(index)
            knots=np.clip(knots,LOW,HIGH)
            r,_,_,metrics,_=simulate(m,q,c,8,knots=knots);r.update(handovers(metrics),middle_clear_deg=middle,index_clear_deg=index)
            r['handover_passed']=bool(r['physical_gate_passed'] and r['ti_to_tm'])
            history.append(r);key=(r['handover_passed'],r['physical_gate_passed'],r['seconds'],-r['non_tip_fraction']-r['inactive_fraction'])
            if best is None or key>best[0]:best=(key,knots.copy(),r)
    r,qs,cs,metrics,contacts=simulate(m,q,c,8,knots=best[1],record=True);r.update(handovers(metrics));r['handover_passed']=bool(r['physical_gate_passed'] and r['ti_to_tm'])
    tests=[];rng=np.random.default_rng(2026101209)
    for i in range(24):
        row,_,_,mm,_=simulate(m,q,c,8,rng=rng,knots=best[1]);row.update(handovers(mm));row['handover_passed']=bool(row['physical_gate_passed'] and row['ti_to_tm']);tests.append(row)
    summary=dict(method='three actuator waypoints; pen free after reset',candidate_count=16,nominal=r,holdout_seed=2026101209,
        holdout_episodes=24,holdout_passed=sum(x['handover_passed'] for x in tests),episodes=tests,human_review='pending')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));(out/'frame_metrics.json').write_text(json.dumps(metrics));(out/'contacts.json').write_text(json.dumps(contacts))
    np.savez_compressed(out/'rollout.npz',qpos=qs,ctrl=cs,dt=.025);np.savez(out/'controller.npz',initial_qpos=q,initial_ctrl=c,knots=best[1])
    (RUNS/'handover_search.json').write_text(json.dumps(history,indent=2))
    print('V5_HANDOVER',r,'holdout',summary['holdout_passed'],flush=True)


if __name__=='__main__':main()
