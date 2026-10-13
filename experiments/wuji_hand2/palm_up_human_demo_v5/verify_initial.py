"""Select on validation only; test new initial poses on held-out perturbations."""
import json
import mujoco
import numpy as np
from .common import ASSETS,RUNS,ART,simulate


def main():
    out=ART/'initial';out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));bank=np.load(ASSETS/'candidate_poses.npz')
    checks=json.loads((RUNS/'candidate_checks.json').read_text());validation=[]
    for row in checks:
        if not row['physical_gate_passed']:continue
        i=row['candidate'];rng=np.random.default_rng(2026101202)
        results=[simulate(m,bank['qpos'][i],bank['ctrl'][i],12,rng)[0] for _ in range(8)]
        fraction=float(np.mean([r['physical_gate_passed'] for r in results]))
        validation.append(dict(candidate=i,success_fraction=fraction,episodes=results))
        print('V5_VALIDATE',i,fraction,flush=True)
    (RUNS/'initial_validation.json').write_text(json.dumps(validation,indent=2))
    best=max(validation,key=lambda x:(x['success_fraction'],-max(r['max_tilt_deg'] for r in x['episodes'])))
    i=best['candidate'];q,c=bank['qpos'][i],bank['ctrl'][i];rng=np.random.default_rng(2026101203)
    tests=[simulate(m,q,c,12,rng)[0] for _ in range(24)]
    nominal,qs,cs,metrics,contacts=simulate(m,q,c,60,record=True)
    summary=dict(candidate=i,validation_seed=2026101202,validation_episodes=8,
        validation_success_fraction=best['success_fraction'],holdout_seed=2026101203,holdout_episodes=24,
        holdout_seconds=12,holdout_passed=sum(r['physical_gate_passed'] for r in tests),
        joint_noise_std_rad=.005,pen_position_noise_std_m=.0004,episodes=tests,nominal_60s=nominal,
        initial_human_review='pending',training_started=False,
        note='Independent candidate poses only. No training bank is approved or generated.')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));(out/'contacts.json').write_text(json.dumps(contacts))
    (out/'frame_metrics.json').write_text(json.dumps(metrics))
    np.savez_compressed(out/'rollout.npz',qpos=qs,ctrl=cs,dt=.025)
    np.savez(out/'candidate_pose.npz',qpos=q,ctrl=c)
    print('V5_INITIAL',i,'holdout',summary['holdout_passed'],'nominal60',nominal,flush=True)


if __name__=='__main__':main()
