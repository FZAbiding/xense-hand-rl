"""Use a physically settled V5 snapshot instead of overlapping IK initialization."""
import json
import mujoco
import numpy as np
from .common import ASSETS,ART,simulate,pen_contacts,order_measure


def main():
    out=ART/'initial_settled';out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));old=np.load(ART/'initial/rollout.npz')
    # This state arose from one second of real free-pen dynamics under fixed control.
    q,c=old['qpos'][39],old['ctrl'][39]
    d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=c;mujoco.mj_forward(m,d)
    depth=max([-contact.dist for contact in d.contact if m.body('pen').id in [m.geom_bodyid[contact.geom1],m.geom_bodyid[contact.geom2]]],default=0)
    if depth>.0005 or not order_measure(m,d)['passed']:raise RuntimeError('Settled pose failed initialization gate')
    rng=np.random.default_rng(2026101206);tests=[simulate(m,q,c,12,rng)[0] for _ in range(24)]
    nominal,qs,cs,metrics,contacts=simulate(m,q,c,60,record=True)
    summary=dict(source='v5 initial/rollout.npz physical frame 39 at 1.0 s; velocities reset to zero',initial_pen_contact_depth_m=depth,
        holdout_seed=2026101206,holdout_episodes=24,holdout_seconds=12,holdout_passed=sum(x['physical_gate_passed'] for x in tests),
        episodes=tests,nominal_60s=nominal,initial_human_review='pending',training_started=False)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));(out/'frame_metrics.json').write_text(json.dumps(metrics));(out/'contacts.json').write_text(json.dumps(contacts))
    np.savez_compressed(out/'rollout.npz',qpos=qs,ctrl=cs,dt=.025);np.savez(out/'candidate_pose.npz',qpos=q,ctrl=c)
    print('V5_SETTLED',summary['holdout_passed'],nominal,flush=True)


if __name__=='__main__':main()
