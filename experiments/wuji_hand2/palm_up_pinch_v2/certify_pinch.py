"""Audit the reused certified bank under the NEW opposing-pinch criterion."""
import json
from pathlib import Path
import numpy as np
import mujoco
from experiments.wuji_hand2.pinch import opposing_strength,normal_vectors
from experiments.wuji_hand2.ring_control import natural_ring_command
from experiments.wuji_hand2.evaluate import contact_forces

ROOT=Path(__file__).resolve().parents[3]


def main():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());assets=ROOT/cfg['assets']
    dest=ROOT/cfg['runs']/'pinch_grasp_certification.json'
    if dest.exists():raise RuntimeError('Refuse to replace certification')
    m=mujoco.MjModel.from_xml_path(str(assets/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8
    quat=m.body_quat[m.body('r_wrist').id];mat=np.zeros(9);mujoco.mju_quat2Mat(mat,quat)
    palm=mat.reshape(3,3)@np.array([0,1,0]);assert np.allclose(palm,[0,0,1]),palm
    bank=np.load(assets/'grasp_bank_60s.npz');rng=np.random.default_rng(20261021);rows=[]
    for gi,(q,ctrl) in enumerate(zip(bank['qpos'],bank['ctrl'])):
      for perturb in range(5):
        d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=ctrl;center=q[20:23].copy()
        if perturb:d.qpos[:20]+=rng.normal(0,.005,20);d.qpos[20:23]+=rng.normal(0,.0004,3)
        pinches=[];tips=[];others=[];palms=[];drifts=[];tilts=[];heights=[]
        for t in range(2400):
            d.ctrl[:]=natural_ring_command(d.ctrl,d.ctrl)
            for k in range(5):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);force=contact_forces(m,d)
            pinches.append(opposing_strength(normal_vectors(m,d))>=.3);tips.append(np.sum(force[:5]>.02)>=2)
            others.append(force[5]>.05);palms.append(force[6]>.05);drifts.append(np.linalg.norm(d.qpos[20:23]-center));heights.append(d.qpos[22])
            axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];tilts.append(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1))))
        row=dict(grasp=gi,perturbation=perturb,opposing_pinch_fraction=float(np.mean(pinches)),two_tip_fraction=float(np.mean(tips)),
                 non_tip_force_frame_fraction=float(np.mean(others)),palm_force_frame_fraction=float(np.mean(palms)),
                 max_drift_m=float(np.max(drifts)),p95_tilt_deg=float(np.percentile(tilts,95)),min_height_m=float(np.min(heights)))
        row['passed']=bool(row['opposing_pinch_fraction']>.8 and row['two_tip_fraction']>.8 and row['non_tip_force_frame_fraction']<.05 and row['palm_force_frame_fraction']<.01 and row['max_drift_m']<=.045 and row['p95_tilt_deg']<=15 and row['min_height_m']>=center[2]-.025)
        rows.append(row)
    result=dict(palm_normal_world=palm.tolist(),seconds=60,perturbations=4,seed=20261021,
                constant_controls=True,ring_profile='natural_v1',passed=all(r['passed'] for r in rows),rows=rows)
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('New pinch criterion failed; stop this experiment')


if __name__=='__main__':main()
