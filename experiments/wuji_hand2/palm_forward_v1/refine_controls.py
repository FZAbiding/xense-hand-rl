"""Search static control preload around isolated, relaxed upright grasps."""
import json
import argparse
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.wuji_hand2.calibrate import contact_forces, FINGERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('control_refinement.json'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assets = ROOT / 'assets/wuji_hand2_palm_forward_v1'
    dest = assets / cfg['output_bank']
    if dest.exists():
        raise RuntimeError('Control refinement bank exists')
    bank = np.load(assets / cfg['source_bank'])
    m = mujoco.MjModel.from_xml_path(str(assets / 'scene.xml')); d = mujoco.MjData(m)
    pen = m.body('pen').id; palm = m.body('r_wrist').id
    tips = {m.body(f'r_{f}_{suffix}').id: i for i, f in enumerate(FINGERS) for suffix in ('distal', 'tip_sensor_frame')}
    low, high = m.actuator_ctrlrange.T
    margin = np.minimum(np.radians(3), (high-low)*.1)
    rng = np.random.default_rng(cfg['seed']); rows = []; accepted = []
    for index in range(cfg['candidates']):
        source_indices = cfg.get('source_indices', list(range(len(bank['qpos']))))
        gi = source_indices[index % len(source_indices)]; q = bank['qpos'][gi].copy(); ctrl = bank['ctrl'][gi].copy()
        if index >= len(bank['qpos']):
            sigma = cfg['control_noise_std_rad'][index % len(cfg['control_noise_std_rad'])]
            ctrl[:12] += rng.normal(0, sigma, 12)
        ctrl = np.clip(ctrl, low+margin, high-margin)
        mujoco.mj_resetData(m, d); d.qpos[:] = q; d.ctrl[:] = ctrl
        center = q[20:23]; counts=[]; others=[]; palms=[]; tilts=[]; forces=[]; max_drift=0.; failed=False
        for step in range(round(cfg['seconds']/m.opt.timestep)):
            mujoco.mj_step(m,d)
            if step % 5: continue
            drift = np.linalg.norm(d.qpos[20:23]-center); max_drift=max(max_drift,float(drift))
            axis=d.xmat[pen].reshape(3,3)[:,2]; tilt=np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1)))
            if drift>.045 or tilt>40 or d.qpos[22]<center[2]-.025:
                failed=True; break
            count,other,palm_force,force=contact_forces(m,d,pen,tips,palm)
            counts.append(count>=2); others.append(other>.05); palms.append(palm_force>.05); tilts.append(tilt); forces.append(force)
        row=dict(candidate=index,source_grasp=gi,failed=failed,max_drift_m=max_drift,
                 two_tip_fraction=float(np.mean(counts)) if counts else 0.,
                 non_tip_force_frame_fraction=float(np.mean(others)) if others else 1.,
                 palm_force_frame_fraction=float(np.mean(palms)) if palms else 1.,
                 p95_tilt_deg=float(np.percentile(tilts,95)) if tilts else 90.,
                 mean_normal_force_n=float(np.mean(forces)) if forces else 0.)
        valid=not failed and row['two_tip_fraction']>.95 and row['non_tip_force_frame_fraction']<.03 and row['palm_force_frame_fraction']<.01 and row['p95_tilt_deg']<10
        row['accepted']=bool(valid); rows.append(row)
        if valid:
            score = row['two_tip_fraction'] - .01*row['p95_tilt_deg'] - max_drift - .1*abs(row['mean_normal_force_n']-1.)
            accepted.append((score,d.qpos.copy(),ctrl.copy(),row))
        if (index+1)%32==0:
            print('CONTROLS',index+1,'accepted',len(accepted),flush=True)
            (assets/cfg.get('report', 'control_refinement_v4.json')).write_text(json.dumps(dict(config=cfg,rows=rows),indent=2))
    (assets/cfg.get('report', 'control_refinement_v4.json')).write_text(json.dumps(dict(config=cfg,rows=rows),indent=2))
    if not accepted: raise RuntimeError('No preloaded grasp passed 12-second gates')
    accepted.sort(key=lambda x:-x[0]); selected=accepted[:cfg['retained']]
    np.savez(dest,qpos=[x[1] for x in selected],ctrl=[x[2] for x in selected])
    print('SAVED',len(selected),dest,flush=True)


if __name__=='__main__':main()
