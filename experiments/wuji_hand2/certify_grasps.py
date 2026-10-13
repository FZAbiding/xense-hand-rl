"""Certify palm-up Wuji grasps over 60 s before using them for PPO resets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.wuji_hand2.calibrate import contact_forces

ASSETS = ROOT / 'assets/wuji_hand2_palm_up'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--perturbations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--orientation', choices=('palm_up', 'palm_forward_v1'), default='palm_up')
    parser.add_argument('--source-bank', default='grasp_bank.npz')
    parser.add_argument('--report-name', default='grasp_bank_60s_certification.json')
    args = parser.parse_args()
    assets = ASSETS if args.orientation == 'palm_up' else ROOT / 'assets/wuji_hand2_palm_forward_v1'
    if args.seconds < 60 or args.perturbations < 1:
        raise ValueError('Certification requires at least 60 s and one noisy rollout per grasp')
    if Path(args.source_bank).name != args.source_bank:
        raise ValueError('Source bank must be a local filename')
    if Path(args.report_name).name != args.report_name:
        raise ValueError('Report must be a local filename')
    if (assets / args.report_name).exists() or (assets / 'grasp_bank_60s.npz').exists():
        raise RuntimeError('Certification output exists; use a new report or experiment version')
    source = np.load(assets / args.source_bank)
    model = mujoco.MjModel.from_xml_path(str(assets / 'scene.xml'))
    if args.orientation == 'palm_forward_v1':
        model.opt.iterations = 8
        model.opt.ls_iterations = 8
    data = mujoco.MjData(model)
    pen = model.body('pen').id
    palm = model.body('r_wrist').id
    tip_bodies = {model.body(f'r_{finger}_{suffix}').id: i
                  for i, finger in enumerate(('thumb', 'index_finger', 'middle_finger',
                                               'ring_finger', 'pinky'))
                  for suffix in ('distal', 'tip_sensor_frame')}
    rng = np.random.default_rng(args.seed)
    rows = []
    selected = []
    steps = round(args.seconds / model.opt.timestep)
    for index, (qpos, ctrl) in enumerate(zip(source['qpos'], source['ctrl'])):
        for perturbation in range(args.perturbations + 1):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = qpos
            data.ctrl[:] = ctrl
            center = qpos[20:23].copy()
            if perturbation:
                data.qpos[:20] += rng.normal(0., .005, 20)
                data.qpos[20:23] += rng.normal(0., .0004, 3)
            mujoco.mj_forward(model, data)
            two_tip = []
            non_tip = []
            palm_contact = []
            tilt = []
            max_drift = 0.
            min_height = float('inf')
            for step in range(steps):
                mujoco.mj_step(model, data)
                if step % 5:
                    continue
                tips, other, palm_force, _ = contact_forces(model, data, pen, tip_bodies, palm)
                axis = data.xmat[pen].reshape(3, 3)[:, 2]
                two_tip.append(tips >= 2)
                non_tip.append(other > .05)
                palm_contact.append(palm_force > .05)
                tilt.append(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0, 1))))
                max_drift = max(max_drift, float(np.linalg.norm(data.qpos[20:23] - center)))
                min_height = min(min_height, float(data.qpos[22]))
            row = dict(grasp_index=index, perturbation=perturbation,
                       seconds=args.seconds, seed=args.seed,
                       two_tip_fraction=float(np.mean(two_tip)),
                       non_tip_force_frame_fraction=float(np.mean(non_tip)),
                       palm_force_frame_fraction=float(np.mean(palm_contact)),
                       p95_tilt_deg=float(np.percentile(tilt, 95)),
                       max_drift_m=max_drift, min_height_m=min_height)
            row['passed'] = bool(row['two_tip_fraction'] > .8
                                 and row['non_tip_force_frame_fraction'] < .05
                                 and row['palm_force_frame_fraction'] < .01
                                 and row['p95_tilt_deg'] <= 15
                                 and row['max_drift_m'] <= .045
                                 and row['min_height_m'] >= center[2] - .025)
            rows.append(row)
            print(json.dumps(row), flush=True)
        if all(row['passed'] for row in rows[-args.perturbations - 1:]):
            selected.append(index)
    (assets / args.report_name).write_text(json.dumps(dict(
        criteria='60 s, untrained constant controls, zero noise plus Gaussian joint 0.005 rad and pen 0.4 mm; two tips >80%, non-tip <5%, palm <1%, P95 tilt <=15 deg, max drift <=45 mm, no >25 mm downward fall',
        source_bank=args.source_bank, selected=selected, rows=rows), indent=2))
    if not selected:
        raise RuntimeError('No grasp passed 60-second certification; PPO must not start')
    dest = assets / 'grasp_bank_60s.npz'
    if dest.exists():
        raise RuntimeError('Certified bank already exists; do not overwrite training assets')
    np.savez(dest, qpos=source['qpos'][selected], ctrl=source['ctrl'][selected])
    if args.orientation == 'palm_forward_v1':
        (assets / 'certified_bank_manifest.json').write_text(json.dumps(dict(
            certification_report=args.report_name, source_bank=args.source_bank,
            scene_sha256=hashlib.sha256((assets / 'scene.xml').read_bytes()).hexdigest(),
            bank_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            selected=selected), indent=2))
    print(f'SAVED {len(selected)} certified grasps: {selected}', flush=True)


if __name__ == '__main__':
    main()
