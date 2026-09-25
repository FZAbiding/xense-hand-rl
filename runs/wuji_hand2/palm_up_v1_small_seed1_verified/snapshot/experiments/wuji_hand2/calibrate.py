"""Search physically stable Wuji Hand 2 grasps for the existing free-rod task."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ASSETS = ROOT / 'assets/wuji_hand2'
PALM_UP_ASSETS = ROOT / 'assets/wuji_hand2_palm_up'
FINGERS = ('thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky')


def contact_forces(model, data, pen, tip_bodies, palm):
    touching = set()
    other = palm_force = total = 0.0
    for i, contact in enumerate(data.contact):
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        partner = b if a == pen else a if b == pen else -1
        if partner < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal = max(float(force[0]), 0.0)
        total += normal
        if partner in tip_bodies and normal > .02:
            touching.add(tip_bodies[partner])
        elif partner not in tip_bodies:
            other += normal
        if partner == palm:
            palm_force += normal
    return len(touching), other, palm_force, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-candidates', type=int, default=96)
    parser.add_argument('--candidate-index', type=int)
    parser.add_argument('--ring-flex', type=float, default=-.5)
    parser.add_argument('--ring-abd', type=float, default=.5)
    parser.add_argument('--pinky-flex', type=float, default=.45)
    parser.add_argument('--pinky-abd', type=float, default=0.)
    parser.add_argument('--middle-x-offset', type=float, default=0.)
    parser.add_argument('--settle-seconds', type=float, default=2.)
    parser.add_argument('--bank-name', default='grasp_bank.npz')
    parser.add_argument('--local-refine', action='store_true')
    parser.add_argument('--orientation', choices=('palm_up', 'legacy_palm_down'), default='palm_up')
    args = parser.parse_args()
    assets = PALM_UP_ASSETS if args.orientation == 'palm_up' else LEGACY_ASSETS
    if Path(args.bank_name).name != args.bank_name or not args.bank_name.endswith('.npz'):
        raise ValueError('--bank-name must be a local .npz filename')
    bank_path = assets / args.bank_name
    metrics_path = assets / ('grasp_metrics.json' if args.bank_name == 'grasp_bank.npz' else args.bank_name.replace('.npz', '_metrics.json'))
    candidates_path = assets / ('calibration_candidates.json' if args.bank_name == 'grasp_bank.npz' else args.bank_name.replace('.npz', '_candidates.json'))
    if args.candidate_index is None and bank_path.exists():
        raise RuntimeError('Grasp bank already exists; do not replace assets beneath a trained policy.')
    model = mujoco.MjModel.from_xml_path(str(assets / 'scene.xml'))
    data = mujoco.MjData(model)
    hand = model.nu
    pen = model.body('pen').id
    sites = [model.site(f'r_{finger}_tip').id for finger in FINGERS[:3]]
    tip_bodies = {}
    for i, finger in enumerate(FINGERS):
        for suffix in ('distal', 'tip_sensor_frame'):
            tip_bodies[model.body(f'r_{finger}_{suffix}').id] = i
    palm = model.body('r_wrist').id
    low, high = model.actuator_ctrlrange.T
    margin = np.minimum(np.radians(5), (high - low) * .12)
    lower, upper = low + margin, high - margin
    seed = np.zeros(hand)
    for start in (4, 8, 12, 16):
        seed[start:start + 4] = [.45, 0, .45, .3]
    seed[12:14] = [args.ring_flex, args.ring_abd]
    seed[16:18] = [args.pinky_flex, args.pinky_abd]
    seed[:4] = [.3, -.4, .4, .2]
    seed = np.clip(seed, lower + 1e-5, upper - 1e-5)
    if args.local_refine and args.orientation == 'palm_up':
        positions = list(itertools.product([.085, .09, .095], [-.005, .0, .005], [.095, .10, .105],
                                           [.014, .018, .022], [.015, .02, .025], [.006, .009, .012],
                                           [.002, .004, .006]))
    elif args.local_refine:
        positions = list(itertools.product([.078, .080, .082], [.003, .005, .007], [.055],
                                           [.010, .012, .014], [.030, .035, .040],
                                           [-.011, -.009, -.007], [.003, .004, .005]))
    elif args.orientation == 'palm_up':
        positions = list(itertools.product([.09, .105, .12], [-.01, 0., .01], [.07, .085, .10],
                                           [.012, .018], [.02, .035], [-.009, 0., .009],
                                           [0., .002, .004]))
    else:
        positions = list(itertools.product([.08, .09, .10], [-.005, .005], [.055, .07, .085],
                                           [.012, .016], [.020, .035], [-.009, -.004, .001],
                                           [0., .001, .002, .004]))
    np.random.default_rng(42).shuffle(positions)
    candidates, accepted = [], []
    indices = range(len(positions)) if args.candidate_index is not None else range(min(args.max_candidates, len(positions)))
    for n in indices:
        if args.candidate_index is not None and n != args.candidate_index:
            continue
        cx, cy, z, inset, thumb_y, z_offset, squeeze = positions[n]
        center = np.array([cx, cy, z])
        tip_z = z + z_offset
        y_sign = -1 if args.orientation == 'palm_up' else 1
        targets = np.array([[cx - inset, cy + y_sign * thumb_y, tip_z],
                            [cx + inset, cy + y_sign * .025, tip_z],
                            [cx + inset + args.middle_x_offset, cy - y_sign * .020, tip_z]])

        def residual(joints):
            data.qpos[:hand] = seed
            data.qpos[:12] = joints
            mujoco.mj_kinematics(model, data)
            return np.r_[(data.site_xpos[sites] - targets).ravel() * 100,
                         .12 * (joints - seed[:12])]

        fit = least_squares(residual, seed[:12], bounds=(lower[:12] + 1e-6, upper[:12] - 1e-6), max_nfev=100)
        pose = seed.copy()
        pose[:12] = fit.x
        residual(fit.x)
        ik_error_mm = float(np.linalg.norm(data.site_xpos[sites] - targets) * 1000)
        fitted_tips = data.site_xpos[sites].copy().tolist()
        closed_targets = targets.copy()
        closed_targets[0, 0] += squeeze
        closed_targets[1:, 0] -= squeeze

        def squeeze_residual(joints):
            data.qpos[:hand] = pose
            data.qpos[:12] = joints
            mujoco.mj_kinematics(model, data)
            return np.r_[(data.site_xpos[sites] - closed_targets).ravel() * 100,
                         .12 * (joints - pose[:12])]

        squeeze_fit = least_squares(squeeze_residual, pose[:12],
                                    bounds=(lower[:12] + 1e-6, upper[:12] - 1e-6), max_nfev=80)
        control = pose.copy()
        control[:12] = squeeze_fit.x
        mujoco.mj_resetData(model, data)
        data.qpos[:hand] = pose
        data.qpos[hand:hand + 3] = center
        data.ctrl[:] = control
        initial_qpos = data.qpos.copy()
        mujoco.mj_forward(model, data)
        initial_contacts = int(data.ncon)
        initial_tip_count = contact_forces(model, data, pen, tip_bodies, palm)[0]
        initial_pen_contact_depth_mm = float(max([-c.dist for c in data.contact if
            pen in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])] + [0.]) * 1000)
        initial_normal_force_n = contact_forces(model, data, pen, tip_bodies, palm)[-1]
        tilt, counts, others, palms, drifts, forces = [], [], [], [], [], []
        trace = []
        other_bodies = {}
        steps = round(args.settle_seconds / model.opt.timestep)
        for step in range(steps):
            mujoco.mj_step(model, data)
            if step in (0, 1, 2, 5, 10, 20, 50, 100, 200, 399, steps - 1):
                trace.append(dict(step=step, pen_pos=data.qpos[hand:hand + 3].tolist(),
                                  tip_count=contact_forces(model, data, pen, tip_bodies, palm)[0],
                                  normal_force_n=contact_forces(model, data, pen, tip_bodies, palm)[-1]))
            if step >= steps // 2 and step % 5 == 0:
                for ci, contact in enumerate(data.contact):
                    a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
                    partner = b if a == pen else a if b == pen else -1
                    if partner >= 0 and partner not in tip_bodies:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, wrench)
                        name = model.body(partner).name
                        other_bodies[name] = other_bodies.get(name, 0.) + max(float(wrench[0]), 0.)
                axis = data.xmat[pen].reshape(3, 3)[:, 2]
                tilt.append(float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0, 1)))))
                count, other, palm_force, force = contact_forces(model, data, pen, tip_bodies, palm)
                counts.append(count)
                others.append(other)
                palms.append(palm_force)
                forces.append(force)
                drifts.append(float(np.linalg.norm(data.qpos[hand:hand + 3] - center)))
        other_bodies = {name: force / len(tilt) for name, force in other_bodies.items()}
        row = dict(candidate=n, center=center.tolist(), inset=inset, thumb_y=thumb_y,
                   z_offset=z_offset,
                   middle_x_offset=args.middle_x_offset, settle_seconds=args.settle_seconds,
                   squeeze_m=squeeze,
                   ik_error_mm=ik_error_mm, fitted_tips=fitted_tips,
                   initial_contacts=initial_contacts, initial_tip_count=initial_tip_count,
                   initial_pen_contact_depth_mm=initial_pen_contact_depth_mm,
                   initial_normal_force_n=initial_normal_force_n, trace=trace,
                   mean_tilt_deg=float(np.mean(tilt)), p95_tilt_deg=float(np.percentile(tilt, 95)),
                   two_tip_fraction=float(np.mean(np.array(counts) >= 2)),
                   mean_other_normal_force_n=float(np.mean(others)),
                   mean_palm_normal_force_n=float(np.mean(palms)),
                   mean_normal_force_sum_n=float(np.mean(forces)), final_drift_m=drifts[-1],
                   other_body_mean_force_n=other_bodies,
                   min_home_limit_margin_deg=float(np.degrees(np.minimum(control - low, high - control)).min()))
        valid = (row['p95_tilt_deg'] < 15 and row['two_tip_fraction'] > .95
                 and row['final_drift_m'] < .025 and row['mean_palm_normal_force_n'] < .02
                 and row['mean_other_normal_force_n'] < .10)
        score = (3 * row['two_tip_fraction'] - .08 * row['mean_tilt_deg']
                 - 2 * row['mean_other_normal_force_n'] - 25 * row['final_drift_m']
                 - .03 * row['mean_normal_force_sum_n'])
        row.update(accepted=bool(valid), score=float(score))
        candidates.append(row)
        if valid:
            # A long-horizon candidate is certified from its pre-settling pose.
            # Reusing the final pose with zeroed velocity can induce slow drift.
            saved_qpos = initial_qpos if args.settle_seconds >= 12 else data.qpos.copy()
            accepted.append((score, saved_qpos, control.copy(), row))
        if args.candidate_index is not None:
            (assets / f'calibration_debug_{n}.json').write_text(json.dumps(row, indent=2))
            print('DEBUG', args.ring_flex, args.ring_abd, args.pinky_flex, args.pinky_abd, row['two_tip_fraction'],
                  row['mean_other_normal_force_n'], row['final_drift_m'], row['other_body_mean_force_n'], flush=True)
        elif (n + 1) % 12 == 0:
            print('PROGRESS', n + 1, 'accepted', len(accepted), 'best', max(c['score'] for c in candidates), flush=True)
            candidates_path.write_text(json.dumps(candidates, indent=2))
    if args.candidate_index is None:
        candidates_path.write_text(json.dumps(candidates, indent=2))
    if args.candidate_index is not None:
        return
    if not accepted:
        raise RuntimeError(f'No Wuji grasp passed gravity/contact gates; inspect {candidates_path.name}.')
    accepted.sort(key=lambda x: x[0], reverse=True)
    selected = accepted[:8]
    np.savez(bank_path, qpos=np.array([x[1] for x in selected]),
             ctrl=np.array([x[2] for x in selected]))
    metrics_path.write_text(json.dumps([x[3] for x in selected], indent=2))
    if args.bank_name == 'grasp_bank.npz':
        provenance = json.loads((assets / 'provenance.json').read_text())
        pose_kind = 'pre-settling pose' if args.settle_seconds >= 12 else 'settled pose'
        provenance.update(grasp_count=len(selected), calibration=f'three-finger IK and {args.settle_seconds:g}s physical validation; {pose_kind} saved')
        (assets / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('SAVED', len(selected), 'grasps', flush=True)


if __name__ == '__main__':
    main()
