"""Independent natural-pad grasp search; keep all official collision geometry."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'assets/wuji_hand2_palm_up_human_demo_v4'
OUT = ROOT / 'runs/wuji_hand2/palm_up_human_demo_v4'
FINGERS = ('thumb', 'index_finger', 'middle_finger')
# Software task envelope, deliberately distinct from official mechanical limits.
LOW_DEG = np.array([0, -75, 0, 5, 0, -30, 10, 15, 0, -30, 10, 15])
HIGH_DEG = np.array([70, 35, 75, 80, 75, 30, 75, 85, 75, 30, 75, 85])


def contacts(m, d):
    """Actual force and local contact position for every pen/hand contact."""
    pen = m.body('pen').id
    rows = []
    for i, c in enumerate(d.contact):
        a, b = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
        body = b if a == pen else a if b == pen else -1
        if body < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        normal = max(0., float(force[0]))
        local = d.xmat[body].reshape(3, 3).T @ (c.pos - d.xpos[body])
        rows.append(dict(body=m.body(body).name, normal_n=normal,
                         local_position_m=local.tolist(), world_position_m=c.pos.tolist()))
    return rows


def assess(m, q, ctrl, seconds, rng=None, record=False):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    center = q[20:23].copy()
    if rng is not None:
        d.qpos[:12] += rng.normal(0, .003, 12)
        d.qpos[20:23] += rng.normal(0, .0003, 3)
    trace, qs, all_contacts = [], [], []
    for t in range(round(seconds / .025)):
        for _ in range(5):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        cc = contacts(m, d)
        tip = np.zeros(3)
        non_tip = inactive = total = 0.
        for c in cc:
            total += c['normal_n']
            name = c['body']
            matched = False
            for j, f in enumerate(FINGERS):
                if name in (f'r_{f}_distal', f'r_{f}_tip_sensor_frame'):
                    tip[j] += c['normal_n']
                    matched = True
            if not matched:
                non_tip += c['normal_n']
            if 'ring_finger' in name or 'pinky' in name:
                inactive += c['normal_n']
        drift = np.linalg.norm(d.qpos[20:23] - center)
        functional = d.qpos[[6, 7, 10, 11]].min() >= 0
        trace.append([drift, np.sum(tip > .02), non_tip, inactive, total, functional])
        if record:
            qs.append(d.qpos.copy())
            all_contacts.append(cc)
        if drift > .05 or not np.isfinite(d.qpos).all():
            break
    a = np.asarray(trace)
    result = dict(frames=len(a), seconds=seconds, max_drift_m=float(a[:, 0].max()),
                  two_tip_fraction=float(np.mean(a[:, 1] >= 2)),
                  non_tip_fraction=float(np.mean(a[:, 2] > .05)),
                  inactive_fraction=float(np.mean(a[:, 3] > .02)),
                  mean_force_n=float(a[:, 4].mean()),
                  reverse_bending_fraction=float(1 - a[:, 5].mean()))
    result['passed'] = bool(len(a) == round(seconds / .025) and result['max_drift_m'] < .025
                            and result['two_tip_fraction'] > .95 and result['non_tip_fraction'] < .05
                            and result['inactive_fraction'] < .01 and result['reverse_bending_fraction'] == 0)
    return result, d.qpos.copy(), np.asarray(qs), all_contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=int, default=384)
    args = parser.parse_args()
    if ASSETS.exists():
        raise RuntimeError('Choose a new search variant; do not overwrite an existing asset search')
    ASSETS.mkdir()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'assets/wuji_hand2_palm_up/scene.xml', ASSETS / 'scene.xml')
    m = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene.xml'))
    m.opt.iterations = 8
    m.opt.ls_iterations = 8
    d = mujoco.MjData(m)
    bodies = [m.body(f'r_{f}_distal').id for f in FINGERS]
    low, high = np.radians(LOW_DEG), np.radians(HIGH_DEG)
    rng = np.random.default_rng(2026101104)
    seed = np.radians([40, -25, 35, 25, 25, 0, 35, 35, 25, 0, 35, 35])
    rows, feasible = [], []
    t0 = time.time()
    for i in range(args.candidates):
        center = np.array([rng.uniform(.035, .095), rng.uniform(-.025, .015), rng.uniform(.085, .145)])
        yaw = rng.uniform(-.45, .45)
        axis = np.array([np.sin(yaw), -np.cos(yaw), 0.])
        radial = np.array([np.cos(yaw), np.sin(yaw), 0.])
        along = np.array([rng.uniform(-.025, .025), -.024, .008])
        radius = .009 + rng.uniform(-.001, .002)
        targets = center + along[:, None] * axis + np.array([-1, 1, 1])[:, None] * radius * radial
        normals = np.array([1, -1, -1])[:, None] * radial
        fold = np.tile(np.radians([rng.choice([55, 70, 85]), 0, rng.choice([80, 95, 110]), 65]), 2)

        def residual(x):
            d.qpos[:12] = x
            d.qpos[12:20] = fold
            mujoco.mj_kinematics(m, d)
            rotations = d.xmat[bodies].reshape(3, 3, 3)
            pads = d.xpos[bodies] + np.einsum('nij,j->ni', rotations, [0, .004, -.019])
            # Fit volar pads, not the end-of-finger sites used by earlier banks.
            return np.r_[((pads - targets) * 100).ravel(),
                         ((rotations[:, :, 1] - normals) * .35).ravel(),
                         .08 * (x - seed), .10 * (x[[7, 11]] - .7 * x[[6, 10]])]

        initial = np.clip(seed + rng.normal(0, .12, 12), low + 1e-5, high - 1e-5)
        fit = least_squares(residual, initial, bounds=(low, high), max_nfev=70)
        residual(fit.x)
        error = float(np.linalg.norm(fit.fun[:9]) / 100)
        q = m.qpos0.copy()
        q[:20] = np.r_[fit.x, fold]
        q[20:23] = center
        # Local pen +z maps to the directed, horizontal source-aligned axis.
        mujoco.mju_quatZ2Vec(q[23:27], axis)
        ctrl = q[:20].copy()
        # Small inward pad squeeze in task space; bounded IK retains natural posture.
        targets += np.array([1, -1, -1])[:, None] * radial * rng.uniform(.0005, .0025)
        squeeze = least_squares(residual, fit.x, bounds=(low, high), max_nfev=35)
        ctrl[:12] = squeeze.x
        q[:12] = squeeze.x
        result, settled, _, _ = assess(m, q, ctrl, 2.)
        result.update(candidate=i, ik_error_m=error, center=center.tolist(), yaw=yaw)
        rows.append(result)
        if result['passed']:
            longer, _, _, _ = assess(m, q, ctrl, 12.)
            result['long_check'] = longer
            if longer['passed']:
                feasible.append((q, ctrl, i))
                np.savez(ASSETS / f'candidate_{i:04d}.npz', qpos=q, ctrl=ctrl)
        if i % 16 == 0 or result['passed']:
            (ASSETS / 'search.json').write_text(json.dumps(rows, indent=2))
            print('SEARCH', i, 'stable', len(feasible), 'elapsed', round(time.time() - t0), 'last', result, flush=True)
        if len(feasible) >= 8:
            break
    (ASSETS / 'search.json').write_text(json.dumps(rows, indent=2))
    accepted, trials = [], []
    for q, c, i in feasible:
        passed = True
        for j in range(5):
            row, _, _, _ = assess(m, q, c, 60., None if j == 0 else rng)
            row.update(candidate=i, perturbation=j)
            trials.append(row)
            passed &= row['passed']
            if not passed:
                break
        if passed:
            accepted.append((q, c, i))
        print('CERTIFY', i, passed, flush=True)
    (ASSETS / 'certification.json').write_text(json.dumps(dict(rows=trials, selected=[x[2] for x in accepted]), indent=2))
    # Preserve diagnostic states even when the strict stage gate rejects all candidates.
    best = sorted(range(len(rows)), key=lambda j: (-rows[j]['two_tip_fraction'], rows[j]['non_tip_fraction'], rows[j]['max_drift_m']))[:8]
    (ASSETS / 'search_summary.json').write_text(json.dumps(dict(accepted=len(accepted), best=[rows[j] for j in best]), indent=2))
    if not accepted:
        raise RuntimeError('No functional, certified natural grasp; do not start rotation training')
    np.savez(ASSETS / 'grasp_bank_60s.npz', qpos=np.array([x[0] for x in accepted]), ctrl=np.array([x[1] for x in accepted]))
    (ASSETS / 'provenance.json').write_text(json.dumps(dict(source='official palm-up scene, unchanged physical model',
        software_low_deg=LOW_DEG.tolist(), software_high_deg=HIGH_DEG.tolist(), calibration='new volar-pad IK and full-geometry dynamics; no v3 poses or policy', grasp_count=len(accepted)), indent=2))
    (ASSETS / 'certified_bank_manifest.json').write_text(json.dumps({name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
        for name in ['scene.xml', 'grasp_bank_60s.npz', 'certification.json']}, indent=2))


if __name__ == '__main__':
    main()
