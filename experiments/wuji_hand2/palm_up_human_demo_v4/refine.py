"""Refine new functional IK grasps without touching prior searches."""
import json
import hashlib
import time
import numpy as np
import mujoco
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ASSETS, LOW_DEG, HIGH_DEG, assess


def main():
    report = ASSETS / 'refinement.json'
    if report.exists():
        raise RuntimeError('No overwrite')
    m = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene.xml'))
    m.opt.iterations = m.opt.ls_iterations = 8
    sources = [np.load(p) for p in sorted(ASSETS.glob('candidate_*.npz'))]
    rng = np.random.default_rng(2026101105)
    low, high = np.radians(LOW_DEG), np.radians(HIGH_DEG)
    rows, trials, accepted = [], [], []
    t0 = time.time()
    for i in range(1536):
        src = sources[i % len(sources)]
        q, c = src['qpos'].copy(), src['ctrl'].copy()
        c[:12] = np.clip(c[:12] + rng.normal(0, [.005, .015, .04, .08][i % 4], 12), low, high)
        row, settled, _, _ = assess(m, q, c, 3.)
        row['candidate'] = i
        if row['passed']:
            long, _, _, _ = assess(m, q, c, 12.)
            row['long_check'] = long
            if long['passed']:
                checks = []
                for j in range(5):
                    rr, _, _, _ = assess(m, q, c, 60., None if j == 0 else rng)
                    rr.update(candidate=i, perturbation=j)
                    checks.append(rr)
                    trials.append(rr)
                    if not rr['passed']:
                        break
                row['certification'] = checks
                if len(checks) == 5 and all(x['passed'] for x in checks):
                    accepted.append((q, c, i))
                    np.savez(ASSETS / f'certified_{i:04d}.npz', qpos=q, ctrl=c)
                    print('CERTIFIED', i, len(accepted), flush=True)
        rows.append(row)
        if i % 32 == 0:
            report.write_text(json.dumps(dict(search=rows, trials=trials, accepted=[x[2] for x in accepted]), indent=2))
            print('REFINE', i, 'accepted', len(accepted), 'elapsed', round(time.time() - t0), flush=True)
        if len(accepted) >= 2:
            break
    report.write_text(json.dumps(dict(search=rows, trials=trials, accepted=[x[2] for x in accepted]), indent=2))
    if not accepted:
        raise RuntimeError('No certified functional grasp; keep diagnostic results')
    np.savez(ASSETS / 'grasp_bank_60s.npz', qpos=np.array([x[0] for x in accepted]), ctrl=np.array([x[1] for x in accepted]))
    (ASSETS / 'provenance.json').write_text(json.dumps(dict(source='official palm-up scene, full physical geometry unchanged',
        software_low_deg=LOW_DEG.tolist(), software_high_deg=HIGH_DEG.tolist(),
        calibration='new natural volar-pad IK, bounded refinement, five independent 60s trials per grasp', grasp_count=len(accepted)), indent=2))
    (ASSETS / 'certified_bank_manifest.json').write_text(json.dumps({name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
        for name in ['scene.xml', 'grasp_bank_60s.npz', 'refinement.json']}, indent=2))


if __name__ == '__main__':
    main()
