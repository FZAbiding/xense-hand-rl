"""Summarize held-out, matched-task CPU evaluations for the three hands."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('mae_deg', 'mean_tilt_deg', 'p95_tilt_deg', 'command_clip_fraction',
          'non_tip_force_frame_fraction', 'palm_force_frame_fraction',
          'two_tip_fraction', 'minimum_half_cycle_gain')
SCENES = {'LEAP': 'assets/scene.xml', 'Revo3': 'assets/revo3_v2/scene.xml',
          'Wuji Hand2': 'assets/wuji_hand2/scene.xml'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for hand in ('leap', 'revo3', 'wuji'):
        parser.add_argument(f'--{hand}', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluations = {}
    expected = None
    for hand, name in (('leap', 'LEAP'), ('revo3', 'Revo3'), ('wuji', 'Wuji Hand2')):
        path = ROOT / getattr(args, hand) / 'metrics.json'
        result = json.loads(path.read_text())
        task = {k: result[k] for k in ('seed', 'episodes', 'seconds', 'amplitude_rad', 'period_seconds')}
        if expected is None:
            expected = task
        elif task != expected:
            raise ValueError(f'Task mismatch for {name}: {task} != {expected}')
        if task['episodes'] != 24 or task['seconds'] != 60 or task['amplitude_rad'] != .6 or task['period_seconds'] != 4.:
            raise ValueError(f'Not the predeclared 24 x 60 s, ±0.6 rad / 4 s test: {task}')
        prefix = 'strict_' if hand == 'leap' else ''
        row = {key: result[prefix + key] for key in FIELDS}
        row.update(quality_success_rate=result[prefix + 'quality_success_rate'],
                   survival_rate=result['survival_rate'],
                   workspace_survival_rate=float(np.mean([
                       ep['strict']['within_workspace'] if hand == 'leap' else ep['within_workspace']
                       for ep in result['per_episode']])),
                   checkpoint=result['checkpoint'], evaluation=str(path.relative_to(ROOT)),
                   checkpoint_sha256=digest(ROOT / result['checkpoint']))
        model = mujoco.MjModel.from_xml_path(str(ROOT / SCENES[name]))
        pen = model.geom('pen_collision').id
        row.update(actuators=model.nu, scene=SCENES[name],
                   scene_sha256=digest(ROOT / SCENES[name]),
                   pen_radius_m=float(model.geom_size[pen, 0]),
                   pen_length_m=float(2 * (model.geom_size[pen, 0] + model.geom_size[pen, 1])),
                   pen_mass_kg=float(model.body_mass[model.body('pen').id]))
        evaluations[name] = row
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(task=expected, evaluations=evaluations), indent=2))
    print('| Hand | Strict pass | Survival | MAE | Non-tip contact frames | Two-tip frames |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, row in evaluations.items():
        print(f"| {name} | {row['quality_success_rate']:.0%} | {row['survival_rate']:.0%} | "
              f"{row['mae_deg']:.2f}° | {row['non_tip_force_frame_fraction']:.1%} | "
              f"{row['two_tip_fraction']:.1%} |")


if __name__ == '__main__':
    main()
