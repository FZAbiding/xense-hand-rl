"""Run paired frozen standard and push validation for one v10 checkpoint."""
import argparse
import json
import subprocess
import sys

import torch

from .env import ROOT
from .train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--no-feedback', action='store_true')
    args = parser.parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ck['config']
    iteration = ck['iteration']
    art = ROOT / config['artifacts']
    outcomes = {}
    for name, seed, pushed in (
        (f'validation_{iteration:07d}', config['validation_seed'], False),
        (f'push_validation_{iteration:07d}', config['push_validation_seed'], True),
    ):
        folder = art / name
        metrics = folder / 'metrics.json'
        video = folder / 'rollout.mp4'
        if not (metrics.is_file() and video.is_file()):
            if folder.exists():
                raise RuntimeError(f'Incomplete validation directory; inspect before retry: {folder}')
            command = [sys.executable, '-m',
                       'experiments.wuji_hand2.palm_up_human_demo_v10.long_training.evaluate',
                       '--checkpoint', args.checkpoint, '--seed', str(seed),
                       '--output-name', name, '--no-feedback']
            if pushed:
                command.append('--diagnostic-push')
            subprocess.run(command, cwd=ROOT, check=True)
        data = json.loads(metrics.read_text())
        if data['checkpoint_sha256'] != outcomes.get('checkpoint_sha256', data['checkpoint_sha256']):
            raise RuntimeError('Paired validation checkpoint mismatch')
        outcomes['checkpoint_sha256'] = data['checkpoint_sha256']
        outcomes['push' if pushed else 'standard'] = data
    standard = outcomes['standard']
    push = outcomes['push']
    failures = [episode['failed_gates'] for episode in standard['episodes']]
    feedback = dict(
        iteration=iteration, level=ck['curriculum_level'],
        success_fraction=standard['success_fraction'],
        standard_success_fraction=standard['success_fraction'],
        push_success_fraction=push['success_fraction'],
        posture_fail_episodes=sum('posture_or_reverse' in gates for gates in failures),
        full_duration_fail_episodes=sum('full_duration' in gates for gates in failures),
        promote=(standard['success_fraction'] >= config['gates']['success_fraction_min'] and
                 push['success_fraction'] >= config['push_validation_success_fraction_min']),
        evidence=[str((art / f'validation_{iteration:07d}' / 'metrics.json').relative_to(ROOT)),
                  str((art / f'push_validation_{iteration:07d}' / 'metrics.json').relative_to(ROOT))],
        checkpoint_sha256=outcomes['checkpoint_sha256'],
    )
    if not args.no_feedback:
        atomic_json(ROOT / config['runs'] / 'evaluation_feedback.json', feedback)
    print(json.dumps(feedback), flush=True)


if __name__ == '__main__':
    main()
