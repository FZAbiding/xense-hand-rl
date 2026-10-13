"""Run the isolated upright curriculum, with CPU gates before expansion."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name('config.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    base = ROOT / cfg['runs']
    base.mkdir(parents=True, exist_ok=True)
    if (base / 'status.json').exists():
        raise RuntimeError('Pipeline already ran; do not overwrite its status')
    shutil.copy2(config_path, base / 'experiment_config.json')
    status = dict(experiment_id=cfg['experiment_id'], config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), stages=[])
    def record():
        (base / 'status.json').write_text(json.dumps(status, indent=2))
    def execute(label, args):
        print('EXECUTE', label, flush=True)
        with (base / f'{label}.log').open('x') as log:
            subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, MUJOCO_GL='egl'))
    previous = cfg.get('initial_checkpoint')
    settings = cfg['training'].copy()
    stages = settings.pop('stages')
    for stage in stages:
        name = stage['name']
        run = base / name
        params = dict(settings, **{k: v for k, v in stage.items() if k != 'name'}, run=str(run.relative_to(ROOT)))
        params['experiment_config'] = str(config_path.relative_to(ROOT))
        if previous:
            params['resume'] = previous
        args = ['experiments/wuji_hand2/train.py']
        for key, value in params.items():
            args.extend(['--' + key.replace('_', '-'), str(value)])
        execute('train_' + name, args)
        results = []
        # Fixed validation seed used only to choose among this stage's models.
        checkpoints = sorted(run.glob('checkpoint_*.pt'))
        for checkpoint in checkpoints:
            output = base / ('eval_' + name + '_' + checkpoint.stem)
            execute(output.name, ['experiments/wuji_hand2/evaluate.py', '--checkpoint', str(checkpoint.relative_to(ROOT)),
                '--output', str(output.relative_to(ROOT)), '--episodes', str(cfg['evaluation']['episodes']),
                '--seconds', str(cfg['evaluation']['seconds']), '--seed', str(cfg['evaluation']['validation_seed'])])
            metrics = json.loads((output / 'metrics.json').read_text())
            results.append((metrics, checkpoint, output))
        passing = [r for r in results if r[0]['quality_success_rate'] == 1]
        pool = passing or results
        best, checkpoint, evaluation = min(pool, key=lambda r: (-r[0]['quality_success_rate'], -r[0]['workspace_survival_rate'], r[0]['mae_deg']))
        previous = str(checkpoint.relative_to(ROOT))
        status['stages'].append(dict(name=name, selected_checkpoint=previous, evaluation=str(evaluation.relative_to(ROOT)),
                                    metrics={k: v for k, v in best.items() if k != 'per_episode'}))
        record()
        print('STAGE_RESULT', name, best['quality_success_rate'], best['survival_rate'], best['mae_deg'], flush=True)
        # Short episodes bootstrap the long stage; all subsequent expansion
        # requires 8/8 independent 60-second CPU quality passes.
        survival_pass = best['survival_rate'] == 1 and best['workspace_survival_rate'] == 1
        gate_pass = bool(passing) if cfg.get('advance_gate', 'quality') == 'quality' else survival_pass
        if name != 'small_short' and not gate_pass:
            status['outcome'] = 'stopped_at_cpu_'+cfg.get('advance_gate', 'quality')+'_gate'
            record()
            break
    else:
        status['outcome'] = 'curriculum_complete'
    holdout = base / 'holdout_cpu60'
    execute('holdout_cpu60', ['experiments/wuji_hand2/evaluate.py', '--checkpoint', previous,
        '--output', str(holdout.relative_to(ROOT)), '--episodes', str(cfg['evaluation']['holdout_episodes']),
        '--seconds', '60', '--seed', str(cfg['evaluation']['holdout_seed']), '--video'])
    status['holdout'] = json.loads((holdout / 'metrics.json').read_text())
    status['selected_checkpoint'] = previous
    record()
    dest = ROOT / cfg['artifacts']
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / previous, dest / 'policy.pt')
    for source, name in [(config_path, 'experiment_config.json'), (base / 'status.json', 'status.json'),
                         ((ROOT / previous).parent / 'config.json', 'training_config.json')]:
        shutil.copy2(source, dest / name)
    for name in ('metrics.json', 'evaluation_manifest.json', 'rollout.npz', 'rollout.mp4', 'preview.png', 'tracking.png'):
        if (holdout / name).exists():
            shutil.copy2(holdout / name, dest / name)
    print('FINISHED', status['outcome'], dest, flush=True)


if __name__ == '__main__':
    main()
