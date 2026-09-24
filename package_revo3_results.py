"""Package an independently validated Revo3 policy without replacing LEAP."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--gpu-eval',required=True);p.add_argument('--cpu-eval',required=True)
    p.add_argument('--output',default='artifacts/revo3/v1.0');args=p.parse_args()
    gpu=json.loads((ROOT/args.gpu_eval/'metrics.json').read_text())
    cpu=json.loads((ROOT/args.cpu_eval/'metrics.json').read_text())
    for result in (gpu,cpu):
        if result['robot']!='revo3' or result['success_rate']<.9:
            raise ValueError('Handoff requires Revo3 success >=90% in both independent evaluations.')
        if Path(result['checkpoint']).resolve()!=Path(args.checkpoint).resolve():
            raise ValueError('The evaluation must correspond to the packaged checkpoint.')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    source={
        'policy.pt':ROOT/args.checkpoint,
        'demo.mp4':ROOT/args.gpu_eval/'rollout.mp4',
        'preview.png':ROOT/args.gpu_eval/'preview.png',
        'tracking.png':ROOT/args.gpu_eval/'tracking.png',
        'gpu_evaluation.json':ROOT/args.gpu_eval/'metrics.json',
        'cpu_evaluation.json':ROOT/args.cpu_eval/'metrics.json',
        'gpu_rollout.npz':ROOT/args.gpu_eval/'rollout.npz',
        'cpu_rollout.npz':ROOT/args.cpu_eval/'rollout.npz',
        'asset_provenance.json':ROOT/'assets/revo3/provenance.json',
    }
    audit=ROOT/args.gpu_eval/'contact_audit.json'
    if audit.exists():source['contact_audit.json']=audit
    for filename,path in source.items():shutil.copy2(path,out/filename)
    manifest={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in source}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    task_files=['assets/revo3/scene.xml','assets/revo3/grasp.npz','assets/pen.obj','assets/pen.mtl',
                'pen_env.py','train.py','evaluate.py','evaluate_cpu.py','hand_config.py','build_revo3.py']
    (out/'task_manifest.json').write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in task_files},indent=2))
    report=dict(gpu=gpu,cpu=cpu)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for path in sorted((ROOT/'runs').glob('revo3_*/metrics.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if not rows or 'iteration' not in rows[0]:continue
        x=[r['iteration'] for r in rows]
        axs[0].plot(x,[np.degrees(r['error_rad']) for r in rows],label=path.parent.name)
        axs[1].plot(x,[r['episode_steps']*.025 for r in rows],label=path.parent.name)
    axs[0].set_ylabel('Training angle MAE (deg)');axs[1].set_ylabel('Training episode length (s)')
    for ax in axs:ax.set_xlabel('PPO iteration');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    print(out)

if __name__=='__main__':main()
