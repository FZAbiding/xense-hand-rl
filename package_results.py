"""Collect the evaluated policy, video and measured results for handoff."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'artifacts/leap';out.mkdir(parents=True,exist_ok=True)
    copies={
        'policy.pt':'runs/ppo_wide/checkpoint_000800.pt',
        'demo.mp4':'runs/eval_wide_800/rollout.mp4',
        'preview.png':'runs/eval_wide_800/preview.png',
        'tracking.png':'runs/eval_wide_800/tracking.png',
        'gpu_evaluation.json':'runs/eval_wide_800/metrics.json',
        'cpu_60s_evaluation.json':'runs/cpu_wide/metrics.json',
    }
    for dest,src in copies.items():shutil.copy2(ROOT/src,out/dest)
    experiments={
        'static_baseline':'runs/static_baseline/metrics.json',
        'ik_baseline':'runs/ik_baseline/metrics.json',
        'ppo_small_300':'runs/eval_300/metrics.json',
        'ppo_small_550':'runs/eval_550/metrics.json',
        'ppo_small_60s_gpu':'runs/eval_550_long/metrics.json',
        'ppo_small_cpu':'runs/cpu_validation/metrics.json',
        'ppo_wide_800':'runs/eval_wide_800/metrics.json',
        'ppo_wide_60s_cpu':'runs/cpu_wide/metrics.json',
    }
    results={}
    for name,src in experiments.items():
        record=json.loads((ROOT/src).read_text())
        record={k:v for k,v in record.items() if not k.startswith('per_episode')}
        if 'success_definition' in record:
            seconds=record['seconds'];period=record['period_seconds'];amp=record['amplitude_rad']
            record['success_definition']=f'{seconds:g}s without falling, >={int(seconds/period)*2} alternating crossings of +/-{max(.15,amp*.6):.3f} rad, MAE<0.25 rad, >=2 fingertips within 24mm for >80% frames'
        record['source']=src;results[name]=record
    (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    # Correct the textual time description in the exported evaluation as well.
    (out/'gpu_evaluation.json').write_text(json.dumps(results['ppo_wide_800'],indent=2))
    manifest={dest:hashlib.sha256((out/dest).read_bytes()).hexdigest() for dest in copies}
    manifest['policy_source']=copies['policy.pt']
    manifest['source_commits']=json.loads((ROOT/'assets/provenance.json').read_text())['source_commits']
    manifest['training_lineage']={'base_iteration':550,'base_steps':550*1024*32,'wide_iterations':250,'wide_steps':250*1024*32,'total_steps':800*1024*32}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for path,label in [('ppo_direct','PPO / +/-34 deg'),('ppo_wide','PPO / +/-57 deg')]:
        rows=[json.loads(line) for line in (ROOT/f'runs/{path}/metrics.jsonl').read_text().splitlines()]
        iteration=[r['iteration'] for r in rows]
        axs[0].plot(iteration,[np.degrees(r['error_rad']) for r in rows],label=label)
        axs[1].plot(iteration,[r['episode_steps']*.025 for r in rows],label=label)
    axs[0].set_ylabel('Training episode mean angle error (deg)');axs[1].set_ylabel('Training episode length (s)')
    for ax in axs:ax.set_xlabel('PPO iteration');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
