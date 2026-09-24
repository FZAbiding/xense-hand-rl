"""Package only explicitly stage-qualified checkpoints; preserve prior versions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--cpu',required=True);p.add_argument('--gpu',required=True);p.add_argument('--holdout',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();out=ROOT/args.output
    if out.exists():raise RuntimeError('Refusing to overwrite an existing stage package')
    evaluations={k:json.loads((ROOT/getattr(args,k)/'metrics.json').read_text()) for k in ['cpu','gpu','holdout']}
    for key,r in evaluations.items():
        assert (ROOT/r['checkpoint']).resolve()==(ROOT/args.checkpoint).resolve(),key
        assert r['seconds']>=60 and r['quality_success_rate']>=.95,key
        assert r['survival_rate']==1 and r['workspace_survival_rate']==1,key
    run=(ROOT/args.checkpoint).parent
    cfg=json.loads((run/'config.json').read_text())
    for r in evaluations.values():assert r['amplitude_rad']==cfg['amplitude'] and r['period_seconds']==cfg['period']
    out.mkdir(parents=True)
    files={ROOT/args.checkpoint:'policy.pt',run/'config.json':'training_config.json',
           run/'source_manifest.json':'training_source_manifest.json',run/'metrics.jsonl':'training_metrics.jsonl'}
    for key in evaluations:
        path=ROOT/getattr(args,key)
        files[path/'metrics.json']=f'{key}_evaluation.json'
        files[path/'rollout.npz']=f'{key}_rollout.npz'
        if (path/'tracking.png').exists():files[path/'tracking.png']=f'{key}_tracking.png'
    path=ROOT/args.cpu
    if (path/'full_episode_00.mp4').exists():files[path/'full_episode_00.mp4']='demo_60s.mp4'
    if (path/'preview_00.png').exists():files[path/'preview_00.png']='preview.png'
    for src,name in files.items():shutil.copy2(src,out/name)
    for name in ['evaluate.py','evaluate_gpu.py','env.py','render.py','package.py']:
        shutil.copy2(ROOT/'experiments/revo3_v2'/name,out/('evaluator_'+name))
    status=dict(version='2.0',stage_amplitude_rad=cfg['amplitude'],period_seconds=cfg['period'],
                status='simulation_stage_qualified',full_task_success=False,hardware_validated=False,
                independent_training_seeds=1,checkpoint=args.checkpoint,
                evaluations={k:{n:v for n,v in r.items() if n!='per_episode'} for k,r in evaluations.items()},
                limitations=['Privileged simulator force observations','Rigid distal links, no identified silicone layer',
                             'Three initial grasps; small initial-state perturbations only','Not a hardware-deployable policy'])
    (out/'stage_status.json').write_text(json.dumps(status,indent=2))
    (out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},indent=2))
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
