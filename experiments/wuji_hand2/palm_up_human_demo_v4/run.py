"""Advance the independent curriculum only after CPU stage gates pass."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='experiments/wuji_hand2/palm_up_human_demo_v4/config_handover.json')
    p.add_argument('--wait-for-handover',action='store_true')
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    runs=ROOT/config['runs'];artifacts=ROOT/config['artifacts']
    artifacts.mkdir(parents=True,exist_ok=True)
    status_path=runs/'curriculum_status.json'
    if status_path.exists():raise RuntimeError('Existing curriculum; inspect its status before restarting')
    status=dict(experiment_id=config['experiment_id'],accepted_demo=config['accepted_demo'],
                accepted_demo_sha256=hashlib.sha256((ROOT/config['accepted_demo']).read_bytes()).hexdigest(),
                config=args.config,stages={},status='running',started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    def save():
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(status_path)
    def call(command,log):
        print('RUN',command,flush=True)
        with log.open('w') as f:subprocess.run([sys.executable,'-m',*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    def evaluate(checkpoint,output,episodes,seconds,seed,video=False):
        command=['experiments.wuji_hand2.palm_up_human_demo_v4.evaluate','--checkpoint',str(checkpoint.relative_to(ROOT)),
                 '--output',str(output.relative_to(ROOT)),'--episodes',str(episodes),'--seconds',str(seconds),'--seed',str(seed)]
        if video:command+=['--video']
        call(command,runs/(output.name+'.log'))
        return json.loads((output/'metrics.json').read_text())
    save()
    try:
        hold=json.loads((artifacts/'hold_verified/metrics.json').read_text())
        if hold['success_fraction']<config['stage_gate']['survival_fraction_min']:
            raise RuntimeError('Hold stage has not passed; do not advance')
        status['stages']['hold']=dict(status='passed',checkpoint=hold['checkpoint'],success_fraction=hold['success_fraction'],
                                     evidence=str((artifacts/'hold_verified/metrics.json').relative_to(ROOT)))
        parent=ROOT/hold['checkpoint'];save()
        for stage in ['handover','half_turn','full_turn','continuous']:
            settings=config['stages'][stage]
            stage_run=runs/stage
            status['current_stage']=stage;save()
            if stage=='handover' and args.wait_for_handover:
                final=stage_run/f'checkpoint_{settings["iterations"]:06d}.pt'
                # The trainer writes checkpoints atomically. Keep each poll short.
                deadline=time.time()+1800
                while not final.exists():
                    if time.time()>deadline:raise RuntimeError('Handover trainer did not produce final checkpoint')
                    time.sleep(10)
            else:
                call(['experiments.wuji_hand2.palm_up_human_demo_v4.train','--config',args.config,'--stage',stage,
                      '--resume',str(parent.relative_to(ROOT))],runs/('train_'+stage+'.log'))
            rows=[]
            for checkpoint in sorted(stage_run.glob('checkpoint_*.pt')):
                output=runs/'validation'/stage/checkpoint.stem
                data=evaluate(checkpoint,output,config['validation_episodes'],settings['episode_seconds'],config['validation_seed'])
                row=dict(checkpoint=str(checkpoint.relative_to(ROOT)),success_fraction=data['success_fraction'],
                         survival_fraction=data['survival_fraction'],mae_deg=data['mean_mae_deg'],metrics=str((output/'metrics.json').relative_to(ROOT)))
                rows.append(row);print('VALIDATION',stage,row,flush=True)
            ranked=sorted(rows,key=lambda r:(-r['success_fraction'],-r['survival_fraction'],r['mae_deg']))
            selected=ranked[0];parent=ROOT/selected['checkpoint']
            # Always export the selected attempt, including an explicitly failed attempt.
            output=artifacts/stage
            held=evaluate(parent,output,config['holdout_episodes'],settings['episode_seconds'],config['holdout_seed'],video=True)
            shutil.copy2(parent,output/'policy.pt')
            shutil.copy2(stage_run/'config.json',output/'training_config.json')
            shutil.copy2(stage_run/'source_manifest.json',output/'training_source_manifest.json')
            passed=selected['success_fraction']>=config['stage_gate']['survival_fraction_min'] and held['success_fraction']>=config['stage_gate']['survival_fraction_min']
            status['stages'][stage]=dict(status='passed' if passed else 'failed_gate',validation=rows,selected=selected,
                                         holdout_success_fraction=held['success_fraction'],holdout_survival_fraction=held['survival_fraction'],
                                         evidence=str((output/'metrics.json').relative_to(ROOT)))
            save()
            if not passed:
                status['status']='stopped_at_failed_gate'
                status['reason']='Do not train larger rotations from a failed handover/motion stage'
                save();print('STAGE_GATE_FAILED',stage,flush=True);return
        status['status']='completed_curriculum';save()
    except Exception as exc:
        status['status']='execution_error';status['error']=repr(exc);save();raise


if __name__=='__main__':main()
