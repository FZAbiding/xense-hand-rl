"""Ensure a final held-out CPU review even if periodic evaluation is still busy."""
import fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[3]
    cfg=json.loads(Path(__file__).with_name('long_training').joinpath('config.json').read_text())
    out=root/cfg['runs'];art=root/cfg['artifacts'];lock=(out/'finalizer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'finalization_status.json'
    def write(value):
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(status_path)
    def preserve_best():
        pinned=art/'evaluation_checkpoints';pinned.mkdir(exist_ok=True)
        # CPU 24x60s review can outlast four GPU checkpoint intervals.
        # Pin every evaluation input before the trainer rotates its recent files.
        for log in out.glob('evaluate_*.log'):
            iteration=int(log.stem.split('_')[-1]);name=f'checkpoint_{iteration:07d}.pt'
            dest=pinned/name
            if dest.exists():continue
            try:content=(out/name).read_bytes()
            except FileNotFoundError:continue
            tmp=dest.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(dest)
        feedback=out/'evaluation_feedback.json'
        if not feedback.exists():return
        value=json.loads(feedback.read_text());evidence=root/value['evidence']
        if not evidence.exists():return
        result=json.loads(evidence.read_text());key=str(result['level'])
        best_path=art/'best_policies.json';best=json.loads(best_path.read_text()) if best_path.exists() else {}
        mae=sum(x['mae_deg'] for x in result['episodes'])/len(result['episodes'])
        previous=best.get(key,dict(success_fraction=-1,mae_deg=1e9))
        if (result['success_fraction'],-mae)<=(previous['success_fraction'],-previous['mae_deg']):return
        checkpoint=out/f'checkpoint_{result["iteration"]:07d}.pt'
        if not checkpoint.exists():checkpoint=pinned/checkpoint.name
        try:content=checkpoint.read_bytes()
        except FileNotFoundError:return
        if hashlib.sha256(content).hexdigest()!=result['checkpoint_sha256']:raise RuntimeError('Best policy hash mismatch')
        dest=art/f'best_level_{result["level"]:02d}.pt';tmp=dest.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(dest)
        best[key]=dict(iteration=result['iteration'],success_fraction=result['success_fraction'],mae_deg=mae,
            qualified=result['success_fraction']>=.95,checkpoint=str(dest.relative_to(root)),evidence=value['evidence'],sha256=result['checkpoint_sha256'])
        tmp=best_path.with_suffix('.tmp');tmp.write_text(json.dumps(best,indent=2));tmp.replace(best_path)
    write(dict(status='waiting_for_training_budget'))
    while True:
        preserve_best()
        if not (out/'training_status.json').exists():
            time.sleep(10)
            continue
        state=json.loads((out/'training_status.json').read_text())
        if state['status'] in ['stopped_by_request','stopped_repeated_runtime_failure']:
            write(dict(status='not_run_training_stopped',training_status=state));return
        if state['status']=='budget_finished':break
        time.sleep(30)
    iteration=state['iteration'];target=art/f'holdout_{iteration:07d}';time.sleep(10)
    if not target.exists():
        write(dict(status='running_final_holdout',iteration=iteration))
        with (out/'final_holdout.log').open('a') as log:
            process=subprocess.run([sys.executable,'-m','experiments.wuji_hand2.palm_up_human_demo_v6b.long_training.evaluate',
                '--checkpoint',str(out/f'checkpoint_{iteration:07d}.pt'),'--holdout'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:write(dict(status='final_holdout_runtime_failed',returncode=process.returncode));return
    deadline=time.time()+1800
    while time.time()<deadline:
        if (target/'metrics.json').exists() and (target/'rollout.mp4').exists() and (art/'final_result.json').exists():
            result=json.loads((target/'metrics.json').read_text())
            write(dict(status='finished',iteration=iteration,success_fraction=result['success_fraction'],evidence=str(target.relative_to(root))));return
        time.sleep(10)
    write(dict(status='final_holdout_incomplete',iteration=iteration,evidence=str(target.relative_to(root))))


if __name__=='__main__':main()
