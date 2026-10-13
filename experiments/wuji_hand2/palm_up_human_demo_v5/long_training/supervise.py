"""Persist beyond the chat, supervise crashes, and enforce the 24-hour budget."""
import datetime,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from .env import ROOT
from .train import atomic_json


def main():
    config=json.loads(Path(__file__).with_name('config.json').read_text());out=ROOT/config['runs'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A supervisor is already running for this experiment')
    atomic_json(out/'supervisor_pid.json',dict(pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    lifecycle_path=out/'lifecycle.json'
    if lifecycle_path.exists():lifecycle=json.loads(lifecycle_path.read_text())
    else:
        now=time.time();lifecycle=dict(started_unix=now,deadline_unix=now+config['wall_hours']*3600,wall_hours=config['wall_hours']);atomic_json(lifecycle_path,lifecycle)
    stop=[False];child=[None]
    def stopping(*_):
        stop[0]=True
        if child[0] and child[0].poll() is None:child[0].terminate()
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    attempts=0;consecutive=0
    if not (out/'training_status.json').exists():atomic_json(out/'training_status.json',dict(status='initializing',heartbeat_unix=time.time(),deadline_unix=lifecycle['deadline_unix']))
    from .evaluate import update_page
    update_page(config)
    while time.time()<lifecycle['deadline_unix'] and not stop[0] and not (out/'STOP').exists():
        before=None
        if (out/'latest.pt').exists():before=(out/'latest.pt').stat().st_mtime
        with (out/'training.log').open('a',buffering=1) as log:
            log.write(f'\nSUPERVISOR_LAUNCH attempt={attempts} unix={time.time()}\n')
            child[0]=subprocess.Popen([sys.executable,'-m','experiments.wuji_hand2.palm_up_human_demo_v5.long_training.train'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('TRAINING_PID',child[0].pid,flush=True)
            while child[0].poll() is None:
                if stop[0] or (out/'STOP').exists():child[0].terminate()
                if (out/'training_status.json').exists():
                    state=json.loads((out/'training_status.json').read_text())
                    if time.time()-state.get('heartbeat_unix',time.time())>1800 and state.get('status') in ['initializing','training']:
                        log.write('WATCHDOG_STALE_HEARTBEAT\n');child[0].terminate()
                time.sleep(10)
            code=child[0].returncode
        if code==0:break
        attempts+=1
        after=(out/'latest.pt').stat().st_mtime if (out/'latest.pt').exists() else None
        consecutive=consecutive+1 if after==before else 0
        atomic_json(out/'restart_status.json',dict(restarts=attempts,consecutive_no_progress=consecutive,last_exit_code=code,unix=time.time()))
        if attempts>=config['max_restarts'] or consecutive>=config['max_consecutive_failed_restarts']:
            atomic_json(out/'training_status.json',dict(status='stopped_repeated_runtime_failure',restarts=attempts,last_exit_code=code));break
        time.sleep(20)
    atomic_json(out/'supervisor_finished.json',dict(unix=time.time(),restarts=attempts,stopped_by_request=stop[0]))
    print('SUPERVISOR_FINISHED',attempts,flush=True)


if __name__=='__main__':main()
