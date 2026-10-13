"""Search and certify a true three-finger grasp with the other fingers folded."""
import hashlib,json,shutil
from pathlib import Path
import mujoco,numpy as np
from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
from experiments.wuji_hand2.pinch import normal_vectors,opposing_strength

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'assets/wuji_hand2_palm_up_tripod_v3'
FOLD=np.tile(np.radians([70.,0.,95.,60.]),2)


def assess(m,q,ctrl,seconds,rng=None,all_links=False):
    d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=ctrl;center=q[20:23].copy()
    if rng is not None:d.qpos[:20]+=rng.normal(0,.005,20);d.qpos[20:23]+=rng.normal(0,.0004,3)
    pinch=[];tip=[];inactive=[];other=[];tilt=[];drift=[]
    for t in range(round(seconds/.025)):
        for k in range(5):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);f=contact_forces(m,d);v=normal_vectors(m,d);v[3:]=0
        pinch.append(opposing_strength(v)>=.3);tip.append(np.sum(f[:3]>.02)>=2)
        inactive.append((inactive_finger_force(m,d) if all_links else f[3:5].sum())>.02);other.append(f[5]>.05)
        axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];tilt.append(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1))))
        drift.append(np.linalg.norm(d.qpos[20:23]-center))
        if drift[-1]>.06 or d.qpos[22]<center[2]-.04:break
    row=dict(seconds=seconds,frames=len(pinch),pinch_fraction=float(np.mean(pinch)),two_tip_fraction=float(np.mean(tip)),
             inactive_contact_fraction=float(np.mean(inactive)),non_tip_fraction=float(np.mean(other)),p95_tilt_deg=float(np.percentile(tilt,95)),max_drift_m=float(max(drift)))
    row['passed']=bool(len(pinch)==round(seconds/.025) and row['pinch_fraction']>.9 and row['two_tip_fraction']>.9 and row['inactive_contact_fraction']<.01 and row['non_tip_fraction']<.05 and row['p95_tilt_deg']<15 and row['max_drift_m']<.03)
    return row,d.qpos.copy()


def main():
    if DEST.exists():raise RuntimeError('Use a new asset directory; no overwrite')
    DEST.mkdir();source=ROOT/'assets/wuji_hand2_palm_up'
    shutil.copy2(source/'scene.xml',DEST/'scene.xml')
    m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8
    b=np.load(source/'grasp_bank_60s.npz');d=mujoco.MjData(m);d.qpos[:]=b['qpos'][0];d.ctrl[:]=b['ctrl'][0]
    for _ in range(800):mujoco.mj_step(m,d)
    seed=d.qpos.copy();seed[12:20]=FOLD;home=b['ctrl'][0].copy();home[12:20]=FOLD
    rng=np.random.default_rng(20261010);rows=[];accepted=[]
    for i in range(512):
        ctrl=home.copy();ctrl[:12]+=rng.normal(0,[.015,.035,.07,.12][i%4],12)
        ctrl[:12]=np.clip(ctrl[:12],m.actuator_ctrlrange[:12,0]+.04,m.actuator_ctrlrange[:12,1]-.04)
        row,q=assess(m,seed,ctrl,3);row['candidate']=i
        if row['passed']:
            longer,q=assess(m,q,ctrl,12);row['long_check']=longer
            if longer['passed']:accepted.append((q,ctrl,i))
        rows.append(row)
        if i%32==0:print('SEARCH',i,'accepted',len(accepted),flush=True)
        if len(accepted)>=12:break
    (DEST/'search.json').write_text(json.dumps(rows,indent=2))
    if not accepted:raise RuntimeError('No true tripod grasp; do not train')
    cert=[];chosen=[]
    for q,ctrl,i in accepted:
        trials=[]
        for p in range(5):
            row,_=assess(m,q,ctrl,60,None if p==0 else rng);row.update(candidate=i,perturbation=p);trials.append(row);cert.append(row)
        if all(r['passed'] for r in trials):chosen.append((q,ctrl,i))
        print('CERTIFY',i,all(r['passed'] for r in trials),flush=True)
        if len(chosen)>=4:break
    (DEST/'tripod_certification.json').write_text(json.dumps(dict(rows=cert,selected=[c[2] for c in chosen],fold_deg=np.degrees(FOLD).tolist()),indent=2))
    if not chosen:raise RuntimeError('No 60-second certified tripod grasp')
    np.savez(DEST/'grasp_bank_60s.npz',qpos=np.array([c[0] for c in chosen]),ctrl=np.array([c[1] for c in chosen]))
    provenance=json.loads((source/'provenance.json').read_text());provenance.update(task_orientation='palm_up',active_policy_joints=12,
        folded_control_deg=np.degrees(FOLD).tolist(),grasp_count=len(chosen),calibration='New three-finger search; real inactive collision geometry retained; five 60s trials per grasp')
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2))
    manifest=dict(certification_report='tripod_certification.json',scene_sha256=hashlib.sha256((DEST/'scene.xml').read_bytes()).hexdigest(),
                  bank_sha256=hashlib.sha256((DEST/'grasp_bank_60s.npz').read_bytes()).hexdigest())
    (DEST/'certified_bank_manifest.json').write_text(json.dumps(manifest,indent=2));print('SAVED',len(chosen),flush=True)


if __name__=='__main__':main()
