"""V2 grasp bank: joint margins, three-finger IK, physical settling and gates."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import itertools
import json
import shutil
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'assets/revo3_v2'


def main():
    p=argparse.ArgumentParser();p.add_argument('--max-candidates',type=int,default=144);args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'grasp_bank.npz').exists():
        raise RuntimeError('Grasp bank already exists; do not replace it underneath a trained policy.')
    shutil.copy2(ROOT/'assets/revo3/scene.xml',DEST/'scene.xml')
    m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));m.opt.iterations=8
    d=mujoco.MjData(m);h=m.nu;pen=m.body('pen').id
    tipids=[m.site(f'{f}_tip').id for f in ['thumb','index','middle']]
    tipbodies={m.body(f'right_{f}_DIP_Link').id for f in ['thumb','index','middle','ring','little']}
    palm=m.body('right_hand_base_link').id
    low,high=m.actuator_ctrlrange.T
    margin=np.minimum(np.radians(5),(high-low)*.12)
    lower=low+margin;upper=high-margin
    seed=np.array([.8,1.2,.5,.4,.3]+[0,.65,.5,.35]*2+[.1,.45,1.05,.8]+[-.1,.45,1.05,.8])
    seed=np.clip(seed,lower+1e-5,upper-1e-5)
    candidates=[];accepted=[];best=-np.inf
    grid=list(itertools.product([.065,.08,.095,.11],[.01,.025],[.075,.09,.105],[.007,.011],[.0,.018]))
    # Fixed deterministic shuffle exposes different heights/centers early.
    np.random.default_rng(42).shuffle(grid)
    for n,(cx,cy,z,inset,thumb_y) in enumerate(grid[:args.max_candidates]):
        center=np.array([cx,cy,z]);targets=np.array([[cx+inset,cy+thumb_y,z],[cx-inset,cy+.024,z],[cx-inset,cy-.018,z]])
        def residual(q):
            d.qpos[:h]=seed;d.qpos[:13]=q;mujoco.mj_kinematics(m,d)
            return np.r_[(d.site_xpos[tipids]-targets).ravel()*100,.15*(q-seed[:13])]
        fit=least_squares(residual,seed[:13],bounds=(lower[:13]+1e-6,upper[:13]-1e-6),max_nfev=90)
        ctrl=seed.copy();ctrl[:13]=fit.x
        mujoco.mj_resetData(m,d);d.qpos[:h]=ctrl;d.qpos[h:h+3]=center;d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
        tilt=[];counts=[];otherforces=[];palmforces=[];drifts=[];forces=[]
        for step in range(400):
            mujoco.mj_step(m,d)
            if step>=200 and step%5==0:
                mujoco.mj_forward(m,d);axis=d.xmat[pen].reshape(3,3)[:,2]
                tilt.append(float(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1)))))
                touching=set();other=0.;palmforce=0.;total=0.
                for ci,c in enumerate(d.contact):
                    a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
                    partner=int(b if a==pen else a if b==pen else -1)
                    if partner<0:continue
                    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);fn=max(float(f[0]),0.);total+=fn
                    if partner in tipbodies and fn>.02:touching.add(partner)
                    elif partner not in tipbodies:other+=fn
                    if partner==palm:palmforce+=fn
                counts.append(len(touching));otherforces.append(other);palmforces.append(palmforce);forces.append(total)
                drifts.append(float(np.linalg.norm(d.qpos[h:h+3]-center)))
        row=dict(candidate=n,requested_center=center.tolist(),inset=inset,thumb_y=thumb_y,
                 mean_tilt_deg=float(np.mean(tilt)),p95_tilt_deg=float(np.percentile(tilt,95)),
                 two_tip_fraction=float(np.mean(np.array(counts)>=2)),mean_other_normal_force_n=float(np.mean(otherforces)),
                 mean_palm_normal_force_n=float(np.mean(palmforces)),mean_normal_force_sum_n=float(np.mean(forces)),
                 final_drift_m=drifts[-1],min_home_limit_margin_deg=float(np.degrees(np.minimum(ctrl-low,high-ctrl)).min()))
        valid=(row['p95_tilt_deg']<15 and row['two_tip_fraction']>.95 and row['final_drift_m']<.025
               and row['mean_palm_normal_force_n']<.02 and row['mean_other_normal_force_n']<.10)
        score=(row['two_tip_fraction']*3-row['mean_tilt_deg']*.08-row['mean_other_normal_force_n']*2
               -row['final_drift_m']*25-.03*row['mean_normal_force_sum_n'])
        row.update(accepted=bool(valid),score=float(score));candidates.append(row)
        if valid:accepted.append((score,d.qpos.copy(),ctrl.copy(),row))
        if score>best:
            best=score;print('BEST',json.dumps(row),flush=True)
        if (n+1)%12==0:
            print('PROGRESS',n+1,'accepted',len(accepted),flush=True)
            (DEST/'calibration_candidates.json').write_text(json.dumps(candidates,indent=2))
    (DEST/'calibration_candidates.json').write_text(json.dumps(candidates,indent=2))
    if not accepted:raise RuntimeError('No grasp passed V2 physical gates; inspect candidates before training.')
    accepted.sort(key=lambda x:x[0],reverse=True);selected=accepted[:8]
    np.savez(DEST/'grasp_bank.npz',qpos=np.array([x[1] for x in selected]),ctrl=np.array([x[2] for x in selected]))
    np.savez(DEST/'grasp.npz',qpos=selected[0][1],ctrl=selected[0][2])
    (DEST/'grasp_metrics.json').write_text(json.dumps([x[3] for x in selected],indent=2))
    provenance=json.loads((ROOT/'assets/revo3/provenance.json').read_text())
    provenance.update(experiment_version='2.0',physics_change='none; V1 model copied, limits unchanged',grasp_count=len(selected),
                      calibration='three-finger IK with joint margins; 2s gravity/contact filtering')
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('SAVED',len(selected),'grasps',flush=True)


if __name__=='__main__':main()
