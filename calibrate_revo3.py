"""Search physically settled Revo3 fingertip grasps for RL initialization."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from PIL import Image

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'assets/revo3'

def main():
    m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));d=mujoco.MjData(m)
    fingers=['thumb','index','middle','ring','little'];ids=[m.site(f+'_tip').id for f in fingers]
    tip_bodies={m.body(f'right_{f}_DIP_Link').id for f in fingers};pen=m.body('pen').id
    default=np.array([.8,1.2,.4,.5,.3]+[0,.6,.5,.3]*4);low,high=m.actuator_ctrlrange.T
    candidates=[];best=-1e9
    for cx in [.075,.09,.105]:
      for z in [.065,.08,.095]:
       for inset in [.006,.009,.012]:
        center=np.array([cx,.01,z]);target=np.array([[cx+inset,.035,z],[cx-inset,.036,z],[cx-inset,.012,z],[cx-inset,-.012,z],[cx-inset,-.036,z]])
        def residual(q):
            d.qpos[:21]=q;mujoco.mj_kinematics(m,d)
            return np.r_[(d.site_xpos[ids]-target).ravel()*100,.03*(q-default)]
        q=least_squares(residual,default,bounds=(low+1e-5,high-1e-5),max_nfev=150).x
        mujoco.mj_resetData(m,d);d.qpos[:21]=q;d.qpos[21:24]=center;d.ctrl[:]=q;mujoco.mj_forward(m,d)
        initial=d.qpos.copy();max_drift=0.;counts=[];non_tip=[]
        for t in range(400):
            mujoco.mj_step(m,d);max_drift=max(max_drift,float(np.linalg.norm(d.qpos[21:24]-center)))
            if t>200:
                touching=set();other=0
                for c in d.contact:
                    a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
                    partner=int(b if a==pen else a if b==pen else -1)
                    if partner in tip_bodies:touching.add(partner)
                    elif partner>=0:other+=1
                counts.append(len(touching));non_tip.append(other)
        mujoco.mj_forward(m,d)
        drift=float(np.linalg.norm(d.qpos[21:24]-center));axis=d.xmat[pen].reshape(3,3)[:,2]
        score=-drift-.3*max_drift-.035*abs(axis[2])+.006*np.mean(counts)-.005*np.mean(non_tip)
        metrics=dict(center=center.tolist(),drift=drift,actual_contacting_fingers=float(np.mean(counts)),non_tip_contacts=float(np.mean(non_tip)),axis=axis.tolist(),score=float(score))
        candidates.append((score,d.qpos.copy(),q.copy(),initial,metrics))
        if score>best:best=score;print('best',metrics,flush=True)
    candidates.sort(key=lambda x:x[0],reverse=True)
    score,qpos,ctrl,initial,metrics=candidates[0]
    np.savez(DEST/'grasp.npz',qpos=qpos,qvel=np.zeros(m.nv),ctrl=ctrl,initial_qpos=initial)
    np.savez(DEST/'grasp_candidates.npz',qpos=np.array([c[1] for c in candidates]),ctrl=np.array([c[2] for c in candidates]))
    (DEST/'grasp_metrics.json').write_text(json.dumps(metrics,indent=2))
    d.qpos[:]=qpos;d.qvel[:]=0;mujoco.mj_forward(m,d)
    r=mujoco.Renderer(m,720,960);cam=mujoco.MjvCamera();cam.lookat[:]=[.12,0,.04];cam.distance=.4;cam.azimuth=130;cam.elevation=-40
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;r.update_scene(d,camera=cam,scene_option=opt);Image.fromarray(r.render()).save(DEST/'grasp.png');r.close()
    print('saved',metrics,flush=True)

if __name__=='__main__':main()
