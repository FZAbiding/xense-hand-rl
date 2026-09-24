"""Find fingertip grasp initial states using kinematics and physical settling."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from PIL import Image

ROOT=Path(__file__).resolve().parent

def main():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/scene.xml'))
    d=mujoco.MjData(m)
    ids=[m.site(n).id for n in ['if_tip','mf_tip','rf_tip','th_tip']]
    default=np.array([.8,0,.8,.8]*3+[.8,.8,.8,0])
    low,high=m.actuator_ctrlrange.T
    best=None
    for cx in [.135,.145,.155]:
      for z in [.05,.065,.08]:
       for inset in [0.004,0.007,0.010]:
        center=np.array([cx,-.012,z])
        target=np.array([[cx+inset,-.042,z],[cx+inset,.0,z],[cx+inset,.041,z],[cx-inset,-.022,z]])
        def residual(q):
            d.qpos[:16]=q
            mujoco.mj_forward(m,d)
            return np.r_[(d.site_xpos[ids]-target).ravel()*100, .02*(q-default)]
        q=least_squares(residual,default,bounds=(low+1e-4,high-1e-4),max_nfev=120).x
        mujoco.mj_resetData(m,d)
        d.qpos[:16]=q;d.qpos[16:19]=center;d.ctrl[:]=q
        mujoco.mj_forward(m,d)
        initial_qpos=d.qpos.copy()
        max_drift=0.;contacts=[]
        for t in range(400):
            mujoco.mj_step(m,d)
            max_drift=max(max_drift,np.linalg.norm(d.qpos[16:19]-center))
            if t>200:
                contacts.append(sum(m.geom_bodyid[c.geom1]==m.body('pen').id or m.geom_bodyid[c.geom2]==m.body('pen').id for c in d.contact))
        drift=np.linalg.norm(d.qpos[16:19]-center)
        axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2]
        score=-drift-max_drift*.5-.03*abs(axis[2])+.001*np.mean(contacts)
        if best is None or score>best[0]:
            best=(score,initial_qpos.copy(),q.copy(),d.qpos.copy(),d.qvel.copy(),dict(center=center.tolist(),drift=float(drift),contacts=float(np.mean(contacts)),axis=axis.tolist()))
            print('best',score,best[-1],flush=True)
    score,initial,ctrl,qpos,qvel,metrics=best
    np.savez(ROOT/'assets/grasp.npz',qpos=qpos,qvel=qvel,ctrl=ctrl,initial_qpos=initial)
    (ROOT/'assets/grasp_metrics.json').write_text(json.dumps(metrics,indent=2))
    d.qpos[:]=qpos;d.qvel[:]=0;mujoco.mj_forward(m,d)
    r=mujoco.Renderer(m,720,960);cam=mujoco.MjvCamera();cam.lookat[:]=[.12,0,.025];cam.distance=.42;cam.azimuth=130;cam.elevation=-35
    r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(ROOT/'assets/grasp.png');r.close()
    print('saved grasp',score,metrics,flush=True)

if __name__=='__main__':main()
