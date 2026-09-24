"""Optional kinematic references for residual RL, never applied to the pen."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parent

def main():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/scene.xml'));d=mujoco.MjData(m)
    g=np.load(ROOT/'assets/grasp.npz');d.qpos[:]=g['qpos'];d.ctrl[:]=g['ctrl'];mujoco.mj_forward(m,d)
    ids=[m.site(n).id for n in ['if_tip','mf_tip','rf_tip','th_tip']]
    center=d.qpos[16:19].copy();offsets=d.site_xpos[ids]-center
    home=g['ctrl'];lo,hi=m.actuator_ctrlrange.T;angles=np.linspace(-.85,.85,101);ctrl=[]
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        target=center+offsets@rot.T
        def residual(q):
            d.qpos[:16]=q;mujoco.mj_forward(m,d)
            return np.r_[(d.site_xpos[ids]-target).ravel()*100,.05*(q-home)]
        q=least_squares(residual,home,bounds=(lo+1e-4,hi-1e-4),max_nfev=120).x
        ctrl.append(q)
    np.savez(ROOT/'assets/reference.npz',angles=angles,ctrl=np.array(ctrl))
    print('saved 101 joint references (no object actuation)')

if __name__=='__main__':main()
