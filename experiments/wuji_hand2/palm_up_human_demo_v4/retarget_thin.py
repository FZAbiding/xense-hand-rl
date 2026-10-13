"""Search real-geometry handover knots before another policy-training attempt.

These are kinematic hypotheses. Every knot and transition must separately pass
free-pen dynamics; moving the freejoint for this diagnostic is never a rollout.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT, ASSETS, LOW_DEG, HIGH_DEG


ASSETS=ROOT/'assets/wuji_hand2_human_demo_v4_thin'

def main():
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/retarget_pads_thin_v4'
    if out.exists():raise RuntimeError('No overwrite')
    out.mkdir()
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));d=mujoco.MjData(m)
    bank=np.load(ROOT/'assets/wuji_hand2_palm_up_human_demo_v4/grasp_seed_bank.npz');q0=bank['qpos'][0];home=bank['ctrl'][0];center=q0[20:23]
    d.qpos[:]=q0;mujoco.mj_kinematics(m,d)
    axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0])
    pitch0=np.arcsin(axis[2]);pen=m.geom('pen_collision').id
    fingers=['thumb','index_finger','middle_finger']
    pads=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==m.body(f'r_{f}_tip_sensor_frame').id) for f in fingers]
    pad_bodies=[m.geom_bodyid[g] for g in pads]
    low=np.r_[np.radians(LOW_DEG),center-.025,np.radians(-35)]
    high=np.r_[np.radians(HIGH_DEG),center+.025,np.radians(35)]
    low[0]=np.radians(-45)  # Thumb CMC extension, not index/middle IP hyperextension.
    high[10]=min(high[10],home[10]+np.radians(35))
    low[10]=max(low[10],home[10]-np.radians(35))
    rng=np.random.default_rng(2026101110)
    all_rows=[];knots=[]
    hand_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]!=m.body('pen').id and m.geom(g).name!='floor']
    self_pairs=[]
    for ia,a in enumerate(hand_geoms):
        for b in hand_geoms[ia+1:]:
            names=[m.body(m.geom_bodyid[g]).name for g in [a,b]]
            if any(any(f'r_{f}' in name for f in fingers) for name in names):self_pairs.append((a,b))
    self_map={tuple(sorted(pair)):i for i,pair in enumerate(self_pairs)}
    def self_penetration():
        mujoco.mj_collision(m,d)
        depths=np.zeros(len(self_pairs))
        for c in d.contact:
            idx=self_map.get(tuple(sorted([int(c.geom1),int(c.geom2)])))
            if idx is not None:depths[idx]=max(depths[idx],-c.dist)
        return depths
    for angle_deg,pair in [(0,(0,1)),(30,(0,1)),(60,(1,2))]:
        excluded=[]
        for g in range(m.ngeom):
            if not m.geom_contype[g] or m.geom_bodyid[g]==m.body('pen').id or m.geom(g).name=='floor':continue
            name=m.body(m.geom_bodyid[g]).name
            active_pad=any(name in (f'r_{fingers[j]}_distal',f'r_{fingers[j]}_tip_sensor_frame') for j in pair)
            if not active_pad:excluded.append(g)
        yaw=yaw0+np.radians(angle_deg)

        def set_pose(x):
            d.qpos[:]=q0;d.qpos[:12]=x[:12];d.qpos[12:20]=np.tile(np.radians([85,0,110,75]),2);d.qpos[20:23]=x[12:15]
            axis=np.array([np.cos(yaw)*np.cos(x[15]),np.sin(yaw)*np.cos(x[15]),np.sin(x[15])])
            mujoco.mju_quatZ2Vec(d.qpos[23:27],axis);mujoco.mj_kinematics(m,d)

        def measure(x):
            set_pose(x);dist=[];normals=[];local=[]
            for j in pair:
                ft=np.zeros(6);v=mujoco.mj_geomDistance(m,d,pads[j],pen,1.,ft);dist.append(v)
                vec=ft[3:]-ft[:3];normals.append(vec/(np.linalg.norm(vec)+1e-8))
                b=pad_bodies[j];local.append(d.xmat[b].reshape(3,3).T@(ft[:3]-d.xpos[b]))
            other=np.array([mujoco.mj_geomDistance(m,d,g,pen,1.,None) for g in excluded])
            return np.array(dist),np.array(normals),np.array(local),other

        def residual(x):
            dist,norm,local,other=measure(x)
            return np.r_[(dist+.0010)*300,
                         np.maximum(self_penetration()-.0002,0)*300,
                         (norm[0]+norm[1])*.5,
                         np.maximum(.0040-local[:,1],0)*300,
                         np.maximum(np.abs(local[:,0])-.004,0)*300,
                         np.maximum(local[:,2]+.012,0)*300,
                         np.maximum(-.026-local[:,2],0)*300,
                         np.maximum(.002-other,0)*120,
                         .015*(x[:12]-home[:12]),
                         (x[12:15]-center)*5,
                         max(np.linalg.norm(x[12:15]-center)-.024,0)*300,
                         .05*(x[15]-pitch0)]

        best=None
        for i in range(64):
            if i==0:x=np.r_[home[:12],center,pitch0]
            elif i%3==0:x=rng.uniform(low,high)
            else:x=np.r_[home[:12]+rng.normal(0,.5,12),center+rng.normal(0,.008,3),pitch0+rng.normal(0,.2)]
            x=np.clip(x,low+1e-7,high-1e-7)
            fit=least_squares(residual,x,bounds=(low,high),max_nfev=160,diff_step=1e-4)
            distances,norm,local,other=measure(fit.x)
            oppose=float(-np.dot(norm[0],norm[1]))
            passed=bool(np.max(abs(distances+.0010))<.0005 and self_penetration().max()<.0005 and oppose>.3 and local[:,1].min()>.0035 and np.abs(local[:,0]).max()<.0045 and local[:,2].max()<-.0115
                        and other.min()>-.0005 and np.linalg.norm(fit.x[12:15]-center)<.025)
            row=dict(angle_deg=angle_deg,pair=list(pair),candidate=i,cost=float(np.sum(fit.fun**2)),
                     pad_distances_m=distances.tolist(),opposing_normal_score=oppose,pad_local_contacts_m=local.tolist(),
                     minimum_other_distance_m=float(other.min()),maximum_active_self_penetration_m=float(self_penetration().max()),passed_kinematic=passed,pose=fit.x.tolist())
            all_rows.append(row)
            if best is None or (passed and not best['passed_kinematic']) or (passed==best['passed_kinematic'] and row['cost']<best['cost']):best=row
            if i%12==0:print('KNOT',angle_deg,i,'best',best['cost'],'passed',best['passed_kinematic'],flush=True)
            if passed:break
        knots.append(best)
        (out/'search.json').write_text(json.dumps(dict(knots=knots,candidates=all_rows),indent=2))
        print('KNOT_RESULT',best,flush=True)
    (out/'status.json').write_text(json.dumps(dict(status='kinematic_diagnostic_only',
        all_knots_pass=all(x['passed_kinematic'] for x in knots),
        warning='No freejoint-pose manipulation may be presented as successful physical pen spinning; dynamic transition certification is still required'),indent=2))


if __name__=='__main__':main()
