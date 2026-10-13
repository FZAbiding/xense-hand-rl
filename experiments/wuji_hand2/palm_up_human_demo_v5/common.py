"""V5 palm-local finger order and actual physics observations. No policy imports."""
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
ASSETS=ROOT/'assets/wuji_hand2_palm_up_human_demo_v5'
RUNS=ROOT/'runs/wuji_hand2/palm_up_human_demo_v5'
ART=ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v5'
FINGERS=('thumb','index_finger','middle_finger')
# Provisional robot task envelope, not a universal human joint range.
LOW=np.radians([-20,-75,5,5, 20,-10,20,15, 20,-10,20,15])
HIGH=np.radians([70,35,80,80, 65,10,60,65, 65,10,60,65])
PARK=np.radians([75,0,95,65]*2)


def chain(m,d,finger):
    points=[d.xpos[m.body(f'r_{finger}_{part}').id] for part in ['proximal','proximal_abd','middle','distal']]
    points.append(d.site_xpos[m.site(f'r_{finger}_tip').id])
    wrist=m.body('r_wrist').id
    return (np.array(points)-d.xpos[wrist])@d.xmat[wrist].reshape(3,3)


def order_measure(m,d):
    a,b=[chain(m,d,f) for f in FINGERS[1:]]
    # Wrist-local +x points from middle to index; -z points along the fingers.
    # Compare whole chains and projected bone intersections, not only endpoints.
    gap=a[:,0]-b[:,0]
    def cross2(u,v):return u[0]*v[1]-u[1]*v[0]
    crossed=False
    for aa,ab in zip(a[:-1,[0,2]],a[1:,[0,2]]):
        for ba,bb in zip(b[:-1,[0,2]],b[1:,[0,2]]):
            u,v=ab-aa,bb-ba;den=cross2(u,v)
            if abs(den)>1e-10:
                t=cross2(ba-aa,v)/den;s=cross2(ba-aa,u)/den
                if 1e-5<t<1-1e-5 and 1e-5<s<1-1e-5:crossed=True
    return dict(min_matched_chain_gap_m=float(gap.min()),tip_order_gap_m=float(gap[-1]),
                projected_bone_crossing=bool(crossed),
                passed=bool(gap.min()>=.006 and not crossed))


def pen_contacts(m,d):
    rows=[];pen=m.body('pen').id
    for i,c in enumerate(d.contact):
        a,b=(int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2])
        body=b if a==pen else a if b==pen else -1
        if body<0:continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        local=d.xmat[body].reshape(3,3).T@(c.pos-d.xpos[body])
        rows.append(dict(body=m.body(body).name,normal_n=float(max(0,force[0])),
            contact_frame_force=force.tolist(),world_position_m=c.pos.tolist(),local_position_m=local.tolist()))
    return rows


def self_depth(m,d):
    maximum=0.
    for c in d.contact:
        names=[m.body(m.geom_bodyid[g]).name for g in [c.geom1,c.geom2]]
        if 'pen' in names or any(m.geom(g).name=='floor' for g in [c.geom1,c.geom2]):continue
        if any(any(f'r_{f}' in n for f in FINGERS) for n in names):maximum=max(maximum,-float(c.dist))
    return maximum


def observe(m,d,initial,prev,angle):
    from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
    from experiments.wuji_hand2.pinch import normal_vectors,opposing_strength
    axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2]
    yaw=float(np.arctan2(axis[1],axis[0]));angle+=np.arctan2(np.sin(yaw-prev),np.cos(yaw-prev))
    forces=contact_forces(m,d);vectors=normal_vectors(m,d);vectors[3:]=0
    order=order_measure(m,d)
    return dict(angle=angle,yaw=yaw,drift=float(np.linalg.norm(d.qpos[20:23]-initial[20:23])),
        tilt=float(np.arcsin(np.clip(abs(axis[2]),0,1))),forces=forces.tolist(),
        inactive_force=float(inactive_finger_force(m,d)),opposing=float(opposing_strength(vectors)),
        reverse=bool((d.qpos[[6,7,10,11]]<0).any()),order=order,self_depth_m=self_depth(m,d))


def simulate(m,q,ctrl,seconds,rng=None,knots=None,record=False):
    d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=ctrl
    if rng is not None:
        d.qpos[:12]+=rng.normal(0,.005,12);d.qpos[20:23]+=rng.normal(0,.0004,3)
    mujoco.mj_forward(m,d);a=d.xmat[m.body('pen').id].reshape(3,3)[:,2];prev=float(np.arctan2(a[1],a[0]));angle=0.
    metrics=[];qs=[];cs=[];cc=[]
    initial_ok=order_measure(m,d)['passed']
    for t in range(round(seconds/.025)):
        if knots is not None:
            targets=np.atleast_2d(knots)
            phase=np.clip(((t+1)*.025-.5)/(seconds-1),0,1)*len(targets)
            index=min(int(phase),len(targets)-1);p=phase-index;p=p*p*(3-2*p)
            left=ctrl[:12] if index==0 else targets[index-1]
            target=left+p*(targets[index]-left)
            d.ctrl[:12]=np.clip(d.ctrl[:12]+np.clip(target-d.ctrl[:12],-.008,.008),LOW,HIGH)
        for _ in range(5):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);r=observe(m,d,q,prev,angle);prev=r['yaw'];angle=r['angle'];metrics.append(r)
        if record:qs.append(d.qpos.copy());cs.append(d.ctrl.copy());cc.append(pen_contacts(m,d))
        if r['drift']>.045 or r['tilt']>np.radians(40) or not r['order']['passed'] or r['reverse'] or not np.isfinite(d.qpos).all():break
    f=np.array([r['forces'] for r in metrics]);n=len(metrics)
    row=dict(frames=n,requested_frames=round(seconds/.025),seconds=n*.025,
        final_angle_deg=float(np.degrees(angle)),max_drift_m=max(r['drift'] for r in metrics),
        max_tilt_deg=float(np.degrees(max(r['tilt'] for r in metrics))),
        min_chain_order_gap_m=min(r['order']['min_matched_chain_gap_m'] for r in metrics),
        order_violation_frames=sum(not r['order']['passed'] for r in metrics)+(not initial_ok),
        reverse_bending_frames=sum(r['reverse'] for r in metrics),
        max_active_self_depth_m=max(r['self_depth_m'] for r in metrics),
        two_tip_fraction=float(((f[:,:3]>.02).sum(1)>=2).mean()),
        non_tip_fraction=float((f[:,5]>.05).mean()),inactive_fraction=float(np.mean([r['inactive_force']>.02 for r in metrics])),
        opposing_fraction=float(np.mean([r['opposing']>=.3 for r in metrics])))
    row['physical_gate_passed']=bool(n==round(seconds/.025) and row['max_drift_m']<.025 and row['order_violation_frames']==0
        and row['reverse_bending_frames']==0 and row['max_active_self_depth_m']<.0005
        and row['two_tip_fraction']>.95 and row['non_tip_fraction']<.05 and row['inactive_fraction']<.01 and row['opposing_fraction']>.9)
    row['human_review_status']='pending'
    return row,np.asarray(qs),np.asarray(cs),metrics,cc
