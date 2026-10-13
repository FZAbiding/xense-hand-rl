"""Use free-pen dynamics to refine the kinematic handover controls."""
import json,time
import mujoco
import numpy as np
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT,ASSETS,LOW_DEG,HIGH_DEG
from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
from experiments.wuji_hand2.pinch import normal_vectors,opposing_strength


def main():
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/retarget_dynamic_v2'
    if out.exists():raise RuntimeError('No overwrite')
    out.mkdir()
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));m.opt.iterations=m.opt.ls_iterations=8
    d=mujoco.MjData(m);b=np.load(ASSETS/'grasp_seed_bank.npz');base=b['qpos'][0]
    d.qpos[:]=base;mujoco.mj_forward(m,d);axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0])
    source=json.loads((ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/retarget_collision_v2/search.json').read_text())['knots']
    low=np.radians(LOW_DEG);high=np.radians(HIGH_DEG);high[10]=min(high[10],b['ctrl'][0,10]+np.radians(35))
    rng=np.random.default_rng(2026101112);results=[];qs=[];cs=[];start=time.time()
    for knot in source:
        if knot['angle_deg'] not in [0,60]:continue
        x=np.array(knot['pose']);q=base.copy();q[:12]=x[:12];q[12:20]=np.tile(np.radians([85,0,110,75]),2);q[20:23]=x[12:15]
        target_yaw=yaw0+np.radians(knot['angle_deg']);a=np.array([np.cos(target_yaw)*np.cos(x[15]),np.sin(target_yaw)*np.cos(x[15]),np.sin(x[15])]);mujoco.mju_quatZ2Vec(q[23:27],a)
        pair=tuple(sorted(knot['pair']));ctrl=q[:20].copy();mean=ctrl[:12].copy();std=np.full(12,.10);best=None;history=[]

        def simulate(c):
            mujoco.mj_resetData(m,d);d.qpos[:]=q;d.ctrl[:]=np.r_[c,ctrl[12:]];trace=[]
            for t in range(120):
                for _ in range(5):mujoco.mj_step(m,d)
                mujoco.mj_forward(m,d);f=contact_forces(m,d);v=normal_vectors(m,d);v[3:]=0;pinch=float(opposing_strength(v));active=tuple(np.flatnonzero(f[:3]>.02))
                a=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(a[1],a[0]);err=abs(np.arctan2(np.sin(yaw-target_yaw),np.cos(yaw-target_yaw)))
                drift=np.linalg.norm(d.qpos[20:23]-q[20:23]);tilt=np.arcsin(np.clip(abs(a[2]),0,1));reverse=(d.qpos[[6,7,10,11]]<0).any();inactive=inactive_finger_force(m,d)
                trace.append([drift,err,tilt,f[5],inactive,active==pair and pinch>=.3,np.sum(f[:3]>.02)>=2,reverse])
                if drift>.05 or reverse or not np.isfinite(d.qpos).all():break
            t=np.array(trace);mean=t.mean(0)
            score=6*mean[5]+mean[6]-300*mean[0]-4*mean[1]-2*mean[2]-4*min(mean[3],5)-8*min(mean[4],5)-20*(1-len(t)/120)-.05*np.mean((c-ctrl[:12])**2)
            row=dict(score=float(score),frames=len(t),max_drift_m=float(t[:,0].max()),mae_deg=float(np.degrees(mean[1])),role_fraction=float(mean[5]),two_tip_fraction=float(mean[6]),non_tip_fraction=float(np.mean(t[:,3]>.05)),inactive_fraction=float(np.mean(t[:,4]>.02)),reverse_bending_fraction=float(mean[7]))
            row['passed']=bool(len(t)==120 and row['max_drift_m']<.025 and row['mae_deg']<10 and row['role_fraction']>.8 and row['two_tip_fraction']>.95 and row['non_tip_fraction']<.05 and row['inactive_fraction']<.01 and row['reverse_bending_fraction']==0)
            return row

        for gen in range(6):
            samples=np.clip(mean+rng.normal(size=(96,12))*std,low,high);samples[0]=mean
            evaluated=[]
            for i,c in enumerate(samples):
                row=simulate(c);row.update(generation=gen,candidate=i);history.append(row);evaluated.append((row['score'],c,row))
                if best is None or row['score']>best[0]:best=(row['score'],c.copy(),row.copy())
            elite=np.array([x[1] for x in sorted(evaluated,key=lambda v:-v[0])[:12]])
            mean=.5*mean+.5*elite.mean(0);std=np.maximum(elite.std(0),.008)
            print('DYNAMIC_KNOT',knot['angle_deg'],gen,best[2],'elapsed',round(time.time()-start),flush=True)
            if best[2]['passed']:break
        qs.append(q);cs.append(np.r_[best[1],ctrl[12:]])
        results.append(dict(angle_deg=knot['angle_deg'],pair=list(pair),selected=best[2],history=history))
        (out/'results.json').write_text(json.dumps(results,indent=2));np.savez(out/'knots.npz',qpos=np.array(qs),ctrl=np.array(cs))
    (out/'status.json').write_text(json.dumps(dict(status='dynamic_knot_diagnostic',all_knots_pass=all(x['selected']['passed'] for x in results),
        note='Three-second static tests at separately initialized poses; not evidence of a continuous handover or pen spin'),indent=2))


if __name__=='__main__':main()
