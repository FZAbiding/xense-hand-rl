"""Optimize a free-pen handover trajectory rather than isolated IK poses."""
import json,time
import mujoco
import numpy as np
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT,ASSETS,LOW_DEG,HIGH_DEG,contacts
from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
from experiments.wuji_hand2.pinch import opposing_strength,normal_vectors


def main():
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/reference_optimization_clockwise'
    if out.exists():raise RuntimeError('No overwrite')
    out.mkdir()
    config=dict(seed=2026101114,population=64,generations=12,duration_seconds=6.,warmup_seconds=.5,
                target_angle_deg=-60.,knots=2,active_fingers=['thumb','index','middle'],
                method='CEM on real actuator trajectories; pen freejoint is only assigned at reset')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));m.opt.iterations=m.opt.ls_iterations=8;d=mujoco.MjData(m)
    bank=np.load(ASSETS/'grasp_seed_bank.npz');initial=bank['qpos'][0];home=bank['ctrl'][0]
    source=json.loads((ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/retarget_collision_v2/search.json').read_text())['knots']
    target=np.array(next(x for x in source if x['angle_deg']==120)['pose'][:12])
    mean=np.array([.5*(target+home[:12]),target]);std=np.full((2,12),.13)
    low=np.radians(LOW_DEG);high=np.radians(HIGH_DEG);high[10]=min(high[10],home[10]+np.radians(35))
    rng=np.random.default_rng(config['seed']);best=None;history=[];started=time.time()

    def simulate(knots,record=False):
        mujoco.mj_resetData(m,d);d.qpos[:]=initial;d.ctrl[:]=home;mujoco.mj_forward(m,d)
        a=d.xmat[m.body('pen').id].reshape(3,3)[:,2];prev=np.arctan2(a[1],a[0]);angle=0.;trace=[];qs=[];cmds=[];cc=[]
        for t in range(280):
            time=(t+1)*.025;p=np.clip((time-.5)/6.,0,1)
            if p<.5:s=p*2;left=home[:12];right=knots[0]
            else:s=(p-.5)*2;left=knots[0];right=knots[1]
            s=s*s*(3-2*s);desired=left+s*(right-left)
            d.ctrl[:12]=np.clip(d.ctrl[:12]+np.clip(desired-d.ctrl[:12],-.008,.008),low,high);d.ctrl[12:]=home[12:]
            for _ in range(5):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);f=contact_forces(m,d);v=normal_vectors(m,d);v[3:]=0;pinch=float(opposing_strength(v))
            a=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(a[1],a[0]);angle+=np.arctan2(np.sin(yaw-prev),np.cos(yaw-prev));prev=yaw
            goal=-np.pi/3*p;desired_pair=int(np.floor(goal/(np.pi/3)+.5))%3;active=tuple(np.flatnonzero(f[:3]>.02))
            actual_pair={(0,1):0,(1,2):1,(0,2):2}.get(active,-1) if pinch>=.3 else -1
            drift=np.linalg.norm(d.qpos[20:23]-initial[20:23]);tilt=np.arcsin(np.clip(abs(a[2]),0,1));reverse=(d.qpos[[6,7,10,11]]<0).any()
            trace.append([angle,goal,angle-goal,drift,tilt,np.sum(f[:3]>.02),f[5],inactive_finger_force(m,d),pinch,reverse,actual_pair==desired_pair])
            if record:qs.append(d.qpos.copy());cmds.append(d.ctrl.copy());cc.append(contacts(m,d))
            if drift>.045 or tilt>np.radians(40) or reverse or not np.isfinite(d.qpos).all():break
        tr=np.array(trace)
        row=dict(frames=len(tr),actual_final_angle_deg=float(np.degrees(tr[-1,0])),mae_deg=float(np.degrees(abs(tr[:,2])).mean()),
                 max_drift_m=float(tr[:,3].max()),two_tip_fraction=float((tr[:,5]>=2).mean()),non_tip_fraction=float((tr[:,6]>.05).mean()),
                 inactive_fraction=float((tr[:,7]>.02).mean()),opposing_pinch_fraction=float((tr[:,8]>=.3).mean()),
                 reverse_bending_fraction=float(tr[:,9].mean()),contact_role_fraction=float(tr[20:,10].mean()) if len(tr)>20 else 0.)
        # Missing time remains a failure; never pad a dropped pen as good motion.
        cost=20*abs(tr[:,2]).mean()+10*abs(tr[-1,2])+200*tr[:,3].mean()+8*row['non_tip_fraction']+8*row['inactive_fraction']+4*(1-row['contact_role_fraction'])+4*(1-row['opposing_pinch_fraction'])+40*(1-len(tr)/280)
        row['cost']=float(cost)
        row['passed']=bool(len(tr)==280 and row['actual_final_angle_deg']<=-54 and row['mae_deg']<10 and row['max_drift_m']<.025
            and row['two_tip_fraction']>.95 and row['non_tip_fraction']<.05 and row['inactive_fraction']<.01
            and row['opposing_pinch_fraction']>.9 and row['reverse_bending_fraction']==0 and row['contact_role_fraction']>.8)
        return row,np.asarray(qs),np.asarray(cmds),tr,cc

    for gen in range(config['generations']):
        samples=np.clip(mean+rng.normal(size=(64,2,12))*std,low,high);samples[0]=mean;samples[1]=home[:12]
        evaluated=[]
        for i,k in enumerate(samples):
            result=simulate(k)[0];result.update(generation=gen,candidate=i);history.append(result);evaluated.append((result['cost'],k,result))
            if best is None or (result['passed'],-result['cost'])>(best[0]['passed'],-best[0]['cost']):best=(result.copy(),k.copy())
        elite=np.array([x[1] for x in sorted(evaluated,key=lambda x:x[0])[:8]])
        mean=.4*mean+.6*elite.mean(0);std=np.maximum(elite.std(0),.02)
        (out/'search.json').write_text(json.dumps(dict(best=best[0],history=history),indent=2));np.savez(out/'best_knots.npz',knots=best[1])
        print('REFERENCE_CEM',gen,'elapsed',round(time.time()-started),best[0],flush=True)
        if best[0]['passed']:break
    result,q,c,tr,cc=simulate(best[1],record=True)
    np.savez_compressed(out/'rollout.npz',qpos=q,ctrl=c,metrics=tr,dt=.025)
    (out/'contacts.json').write_text(json.dumps(cc));(out/'metrics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
