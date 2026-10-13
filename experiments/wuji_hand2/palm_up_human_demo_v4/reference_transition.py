"""Test a small physical tip-pinch rotation before attempting a full handover.

Only hand actuator commands change after reset. Pen poses from IK are never
assigned during simulation. This is a separate 30-degree feasibility probe.
"""
import json
import mujoco
import numpy as np
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT,ASSETS,LOW_DEG,HIGH_DEG,contacts
from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
from experiments.wuji_hand2.pinch import opposing_strength,normal_vectors


def main():
    out=ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/reference_transition_30deg'
    if out.exists():raise RuntimeError('No overwrite')
    out.mkdir()
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));m.opt.iterations=m.opt.ls_iterations=8;d=mujoco.MjData(m)
    bank=np.load(ASSETS/'grasp_seed_bank.npz');initial=bank['qpos'][0];home=bank['ctrl'][0]
    source=json.loads((ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/retarget_collision_v2/search.json').read_text())['knots']
    knot=next(x for x in source if x['angle_deg']==30);target=np.array(knot['pose'][:12]);low=np.radians(LOW_DEG);high=np.radians(HIGH_DEG)
    rows=[];best=None
    for duration in [3.,6.,10.]:
        for scale in [.8,1.,1.2]:
            mujoco.mj_resetData(m,d);d.qpos[:]=initial;d.ctrl[:]=home;mujoco.mj_forward(m,d)
            axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];prev=np.arctan2(axis[1],axis[0]);angle=0.
            endpoint=np.clip(home[:12]+scale*(target-home[:12]),low,high)
            qs=[];ctrls=[];trace=[];cc=[]
            for t in range(round((duration+1.5)/.025)):
                time=(t+1)*.025;s=np.clip((time-.5)/duration,0,1);s=s*s*(3-2*s)
                desired=home[:12]+s*(endpoint-home[:12]);delta=np.clip(desired-d.ctrl[:12],-.008,.008)
                d.ctrl[:12]=np.clip(d.ctrl[:12]+delta,low,high);d.ctrl[12:]=home[12:]
                for _ in range(5):mujoco.mj_step(m,d)
                mujoco.mj_forward(m,d);f=contact_forces(m,d);v=normal_vectors(m,d);v[3:]=0;pinch=float(opposing_strength(v));a=d.xmat[m.body('pen').id].reshape(3,3)[:,2]
                yaw=np.arctan2(a[1],a[0]);angle+=np.arctan2(np.sin(yaw-prev),np.cos(yaw-prev));prev=yaw;goal=np.radians(30)*s
                drift=np.linalg.norm(d.qpos[20:23]-initial[20:23]);tilt=np.arcsin(np.clip(abs(a[2]),0,1));reverse=(d.qpos[[6,7,10,11]]<0).any()
                trace.append([angle,goal,angle-goal,drift,tilt,np.sum(f[:3]>.02),f[5],inactive_finger_force(m,d),pinch,reverse])
                qs.append(d.qpos.copy());ctrls.append(d.ctrl.copy());cc.append(contacts(m,d))
                if drift>.045 or tilt>np.radians(40) or reverse or not np.isfinite(d.qpos).all():break
            tr=np.asarray(trace);qarr=np.asarray(qs)
            row=dict(duration_seconds=duration,target_scale=scale,frames=len(tr),expected_frames=round((duration+1.5)/.025),
                     actual_final_angle_deg=float(np.degrees(tr[-1,0])),mae_deg=float(np.degrees(abs(tr[:,2])).mean()),
                     max_drift_m=float(tr[:,3].max()),two_tip_fraction=float((tr[:,5]>=2).mean()),non_tip_fraction=float((tr[:,6]>.05).mean()),
                     inactive_fraction=float((tr[:,7]>.02).mean()),opposing_pinch_fraction=float((tr[:,8]>=.3).mean()),reverse_bending_fraction=float(tr[:,9].mean()),
                     middle_pip_excursion_deg=float(np.degrees(np.ptp(qarr[:,10]))))
            row['passed_small_rotation']=bool(row['frames']==row['expected_frames'] and row['actual_final_angle_deg']>=27 and row['mae_deg']<10
                and row['max_drift_m']<.025 and row['two_tip_fraction']>.95 and row['non_tip_fraction']<.05 and row['inactive_fraction']<.01
                and row['opposing_pinch_fraction']>.9 and row['reverse_bending_fraction']==0 and row['middle_pip_excursion_deg']<35)
            row['score']=float(-row['mae_deg']-50*row['non_tip_fraction']-50*row['inactive_fraction']-200*row['max_drift_m']-50*(1-row['frames']/row['expected_frames']))
            rows.append(row);print('PHYSICAL_TRANSITION',row,flush=True)
            if best is None or (row['passed_small_rotation'],row['score'])>(best[0]['passed_small_rotation'],best[0]['score']):best=(row,qarr,np.asarray(ctrls),tr,cc)
    (out/'metrics.json').write_text(json.dumps(dict(probe='30deg tip-pinch rotation; not full handover gate',candidates=rows,selected=best[0]),indent=2))
    np.savez_compressed(out/'rollout.npz',qpos=best[1],ctrl=best[2],metrics=best[3],dt=.025)
    (out/'contacts.json').write_text(json.dumps(best[4]))


if __name__=='__main__':main()
