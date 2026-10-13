"""Calibrate finger order and search new V5 initial poses. Does not train."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .common import ROOT,ASSETS,RUNS,ART,FINGERS,LOW,HIGH,PARK,order_measure,chain,self_depth,simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=int,default=160);args=p.parse_args()
    if ASSETS.exists():raise RuntimeError('Existing v5 assets: do not overwrite')
    for path in [ASSETS,RUNS,ART]:path.mkdir(parents=True,exist_ok=True)
    source=ROOT/'assets/wuji_hand2_palm_up/scene.xml';shutil.copy2(source,ASSETS/'scene.xml')
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));d=mujoco.MjData(m)
    joints=[dict(index=i,name=m.joint(i).name,axis_local=m.jnt_axis[i].tolist(),range_deg=np.degrees(m.jnt_range[i]).tolist()) for i in range(20)]
    d.qpos[:]=m.qpos0;mujoco.mj_forward(m,d)
    calibration=dict(joints=joints,wrist_rotation=d.xmat[m.body('r_wrist').id].reshape(3,3).tolist(),
        palm_local_axes=dict(lateral_to_index='+x',distal='-z',palm_normal='+y'),
        straight=order_measure(m,d),minimum_chain_gap_m=.006,
        software_low_deg=np.degrees(LOW).tolist(),software_high_deg=np.degrees(HIGH).tolist(),
        thresholds_status='provisional task constraints from neutral model geometry; await human review',
        collision_model='unchanged official palm-up scene; all hand collisions retained',
        policy_initialization='none; no v4 grasp bank, policy, or trajectory used for candidates')
    # Rejected V4 is used ONLY as a negative calibration example.
    old=np.load(ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v4/partial_rotation/rollout.npz')['qpos_0']
    negative=[]
    for q in old:
        d.qpos[:]=q;mujoco.mj_forward(m,d);negative.append(order_measure(m,d))
    calibration['rejected_v4_frames']=len(negative);calibration['rejected_v4_detected']=sum(not r['passed'] for r in negative)
    if calibration['straight']['passed'] is not True or calibration['rejected_v4_detected']!=len(negative):raise RuntimeError('Order calibration regression')
    (RUNS/'calibration.json').write_text(json.dumps(calibration,indent=2))
    provenance=dict(scene_source=str(source.relative_to(ROOT)),scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        accepted_demo='artifacts/human_pen_spinning/review_20261011/A_triangle_pass_clean.mp4',status='candidate_search_only_no_training')
    (ASSETS/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,RUNS/f.name)
    rng=np.random.default_rng(2026101201);rows=[];candidates=[];started=time.time()
    bodies=[m.body('r_'+f+'_distal').id for f in FINGERS]
    seed=np.radians([45,-25,30,25, 45,0,40,30, 45,0,40,30])
    for i in range(args.candidates):
        center=np.array([rng.uniform(.055,.095),rng.uniform(-.017,-.003),rng.uniform(.105,.145)])
        yaw=rng.uniform(-.2,.2);axis=np.array([np.sin(yaw),-np.cos(yaw),0]);radial=np.array([np.cos(yaw),np.sin(yaw),0])
        along=np.array([rng.uniform(-.008,.008),.013,-.011])
        targets=center+along[:,None]*axis+np.array([-1,1,1])[:,None]*.009*radial
        normals=np.array([1,-1,-1])[:,None]*radial
        def residual(x):
            d.qpos[:]=m.qpos0;d.qpos[:20]=np.r_[x,PARK];mujoco.mj_kinematics(m,d)
            rot=d.xmat[bodies].reshape(3,3,3);pads=d.xpos[bodies]+np.einsum('nij,j->ni',rot,[0,.004,-.019])
            gap=chain(m,d,'index_finger')[:,0]-chain(m,d,'middle_finger')[:,0]
            return np.r_[((pads-targets)*150).ravel(),((rot[:,:,1]-normals)*.5).ravel(),
                np.maximum(.009-gap,0)*300,.05*(x-seed),.1*(x[[7,11]]-.8*x[[6,10]])]
        initial=np.clip(seed+rng.normal(0,.16,12),LOW+1e-6,HIGH-1e-6)
        fit=least_squares(residual,initial,bounds=(LOW,HIGH),max_nfev=90)
        q=m.qpos0.copy();q[:20]=np.r_[fit.x,PARK];q[20:23]=center;mujoco.mju_quatZ2Vec(q[23:27],axis)
        d.qpos[:]=q;mujoco.mj_forward(m,d);order=order_measure(m,d)
        # Squeeze inside the physical capsule after fitting opposing volar pads.
        targets+=np.array([1,-1,-1])[:,None]*radial*rng.uniform(.0003,.0012)
        squeeze=least_squares(residual,fit.x,bounds=(LOW,HIGH),max_nfev=45)
        ctrl=np.r_[squeeze.x,PARK]
        if order['passed']:
            row=simulate(m,q,ctrl,2)[0]
        else:row=dict(physical_gate_passed=False,order_violation_frames=1,frames=0,two_tip_fraction=0.,opposing_fraction=0.,non_tip_fraction=1.,inactive_fraction=1.,max_drift_m=1.)
        row.update(candidate=i,ik_position_error_m=float(np.linalg.norm(fit.fun[:9])/150),initial_order=order)
        rows.append(row);candidates.append((q.copy(),ctrl.copy()))
        if i%16==0 or row['physical_gate_passed']:
            (RUNS/'search.json').write_text(json.dumps(rows,indent=2));print('V5_SEARCH',i,'elapsed',round(time.time()-started),row,flush=True)
    (RUNS/'search.json').write_text(json.dumps(rows,indent=2))
    rank=sorted(range(len(rows)),key=lambda j:(rows[j]['physical_gate_passed'],rows[j].get('frames',0),rows[j]['opposing_fraction'],rows[j]['two_tip_fraction'],-rows[j]['non_tip_fraction'],-rows[j]['max_drift_m']),reverse=True)
    np.savez_compressed(ASSETS/'candidate_poses.npz',qpos=np.array([x[0] for x in candidates]),ctrl=np.array([x[1] for x in candidates]),ranking=rank)
    longer=[]
    for i in rank[:8]:
        row=simulate(m,*candidates[i],8)[0];row['candidate']=i;longer.append(row)
    (RUNS/'candidate_checks.json').write_text(json.dumps(longer,indent=2))
    (RUNS/'status.json').write_text(json.dumps(dict(status='candidate_search_complete_awaiting_preview',physical_short_pass_count=sum(x['physical_gate_passed'] for x in rows),eight_second_pass_count=sum(x['physical_gate_passed'] for x in longer),human_review='pending',training_started=False),indent=2))
    print('V5_DONE',json.loads((RUNS/'status.json').read_text()),flush=True)


if __name__=='__main__':main()
