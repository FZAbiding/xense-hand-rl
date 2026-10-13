"""Independent CPU validation and live offline HTML, while GPU PPO continues."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
import torch
from train import ActorCritic
from experiments.wuji_hand2.palm_up_human_demo_v5.common import order_measure,self_depth,pen_contacts
from experiments.wuji_hand2.evaluate import contact_forces,inactive_finger_force
from experiments.wuji_hand2.pinch import normal_vectors,opposing_strength
from experiments.wuji_hand2.palm_up_human_demo_v5.visualize import render
from tools.render_experiment_review import render_review
from .env import ROOT,LOW_DEG,HIGH_DEG
from .train import atomic_json


def cpu_observation(m,d,home,center,yaw0,time,last,forces,angle,pinch,theta,duration,continuous):
    axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(axis[1],axis[0])-yaw0
    fraction=max(0,time-.5)/duration;goal=theta*(fraction if continuous else min(fraction,1));omega=theta/duration if time>.5 and (continuous or fraction<1) else 0.
    base=np.r_[d.qpos[:20]-home,d.qvel[:20]*.1,(d.qpos[20:23]-center)*20,axis,d.qvel[20:23],d.qvel[23:26]*.1,
        np.sin(yaw),np.cos(yaw),np.arctan2(np.sin(goal),np.cos(goal)),omega,np.sin(fraction*2*np.pi),np.cos(fraction*2*np.pi),last,
        2*(d.ctrl-m.actuator_ctrlrange[:,0])/np.ptp(m.actuator_ctrlrange,axis=1)-1,forces*.2]
    wrist=m.body('r_wrist').id;rotation=d.xmat[wrist].reshape(3,3)
    gap=np.dot(d.site_xpos[m.site('r_index_finger_tip').id]-d.site_xpos[m.site('r_middle_finger_tip').id],rotation[:,0])
    return np.r_[base,gap*20,pinch,np.clip(angle-goal,-6,6),theta/(2*np.pi),duration/10,float(continuous)].clip(-20,20)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--holdout',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);checkpoint=Path(args.checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];level=ck['curriculum_level'];task=cfg['levels'][level]
    art=ROOT/cfg['artifacts'];art.mkdir(parents=True,exist_ok=True)
    out=art/f'{"holdout" if args.holdout else "validation"}_{ck["iteration"]:07d}';out.mkdir(exist_ok=False)
    assets=ROOT/cfg['assets']
    for name in ['scene.xml','grasp_seed_bank.npz']:
        path=assets/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ck['source_manifest'][str(path.relative_to(ROOT))]
    model=mujoco.MjModel.from_xml_path(str(assets/'scene.xml'));data=mujoco.MjData(model);bank=np.load(assets/'grasp_seed_bank.npz')
    agent=ActorCritic(ck['nobs'],12);agent.load_state_dict(ck['model']);agent.eval();low,high=np.radians(LOW_DEG),np.radians(HIGH_DEG)
    seed=cfg['holdout_seed'] if args.holdout else cfg['validation_seed'];rng=np.random.default_rng(seed);theta=np.radians(task['angle_deg']);duration=task['duration'];continuous=task.get('continuous',False)
    seconds=60 if level==0 or continuous else task['episode_seconds'];episodes=[];saved_q=[];saved_metrics=[];saved_cc=[]
    with torch.no_grad():
        for episode in range(cfg['validation_episodes']):
            gi=episode%len(bank['qpos']);q=bank['qpos'][gi];home=bank['ctrl'][gi];mujoco.mj_resetData(model,data)
            data.qpos[:]=q;data.qpos[:12]+=rng.normal(0,.005,12);data.qpos[20:23]+=rng.normal(0,.0004,3);data.ctrl[:]=home
            center=q[20:23];mujoco.mj_forward(model,data)
            w,x,y,z=q[23:27];yaw0=np.arctan2(2*(y*z-w*x),2*(x*z+w*y));prev=yaw0;angle=0.;last=np.zeros(20);rows=[];qs=[];cc=[];pairs=[];last_pair=None;dwell=0
            for i in range(round(seconds/.025)):
                forces=contact_forces(model,data);vectors=normal_vectors(model,data);vectors[3:]=0;pinch=float(opposing_strength(vectors))
                obs=cpu_observation(model,data,home,center,yaw0,i*.025,last,forces,angle,pinch,theta,duration,continuous)
                action=agent(torch.tensor(obs,dtype=torch.float32)[None])[0][0].numpy().clip(-1,1)
                desired=data.ctrl[:12]+action*.012;command=np.clip(desired,low,high);clipped=float(np.mean(abs(desired-command)>1e-7))
                data.ctrl[:12]=command;data.ctrl[12:]=home[12:];last=np.r_[action,np.zeros(8)];posture=False;reverse=False
                for _ in range(5):
                    mujoco.mj_step(model,data);mujoco.mj_kinematics(model,data)
                    posture=posture or not order_measure(model,data)['passed'];reverse=reverse or bool((data.qpos[[6,7,10,11]]<0).any())
                mujoco.mj_forward(model,data);axis=data.xmat[model.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(axis[1],axis[0]);angle+=np.arctan2(np.sin(yaw-prev),np.cos(yaw-prev));prev=yaw
                fraction=max(0,(i+1)*.025-.5)/duration;goal=theta*(fraction if continuous else min(fraction,1))
                forces=contact_forces(model,data);vectors=normal_vectors(model,data);vectors[3:]=0;pinch=float(opposing_strength(vectors));inactive=float(inactive_finger_force(model,data));depth=self_depth(model,data)
                drift=float(np.linalg.norm(data.qpos[20:23]-center));tilt=float(np.arcsin(np.clip(abs(axis[2]),0,1)))
                pair=tuple(int(j) for j in np.flatnonzero(forces[:3]>.02));pair=pair if len(pair)==2 and pinch>=.3 else None
                dwell=dwell+1 if pair==last_pair else 1;last_pair=pair
                if pair is not None and dwell>=4 and (not pairs or pairs[-1]!=pair):pairs.append(pair)
                row=dict(angle=float(angle),goal=float(goal),error=float(angle-goal),drift=drift,tilt=tilt,forces=forces.tolist(),inactive_force=inactive,
                    opposing=pinch,order=order_measure(model,data),posture=posture,reverse=reverse,self_depth_m=depth,command_clip=clipped)
                rows.append(row);qs.append(data.qpos.copy())
                if episode==0:cc.append(pen_contacts(model,data))
                if drift>.045 or tilt>np.radians(40) or posture or reverse or depth>.001 or not np.isfinite(data.qpos).all():break
            n=len(rows);f=np.array([r['forces'] for r in rows]);mae=float(np.degrees(np.mean([abs(r['error']) for r in rows])));actual=float(np.degrees(angle));goal_final=float(np.degrees(rows[-1]['goal']))
            progress=actual/goal_final if goal_final else None
            gates=dict(full_duration=n==round(seconds/.025),max_drift_m=max(r['drift'] for r in rows),
                angle_mae_deg=mae,posture_or_reverse=any(r['posture'] or r['reverse'] for r in rows),
                max_self_depth_m=max(r['self_depth_m'] for r in rows),
                two_tip_fraction=float(((f[:,:3]>.02).sum(1)>=2).mean()),
                non_tip_fraction=float((f[:,5]>.05).mean()),
                inactive_fraction=float(np.mean([r['inactive_force']>.02 for r in rows])),
                opposing_fraction=float(np.mean([r['opposing']>=.3 for r in rows])),
                command_clip_fraction=float(np.mean([r['command_clip'] for r in rows])),
                middle_pip_excursion_deg=float(np.degrees(np.ptp(np.array(qs)[:,10]))),
                rotation_progress=progress,contact_handovers=max(0,len(pairs)-1))
            failures=[]
            def require(name,condition):
                if not condition:failures.append(name)
            require('full_duration',gates['full_duration']);require('max_drift_m',gates['max_drift_m']<.025)
            require('angle_mae_deg',mae<=(5 if not theta else cfg['gates']['mae_deg_max']))
            require('posture_or_reverse',not gates['posture_or_reverse'])
            require('max_self_depth_m',gates['max_self_depth_m']<.0005)
            require('two_tip_fraction',gates['two_tip_fraction']>.95)
            require('non_tip_fraction',gates['non_tip_fraction']<.05)
            require('inactive_fraction',gates['inactive_fraction']<.01)
            require('opposing_fraction',gates['opposing_fraction']>.9)
            require('command_clip_fraction',gates['command_clip_fraction']<.05)
            require('middle_pip_excursion_deg',gates['middle_pip_excursion_deg']<=35)
            require('rotation_progress',not theta or .9<=progress<=1.15)
            require('contact_handovers',task['angle_deg']<180 or gates['contact_handovers']>=(1 if task['angle_deg']<360 else 3))
            strict=not failures
            episodes.append(dict(episode=episode,seconds=n*.025,actual_angle_deg=actual,goal_angle_deg=goal_final,mae_deg=mae,rotation_progress=progress,
                strict_passed=strict,order_violation_frames=sum(r['posture'] for r in rows),inactive_fraction=float(np.mean([r['inactive_force']>.02 for r in rows])),
                non_tip_fraction=float((f[:,5]>.05).mean()),stable_contact_pairs=[list(p) for p in pairs],contact_handovers=max(0,len(pairs)-1),
                gates=gates,failed_gates=failures))
            if episode==0:saved_q=qs;saved_metrics=rows;saved_cc=cc
    success=float(np.mean([r['strict_passed'] for r in episodes]));summary=dict(iteration=ck['iteration'],level=level,task=task,seed=seed,holdout=args.holdout,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),success_fraction=success,episodes=episodes,backend='CPU MuJoCo',full_task_achieved=bool(level==len(cfg['levels'])-1 and success>=.95))
    atomic_json(out/'metrics.json',summary);np.savez_compressed(out/'rollout.npz',qpos=saved_q,dt=.025);atomic_json(out/'frame_metrics.json',saved_metrics);atomic_json(out/'contacts.json',saved_cc)
    feedback=dict(iteration=ck['iteration'],level=level,success_fraction=success,promote=success>=cfg['gates']['success_fraction_min'],evidence=str((out/'metrics.json').relative_to(ROOT)))
    if not args.holdout:atomic_json(ROOT/cfg['runs']/'evaluation_feedback.json',feedback)
    render(saved_q,out/'rollout.mp4',f'PPO iteration {ck["iteration"]}; {task["name"]}; CPU strict {success:.0%}',metrics=saved_metrics)
    atomic_json(art/'latest_evaluation.json',summary)
    atomic_json(art/'final_result.json',summary) if args.holdout else None
    update_page(cfg)
    print(json.dumps(feedback),flush=True)


def update_page(cfg):
    art=ROOT/cfg['artifacts'];art.mkdir(parents=True,exist_ok=True);reference=art/'human_A.mp4'
    if not reference.exists():reference.write_bytes((ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v5/reference/human_A.mp4').read_bytes())
    cards=[dict(title='真人参考（已批准）',status='reference',video='human_A.mp4',description='训练目标参考；不表示当前策略已达到此效果。')]
    evaluations=sorted(art.glob('validation_*/metrics.json'),reverse=True)[:8]+sorted(art.glob('holdout_*/metrics.json'),reverse=True)[:1]
    for p in evaluations:
        data=json.loads(p.read_text());movie=p.parent/'rollout.mp4'
        if not movie.exists():continue
        cards.append(dict(title=f'第 {data["iteration"]} 轮 / {data["task"]["name"]}',status='passed' if data['success_fraction']>=.95 else 'failed',
            video=str(movie.relative_to(art)),description=f'CPU 严格通过 {data["success_fraction"]:.1%}；此结论仅适用于所标任务。失败不停止长期训练。',
            links=[dict(label='24 回合明细',path=str(p.relative_to(art)))]))
    manifest=dict(title='Wuji v6：整圈质量与提速训练',summary='从 v5 最终整圈策略热启动；优先降低中指 PIP 活动、笔漂移和手部自碰撞，再分阶段提速。GPU PPO 持续训练，CPU 独立验收；失败评估保留。',cards=cards,
        links=[dict(label='训练状态',path='../../../../..'+'/'+cfg['runs']+'/training_status.json'),
               dict(label='详细长期训练计划',path='../../../../..'+'/experiments/wuji_hand2/palm_up_human_demo_v6/long_training/PLAN.md')])
    # This artifact directory is four components below the repository root.
    for link in manifest['links']:link['path']=link['path'].replace('../../../../..','../../../..',1)
    atomic_json(art/'review_manifest.json',manifest);render_review(manifest,art/'review.html')


if __name__=='__main__':main()
