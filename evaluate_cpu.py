"""Cross-check learned motor actions in ordinary CPU MuJoCo (not replay)."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from train import ActorCritic
from hand_config import hand_config
from experiments.revo3_v2.evaluate import quality_pass

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--output',default='runs/cpu_validation');p.add_argument('--seed',type=int,default=901)
    args=p.parse_args();torch.set_num_threads(1)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False);agent=ActorCritic(ckpt['nobs'],ckpt.get('naction',16));agent.load_state_dict(ckpt['model']);agent.eval()
    cfg=ckpt['config'];assert not cfg['reference'],'This cross-check currently tests direct PPO only.'
    amp,period=cfg['amplitude'],cfg['period'];dt=.025
    spec=hand_config(cfg.get('robot','leap'))
    m=mujoco.MjModel.from_xml_path(str(spec['assets']/'scene.xml'));m.opt.timestep=.005;m.opt.iterations=8
    h=m.nu
    d=mujoco.MjData(m);g=np.load(spec['assets']/'grasp.npz');home=g['ctrl'];center=g['qpos'][h:h+3]
    pen=m.body('pen').id;tip_ids=[m.site(n).id for n in spec['tips']]
    tip_bodies={m.geom(n).bodyid[0] for n in spec['geoms']}
    tip_body_index={body:i for i,body in enumerate(sorted(tip_bodies))}
    palm=m.body('palm').id if cfg.get('robot','leap')=='leap' else m.body('right_hand_base_link').id
    d.qpos[:]=g['qpos'];mujoco.mj_forward(m,d);axis=d.xmat[pen].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0])
    rng=np.random.default_rng(args.seed);rows=[];trajectories=[]
    def wrap(a):return np.arctan2(np.sin(a),np.cos(a))
    with torch.no_grad():
      for ep in range(args.episodes):
        mujoco.mj_resetData(m,d);d.qpos[:]=g['qpos'];d.qpos[:h]+=rng.normal(0,.005,h);d.qpos[h:h+3]+=rng.normal(0,.0004,3);d.ctrl[:]=home;mujoco.mj_forward(m,d)
        last=np.zeros(h);errors=[];angles=[];near=[];contacts=[];qposes=[];failed=False;expect=1;crossings=0
        strict_samples=[];strict_dropped=False;strict_workspace_failed=False
        for t in range(round(args.seconds/dt)):
            axis=d.xmat[pen].reshape(3,3)[:,2];angle=wrap(np.arctan2(axis[1],axis[0])-yaw0)
            phase=t*dt*2*np.pi/period;goal=amp*np.sin(phase);omega=amp*2*np.pi/period*np.cos(phase)
            obs=np.r_[d.qpos[:h]-home,d.qvel[:h]*.1,(d.qpos[h:h+3]-center)*20,axis,d.qvel[h:h+3],d.qvel[h+3:h+6]*.1,
                      np.sin(angle),np.cos(angle),goal,omega,np.sin(phase),np.cos(phase),last]
            action=agent(torch.tensor(obs,dtype=torch.float32).clamp(-20,20)[None])[0][0].numpy().clip(-1,1)
            desired=home+.65*action
            command=np.clip(desired,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
            clip=float(np.mean(abs(desired-command)>1e-7));d.ctrl[:]=command
            for _ in range(5):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            last=action;axis=d.xmat[pen].reshape(3,3)[:,2];angle=wrap(np.arctan2(axis[1],axis[0])-yaw0)
            next_goal=amp*np.sin((t+1)*dt*2*np.pi/period)
            error=wrap(angle-next_goal);drift=np.linalg.norm(d.qpos[h:h+3]-center)
            tilt=np.arcsin(np.clip(abs(axis[2]),0,1))
            strict_workspace_failed|=bool(drift>.045 or d.qpos[h+2]<center[2]-.025 or tilt>np.radians(40) or not np.isfinite(d.qpos).all())
            strict_dropped|=bool(drift>.075 or d.qpos[h+2]<.015 or not np.isfinite(d.qpos).all())
            failed|=bool(d.qpos[h+2]<.015 or drift>.075 or not np.isfinite(d.qpos).all())
            if cfg.get('objective','legacy')=='fingertip':failed|=bool(d.qpos[h+2]<center[2]-.018 or drift>.045)
            offset=d.site_xpos[tip_ids]-d.qpos[h:h+3]
            distance=np.linalg.norm(offset-np.clip(offset@axis,-.075,.075)[:,None]*axis,axis=1)
            contacting=set();tip_forces=np.zeros(len(tip_body_index));other_force=palm_force=0.
            for ci,c in enumerate(d.contact):
                a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
                if a==pen and b in tip_bodies:contacting.add(int(b))
                if b==pen and a in tip_bodies:contacting.add(int(a))
                partner=b if a==pen else a if b==pen else -1
                if partner>=0:
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench);fn=max(0.,float(wrench[0]))
                    if partner in tip_body_index:tip_forces[tip_body_index[partner]]+=fn
                    else:other_force+=fn
                    if partner==palm:palm_force+=fn
            if not failed and angle*expect>max(.15,amp*.6):crossings+=1;expect*=-1
            errors.append(abs(error));angles.append(angle);near.append(int((distance<.024).sum()));contacts.append(len(contacting))
            strict_samples.append([angle,next_goal,error,tilt,drift,clip,tip_forces.sum()+other_force,
                                   other_force,palm_force,int(np.sum(tip_forces>.02))])
            if ep<4:qposes.append(d.qpos.copy())
        success=not failed and crossings>=int(args.seconds/period)*2 and np.mean(errors)<.25 and np.mean(np.array(near)>=2)>.8
        if cfg.get('robot','leap')=='revo3':success=success and np.mean(np.array(contacts)>=2)>.8
        a=np.array(strict_samples);cycle_len=round(period/dt);gains=[]
        for c in range(int(args.seconds/period)):
            cycle=a[c*cycle_len:(c+1)*cycle_len,0];half=cycle_len//2
            gains.extend([float(cycle[:half].max()/amp),float(-cycle[half:].min()/amp)])
        strict=dict(survived=not strict_dropped,within_workspace=not strict_workspace_failed,
                    mae_deg=float(np.degrees(abs(a[:,2])).mean()),mean_tilt_deg=float(np.degrees(a[:,3]).mean()),
                    p95_tilt_deg=float(np.percentile(np.degrees(a[:,3]),95)),command_clip_fraction=float(a[:,5].mean()),
                    normal_force_sum_n=float(a[:,6].mean()),non_tip_force_frame_fraction=float(np.mean(a[:,7]>.05)),
                    palm_force_frame_fraction=float(np.mean(a[:,8]>.05)),two_tip_fraction=float(np.mean(a[:,9]>=2)),
                    minimum_half_cycle_gain=min(gains) if gains else 0.)
        strict['quality_success']=quality_pass(strict)
        rows.append(dict(success=bool(success),survived=not failed,mae_rad=float(np.mean(errors)),angle_span_rad=float(np.ptp(angles)),crossings=crossings,
                         strict=strict,
                         at_least_two_actual_fingertip_contacts=float(np.mean(np.array(contacts)>=2)),mean_actual_contacting_fingers=float(np.mean(contacts))))
        if ep<4:trajectories.append(qposes)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=dict(robot=cfg.get('robot','leap'),checkpoint=args.checkpoint,backend='CPU MuJoCo closed-loop policy',seed=args.seed,episodes=args.episodes,seconds=args.seconds,
        amplitude_rad=amp,period_seconds=period,objective=cfg.get('objective','legacy'),
        success_rate=float(np.mean([r['success'] for r in rows])),survival_rate=float(np.mean([r['survived'] for r in rows])),
        mean_abs_error_rad=float(np.mean([r['mae_rad'] for r in rows])),
        mean_actual_contacting_fingers=float(np.mean([r['mean_actual_contacting_fingers'] for r in rows])),
        actual_two_fingertip_contact_fraction=float(np.mean([r['at_least_two_actual_fingertip_contacts'] for r in rows])),
        strict_quality_success_rate=float(np.mean([r['strict']['quality_success'] for r in rows])),
        strict_quality_definition='survive; workspace; MAE<=3deg; mean tilt<=10deg; P95 tilt<=15deg; clip<5%; non-tip Fn>0.05N <5%frames; palm Fn>0.05N <1%; two tips Fn>0.02N >80%; each half-cycle peak>=90% target',
        per_episode=rows)
    for key in ['mae_deg','mean_tilt_deg','p95_tilt_deg','command_clip_fraction','normal_force_sum_n',
                'non_tip_force_frame_fraction','palm_force_frame_fraction','two_tip_fraction','minimum_half_cycle_gain']:
        summary['strict_'+key]=float(np.mean([r['strict'][key] for r in rows]))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'rollout.npz',qpos=np.array(trajectories),dt=dt)
    print(json.dumps({k:v for k,v in summary.items() if k!='per_episode'},indent=2))

if __name__=='__main__':main()
