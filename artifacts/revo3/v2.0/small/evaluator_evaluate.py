"""CPU closed-loop V1/V2 comparison under the same, stricter quality gates."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train import ActorCritic


def quality_pass(row):
    """Shared CPU/GPU gate; thresholds are fixed before checkpoint selection."""
    return bool(row['survived'] and row['within_workspace'] and row['mae_deg']<=3
        and row['mean_tilt_deg']<=10 and row['p95_tilt_deg']<=15
        and row['command_clip_fraction']<.05 and row['non_tip_force_frame_fraction']<.05
        and row['palm_force_frame_fraction']<.01 and row['two_tip_fraction']>.8
        and row['minimum_half_cycle_gain']>=.9)


def contact_forces(m,d):
    pen=m.body('pen').id;palm=m.body('right_hand_base_link').id
    tips={m.body(f'right_{f}_DIP_Link').id:i for i,f in enumerate(['thumb','index','middle','ring','little'])}
    forces=np.zeros(7)
    for i,c in enumerate(d.contact):
        a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2];partner=b if a==pen else a if b==pen else -1
        if partner<0:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);fn=max(0.,float(f[0]))
        forces[tips[partner] if partner in tips else 5]+=fn
        if partner==palm:forces[6]+=fn
    return forces


def observation(m,d,home,center,yaw0,time,last,forces,cfg):
    axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(axis[1],axis[0])-yaw0;angle=np.arctan2(np.sin(yaw),np.cos(yaw))
    phase=time*2*np.pi/cfg['period'];amp=cfg['amplitude'];h=m.nu
    base=np.r_[d.qpos[:h]-home,d.qvel[:h]*.1,(d.qpos[h:h+3]-center)*20,axis,d.qvel[h:h+3],d.qvel[h+3:h+6]*.1,
               np.sin(angle),np.cos(angle),amp*np.sin(phase),amp*2*np.pi/cfg['period']*np.cos(phase),np.sin(phase),np.cos(phase),last]
    if cfg.get('experiment_version')=='2.0':base=np.r_[base,2*(d.ctrl-m.actuator_ctrlrange[:,0])/np.ptp(m.actuator_ctrlrange,axis=1)-1,forces*.2]
    return base.clip(-20,20)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=12);p.add_argument('--seconds',type=float,default=12);p.add_argument('--seed',type=int,default=20260924)
    p.add_argument('--video',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    checkpoint=ROOT/args.checkpoint;ck=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];v2=cfg.get('experiment_version')=='2.0'
    assets=ROOT/('assets/revo3_v2' if v2 else 'assets/revo3')
    if v2:
        for name in ['assets/revo3_v2/scene.xml','assets/revo3_v2/grasp_bank.npz']:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==ck['source_manifest'][name],name
    m=mujoco.MjModel.from_xml_path(str(assets/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8;d=mujoco.MjData(m);h=m.nu
    if v2 and cfg['grasp_version']=='v2':bank=np.load(assets/'grasp_bank.npz');bq=bank['qpos'];bc=bank['ctrl']
    else:g=np.load(ROOT/'assets/revo3/grasp.npz');bq=g['qpos'][None];bc=g['ctrl'][None]
    net=ActorCritic(ck['nobs'],ck.get('naction',21));net.load_state_dict(ck['model']);net.eval()
    low,high=m.actuator_ctrlrange.T;safe_margin=np.minimum((high-low)*.05,np.radians(2))
    delta_scale=np.clip((high-low)*.03,.012,.035);delta_scale[13:]*=.5
    amp=cfg['amplitude'];period=cfg['period'];dt=.025;T=round(args.seconds/dt)
    rng=np.random.default_rng(args.seed);rows=[];all_arrays=[];qposes=[]
    with torch.no_grad():
      for ep in range(args.episodes):
        gi=ep%len(bq);home=bc[gi];center=bq[gi,21:24];mujoco.mj_resetData(m,d);d.qpos[:]=bq[gi];d.ctrl[:]=home
        mujoco.mj_forward(m,d);axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0])
        d.qpos[:21]+=rng.normal(0,.005,21);d.qpos[21:24]+=rng.normal(0,.0004,3);mujoco.mj_forward(m,d)
        last=np.zeros(21);forces=contact_forces(m,d);samples=[];qs=[];failed=False;dropped=False;clipped_by_joint=np.zeros(21)
        for t in range(T):
            obs=observation(m,d,home,center,yaw0,t*dt,last,forces,cfg)
            raw=net(torch.tensor(obs,dtype=torch.float32)[None])[0][0].numpy();action=raw.clip(-1,1)
            if v2 and cfg['action_mode']=='delta':desired=d.ctrl+action*delta_scale;command=np.clip(desired,low+safe_margin,high-safe_margin)
            else:desired=home+.65*action;command=np.clip(desired,low,high)
            clip=float(np.mean(abs(desired-command)>1e-7));d.ctrl[:]=command
            clipped_by_joint+=(abs(desired-command)>1e-7)
            for _ in range(5):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);last=action;forces=contact_forces(m,d)
            axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw=np.arctan2(axis[1],axis[0])-yaw0;angle=np.arctan2(np.sin(yaw),np.cos(yaw))
            goal=amp*np.sin((t+1)*dt*2*np.pi/period);error=np.arctan2(np.sin(angle-goal),np.cos(angle-goal));tilt=np.arcsin(np.clip(abs(axis[2]),0,1))
            drift=np.linalg.norm(d.qpos[21:24]-center)
            failed|=bool(drift>.045 or d.qpos[23]<center[2]-.025 or tilt>np.radians(40) or not np.isfinite(d.qpos).all())
            dropped|=bool(drift>.075 or d.qpos[23]<.015 or not np.isfinite(d.qpos).all())
            samples.append([angle,goal,error,tilt,drift,clip,forces[:6].sum(),forces[5],forces[6],np.sum(forces[:5]>.02)])
            if ep<4:qs.append(d.qpos.copy())
        a=np.array(samples);cycles=int(args.seconds/period);cycle_len=round(period/dt)
        gains=[]
        for c in range(cycles):
            cycle=a[c*cycle_len:(c+1)*cycle_len,0];half=cycle_len//2
            gains.extend([float(cycle[:half].max()/amp),float(-cycle[half:].min()/amp)])
        row=dict(episode=ep,grasp_index=gi,survived=not dropped,within_workspace=not failed,mae_deg=float(np.degrees(abs(a[:,2])).mean()),
                 mean_tilt_deg=float(np.degrees(a[:,3]).mean()),p95_tilt_deg=float(np.percentile(np.degrees(a[:,3]),95)),
                 command_clip_fraction=float(a[:,5].mean()),normal_force_sum_n=float(a[:,6].mean()),
                 non_tip_force_frame_fraction=float(np.mean(a[:,7]>.05)),palm_force_frame_fraction=float(np.mean(a[:,8]>.05)),
                 two_tip_fraction=float(np.mean(a[:,9]>=2)),minimum_half_cycle_gain=min(gains) if gains else 0.,half_cycle_gains=gains)
        row['quality_success']=quality_pass(row)
        row['joint_clip_fractions']={m.joint(j).name:float(clipped_by_joint[j]/T) for j in range(21)}
        rows.append(row);all_arrays.append(a)
        if ep<4:qposes.append(qs)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    summary=dict(checkpoint=args.checkpoint,experiment_version='2.0' if v2 else '1.0',backend='CPU MuJoCo closed-loop',
                 seed=args.seed,episodes=args.episodes,seconds=args.seconds,amplitude_rad=amp,period_seconds=period,
                 quality_success_rate=float(np.mean([r['quality_success'] for r in rows])),survival_rate=float(np.mean([r['survived'] for r in rows])),
                 workspace_survival_rate=float(np.mean([r['within_workspace'] for r in rows])),
                 quality_definition='survive; MAE<=3deg; mean tilt<=10deg; P95 tilt<=15deg; clip<5%; non-tip Fn>0.05N <5%frames; palm Fn>0.05N <1%; two tips Fn>0.02N >80%; each half-cycle peak>=90% target')
    for key in ['mae_deg','mean_tilt_deg','p95_tilt_deg','command_clip_fraction','normal_force_sum_n','non_tip_force_frame_fraction','palm_force_frame_fraction','two_tip_fraction','minimum_half_cycle_gain']:
        summary[key]=float(np.mean([r[key] for r in rows]))
    (out/'metrics.json').write_text(json.dumps(dict(summary,per_episode=rows),indent=2));np.savez_compressed(out/'rollout.npz',metrics=np.array(all_arrays),qpos=np.array(qposes),dt=dt)
    print(json.dumps(summary,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True);ts=(np.arange(T)+1)*dt
    axs[0].plot(ts,np.degrees(all_arrays[0][:,1]),'k--',label='target')
    for i,a in enumerate(all_arrays[:3]):
        axs[0].plot(ts,np.degrees(a[:,0]),label=f'grasp {rows[i]["grasp_index"]}')
        axs[1].plot(ts,np.degrees(a[:,3]));axs[2].plot(ts,a[:,6])
    axs[0].set_ylabel('Yaw (deg)');axs[0].legend();axs[1].set_ylabel('Tilt (deg)');axs[2].set_ylabel('Normal force sum (N)');axs[2].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'tracking.png');plt.close(fig)
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        renderer=mujoco.Renderer(m,720,960);camera=mujoco.MjvCamera();camera.lookat[:]=[.125,0,.025];camera.distance=.40;camera.azimuth=130;camera.elevation=-35
        option=mujoco.MjvOption();option.geomgroup[3]=0
        with imageio.get_writer(out/'rollout.mp4',fps=40,codec='libx264',quality=8) as writer:
            for t,q in enumerate(qposes[0][:480]):
                d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=option)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,55),fill=(18,23,32))
                draw.text((12,10),f'Revo3 V{summary["experiment_version"]} | t={(t+1)*dt:.2f}s | target={np.degrees(all_arrays[0][t,1]):+.1f} | angle={np.degrees(all_arrays[0][t,0]):+.1f} | tilt={np.degrees(all_arrays[0][t,3]):.1f}',fill='white')
                draw.text((12,30),f'CPU closed-loop | quality success={rows[0]["quality_success"]} | no resets',fill='white')
                writer.append_data(np.asarray(frame))
                if t==0:frame.save(out/'preview.png')
        renderer.close()


if __name__=='__main__':main()
