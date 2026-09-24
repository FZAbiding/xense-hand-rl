"""Independent deterministic policy evaluation; no reset hides dropped pens."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from pen_env import PenEnv
from train import ActorCritic

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',default='runs/evaluation');p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--amplitude',type=float,default=.6);p.add_argument('--period',type=float,default=4);p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--reference',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--seed',type=int,default=777)
    p.add_argument('--robot',choices=['leap','revo3'],default='leap')
    args=p.parse_args();torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.checkpoint,weights_only=False) if args.checkpoint else None
    if ckpt:
        args.reference=ckpt['config']['reference'];args.amplitude=ckpt['config']['amplitude'];args.period=ckpt['config']['period']
        args.robot=ckpt['config'].get('robot','leap')
    env=PenEnv(args.num_envs,args.amplitude,args.period,args.seconds,seed=args.seed,reference=args.reference,robot=args.robot,objective=ckpt['config'].get('objective','legacy') if ckpt else 'legacy')
    env.reset(torch.ones(args.num_envs,device='cuda',dtype=torch.bool))
    agent=None
    if ckpt:
        agent=ActorCritic(ckpt['nobs'],ckpt.get('naction',16)).cuda();agent.load_state_dict(ckpt['model']);agent.eval()
    failed=torch.zeros(args.num_envs,device='cuda',dtype=torch.bool)
    alive=[];angles=[];goals=[];errors=[];near=[];drifts=[];qposes=[];actual_contacts=[]
    with torch.no_grad():
      for t in range(round(args.seconds/env.dt)):
        action=agent(env.obs())[0] if agent else torch.zeros(args.num_envs,env.naction,device='cuda')
        _,_,_,info=env.step(action,auto_reset=False)
        failed|=info['fail'];alive.append((~failed).cpu().numpy())
        angles.append(info['angle'].cpu().numpy());goals.append(info['goal'].cpu().numpy());errors.append(info['error'].cpu().numpy())
        near.append(info['near_tips'].cpu().numpy());drifts.append(info['drift'].cpu().numpy());qposes.append(env.qpos[:4].cpu().numpy().copy())
        if args.robot=='revo3':actual_contacts.append(env.contact_fingers()[0].cpu().numpy())
    angles=np.array(angles);goals=np.array(goals);errors=np.array(errors);alive=np.array(alive);near=np.array(near);drifts=np.array(drifts);qposes=np.array(qposes)
    # Count actual, alternating threshold crossings, not commanded reversals.
    threshold=max(.15,args.amplitude*.60)
    crossings=np.zeros(args.num_envs,int);expect=np.ones(args.num_envs)
    for t,a in enumerate(angles):
        hit=(a*expect>threshold)&alive[t]
        crossings[hit]+=1;expect[hit]*=-1
    mean_error=np.mean(np.abs(errors),axis=0)
    success=alive[-1]&(crossings>=int(args.seconds/args.period)*2)&(mean_error<.25)&((near>=2).mean(0)>.8)
    if actual_contacts:
        actual_contacts=np.array(actual_contacts)
        success=success&((actual_contacts>=2).mean(0)>.8)
    metrics=dict(robot=args.robot,checkpoint=args.checkpoint,seed=args.seed,episodes=args.num_envs,seconds=args.seconds,amplitude_rad=args.amplitude,period_seconds=args.period,
        survival_rate=float(alive[-1].mean()),success_rate=float(success.mean()),mean_abs_error_rad=float(mean_error.mean()),
        mean_angle_span_rad=float(np.ptp(angles,axis=0).mean()),mean_alternating_crossings=float(crossings.mean()),
        minimum_alternating_crossings=int(crossings.min()),fingertip_proximity_fraction=float((near>=2).mean()),
        success_definition=f'{args.seconds:g}s without falling, >={int(args.seconds/args.period)*2} alternating crossings of +/-{threshold:.3f} rad, MAE<0.25 rad, >=2 fingertips within 24mm for >80% frames',
        per_episode_success=success.tolist(),per_episode_crossings=crossings.tolist())
    if args.robot=='revo3':
        metrics['actual_two_fingertip_contact_fraction']=float((actual_contacts>=2).mean())
        metrics['success_definition']+='; >=2 actual contacting fingertip bodies for >80% frames'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'rollout.npz',angles=angles,goals=goals,errors=errors,alive=alive,near=near,drifts=drifts,qpos=qposes,actual_contacts=np.asarray(actual_contacts),dt=env.dt)
    print(json.dumps({k:v for k,v in metrics.items() if not k.startswith('per_episode')},indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True);times=np.arange(len(angles))*env.dt
    axs[0].plot(times,np.degrees(goals[:,0]),'k--',label='target');axs[0].plot(times,np.degrees(angles[:,:4]),alpha=.7);axs[0].set_ylabel('Pen angle (deg)');axs[0].legend()
    axs[1].plot(times,drifts[:,:4]*1000);axs[1].set_ylabel('Center drift (mm)');axs[1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'tracking.png');plt.close(fig)
    if args.video:
        model=env.cpu_model;data=mujoco.MjData(model);renderer=mujoco.Renderer(model,720,960)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.125,0,.025];camera.distance=.40;camera.azimuth=130;camera.elevation=-35
        opt=mujoco.MjvOption();opt.geomgroup[3]=0
        contact_counts=[]
        with imageio.get_writer(out/'rollout.mp4',fps=40,codec='libx264',quality=8) as writer:
          for t,q in enumerate(qposes[:,0]):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            count=int(sum((model.geom_bodyid[c.geom1]==env.pen_id or model.geom_bodyid[c.geom2]==env.pen_id) for c in data.contact))
            contact_counts.append(count)
            renderer.update_scene(data,camera=camera,scene_option=opt)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,64),fill=(18,23,32))
            draw.text((16,10),f'{env.spec["label"]} / PPO  |  t={(t+1)*env.dt:.2f}s  |  target={np.degrees(goals[t,0]):+.1f} deg  |  pen={np.degrees(angles[t,0]):+.1f} deg',fill='white')
            draw.text((16,34),f'Free rigid body / gravity / fingertip contact  |  contacts={count}  |  alive={alive[t,0]}',fill='white')
            writer.append_data(np.asarray(frame))
            if t==0:frame.save(out/'preview.png')
        renderer.close()
        (out/'render_contact_counts.json').write_text(json.dumps(contact_counts))

if __name__=='__main__':main()
