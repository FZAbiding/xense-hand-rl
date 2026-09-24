"""Read-only policy diagnostics. Writes reports, never changes training/assets."""
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train import ActorCritic
from hand_config import hand_config


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def logs():
    rows = []
    for run, end in [('ppo_direct', 550), ('ppo_wide', 800), ('revo3_first', 600),
                     ('revo3_fingertip', 300), ('revo3_medium', 550)]:
        all_rows = [json.loads(x) for x in (ROOT / 'runs' / run / 'metrics.jsonl').read_text().splitlines()]
        selected = next(x for x in all_rows if x['iteration'] == end)
        recent = [x for x in all_rows if end - 100 < x['iteration'] <= end]
        rows.append(dict(run=run, **selected,
                         loss_median_last_100_iterations=float(np.median([x['loss'] for x in recent])),
                         max_logged_kl_entire_run=max(x['kl'] for x in all_rows),
                         config=json.loads((ROOT / 'runs' / run / 'config.json').read_text())))
    save_json('training_summary.json', rows)
    flat = [{k: v for k, v in r.items() if k != 'config'} for r in rows]
    with (OUT / 'training_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return rows


def rollout(label, checkpoint, episodes=4, seconds=12, seed=20260923,
            physics_dt=.005, solver_iterations=8):
    ck = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)
    robot = ck['config'].get('robot', 'leap'); spec = hand_config(robot)
    m = mujoco.MjModel.from_xml_path(str(spec['assets'] / 'scene.xml'))
    m.opt.timestep = physics_dt; m.opt.iterations = solver_iterations; m.opt.ls_iterations = 8
    d = mujoco.MjData(m); h = m.nu
    g = np.load(spec['assets'] / 'grasp.npz'); home = g['ctrl']; center = g['qpos'][h:h+3]
    pen = m.body('pen').id
    tips = {int(m.geom(n).bodyid[0]) for n in spec['geoms']}
    palm = {i for i in range(1, m.nbody) if m.body_parentid[i] == 0 and i != pen}
    net = ActorCritic(ck['nobs'], ck.get('naction', 16)); net.load_state_dict(ck['model']); net.eval()
    d.qpos[:] = g['qpos']; mujoco.mj_forward(m, d)
    axis = d.xmat[pen].reshape(3, 3)[:, 2]; yaw0 = np.arctan2(axis[1], axis[0])
    amp = ck['config']['amplitude']; period = ck['config']['period']; dt = .025
    rng = np.random.default_rng(seed)
    limits = m.jnt_actfrcrange[:h, 1]
    stats = {k: [] for k in ['angles', 'goals', 'tilt', 'drift', 'height', 'tip_count', 'palm', 'other',
                             'raw_action', 'action', 'command_clipped', 'command', 'q', 'torque', 'tip_force',
                             'palm_force', 'other_force', 'penetration', 'joint_velocity']}
    bodies = {}; failed_episodes = 0
    with torch.no_grad():
      for ep in range(episodes):
        mujoco.mj_resetData(m, d); d.qpos[:] = g['qpos']
        d.qpos[:h] += rng.normal(0, .005, h); d.qpos[h:h+3] += rng.normal(0, .0004, 3)
        d.ctrl[:] = home; mujoco.mj_forward(m, d); last = np.zeros(h); failed = False
        for t in range(round(seconds / dt)):
            axis = d.xmat[pen].reshape(3, 3)[:, 2]
            yaw = np.arctan2(axis[1], axis[0]) - yaw0; angle = np.arctan2(np.sin(yaw), np.cos(yaw))
            phase = t * dt * 2 * np.pi / period
            obs = np.r_[d.qpos[:h]-home, d.qvel[:h]*.1, (d.qpos[h:h+3]-center)*20, axis,
                        d.qvel[h:h+3], d.qvel[h+3:h+6]*.1, np.sin(angle), np.cos(angle),
                        amp*np.sin(phase), amp*2*np.pi/period*np.cos(phase), np.sin(phase), np.cos(phase), last]
            raw = net(torch.tensor(obs, dtype=torch.float32).clamp(-20, 20)[None])[0][0].numpy()
            action = raw.clip(-1, 1); desired = home + .65 * action
            command = np.clip(desired, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            d.ctrl[:] = command
            for _ in range(round(dt/physics_dt)): mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d); last = action
            axis = d.xmat[pen].reshape(3, 3)[:, 2]; yaw = np.arctan2(axis[1], axis[0]) - yaw0
            angle = np.arctan2(np.sin(yaw), np.cos(yaw)); drift = np.linalg.norm(d.qpos[h:h+3]-center)
            failed |= bool(d.qpos[h+2] < .015 or drift > .075 or not np.isfinite(d.qpos).all())
            if ck['config'].get('objective', 'legacy') == 'fingertip':
                failed |= bool(d.qpos[h+2] < center[2]-.018 or drift > .045)
            partners = set(); force = dict(tip=0., palm=0., other=0.); deepest = 0.
            for i, c in enumerate(d.contact):
                a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                partner = b if a == pen else a if b == pen else -1
                if partner < 0: continue
                partners.add(partner); deepest = min(deepest, c.dist)
                wrench = np.zeros(6); mujoco.mj_contactForce(m, d, i, wrench)
                kind = 'tip' if partner in tips else 'palm' if partner in palm else 'other'
                force[kind] += max(0., wrench[0])
            for b in partners:
                name = m.body(b).name; bodies[name] = bodies.get(name, 0) + 1
            values = dict(angles=angle, goals=amp*np.sin((t+1)*dt*2*np.pi/period),
                          tilt=np.arcsin(np.clip(abs(axis[2]), 0, 1)), drift=drift, height=d.qpos[h+2],
                          tip_count=len(partners & tips), palm=bool(partners & palm), other=bool(partners-tips),
                          raw_action=raw, action=action, command_clipped=np.abs(command-desired)>1e-7,
                          command=command, q=d.qpos[:h], torque=d.qfrc_actuator[:h], joint_velocity=d.qvel[:h],
                          tip_force=force['tip'], palm_force=force['palm'], other_force=force['other'], penetration=-deepest)
            for k, v in values.items(): stats[k].append(np.array(v, copy=True))
        failed_episodes += failed
    arrays = {k: np.array(v).reshape((episodes, round(seconds/dt)) + np.array(v).shape[1:]) for k, v in stats.items()}
    np.savez_compressed(OUT / (label+'_diagnostic.npz'), **arrays, dt=dt)
    error = np.arctan2(np.sin(arrays['angles']-arrays['goals']), np.cos(arrays['angles']-arrays['goals']))
    cycle = round(period/dt); angle_cycles = arrays['angles'].reshape(episodes, -1, cycle)
    peak_pos = angle_cycles.max(-1); peak_neg = angle_cycles.min(-1)
    joint_rows = []
    for j in range(h):
        joint_rows.append(dict(joint=m.joint(j).name,
            range_low_deg=float(np.degrees(m.jnt_range[j, 0])), range_high_deg=float(np.degrees(m.jnt_range[j, 1])),
            home_deg=float(np.degrees(home[j])), command_clip_fraction=float(arrays['command_clipped'][:,:,j].mean()),
            action_clip_fraction=float((abs(arrays['raw_action'][:,:,j])>1).mean()),
            torque_saturation_fraction=float((abs(arrays['torque'][:,:,j]) >= .99*limits[j]).mean()),
            motion_span_deg=float(np.degrees(np.ptp(arrays['q'][:,:,j], axis=1)).mean())))
    with (OUT / (label+'_joints.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    total_force = arrays['tip_force'].sum()+arrays['palm_force'].sum()+arrays['other_force'].sum()
    result = dict(label=label, robot=robot, checkpoint=checkpoint, episodes=episodes, seconds=seconds, seed=seed,
        physics_dt=physics_dt,solver_iterations=solver_iterations,
        amplitude_deg=float(np.degrees(amp)), survived_fraction=1-failed_episodes/episodes,
        mae_deg=float(np.degrees(abs(error)).mean()), p95_error_deg=float(np.percentile(np.degrees(abs(error)),95)),
        normalized_mae=float(abs(error).mean()/amp),
        mean_positive_peak_deg=float(np.degrees(peak_pos).mean()), mean_negative_peak_deg=float(np.degrees(peak_neg).mean()),
        min_cycle_direction_gain=float(np.minimum(peak_pos/amp,-peak_neg/amp).min()),
        mean_tilt_deg=float(np.degrees(arrays['tilt']).mean()), p95_tilt_deg=float(np.percentile(np.degrees(arrays['tilt']),95)),
        mean_drift_mm=float(arrays['drift'].mean()*1000), p95_drift_mm=float(np.percentile(arrays['drift'],95)*1000),
        actual_two_fingertip_fraction=float((arrays['tip_count']>=2).mean()),
        mean_contacting_fingers=float(arrays['tip_count'].mean()), palm_contact_fraction=float(arrays['palm'].mean()),
        any_non_tip_contact_fraction=float(arrays['other'].mean()),
        command_clip_fraction=float(arrays['command_clipped'].mean()), action_clip_fraction=float((abs(arrays['raw_action'])>1).mean()),
        torque_saturation_fraction=float((abs(arrays['torque'])>=.99*limits).mean()),
        joint_velocity_rms_rad_s=float(np.sqrt(np.mean(arrays['joint_velocity']**2))),
        command_step_rms_rad=float(np.sqrt(np.mean(np.diff(arrays['command'],axis=1)**2))),
        normal_force_sum_mean_n=float(total_force/(episodes*round(seconds/dt))),
        palm_normal_force_sum_fraction=float(arrays['palm_force'].sum()/max(total_force,1e-9)),
        nontip_normal_force_sum_fraction=float((arrays['palm_force'].sum()+arrays['other_force'].sum())/max(total_force,1e-9)),
        p95_penetration_mm=float(np.percentile(arrays['penetration'],95)*1000),
        body_contact_fractions={k: v/(episodes*round(seconds/dt)) for k,v in bodies.items()}, joints=joint_rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['joints','body_contact_fractions']},indent=2),flush=True)
    return result


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for run, end, label in [('ppo_direct',550,'LEAP small'),('ppo_wide',800,'LEAP final'),
                            ('revo3_first',600,'Revo3 failed'),('revo3_fingertip',450,'Revo3 small'),('revo3_medium',550,'Revo3 final')]:
        rows=[json.loads(x) for x in (ROOT/'runs'/run/'metrics.jsonl').read_text().splitlines()]
        rows=[x for x in rows if x['iteration']<=end]; x=[r['iteration'] for r in rows]
        for a,key in zip(ax.ravel(),['loss','kl','std','error_rad','episode_steps','return_mean']):
            y=np.array([r[key] for r in rows]);y=np.degrees(y) if key=='error_rad' else y*.025 if key=='episode_steps' else y
            a.plot(x,y,label=label);a.set_title(key);a.grid(alpha=.2);a.set_xlabel('PPO iteration')
    ax[0,0].set_yscale('symlog',linthresh=1);ax[0,1].set_yscale('log');ax[0,0].legend(fontsize=7)
    fig.suptitle('Logged total loss is NOT comparable across different rewards');fig.tight_layout();fig.savefig(OUT/'training_comparison.png',dpi=150);plt.close(fig)
    fig, ax=plt.subplots(2,2,figsize=(12,7))
    for label in ['leap_final','leap_same_amplitude','revo3_final']:
        z=np.load(OUT/(label+'_diagnostic.npz'));t=(np.arange(z['angles'].shape[1])+1)*.025
        ax[0,0].plot(t,np.degrees(z['angles'][0]),label=label)
        ax[0,1].plot(t,np.degrees(z['tilt'][0]),label=label)
        ax[1,0].plot(t,z['drift'][0]*1000,label=label)
        ax[1,1].plot(t,np.mean(z['command_clipped'][0],axis=1),label=label)
    for a,title in zip(ax.ravel(),['Pen yaw (deg)','Pen tilt (deg)','Center drift (mm)','Fraction of target commands clipped']):
        a.set_title(title);a.set_xlabel('Time (s)');a.grid(alpha=.2);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'motion_comparison.png',dpi=150);plt.close(fig)
    canvas=Image.new('RGB',(1920,720))
    for row,hand in enumerate(['leap','revo3']):
        reader=imageio.get_reader(ROOT/'artifacts'/hand/'demo.mp4');fps=reader.get_meta_data()['fps']
        # Decode the entire video and retain selected times.
        picks={round(t*fps): (i,t) for i,t in enumerate([1.,3.,5.,7.])}
        for idx,frame in enumerate(reader):
            if idx in picks:
                col,t=picks[idx];im=Image.fromarray(frame).resize((480,360));ImageDraw.Draw(im).text((10,70),f'{hand} / {t:.1f}s',fill='yellow');canvas.paste(im,(480*col,360*row))
        reader.close()
    canvas.save(OUT/'video_keyframes.jpg')


if __name__ == '__main__':
    torch.set_num_threads(1)
    logs()
    cases=[('leap_final','artifacts/leap/policy.pt'),('leap_same_amplitude','runs/ppo_direct/checkpoint_000550.pt'),
           ('revo3_final','artifacts/revo3/v1.0/policy.pt'),('revo3_small','artifacts/revo3/v1.0/small/policy.pt')]
    results=[rollout(label,path) for label,path in cases]
    save_json('diagnostics.json',results)
    flat=[{k:v for k,v in r.items() if k not in ['joints','body_contact_fractions']} for r in results]
    with (OUT/'diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    plots()
