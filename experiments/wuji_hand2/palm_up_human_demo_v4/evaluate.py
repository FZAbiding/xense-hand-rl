"""CPU verification of the exact V4 controller, joints and per-link contacts."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from train import ActorCritic
from experiments.wuji_hand2.pinch import opposing_strength, normal_vectors
from experiments.wuji_hand2.evaluate import contact_forces, inactive_finger_force
from experiments.wuji_hand2.palm_up_human_demo_v4.build import ROOT, ASSETS, LOW_DEG, HIGH_DEG, contacts


def observation(m, d, home, center, yaw0, time, last, force, cfg):
    axis = d.xmat[m.body('pen').id].reshape(3, 3)[:, 2]
    angle = np.arctan2(axis[1], axis[0]) - yaw0
    phase = time * 2 * np.pi / cfg['period']
    goal = cfg['amplitude'] * phase
    goal = np.arctan2(np.sin(goal), np.cos(goal))
    omega = cfg['amplitude'] * 2 * np.pi / cfg['period']
    base = np.r_[d.qpos[:20] - home, d.qvel[:20] * .1,
                 (d.qpos[20:23] - center) * 20, axis, d.qvel[20:23], d.qvel[23:26] * .1,
                 np.sin(angle), np.cos(angle), goal, omega, np.sin(phase), np.cos(phase), last,
                 2 * (d.ctrl - m.actuator_ctrlrange[:, 0]) / np.ptp(m.actuator_ctrlrange, axis=1) - 1,
                 force * .2]
    return base.clip(-20, 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--seed', type=int, default=2026101106)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    out = ROOT / args.output
    if (out / 'metrics.json').exists():
        raise RuntimeError('No overwrite')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg['experiment_version'] != 'human_demo_v4':
        raise ValueError('V4 checkpoint required')
    for name in ['scene.xml', 'grasp_seed_bank.npz']:
        relative = str((ASSETS / name).relative_to(ROOT))
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != ck['source_manifest'][relative]:
            raise ValueError('Assets differ from checkpoint')
    net = ActorCritic(ck['nobs'], ck['naction'])
    net.load_state_dict(ck['model'])
    net.eval()
    m = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene.xml'))
    m.opt.iterations = m.opt.ls_iterations = 8
    d = mujoco.MjData(m)
    bank = np.load(ASSETS / 'grasp_seed_bank.npz')
    low, high = np.radians(LOW_DEG), np.radians(HIGH_DEG)
    rng = np.random.default_rng(args.seed)
    rows, saved_q, traces, saved_contacts = [], [], [], []
    gates = cfg.get('stage_gate') or json.loads((Path(__file__).parent / 'config.json').read_text())['stage_gate']
    with torch.no_grad():
        for ep in range(args.episodes):
            mujoco.mj_resetData(m, d)
            gi = ep % len(bank['qpos'])
            d.qpos[:] = bank['qpos'][gi]
            home = bank['ctrl'][gi].copy()
            center = d.qpos[20:23].copy()
            d.ctrl[:] = home
            d.qpos[:12] += rng.normal(0, .005, 12)
            d.qpos[20:23] += rng.normal(0, .0004, 3)
            mujoco.mj_forward(m, d)
            axis = d.xmat[m.body('pen').id].reshape(3, 3)[:, 2]
            # GPU reset uses unperturbed initial yaw, so use bank quaternion here too.
            w, x, y, z = bank['qpos'][gi, 23:27]
            yaw0 = np.arctan2(2 * (y*z-w*x), 2 * (x*z+w*y))
            prev = yaw0
            angle = 0.
            last = np.zeros(20)
            samples, qs, cc_trace, joint_qs, roles = [], [], [], [], []
            failed = False
            for t in range(round(args.seconds / .025)):
                force = contact_forces(m, d)
                obs = observation(m, d, home, center, yaw0, t*.025, last, force, cfg)
                raw = net(torch.tensor(obs, dtype=torch.float32)[None])[0][0].numpy().clip(-1, 1)
                desired = d.ctrl[:12] + raw * .008
                command = np.clip(desired, low, high)
                clip = np.mean(abs(command-desired) > 1e-7)
                d.ctrl[:12] = command
                d.ctrl[12:] = home[12:]
                last = np.r_[raw, np.zeros(8)]
                for _ in range(5):
                    mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                force = contact_forces(m, d)
                vectors = normal_vectors(m, d)
                vectors[3:] = 0
                pinch = float(opposing_strength(vectors))
                axis = d.xmat[m.body('pen').id].reshape(3, 3)[:, 2]
                yaw = np.arctan2(axis[1], axis[0])
                angle += np.arctan2(np.sin(yaw-prev), np.cos(yaw-prev))
                prev = yaw
                goal = cfg['amplitude'] * (t+1)*.025 * 2*np.pi/cfg['period']
                active=tuple(np.flatnonzero(force[:3]>.02))
                actual_pair={(0,1):0,(1,2):1,(0,2):2}.get(active,-1) if pinch>=.3 else -1
                desired_pair=int(np.floor(goal/(np.pi/3)+.5))%3
                roles.append([desired_pair,actual_pair])
                drift = np.linalg.norm(d.qpos[20:23]-center)
                tilt = np.arcsin(np.clip(abs(axis[2]), 0, 1))
                reverse = bool((d.qpos[[6,7,10,11]] < 0).any())
                inactive = inactive_finger_force(m, d)
                samples.append([angle, goal, angle-goal, drift, tilt, clip, np.sum(force[:3]>.02),
                                force[5], inactive, pinch, reverse])
                joint_qs.append(d.qpos[:20].copy())
                if ep < 4:
                    qs.append(d.qpos.copy())
                if ep == 0:
                    cc_trace.append(contacts(m, d))
                if drift > .045 or d.qpos[22] < center[2]-.025 or tilt > np.radians(40) or reverse or not np.isfinite(d.qpos).all():
                    failed = True
                    break
            a = np.asarray(samples)
            qarr = np.asarray(qs)
            progress_ratio = float(a[-1,0]/a[-1,1]) if cfg['amplitude'] else None
            row = dict(episode=ep, seconds=len(a)*.025, survived=not failed,
                       mae_deg=float(np.rad2deg(abs(a[:,2])).mean()), max_drift_m=float(a[:,3].max()),
                       p95_tilt_deg=float(np.percentile(np.rad2deg(a[:,4]),95)),
                       command_clip_fraction=float(a[:,5].mean()), two_tip_fraction=float((a[:,6]>=2).mean()),
                       non_tip_fraction=float((a[:,7]>.05).mean()), inactive_fraction=float((a[:,8]>.02).mean()),
                       opposing_pinch_fraction=float((a[:,9]>=.3).mean()), reverse_bending_fraction=float(a[:,10].mean()),
                       signed_turns=float(a[-1,0]/(2*np.pi)), rotation_progress_ratio=progress_ratio)
            joint_qs = np.asarray(joint_qs)
            row['middle_pip_excursion_deg'] = float(np.degrees(np.ptp(joint_qs[:,10])))
            row['middle_dip_range_deg'] = np.degrees([joint_qs[:,11].min(),joint_qs[:,11].max()]).tolist()
            role_array=np.asarray(roles)
            row['contact_role_fraction']=float(np.mean(role_array[12:,0]==role_array[12:,1])) if len(role_array)>12 else 0.
            stable_pairs=[]
            for j in range(3,len(role_array)):
                pair=int(role_array[j,1])
                if pair>=0 and np.all(role_array[j-3:j+1,1]==pair) and (not stable_pairs or stable_pairs[-1]!=pair):stable_pairs.append(pair)
            row['observed_stable_contact_pairs']=stable_pairs
            row['completed_contact_handovers']=max(0,len(stable_pairs)-1)
            functional = row['survived'] and row['max_drift_m'] < gates['max_drift_m'] and row['two_tip_fraction'] >= gates['two_tip_fraction_min'] and row['non_tip_fraction'] < gates['non_tip_fraction_max'] and row['inactive_fraction'] < gates['inactive_fraction_max'] and row['opposing_pinch_fraction'] >= gates['opposing_pinch_fraction_min'] and row['reverse_bending_fraction'] == 0 and row['command_clip_fraction'] < gates['command_clip_fraction_max']
            row['stage_passed'] = bool(functional and row['middle_pip_excursion_deg'] <= gates['middle_pip_excursion_deg_max'] and row['mae_deg'] <= (gates['hold_mae_deg_max'] if not cfg['amplitude'] else gates['rotation_mae_deg_max']) and (not cfg['amplitude'] or progress_ratio>=gates['rotation_progress_ratio_min']))
            if cfg.get('contact_schedule')=='TI_IM_MT_60deg':
                row['stage_passed'] &= row['contact_role_fraction']>=gates['handover_role_fraction_min'] and row['completed_contact_handovers']>=gates['handover_completed_min']
            rows.append(row)
            traces.append(a)
            if ep < 4:
                saved_q.append(qarr)
            if ep == 0:
                saved_contacts = cc_trace
    metrics = dict(checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256((ROOT/args.checkpoint).read_bytes()).hexdigest(),
                   stage=cfg['stage'], seed=args.seed, episodes=args.episodes, requested_seconds=args.seconds,
                   success_fraction=float(np.mean([x['stage_passed'] for x in rows])),
                   survival_fraction=float(np.mean([x['survived'] for x in rows])),
                   mean_mae_deg=float(np.mean([x['mae_deg'] for x in rows])), per_episode=rows,
                   stage_gate=gates, reference='accepted human A; current stage checks robot feasibility, not a claim of full imitation')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # Ragged episodes are explicit separate arrays, never padded as successful time.
    np.savez_compressed(out/'rollout.npz',**{f'qpos_{i}':q for i,q in enumerate(saved_q)},
                        **{f'metrics_{i}':a for i,a in enumerate(traces)},dt=.025)
    (out/'contacts_episode_0.json').write_text(json.dumps(saved_contacts))
    if args.video:
        for geom in range(m.ngeom):
            if m.geom_group[geom] == 2:
                m.geom_group[geom] = 3
        for name in ['pen_visual','pen_cap','pen_ring']:
            m.geom_group[m.geom(name).id] = 1
        option=mujoco.MjvOption();option.geomgroup[3]=0
        camera=mujoco.MjvCamera();camera.lookat[:]=[.065,0,.10];camera.distance=.34;camera.azimuth=130;camera.elevation=35
        renderer=mujoco.Renderer(m,height=720,width=960)
        with imageio.get_writer(out/'rollout.mp4',fps=40,codec='libx264',quality=8,macro_block_size=None) as writer:
            for i,q in enumerate(saved_q[0]):
                d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=option)
                frame=renderer.render()
                if i==0:imageio.imwrite(out/'preview.png',frame)
                writer.append_data(frame)
        renderer.close()
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['per_episode','stage_gate']},indent=2),flush=True)


if __name__=='__main__':
    main()
