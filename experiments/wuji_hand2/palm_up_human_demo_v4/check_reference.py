"""Independent perturbed replay of the physically optimized partial rotation."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import hashlib
import json
import shutil
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from .build import ROOT, ASSETS, LOW_DEG, HIGH_DEG, contacts
from experiments.wuji_hand2.evaluate import contact_forces, inactive_finger_force
from experiments.wuji_hand2.pinch import opposing_strength, normal_vectors


def main():
    source = ROOT/'runs/wuji_hand2/palm_up_human_demo_v4/reference_optimization_clockwise'
    out = ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v4/partial_rotation'
    out.mkdir(exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'))
    m.opt.iterations = m.opt.ls_iterations = 8
    d = mujoco.MjData(m)
    bank = np.load(ASSETS/'grasp_seed_bank.npz')
    initial, home = bank['qpos'][0], bank['ctrl'][0]
    knots = np.load(source/'best_knots.npz')['knots']
    low, high = np.radians(LOW_DEG), np.radians(HIGH_DEG)
    high[10] = min(high[10], home[10] + np.radians(35))
    rng = np.random.default_rng(2026101115)
    results, recorded = [], {}
    for episode in range(25):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = initial
        if episode:
            d.qpos[:12] += rng.normal(0, .005, 12)
            d.qpos[20:23] += rng.normal(0, .0004, 3)
        d.ctrl[:] = home
        mujoco.mj_forward(m, d)
        axis = d.xmat[m.body('pen').id].reshape(3, 3)[:, 2]
        previous = np.arctan2(axis[1], axis[0])
        angle = 0.
        trace, qs, cs, fs, cc = [], [], [], [], []
        for t in range(280):
            p = np.clip(((t+1)*.025 - .5)/6, 0, 1)
            if p < .5:
                s, left, right = 2*p, home[:12], knots[0]
            else:
                s, left, right = 2*(p-.5), knots[0], knots[1]
            s = s*s*(3-2*s)
            desired = left + s*(right-left)
            d.ctrl[:12] = np.clip(d.ctrl[:12] + np.clip(desired-d.ctrl[:12], -.008, .008), low, high)
            d.ctrl[12:] = home[12:]
            for _ in range(5):
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            f = contact_forces(m, d)
            vectors = normal_vectors(m, d)
            vectors[3:] = 0
            pinch = float(opposing_strength(vectors))
            axis = d.xmat[m.body('pen').id].reshape(3, 3)[:, 2]
            yaw = np.arctan2(axis[1], axis[0])
            angle += np.arctan2(np.sin(yaw-previous), np.cos(yaw-previous))
            previous = yaw
            goal = -np.pi/3*p
            drift = np.linalg.norm(d.qpos[20:23]-initial[20:23])
            tilt = np.arcsin(np.clip(abs(axis[2]), 0, 1))
            reverse = bool((d.qpos[[6, 7, 10, 11]] < 0).any())
            active = tuple(np.flatnonzero(f[:3] > .02))
            pair = {(0, 1): 0, (1, 2): 1, (0, 2): 2}.get(active, -1) if pinch >= .3 else -1
            role = pair == int(np.floor(goal/(np.pi/3)+.5)) % 3
            trace.append([angle, goal, angle-goal, drift, tilt, np.sum(f[:3] > .02), f[5], inactive_finger_force(m,d), pinch, reverse, role])
            qs.append(d.qpos.copy()); cs.append(d.ctrl.copy()); fs.append(f.copy())
            if episode == 0:
                cc.append(contacts(m,d))
            if drift > .045 or tilt > np.radians(40) or reverse or not np.isfinite(d.qpos).all():
                break
        tr, q = np.asarray(trace), np.asarray(qs)
        row = dict(episode=episode, perturbed=bool(episode), frames=len(tr),
                   actual_final_angle_deg=float(np.degrees(tr[-1,0])),
                   mae_deg=float(np.degrees(abs(tr[:,2])).mean()),
                   max_drift_m=float(tr[:,3].max()),
                   two_tip_fraction=float((tr[:,5]>=2).mean()),
                   non_tip_fraction=float((tr[:,6]>.05).mean()),
                   inactive_fraction=float((tr[:,7]>.02).mean()),
                   opposing_pinch_fraction=float((tr[:,8]>=.3).mean()),
                   reverse_bending_fraction=float(tr[:,9].mean()),
                   contact_role_fraction=float(tr[20:,10].mean()) if len(tr)>20 else 0.,
                   middle_pip_excursion_deg=float(np.degrees(np.ptp(q[:,10]))))
        row['partial_rotation_passed'] = bool(len(tr)==280 and row['actual_final_angle_deg']<=-54 and row['mae_deg']<10
            and row['max_drift_m']<.025 and row['two_tip_fraction']>.95 and row['non_tip_fraction']<.05
            and row['inactive_fraction']<.01 and row['opposing_pinch_fraction']>.9 and not reverse
            and row['reverse_bending_fraction']==0)
        row['handover_passed'] = bool(row['partial_rotation_passed'] and row['contact_role_fraction']>.8)
        results.append(row)
        if episode < 4:
            recorded.update({f'qpos_{episode}':q, f'ctrl_{episode}':np.asarray(cs), f'metrics_{episode}':tr, f'forces_{episode}':np.asarray(fs)})
        if episode == 0:
            (out/'contacts_episode_0.json').write_text(json.dumps(cc))
        print('REFERENCE_CHECK',episode,row['actual_final_angle_deg'],row['partial_rotation_passed'], flush=True)
    summary = dict(controller_type='open-loop actuator trajectory; not a learned policy',
        target_angle_deg=-60, seconds=7, seed=2026101115, perturbed_episodes=24,
        perturbations=dict(active_joint_std_rad=.005, pen_position_std_m=.0004),
        partial_rotation_success_fraction=float(np.mean([x['partial_rotation_passed'] for x in results[1:]])),
        handover_success_fraction=float(np.mean([x['handover_passed'] for x in results[1:]])),
        note='Partial rotation is a separate diagnostic. It does not replace or relax the handover gate.', episodes=results)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'rollout.npz',dt=.025, **recorded)
    for name in ('config.json','best_knots.npz','search.json'):
        shutil.copy2(source/name, out/name)
    shutil.copy2(Path(__file__),out/'check_reference.py')
    shutil.copy2(Path(__file__).with_name('optimize_reference.py'),out/'optimize_reference.py')
    cams = []
    for az, el in [(130,35), (90,85)]:
        camera=mujoco.MjvCamera();camera.lookat[:]=[.065,0,.10];camera.distance=.34;camera.azimuth=az;camera.elevation=el
        cams.append(camera)
    renderer=mujoco.Renderer(m,height=480,width=640)
    options=mujoco.MjvOption();options.geomgroup[3]=0
    with imageio.get_writer(out/'rollout.mp4',fps=40,codec='libx264',quality=8,macro_block_size=None) as writer:
        for i,q in enumerate(recorded['qpos_0']):
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            views=[]
            for camera in cams:
                renderer.update_scene(d,camera=camera,scene_option=options);views.append(renderer.render().copy())
            frame=Image.fromarray(np.concatenate(views,axis=1));draw=ImageDraw.Draw(frame)
            actual=np.degrees(recorded['metrics_0'][i,0])
            draw.text((15,15),f'v4 partial rotation | t={(i+1)*.025:.2f}s | actual={actual:.1f} deg | target=-60 deg',fill='white')
            draw.text((660,15),'Top view | No finger handover achieved',fill='white')
            writer.append_data(np.asarray(frame))
            if i in [0,139,279]:frame.save(out/f'frame_{i:03d}.png')
    renderer.close()
    paths=[ASSETS/'scene.xml',ASSETS/'grasp_seed_bank.npz',out/'best_knots.npz',out/'check_reference.py',out/'optimize_reference.py']
    (out/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':
    main()
