"""Archive all 24 fixed-seed v7/v8 contact traces without changing frozen evaluators.

This replays the same CPU control/physics loop as v8 evaluate.py, checks every
episode's duration and final signed angle against archived metrics, and saves
pen-contact points for later human-calibrated region adjudication. It does not
change or run a new acceptance gate and never opens the unseen seed.
"""

import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiments.wuji_hand2.evaluate import contact_forces as legacy_contact_forces
from experiments.wuji_hand2.palm_up_human_demo_v5.common import order_measure, pen_contacts, self_depth
from experiments.wuji_hand2.palm_up_human_demo_v8.long_training.env import HIGH_DEG, LOW_DEG, ROOT
from experiments.wuji_hand2.palm_up_human_demo_v8.long_training.evaluate import cpu_observation
from experiments.wuji_hand2.palm_up_human_demo_v8.long_training.train import ActorCritic
from experiments.wuji_hand2.pinch import normal_vectors as legacy_normal_vectors, opposing_strength


OUTPUT = ROOT / "reports/wuji_hand2/palm_up_human_demo_v9/gate0_20261019"
SOURCES = {
    "v7_baseline": (
        ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/preflight_20261017/baseline_v7_wrapped_for_v8.pt",
        ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/preflight_20261017/baseline_v7_legacy_obs_sensor_gate/metrics.json",
    ),
    "v8_iteration_996": (
        ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/pilot_20261017/final_0000996/model.pt",
        ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/pilot_20261017/final_0000996/fixed_validation/metrics.json",
    ),
}


def replay(checkpoint, archived_metrics):
    torch.set_num_threads(1)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(archived_metrics.read_text())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metrics["checkpoint_sha256"] == checkpoint_sha
    assert metrics["seed"] == 2026101701 and not metrics["holdout"]
    cfg = ck["config"]
    assert cfg["validation_episodes"] == len(metrics["episodes"]) == 24
    task = cfg["levels"][ck["curriculum_level"]]
    assert task["angle_deg"] == 360 and task["duration"] == 3 and task["episode_seconds"] == 5.5
    assets = ROOT / cfg["assets"]
    for filename in ("scene.xml", "grasp_seed_bank.npz"):
        path = assets / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ck["source_manifest"][str(path.relative_to(ROOT))]
    model = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
    data = mujoco.MjData(model)
    bank = np.load(assets / "grasp_seed_bank.npz")
    agent = ActorCritic(ck["nobs"], 12)
    agent.load_state_dict(ck["model"])
    agent.eval()
    low, high = np.radians(LOW_DEG), np.radians(HIGH_DEG)
    rng = np.random.default_rng(metrics["seed"])
    theta = np.radians(task["angle_deg"])
    duration = task["duration"]
    all_contacts = []
    checks = []
    with torch.no_grad():
        for episode in range(24):
            gi = episode % len(bank["qpos"])
            q = bank["qpos"][gi]
            home = bank["ctrl"][gi]
            mujoco.mj_resetData(model, data)
            data.qpos[:] = q
            data.qpos[:12] += rng.normal(0, 0.005, 12)
            data.qpos[20:23] += rng.normal(0, 0.0004, 3)
            data.ctrl[:] = home
            center = q[20:23]
            mujoco.mj_forward(model, data)
            w, x, y, z = q[23:27]
            yaw0 = np.arctan2(2 * (y * z - w * x), 2 * (x * z + w * y))
            prev = yaw0
            angle = 0.0
            last = np.zeros(20)
            contacts_this_episode = []
            for i in range(round(task["episode_seconds"] / 0.025)):
                policy_forces = legacy_contact_forces(model, data)
                policy_vectors = legacy_normal_vectors(model, data)
                policy_vectors[3:] = 0
                policy_pinch = float(opposing_strength(policy_vectors))
                obs = cpu_observation(model, data, home, center, yaw0, i * 0.025, last,
                                      policy_forces, angle, policy_pinch, theta, duration, False)
                action = agent(torch.tensor(obs, dtype=torch.float32)[None])[0][0].numpy().clip(-1, 1)
                desired = data.ctrl[:12] + action * 0.012
                data.ctrl[:12] = np.clip(desired, low, high)
                data.ctrl[12:] = home[12:]
                last = np.r_[action, np.zeros(8)]
                posture = False
                reverse = False
                data.xfrc_applied[:] = 0
                for _ in range(5):
                    mujoco.mj_step(model, data)
                    mujoco.mj_kinematics(model, data)
                    posture = posture or not order_measure(model, data)["passed"]
                    reverse = reverse or bool((data.qpos[[6, 7, 10, 11]] < 0).any())
                mujoco.mj_forward(model, data)
                axis = data.xmat[model.body("pen").id].reshape(3, 3)[:, 2]
                yaw = np.arctan2(axis[1], axis[0])
                angle += np.arctan2(np.sin(yaw - prev), np.cos(yaw - prev))
                prev = yaw
                contacts_this_episode.append({
                    "seconds": (i + 1) * 0.025,
                    "contacts": pen_contacts(model, data),
                })
                drift = float(np.linalg.norm(data.qpos[20:23] - center))
                tilt = float(np.arcsin(np.clip(abs(axis[2]), 0, 1)))
                depth = self_depth(model, data)
                if drift > 0.045 or tilt > np.radians(40) or posture or reverse or depth > 0.001 or not np.isfinite(data.qpos).all():
                    break
            expected = metrics["episodes"][episode]
            seconds = len(contacts_this_episode) * 0.025
            actual_angle_deg = float(np.degrees(angle))
            seconds_error = abs(seconds - expected["seconds"])
            angle_error = abs(actual_angle_deg - expected["actual_angle_deg"])
            if seconds_error > 1e-9 or angle_error > 1e-5:
                raise ValueError(f"Replay diverged at episode {episode}: seconds error {seconds_error}, angle error {angle_error}")
            all_contacts.append(contacts_this_episode)
            checks.append({"episode": episode, "frames": len(contacts_this_episode),
                           "angle_error_deg": angle_error})
    return all_contacts, {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "archived_metrics": str(archived_metrics.relative_to(ROOT)),
        "archived_metrics_sha256": hashlib.sha256(archived_metrics.read_bytes()).hexdigest(),
        "seed": metrics["seed"],
        "episodes": checks,
        "max_angle_replay_error_deg": max(item["angle_error_deg"] for item in checks),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {"status": "fixed-validation replay only; not a pad classifier or new acceptance result"}
    for label, (checkpoint, metrics) in SOURCES.items():
        contacts, details = replay(checkpoint, metrics)
        path = OUTPUT / f"{label}_all_contacts.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump(contacts, file, separators=(",", ":"))
        details["contact_archive"] = str(path.relative_to(ROOT))
        details["contact_archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary[label] = details
    (OUTPUT / "replay_checks.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
