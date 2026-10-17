"""Short v8 contact-classification and GPU stepping preflight."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiments.wuji_hand2.evaluate import contact_forces as legacy_forces
from .contact import contact_forces, sensor_body_channels
from .env import Env, ROOT


def main():
    cfg = json.loads(Path(__file__).with_name("config.json").read_text())
    env = Env(num_envs=8, pinch_reward_weight=1.5, command_clip_penalty=8.,
              tracking_sigma=.35, config=cfg)
    assert env.obs().shape == (8, 111)
    model = env.cpu_model
    channels = sensor_body_channels(model)
    mapping = env.geom_finger.cpu().numpy()
    legacy_mapping = env.geom_legacy_finger.cpu().numpy()
    legacy_bodies = dict(channels)
    for finger_index, finger in enumerate(("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")):
        legacy_bodies[model.body(f"r_{finger}_distal").id] = finger_index
    for geom_id, body_id in enumerate(model.geom_bodyid):
        expected = channels.get(int(body_id), -1) + 1
        assert int(mapping[geom_id]) == expected, (geom_id, expected, mapping[geom_id])
        expected_legacy = legacy_bodies.get(int(body_id), -1) + 1
        assert int(legacy_mapping[geom_id]) == expected_legacy
    data = mujoco.MjData(model)
    bank = np.load(ROOT / cfg["assets"] / "grasp_seed_bank.npz")
    data.qpos[:] = bank["qpos"][0]
    data.ctrl[:] = bank["ctrl"][0]
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    sensor = contact_forces(model, data)
    legacy = legacy_forces(model, data)
    assert np.all(sensor[:5] <= legacy[:5] + 1e-8)
    assert sensor[5] >= legacy[5] - 1e-8
    archived = np.load(ROOT / "reports/wuji_hand2/palm_up_human_demo_v7/stage_a_20261016/holdout_best9000/rollout.npz")
    data.qpos[:] = archived["qpos"][120]
    mujoco.mj_forward(model, data)
    known_distal_sensor = contact_forces(model, data)
    known_distal_legacy = legacy_forces(model, data)
    assert known_distal_sensor[5] > 0.05
    assert known_distal_legacy[2] > known_distal_sensor[2]
    for _ in range(5):
        obs, reward, _, _ = env.step(torch.zeros(8, 12, device="cuda"))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    result = {
        "observation_shape": list(env.obs().shape),
        "actions": 12,
        "sensor_bodies": [model.body(body).name for body in channels],
        "cpu_sensor_forces_n": sensor.tolist(),
        "cpu_legacy_forces_n": legacy.tolist(),
        "archived_middle_distal_frame_sensor_forces_n": known_distal_sensor.tolist(),
        "archived_middle_distal_frame_legacy_forces_n": known_distal_legacy.tolist(),
        "all_gpu_geom_channels_match_cpu_classifier": True,
        "legacy_policy_observation_mapping_verified": True,
        "finite_gpu_steps": 5,
        "capacity": env.check_capacity(),
    }
    out = ROOT / cfg["runs"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
