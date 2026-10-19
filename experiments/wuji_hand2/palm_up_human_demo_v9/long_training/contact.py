"""V9 frozen, locally bounded pad classifier shared with the GPU environment.

The sensor bodies retain the prior pad proxy. Only small neighborhoods around
the user's accepted v8 middle/thumb distal contact examples are additionally
counted as pad; other distal-body contacts remain non-pad.
"""

import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PAD_MAP_PATH = ROOT / "assets/wuji_hand2_palm_up_human_demo_v9/pad_regions.json"
PAD_MAP = json.loads(PAD_MAP_PATH.read_text())
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def sensor_body_channels(model):
    return {model.body(f"r_{finger}_tip_sensor_frame").id: i
            for i, finger in enumerate(FINGERS)}


def distal_regions(model):
    return {
        model.body(body_name).id: (
            int(spec["finger_channel"]),
            np.asarray(spec["accepted_example_centers_local_m"], dtype=np.float64),
            float(spec["neighborhood_radius_m"]) ** 2,
        )
        for body_name, spec in PAD_MAP["distal_regions"].items()
    }


def classify_local_position(model, body_id, local_position_m):
    """Return zero-based pad finger channel, or 5 for non-pad pen contact."""
    sensor = sensor_body_channels(model).get(body_id)
    if sensor is not None:
        return sensor
    region = distal_regions(model).get(body_id)
    if region is None:
        return 5
    channel, centers, radius_sq = region
    position = np.asarray(local_position_m)
    return channel if np.min(np.sum((centers - position) ** 2, axis=1)) <= radius_sq else 5


def classify_contact(model, data, body_id, world_position_m):
    rotation = data.xmat[body_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(world_position_m) - data.xpos[body_id])
    return classify_local_position(model, body_id, local)


def contact_forces(model, data):
    pen = model.body("pen").id
    palm = model.body("r_wrist").id
    forces = np.zeros(7)
    for index, contact in enumerate(data.contact):
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        partner = b if a == pen else a if b == pen else -1
        if partner < 0:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = max(0., float(wrench[0]))
        channel = classify_contact(model, data, partner, contact.pos)
        forces[channel] += normal
        if partner == palm:
            forces[6] += normal
    return forces


def normal_vectors(model, data):
    pen = model.body("pen").id
    vectors = np.zeros((5, 3))
    for index, contact in enumerate(data.contact):
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        partner = b if a == pen else a if b == pen else -1
        if partner < 0:
            continue
        channel = classify_contact(model, data, partner, contact.pos)
        if channel == 5:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        vectors[channel] += max(0., float(wrench[0])) * contact.frame[:3] * (1 if b == pen else -1)
    return vectors
