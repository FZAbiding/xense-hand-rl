"""Conservative v8 contact proxy: only tip-sensor collision bodies count.

This is an experiment-specific, auditable proxy. It does not assert that the
sensor mesh spans every anatomically acceptable pad surface.
"""

import mujoco
import numpy as np


FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def sensor_body_channels(model):
    """Return body-id -> zero-based finger channel for the five sensor meshes."""
    return {model.body(f"r_{finger}_tip_sensor_frame").id: i
            for i, finger in enumerate(FINGERS)}


def contact_forces(model, data):
    """Five sensor forces, other pen-contact force, and palm force (new gate)."""
    pen = model.body("pen").id
    palm = model.body("r_wrist").id
    sensors = sensor_body_channels(model)
    forces = np.zeros(7)
    for i, contact in enumerate(data.contact):
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        partner = b if a == pen else a if b == pen else -1
        if partner < 0:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = max(0., float(wrench[0]))
        forces[sensors.get(partner, 5)] += normal
        if partner == palm:
            forces[6] += normal
    return forces


def normal_vectors(model, data):
    """Net pen-contact normals on the five sensor meshes, for opposing grip."""
    pen = model.body("pen").id
    sensors = sensor_body_channels(model)
    vectors = np.zeros((5, 3))
    for i, contact in enumerate(data.contact):
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        partner = b if a == pen else a if b == pen else -1
        if partner not in sensors:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        vectors[sensors[partner]] += max(0., float(wrench[0])) * contact.frame[:3] * (1 if b == pen else -1)
    return vectors
