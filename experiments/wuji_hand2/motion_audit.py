"""40 Hz joint-motion diagnostics for Wuji Hand2 evaluation trajectories."""
import numpy as np

GROUPS = {'thumb': (0, 4), 'index': (4, 8), 'middle': (8, 12),
          'ring': (12, 16), 'pinky': (16, 20)}


def joint_motion(qpos, dt):
    q = np.degrees(np.asarray(qpos)[..., :20])
    dq = np.diff(q, axis=-2)
    velocity = dq / dt
    acceleration = np.diff(velocity, axis=-2) / dt
    return {name: dict(
        mean_abs_velocity_deg_s=float(abs(velocity[..., lo:hi]).mean()),
        rms_acceleration_deg_s2=float(np.sqrt(np.mean(acceleration[..., lo:hi] ** 2))),
        p95_step_deg=float(np.percentile(abs(dq[..., lo:hi]), 95)),
        max_step_deg=float(abs(dq[..., lo:hi]).max()))
        for name, (lo, hi) in GROUPS.items()}
