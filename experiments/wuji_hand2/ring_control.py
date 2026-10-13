"""Optional software posture envelope; official mechanical ranges stay unchanged.

The certified initial grasp is preserved. Out-of-envelope initial commands
move into the envelope at at most 20 deg/s rather than jumping at reset.
"""
import numpy as np

RING_LOW = np.radians([0., -15., 0., 0.])
RING_HIGH = np.radians([65., 15., 85., 60.])
RING_REST = np.radians([20., 0., 30., 20.])
RING_STEP = np.radians(20.) * .025


def natural_ring_command(current, command):
    result = command.copy() if isinstance(command, np.ndarray) else command.clone()
    for offset, j in enumerate(range(12, 16)):
        target = command[..., j].clip(float(RING_LOW[offset]), float(RING_HIGH[offset]))
        result[..., j] = current[..., j] + (target-current[..., j]).clip(-RING_STEP, RING_STEP)
    return result


def ring_pose_audit(qpos, dt):
    q = np.degrees(np.asarray(qpos)[:, 12:16])
    steady = q[min(round(3./dt), len(q)-1):]
    return dict(joint_min_deg=q.min(0).tolist(), joint_max_deg=q.max(0).tolist(),
                settling_seconds=3.,
                steady_dip_hyperextension_fraction=float(np.mean(steady[:, 3] < -5)),
                steady_mcp_hyperextension_fraction=float(np.mean(steady[:, 0] < -5)),
                steady_abduction_outside_20deg_fraction=float(np.mean(abs(steady[:, 1]) > 20)),
                steady_coordination_mae_deg=float(abs(steady[:, 3]-.65*steady[:, 2]).mean()))
