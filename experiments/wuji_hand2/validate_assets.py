"""Check that the Wuji task scene retains the official robot mechanics."""
import argparse
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'third_party/wuji-description/hand2/hand2_beta2/body/mjcf/right.xml'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orientation', choices=('palm_up', 'legacy_palm_down'), default='palm_up')
    args = parser.parse_args()
    task_path = ROOT / ('assets/wuji_hand2_palm_up' if args.orientation == 'palm_up' else 'assets/wuji_hand2') / 'scene.xml'
    source = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    task = mujoco.MjModel.from_xml_path(str(task_path))
    assert source.nu == task.nu == 20
    assert task.nq == source.nq + 7
    assert task.nbody == source.nbody + 1
    np.testing.assert_allclose(task.jnt_range[:20], source.jnt_range)
    np.testing.assert_allclose(task.actuator_ctrlrange, source.actuator_ctrlrange)
    np.testing.assert_allclose(task.actuator_forcerange, source.actuator_forcerange)
    np.testing.assert_allclose(task.body_mass[:source.nbody], source.body_mass)
    np.testing.assert_allclose(task.body_inertia[:source.nbody], source.body_inertia)
    for i in range(source.nbody):
        assert source.body(i).name == task.body(i).name
    data = mujoco.MjData(task)
    mujoco.mj_forward(task, data)
    wrist = data.xmat[task.body('r_wrist').id].reshape(3, 3)
    expected_palm_normal = np.array([0, 0, 1 if args.orientation == 'palm_up' else -1])
    np.testing.assert_allclose(wrist[:, 1], expected_palm_normal)
    np.testing.assert_allclose(wrist[:, 2], [-1, 0, 0])
    print(f'Wuji Hand 2 {args.orientation}: official mechanics preserved; base horizontal, palm normal {expected_palm_normal.tolist()}')


if __name__ == '__main__':
    main()
