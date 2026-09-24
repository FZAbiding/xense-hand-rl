"""Check that the Wuji task scene retains the official robot mechanics."""
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'third_party/wuji-description/hand2/hand2_beta2/body/mjcf/right.xml'
TASK = ROOT / 'assets/wuji_hand2/scene.xml'


def main():
    source = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    task = mujoco.MjModel.from_xml_path(str(TASK))
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
    print('Wuji Hand 2 official joint limits, actuator ranges, link masses and inertias preserved')


if __name__ == '__main__':
    main()
