"""CPU-only structural regression checks for the two pen manipulation tasks."""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from hand_config import hand_config

ROOT = Path(__file__).resolve().parent


class AssetTests(unittest.TestCase):
    def test_free_pen_and_policy_dimensions(self):
        for robot, actions in [('leap', 16), ('revo3', 21)]:
            with self.subTest(robot=robot):
                spec = hand_config(robot)
                model = mujoco.MjModel.from_xml_path(str(spec['assets'] / 'scene.xml'))
                grasp = np.load(spec['assets'] / 'grasp.npz')
                self.assertEqual((model.nu, model.nq, model.nv), (actions, actions + 7, actions + 6))
                pen = model.joint('pen_free').id
                self.assertEqual(model.jnt_type[pen], mujoco.mjtJoint.mjJNT_FREE)
                self.assertEqual(model.jnt_qposadr[pen], actions)
                self.assertNotIn(pen, model.actuator_trnid[:, 0])
                self.assertEqual(model.neq, 0)
                np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
                self.assertEqual(grasp['ctrl'].shape, (actions,))
                self.assertEqual(grasp['qpos'].shape, (actions + 7,))
                self.assertTrue(np.isfinite(grasp['qpos']).all())
                for name in spec['tips']:
                    self.assertGreaterEqual(model.site(name).id, 0)

    def test_revo3_official_geometry_and_inertia_preserved(self):
        original = mujoco.MjModel.from_xml_path(str(
            ROOT / 'third_party/brainco-description/revo3_system/mjcf/revo3_right.xml'))
        task = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/revo3/scene.xml'))
        self.assertEqual(original.njnt, 21)
        for i in range(original.njnt):
            j = task.joint(original.joint(i).name).id
            for field in ['jnt_axis', 'jnt_pos', 'jnt_range']:
                np.testing.assert_allclose(getattr(original, field)[i], getattr(task, field)[j])
        for i in range(1, original.nbody):
            j = task.body(original.body(i).name).id
            for field in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat']:
                np.testing.assert_allclose(getattr(original, field)[i], getattr(task, field)[j])
        for i in range(original.ngeom):
            j = task.geom(original.geom(i).name).id
            for field in ['geom_type', 'geom_size', 'geom_pos', 'geom_quat']:
                np.testing.assert_allclose(getattr(original, field)[i], getattr(task, field)[j])


if __name__ == '__main__':
    unittest.main(verbosity=2)
