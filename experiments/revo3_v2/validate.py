"""CPU-only regression checks for version isolation and V2 control/observations."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mujoco
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.revo3_v2.evaluate import observation,contact_forces,quality_pass
from experiments.revo3_v2.motion_audit import joint_motion
from experiments.revo3_v2.env import Env


class V2Tests(unittest.TestCase):
    def test_capacity_guard(self):
        dummy=SimpleNamespace(capacity_peaks=None,data=SimpleNamespace(njmax=512,naconmax=100))
        for peaks in [[512,1],[1,100],[513,101]]:
            with patch('experiments.revo3_v2.env.wp.to_torch',return_value=torch.tensor(peaks)):
                with self.assertRaises(RuntimeError):Env.check_capacity(dummy)
        with patch('experiments.revo3_v2.env.wp.to_torch',return_value=torch.tensor([89,40])):
            self.assertEqual(Env.check_capacity(dummy)['peak_constraints_per_world'],89)

    def test_motion_audit_constant_velocity(self):
        q=np.tile(np.arange(10)[:,None]*np.radians(.25),(1,21))
        metrics=joint_motion(q,.025)
        for finger in metrics.values():
            self.assertAlmostEqual(finger['mean_abs_velocity_deg_s'],10.)
            self.assertAlmostEqual(finger['p95_step_deg'],.25)
            self.assertLess(finger['rms_acceleration_deg_s2'],1e-9)

    def test_quality_gate_does_not_accept_survival_only(self):
        row=dict(survived=True,within_workspace=True,mae_deg=.3,mean_tilt_deg=.5,
                 p95_tilt_deg=1.,command_clip_fraction=.01,non_tip_force_frame_fraction=0.,
                 palm_force_frame_fraction=0.,two_tip_fraction=1.,minimum_half_cycle_gain=.98)
        self.assertTrue(quality_pass(row))
        for key,value in dict(survived=False,within_workspace=False,mae_deg=3.1,
                              mean_tilt_deg=10.1,p95_tilt_deg=15.1,command_clip_fraction=.05,
                              non_tip_force_frame_fraction=.05,palm_force_frame_fraction=.01,
                              two_tip_fraction=.8,minimum_half_cycle_gain=.89).items():
            with self.subTest(key=key):self.assertFalse(quality_pass(dict(row,**{key:value})))

    def test_physics_and_limits_not_changed(self):
        self.assertEqual((ROOT/'assets/revo3/scene.xml').read_bytes(),(ROOT/'assets/revo3_v2/scene.xml').read_bytes())
        m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/revo3_v2/scene.xml'))
        self.assertEqual((m.nu,m.nq,m.nv,m.neq),(21,28,27,0))
        self.assertNotIn(m.joint('pen_free').id,m.actuator_trnid[:,0])
        np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81])

    def test_grasp_margin_and_observation(self):
        bank=np.load(ROOT/'assets/revo3_v2/grasp_bank.npz')
        m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/revo3_v2/scene.xml'));m.opt.iterations=8
        d=mujoco.MjData(m);low,high=m.actuator_ctrlrange.T
        self.assertGreaterEqual(len(bank['qpos']),1)
        for q,home in zip(bank['qpos'],bank['ctrl']):
            self.assertGreater(float(np.minimum(home-low,high-home).min()),np.radians(3))
            d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=home;mujoco.mj_forward(m,d)
            axis=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw0=np.arctan2(axis[1],axis[0])
            forces=contact_forces(m,d)
            cfg=dict(experiment_version='2.0',amplitude=.2,period=6)
            obs=observation(m,d,home,q[21:24],yaw0,0,np.zeros(21),forces,cfg)
            self.assertEqual(obs.shape,(109,));self.assertTrue(np.isfinite(obs).all())
            np.testing.assert_allclose(obs[81:102],2*(home-low)/(high-low)-1)
            np.testing.assert_allclose(obs[102:],forces*.2)
            self.assertLess(forces[5],.1)

    def test_zero_delta_preserves_target(self):
        bank=np.load(ROOT/'assets/revo3_v2/grasp_bank.npz');m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/revo3_v2/scene.xml'))
        low,high=m.actuator_ctrlrange.T;safe=np.minimum((high-low)*.05,np.radians(2))
        for home in bank['ctrl']:
            np.testing.assert_allclose(np.clip(home,low+safe,high-safe),home)


if __name__=='__main__':unittest.main(verbosity=2)
