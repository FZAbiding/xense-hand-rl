"""Functional contact-role curriculum for the accepted three-finger primitive."""
import torch
from experiments.wuji_hand2.palm_up_human_demo_v4.env import Env


class HandoverEnv(Env):
    """Robot contact schedule is a hypothesis to verify, not RGB force labels.

    TI -> IM -> MT every ~60 degrees. The third finger must actually clear
    the pen; merely rotating a permanently closed tripod is not a handover.
    """
    def step(self, action, auto_reset=True):
        obs, reward, done, info = super().step(action, auto_reset=False)
        phase = torch.floor(info['goal'] / (torch.pi / 3) + .5).long() % 3
        pairs = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]],device=action.device)[phase]
        f = self.forces[:, :3]
        rows = torch.arange(self.n,device=action.device)
        a, b, clear = (f[rows,pairs[:,j]] for j in range(3))
        touch = (a/.05).clamp(0,1)*(b/.05).clamp(0,1)
        terms = dict(contact_role=1.5*touch,
                     released_finger=-3*clear.clamp(max=2)-.5*(clear>.02).float())
        extra = sum(terms.values())
        reward += extra
        self.total_return += extra
        info['reward_terms'].update(terms)
        info['episode_return'] = self.total_return.clone()
        if auto_reset:
            self.reset(done)
            obs = self.obs()
        return obs, reward, done, info
