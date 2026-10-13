"""Save relaxed reset poses separately; retain the original IK candidates."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'assets/wuji_hand2_palm_forward_v1'


def main():
    source = np.load(ASSETS / 'grasp_search_v2.npz')
    dest = ASSETS / 'grasp_search_v2_settled.npz'
    if dest.exists():
        raise RuntimeError('Settled candidate bank exists')
    m = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene.xml'))
    d = mujoco.MjData(m)
    qs = []
    for q, ctrl in zip(source['qpos'], source['ctrl']):
        mujoco.mj_resetData(m, d); d.qpos[:] = q; d.ctrl[:] = ctrl
        for _ in range(round(12 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        qs.append(d.qpos.copy())
    np.savez(dest, qpos=qs, ctrl=source['ctrl'])
    print(dest)


if __name__ == '__main__':
    main()
