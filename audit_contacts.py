"""Audit contact geometry in saved trajectories; this is not a dynamics test."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from hand_config import hand_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--robot', choices=['leap', 'revo3'], default='revo3')
    args = parser.parse_args()
    spec = hand_config(args.robot)
    model = mujoco.MjModel.from_xml_path(str(spec['assets'] / 'scene.xml'))
    data = mujoco.MjData(model)
    qpos = np.load(args.input)['qpos'].reshape(-1, model.nq)
    pen = model.body('pen').id
    tips = {int(model.geom(name).bodyid[0]) for name in spec['geoms']}
    counts = {}
    two_tips = 0
    any_other = 0
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        others = set()
        for contact in data.contact:
            a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            if a == pen:
                others.add(int(b))
            if b == pen:
                others.add(int(a))
        two_tips += len(others & tips) >= 2
        any_other += bool(others - tips)
        for body in others:
            name = model.body(body).name
            counts[name] = counts.get(name, 0) + 1
    result = dict(
        robot=args.robot, source=args.input,
        method='CPU contact-geometry audit of saved positions, not independent dynamics or contact-force measurement',
        sampled_frames=len(qpos), two_fingertip_fraction=two_tips / len(qpos),
        any_non_fingertip_contact_fraction=any_other / len(qpos),
        body_contact_frame_counts=counts,
        body_contact_frame_fractions={name: count / len(qpos) for name, count in counts.items()},
        minimum_pen_center_height_m=float(qpos[:, model.nu + 2].min()),
        maximum_pen_center_height_m=float(qpos[:, model.nu + 2].max()),
    )
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
