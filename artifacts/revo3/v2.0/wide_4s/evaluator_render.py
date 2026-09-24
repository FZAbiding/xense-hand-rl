"""Render a saved CPU closed-loop trajectory in full; never rerun or reset physics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True)
    p.add_argument('--episode',type=int,default=0);p.add_argument('--stride',type=int,default=2)
    args=p.parse_args();path=ROOT/args.evaluation
    record=json.loads((path/'metrics.json').read_text());archive=np.load(path/'rollout.npz')
    qs=archive['qpos'][args.episode];a=archive['metrics'][args.episode];dt=float(archive['dt'])
    dest=path/f'full_episode_{args.episode:02d}.mp4'
    if dest.exists():raise RuntimeError('Refusing to overwrite existing video')
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/revo3_v2/scene.xml'));d=mujoco.MjData(m)
    renderer=mujoco.Renderer(m,720,960);camera=mujoco.MjvCamera()
    camera.lookat[:]=[.125,0,.025];camera.distance=.40;camera.azimuth=130;camera.elevation=-35
    option=mujoco.MjvOption();option.geomgroup[3]=0
    with imageio.get_writer(dest,fps=1/(dt*args.stride),codec='libx264',quality=8) as writer:
        for t in range(0,len(qs),args.stride):
            d.qpos[:]=qs[t];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=option)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,55),fill=(18,23,32))
            draw.text((12,10),f'Revo3 V2.0 | t={(t+1)*dt:.2f}s | target={np.degrees(a[t,1]):+.1f} | angle={np.degrees(a[t,0]):+.1f} | tilt={np.degrees(a[t,3]):.1f}',fill='white')
            draw.text((12,30),f'CPU closed loop | full {record["seconds"]:g}s | grasp={record["per_episode"][args.episode]["grasp_index"]} | no resets | strict pass={record["per_episode"][args.episode]["quality_success"]}',fill='white')
            writer.append_data(np.asarray(frame))
            if t==0:frame.save(path/f'preview_{args.episode:02d}.png')
    renderer.close();print(dest,flush=True)


if __name__=='__main__':main()
