"""Three simultaneous physical views with stable finger colors."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from .common import ASSETS,order_measure


def render(qs,path,title,video=True,metrics=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));d=mujoco.MjData(m)
    colors={'thumb':[1,.72,.18,1],'index_finger':[1,.35,.25,1],'middle_finger':[.2,.85,.7,1]}
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if name=='pen' and not m.geom_contype[g]:m.geom_group[g]=1
        if not name.startswith('r_'):continue
        m.geom_rgba[g]=[.55,.58,.65,1]
        for f,c in colors.items():
            if f'r_{f}' in name:m.geom_rgba[g]=c
    options=mujoco.MjvOption();options.geomgroup[2]=0;options.geomgroup[3]=0
    options.sitegroup[:]=0
    cameras=[]
    for az,el in [(130,35),(90,85),(0,10)]:
        c=mujoco.MjvCamera();c.lookat[:]=[.065,-.005,.10];c.distance=.34;c.azimuth=az;c.elevation=el;cameras.append(c)
    renderer=mujoco.Renderer(m,height=480,width=640)
    writer=imageio.get_writer(path,fps=40,codec='libx264',quality=8,macro_block_size=None) if video else None
    for i,q in enumerate(qs):
        d.qpos[:]=q;mujoco.mj_forward(m,d);views=[]
        for c in cameras:
            renderer.update_scene(d,camera=c,scene_option=options);views.append(renderer.render().copy())
        frame=Image.fromarray(np.concatenate(views,axis=1));draw=ImageDraw.Draw(frame)
        r=order_measure(m,d)
        draw.text((12,12),f'V5 | {title} | t={(i+1)*.025:.2f}s',fill='white')
        draw.text((12,35),'T=gold | I=orange | M=green | other fingers=grey',fill='white')
        draw.text((652,12),f'Top | order gap={r["min_matched_chain_gap_m"]*1000:.1f} mm | valid={r["passed"]}',fill='white')
        draw.text((1292,12),'Side | all physical collisions retained',fill='white')
        if metrics:
            draw.text((652,35),f'Actual pen angle={np.degrees(metrics[i]["angle"]):.1f} deg',fill='white')
        if writer:writer.append_data(np.asarray(frame))
        if not video or i in [0,len(qs)//2,len(qs)-1]:frame.save(path.with_name(path.stem+f'_frame_{i:04d}.png') if video else path)
    if writer:writer.close()
    renderer.close()


if __name__=='__main__':
    from .common import ART
    z=np.load(ART/'initial/candidate_pose.npz')
    render([z['qpos']],ART/'initial/initial_pose.png','new candidate; awaiting human review',video=False)
