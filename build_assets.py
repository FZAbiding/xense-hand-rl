"""Build a freely moving pen and scene from the downloaded LEAP hand assets."""
from pathlib import Path
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
UPSTREAM = ROOT / 'third_party/mujoco_playground/mujoco_playground/_src/manipulation/leap_hand/xmls'


def build(length=0.18, radius=0.009, mass=0.025):
    ASSETS.mkdir(exist_ok=True)
    # Copy the XML structure, while keeping mesh references to the source checkout.
    hand = ET.parse(UPSTREAM / 'leap_rh_mjx.xml').getroot()
    for mesh in hand.findall('.//mesh'):
        filename = Path(mesh.get('file')).name
        path = ROOT / 'third_party/mujoco_menagerie/leap_hand/assets' / filename
        if not path.exists():
            path = UPSTREAM / 'meshes' / filename
        mesh.set('file', os.path.relpath(path, ASSETS))
    option = hand.find('option')
    option.set('timestep', '0.005')
    option.set('iterations', '8')
    option.set('ls_iterations', '8')
    option.set('integrator', 'implicitfast')
    # Stable, soft rubber fingertip contacts; no constraints attach pen to hand.
    for cls in ['tip', 'thumb_tip']:
        geom = hand.find(f".//default[@class='{cls}']/geom")
        geom.set('friction', '1.2 0.005 0.0001')
        geom.set('condim', '3')
    ET.indent(hand)
    ET.ElementTree(hand).write(ASSETS / 'hand.xml')
    (ASSETS/'LICENSE.playground').write_bytes((ROOT/'third_party/mujoco_playground/LICENSE').read_bytes())
    (ASSETS/'LICENSE.leap_hand').write_bytes((ROOT/'third_party/mujoco_menagerie/leap_hand/LICENSE').read_bytes())
    # Export an independently usable, colored OBJ + MTL asset along the local z axis.
    vertices, faces = [], []
    rings = [(-length/2, radius*.35), (-length/2+.008, radius),
             (length/2-.012, radius), (length/2, radius*.8)]
    n = 64
    for z, r in rings:
        for i in range(n):
            a = 2*math.pi*i/n
            vertices.append((r*math.cos(a), r*math.sin(a), z))
    for j in range(len(rings)-1):
        for i in range(n):
            a, b = j*n+i+1, j*n+(i+1)%n+1
            faces += [(a, b, b+n), (a, b+n, a+n)]
    faces += [tuple(range(n, 0, -1)), tuple(range(3*n+1, 4*n+1))]
    (ASSETS/'pen.obj').write_text('mtllib pen.mtl\nusemtl barrel\n' +
        ''.join('v %.7f %.7f %.7f\n' % v for v in vertices) +
        ''.join('f '+' '.join(map(str,f))+'\n' for f in faces))
    (ASSETS/'pen.mtl').write_text('newmtl barrel\nKd 0.05 0.4 0.85\nKs 0.3 0.3 0.3\nNs 50\n')
    # Analytic collision shape gives smooth, fast contact; mesh is visual only.
    xml = f'''<mujoco model="LEAP fingertip reciprocal pen rotation">
  <include file="hand.xml"/>
  <statistic center="0.10 0 0.02" extent="0.38"/>
  <visual><global offwidth="1280" offheight="720" azimuth="120" elevation="-30"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/></visual>
  <asset><mesh name="pen_mesh" file="pen.obj"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1=".16 .19 .23" rgb2=".20 .23 .28" width="128" height="128"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="2 2" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 -0.24" size="1 1 .01" material="floor_mat" contype="2" conaffinity="2"/>
    <camera name="overview" pos="-.25 .38 .32" xyaxes="-.74 -.67 0 .37 -.41 .83"/>
    <body name="pen" pos=".10 0 .065" quat=".70710678 .70710678 0 0">
      <freejoint name="pen_free"/>
      <geom name="pen_collision" type="capsule" size="{radius} {length/2-radius}" mass="{mass}" friction="1.2 .005 .0001" condim="3" solref=".01 1" group="3"/>
      <geom name="pen_visual" type="mesh" mesh="pen_mesh" density="0" contype="0" conaffinity="0" rgba=".06 .42 .90 1" group="2"/>
      <geom name="pen_cap" type="cylinder" size="{radius*1.01} .009" pos="0 0 {length/2-.009}" density="0" contype="0" conaffinity="0" rgba="1 .30 .08 1" group="2"/>
      <geom name="pen_ring" type="cylinder" size="{radius*1.01} .002" pos="0 0 {-length/2+.012}" density="0" contype="0" conaffinity="0" rgba=".9 .9 .95 1" group="2"/>
      <site name="pen_tip_a" pos="0 0 {length/2}" size=".002"/>
      <site name="pen_tip_b" pos="0 0 {-length/2}" size=".002"/>
    </body>
  </worldbody>
  <sensor><framepos name="pen_pos" objtype="body" objname="pen"/>
    <framezaxis name="pen_axis" objtype="body" objname="pen"/>
    <framelinvel name="pen_vel" objtype="body" objname="pen"/>
    <frameangvel name="pen_omega" objtype="body" objname="pen"/></sensor>
</mujoco>'''
    (ASSETS/'scene.xml').write_text(xml)
    sources = {}
    for name in ['mujoco_playground','mujoco_menagerie']:
        sources[name] = subprocess.check_output(['git','-C',str(ROOT/'third_party'/name),'rev-parse','HEAD'],text=True).strip()
    (ASSETS/'provenance.json').write_text(json.dumps(dict(length_m=length,radius_m=radius,mass_kg=mass,source_commits=sources), indent=2))
    print(ASSETS/'scene.xml')

if __name__ == '__main__':
    build()
