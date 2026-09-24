"""Adapt the official BrainCo Revo3 right hand for free-body pen manipulation."""
from pathlib import Path
import copy
import json
import os
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'assets/revo3'
SOURCE=ROOT/'third_party/brainco-description/revo3_system/mjcf/revo3_right.xml'
FINGERS=('thumb','index','middle','ring','little')

def main():
    DEST.mkdir(exist_ok=True,parents=True)
    root=ET.parse(SOURCE).getroot()
    option=root.find('option')
    option.attrib.update(timestep='.005',integrator='implicitfast',iterations='10',ls_iterations='8',cone='pyramidal',impratio='1')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file',os.path.relpath((SOURCE.parent/mesh.get('file')).resolve(),DEST))
    defaults=root.find('default')
    ET.SubElement(defaults,'joint',damping='.08',armature='.0002',frictionloss='.01')
    col=defaults.find("default[@class='collision']/geom")
    col.attrib.update(friction='1.2 .005 .0001',condim='3',solref='.01 1',solimp='.9 .99 .001')
    # Preserve all official collision meshes, joint axes, limits, masses and inertia.
    base=root.find('./worldbody/body');base.set('pos','.22 0 0');base.set('quat','.70710678 0 -.70710678 0')
    for finger in FINGERS:
        tip=root.find(f".//body[@name='right_{finger}_tip_Link']")
        ET.SubElement(tip,'site',name=f'{finger}_tip',size='.002',rgba='1 0 0 1',group='4')
    actuators=ET.SubElement(root,'actuator');joint_names=[]
    for joint in root.findall('.//worldbody//joint'):
        name=joint.get('name');joint_names.append(name)
        ET.SubElement(actuators,'position',name=name+'_motor',joint=name,kp='3',ctrlrange=joint.get('range'),forcerange='-.35 .35')
    # Reuse exactly the previous freely moving pen, ground and lighting.
    scene=ET.parse(ROOT/'assets/scene.xml').getroot()
    for name in ['statistic','visual']:
        root.append(copy.deepcopy(scene.find(name)))
    for item in scene.find('asset'):
        item=copy.deepcopy(item)
        if item.tag=='mesh':item.set('file','../pen.obj')
        root.find('asset').append(item)
    for item in scene.find('worldbody'):
        item=copy.deepcopy(item)
        if item.get('name')=='pen':item.set('pos','.10 .01 .07')
        root.find('worldbody').append(item)
    root.append(copy.deepcopy(scene.find('sensor')))
    ET.indent(root);ET.ElementTree(root).write(DEST/'scene.xml',encoding='utf-8',xml_declaration=True)
    commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/brainco-description'),'rev-parse','HEAD'],text=True).strip()
    metadata=dict(robot='BrainCo Revo3 right hand',source='https://github.com/BrainCoTech/brainco-description',commit=commit,
        joint_names=joint_names,actuators=21,collision='all official convex mesh geometries retained',kp=3,damping=.08,armature=.0002,torque_limit_nm=.35,
        pen_length_m=.18,pen_radius_m=.009,pen_mass_kg=.025,
        license_note='Upstream currently has no license file; assets remain in the source checkout and are not relicensed.')
    (DEST/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(DEST/'scene.xml')

if __name__=='__main__':main()
