"""Build the same free-pen task around the official Wuji Hand 2 Beta 2 right hand."""
from pathlib import Path
import copy
import json
import os
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'third_party/wuji-description/hand2/hand2_beta2/body/mjcf/right.xml'
DEST = ROOT / 'assets/wuji_hand2'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SOURCE).getroot()
    root.set('model', 'wuji_hand2_beta2_pen')
    root.find('compiler').set('meshdir', '.')
    root.find('option').attrib.update(
        timestep='.005', integrator='implicitfast', iterations='10',
        ls_iterations='8', cone='pyramidal', impratio='1')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', os.path.relpath(SOURCE.parent / '../meshes/right' / mesh.get('file'), DEST))
    # Match the Revo3 task's position-controller stiffness while retaining
    # the official force limits, velocity gains and all link parameters.
    for actuator in root.findall('./actuator/position'):
        actuator.set('kp', '3')
    for geom in root.findall('.//worldbody/body//geom'):
        if geom.get('contype', '1') != '0':
            geom.attrib.update(friction='1.2 .005 .0001', condim='3',
                               solref='.01 1', solimp='.9 .99 .001')
    wrist = root.find('./worldbody/body')
    wrist.attrib.update(pos='-.04 0 .05', quat='.5 -.5 -.5 .5')

    task = ET.parse(ROOT / 'assets/scene.xml').getroot()
    for name in ('statistic', 'visual'):
        root.append(copy.deepcopy(task.find(name)))
    for item in task.find('asset'):
        item = copy.deepcopy(item)
        if item.tag == 'mesh':
            item.set('file', '../pen.obj')
        root.find('asset').append(item)
    for item in task.find('worldbody'):
        item = copy.deepcopy(item)
        if item.get('name') == 'pen':
            item.set('pos', '.10 0 .065')
        root.find('worldbody').append(item)
    root.append(copy.deepcopy(task.find('sensor')))
    ET.indent(root)
    ET.ElementTree(root).write(DEST / 'scene.xml', encoding='utf-8', xml_declaration=True)
    commit = subprocess.check_output(
        ['git', '-C', str(ROOT / 'third_party/wuji-description'), 'rev-parse', 'HEAD'],
        text=True).strip()
    joints = [j.get('name') for j in root.findall('.//worldbody//joint')]
    (DEST / 'provenance.json').write_text(json.dumps(dict(
        robot='Wuji Hand 2 Beta 2 right hand',
        source='https://github.com/wuji-technology/wuji-description', commit=commit,
        source_mjcf=str(SOURCE.relative_to(ROOT)), joint_names=joints,
        actuators=len(root.findall('./actuator/position')),
        official_geometry_inertia_joint_limits_and_actuator_force_limits='preserved',
        position_actuator_kp=3,
        task_changes=['fixed upward-facing wrist pose', '5 ms implicitfast integration',
                      'position-controller kp matched to Revo3 task; official kv and force limits retained',
                      'pen contact friction and compliance matched to Revo3 task',
                      'same free training rod, floor and lighting'],
        pen_length_m=.18, pen_radius_m=.009, pen_mass_kg=.025,
        license='MIT; upstream LICENSE retained in submodule'), indent=2))
    print(DEST / 'scene.xml')


if __name__ == '__main__':
    main()
