"""Build an isolated upright Wuji scene, with the wrist base on the table."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name('config.json')


def main():
    cfg = json.loads(CONFIG.read_text())
    dest = ROOT / cfg['assets']
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'scene.xml').exists():
        raise RuntimeError('Scene exists; use a new experiment version to change assets')
    source = ROOT / 'third_party/wuji-description/hand2/hand2_beta2/body/mjcf/right.xml'
    root = ET.parse(source).getroot()
    root.set('model', cfg['experiment_id'])
    root.find('compiler').set('meshdir', '.')
    root.find('option').attrib.update(timestep='.005', integrator='implicitfast', iterations='10', ls_iterations='8', cone='pyramidal', impratio='1')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', os.path.relpath(source.parent / '../meshes/right' / mesh.get('file'), dest))
    for actuator in root.findall('./actuator/position'):
        actuator.set('kp', '3')
    for geom in root.findall('.//worldbody/body//geom'):
        if geom.get('contype', '1') != '0':
            geom.attrib.update(friction='1.2 .005 .0001', condim='3', solref='.01 1', solimp='.9 .99 .001')
    wrist = root.find('./worldbody/body')
    wrist.set('quat', ' '.join(map(str, cfg['layout']['wrist_quat'])))
    wrist.set('pos', '0 0 0')
    # Compile in memory to find the bottom of the official wrist mesh. Its
    # mesh transform is applied by MuJoCo, including mesh centering/rotation.
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((dest / mesh.get('file')).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wid = model.body('r_wrist').id
    points = []
    for gid in np.flatnonzero(model.geom_bodyid == wid):
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        points.append(vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    bottom = float(np.concatenate(points)[:, 2].min())
    z = cfg['layout']['table_height_m'] - bottom
    wrist.set('pos', ' '.join(map(str, [*cfg['layout']['wrist_xy_m'], z])))
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', os.path.relpath(mesh.get('file'), dest))
    task = ET.parse(ROOT / 'assets/scene.xml').getroot()
    for name in ('statistic', 'visual'):
        root.append(copy.deepcopy(task.find(name)))
    root.find('statistic').set('center', '.12 0 .14')
    for item in task.find('asset'):
        item = copy.deepcopy(item)
        if item.tag == 'mesh':
            item.set('file', '../pen.obj')
        root.find('asset').append(item)
    for item in task.find('worldbody'):
        item = copy.deepcopy(item)
        if item.get('name') == 'floor':
            item.set('pos', f"0 0 {cfg['layout']['table_height_m']}")
        if item.get('name') == 'pen':
            item.set('pos', f'.105 0 {z + .105}')
        root.find('worldbody').append(item)
    root.append(copy.deepcopy(task.find('sensor')))
    ET.indent(root)
    ET.ElementTree(root).write(dest / 'scene.xml', encoding='utf-8', xml_declaration=True)
    commit = subprocess.check_output(['git', '-C', str(ROOT / 'third_party/wuji-description'), 'rev-parse', 'HEAD'], text=True).strip()
    (dest / 'provenance.json').write_text(json.dumps(dict(
        experiment_id=cfg['experiment_id'], task_orientation='palm_forward',
        config=str(CONFIG.relative_to(ROOT)), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        wrist_pos=[*cfg['layout']['wrist_xy_m'], z], wrist_quat=cfg['layout']['wrist_quat'],
        palm_normal_world=cfg['layout']['palm_normal_world'], finger_direction_world=cfg['layout']['finger_direction_world'],
        table_height_m=cfg['layout']['table_height_m'], base_bottom_world_z=z + bottom,
        source='https://github.com/wuji-technology/wuji-description', commit=commit,
        official_geometry_inertia_joint_limits_and_actuator_force_limits='preserved',
        position_actuator_kp=3, pen_length_m=.18, pen_radius_m=.009, pen_mass_kg=.025,
        license='MIT; upstream LICENSE retained in submodule'), indent=2))
    # Verify mechanics and the two independent orientation axes.
    official = mujoco.MjModel.from_xml_path(str(source))
    model = mujoco.MjModel.from_xml_path(str(dest / 'scene.xml'))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    rotation = data.xmat[model.body('r_wrist').id].reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 1], cfg['layout']['palm_normal_world'], atol=1e-12)
    np.testing.assert_allclose(-rotation[:, 2], cfg['layout']['finger_direction_world'], atol=1e-12)
    for field in ('actuator_ctrlrange', 'actuator_forcerange'):
        np.testing.assert_allclose(getattr(model, field), getattr(official, field))
    np.testing.assert_allclose(model.jnt_range[:20], official.jnt_range)
    np.testing.assert_allclose(model.body_mass[:official.nbody], official.body_mass)
    np.testing.assert_allclose(model.body_inertia[:official.nbody], official.body_inertia)
    print('VALIDATED upright fingers +Z, palm -X, wrist mesh bottom on table', z, flush=True)


if __name__ == '__main__':
    main()
