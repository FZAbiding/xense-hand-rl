"""Archive only this continuation's configuration, policy and evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]


def main():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    base=ROOT/cfg['runs'];dest=ROOT/cfg['artifacts'];assets=ROOT/cfg['assets']
    status=json.loads((base/'status.json').read_text())
    if 'holdout' not in status:raise RuntimeError('Evaluation incomplete')
    selected=(ROOT/status['selected_checkpoint']).parent
    shutil.copytree(selected/'snapshot',dest/'snapshot',dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parent,dest/'experiment',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['config.json','source_manifest.json','metrics.jsonl']:
        shutil.copy2(selected/name,dest/('training_'+name))
    for name in ['ring_control.py','evaluate.py','motion_audit.py']:
        p=dest/'evaluation_source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'experiments/wuji_hand2'/name,p)
    for name in ['provenance.json','grasp_bank_60s.npz','certified_bank_manifest.json','orientation_preview.png']:
        shutil.copy2(assets/name,dest/name)
    certification=json.loads((assets/'certified_bank_manifest.json').read_text())
    shutil.copy2(assets/certification['certification_report'],dest/'grasp_certification.json')
    shutil.copy2(assets/'scene.xml',dest/'original_scene.xml')
    scene=ET.parse(assets/'scene.xml')
    for mesh in scene.getroot().findall('./asset/mesh'):
        mesh.set('file',os.path.relpath((assets/mesh.get('file')).resolve(),dest))
    scene.write(dest/'scene.xml',encoding='utf-8',xml_declaration=True)
    shutil.copy2(base/'baseline_cpu60/metrics.json',dest/'baseline_metrics.json')
    shutil.copy2(base/'static_ring_diagnostic.json',dest/'static_ring_diagnostic.json')
    validation=dest/'validation';validation.mkdir(exist_ok=True)
    for p in sorted(base.glob('eval_*/metrics.json')):
        shutil.copy2(p,validation/(p.parent.name+'.json'))
    lineage=dict(parent_checkpoint=cfg['initial_checkpoint'],parent_sha256=hashlib.sha256((ROOT/cfg['initial_checkpoint']).read_bytes()).hexdigest(),
                 selected_checkpoint=status['selected_checkpoint'],selected_sha256=hashlib.sha256((ROOT/status['selected_checkpoint']).read_bytes()).hexdigest(),
                 new_iterations=600,new_environment_steps=600*1024*32,initial_bank_unchanged=True,
                 ring_output_rows_reset=[12,13,14,15],optimizer_state_reset=True)
    (dest/'lineage.json').write_text(json.dumps(lineage,indent=2))
    (dest/'README.md').write_text('# Wuji Hand2 无名指修正续训\n\n独立配置、600 轮续训、24 次 60 秒留出评估。结果见 REPORT.md 与 comparison.json；完整视频为 rollout.mp4。\n\nscene.xml 使用仓库内相对网格路径；机械参数与原掌心向前场景相同。policy.pt 必须与此配置的 natural_v1 无名指控制器一起使用，不能直接替换旧版本权重。snapshot 保存精确训练源码和原始认证资源来源；experiment 保存本轮配置。软件范围和限速不改变官方机械极限。\n\n严格任务门槛未调整；请以 metrics.json 的实际结果为准。\n')
    print('Archived',dest)


if __name__=='__main__':main()
