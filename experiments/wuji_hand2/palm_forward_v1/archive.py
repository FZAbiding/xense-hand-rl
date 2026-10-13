"""Archive the selected upright policy together with its exact experiment evidence."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    config_path = args.config
    cfg = json.loads(config_path.read_text())
    runs = ROOT / cfg['runs']; assets = ROOT / cfg['assets']; dest = ROOT / cfg['artifacts']
    status = json.loads((runs/'status.json').read_text())
    if 'holdout' not in status:
        raise RuntimeError('Wait for the pipeline and independent holdout to finish')
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / status['selected_checkpoint']
    selected = checkpoint.parent
    for name in ('config.json', 'source_manifest.json', 'metrics.jsonl'):
        shutil.copy2(selected/name, dest/('training_'+name))
    shutil.copytree(selected/'snapshot', dest/'snapshot', dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parent, dest/'experiment', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('provenance.json', 'certified_bank_manifest.json', 'grasp_bank_60s.npz', 'orientation_preview.png'):
        shutil.copy2(assets/name, dest/name)
    certification = json.loads((assets/'certified_bank_manifest.json').read_text())
    shutil.copy2(assets/certification['certification_report'], dest/'grasp_certification.json')
    shutil.copy2(assets/'scene.xml', dest/'original_scene.xml')
    # Export the scene with mesh paths relative to the archive directory.
    root = ET.parse(assets/'scene.xml').getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', os.path.relpath((assets/mesh.get('file')).resolve(),dest))
    ET.ElementTree(root).write(dest/'scene.xml', encoding='utf-8', xml_declaration=True)
    for stage in status['stages']:
        target = dest/'stages'/stage['name']; target.mkdir(parents=True,exist_ok=True)
        source = ROOT/stage['selected_checkpoint']
        for name in ('config.json','metrics.jsonl','source_manifest.json'):
            shutil.copy2(source.parent/name,target/name)
        for name in ('metrics.json','evaluation_manifest.json','tracking.png'):
            shutil.copy2(ROOT/stage['evaluation']/name,target/('evaluation_'+name))
    if (runs/'gpu_static_diagnostic.json').exists():
        shutil.copy2(runs/'gpu_static_diagnostic.json',dest/'gpu_static_diagnostic.json')
    hashes = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
    (dest/'archive_manifest.json').write_text(json.dumps(hashes,indent=2))
    print('ARCHIVED',dest,'files',len(hashes),flush=True)


if __name__=='__main__':main()
