"""Final all-link, perturbed, 60-second folded-finger clearance certification."""
import hashlib,json,shutil
import numpy as np,mujoco
from experiments.wuji_hand2.palm_up_tripod_v3.build import DEST,assess


def main():
    report=DEST/'all_link_certification.json'
    if report.exists():raise RuntimeError('Refuse to replace certification')
    source=DEST/'grasp_bank_60s_initial_search.npz'
    if not source.exists():shutil.copy2(DEST/'grasp_bank_60s.npz',source)
    b=np.load(source);m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8
    rng=np.random.default_rng(202610104);rows=[];selected=[]
    for i,(q,c) in enumerate(zip(b['qpos'],b['ctrl'])):
        trials=[]
        for j in range(5):
            r,_=assess(m,q,c,60,None if j==0 else rng,all_links=True);r.update(grasp=i,perturbation=j);trials.append(r);rows.append(r)
        if all(r['passed'] for r in trials):selected.append(i)
        print('ALL_LINK_CERT',i,all(r['passed'] for r in trials),flush=True)
    report.write_text(json.dumps(dict(source_bank=source.name,selected=selected,rows=rows,inactive_measurement='all ring/pinky links, summed normal force >0.02N'),indent=2))
    if not selected:raise RuntimeError('No fully certified tripod bank')
    np.savez(DEST/'grasp_bank_60s.npz',qpos=b['qpos'][selected],ctrl=b['ctrl'][selected])
    manifest=dict(certification_report=report.name,source_bank=source.name,scene_sha256=hashlib.sha256((DEST/'scene.xml').read_bytes()).hexdigest(),bank_sha256=hashlib.sha256((DEST/'grasp_bank_60s.npz').read_bytes()).hexdigest())
    (DEST/'certified_bank_manifest.json').write_text(json.dumps(manifest,indent=2))
    p=json.loads((DEST/'provenance.json').read_text());p['grasp_count']=len(selected);p['all_link_certification']=report.name;(DEST/'provenance.json').write_text(json.dumps(p,indent=2))
    print('FULLY_CERTIFIED',len(selected),flush=True)


if __name__=='__main__':main()
