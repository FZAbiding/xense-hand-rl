"""Refine the policy-derived tripod candidate without touching previous results."""
import json,hashlib
import numpy as np,mujoco
from experiments.wuji_hand2.palm_up_tripod_v3.build import assess,DEST,ROOT,FOLD


def main():
    report=DEST/'refinement_certification.json'
    if report.exists():raise RuntimeError('No overwrite')
    b=np.load(DEST/'policy_seed_candidates.npz');m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8
    rng=np.random.default_rng(202610103);accepted=[];search=[];rows=[];chosen=[]
    for i in range(1024):
        q=b['qpos'][0].copy();c=b['ctrl'][0].copy();c[:12]+=rng.normal(0,[.005,.015,.03,.06][i%4],12)
        c[:12]=np.clip(c[:12],m.actuator_ctrlrange[:12,0]+.04,m.actuator_ctrlrange[:12,1]-.04)
        r,settled=assess(m,q,c,3);r['candidate']=i
        if r['passed']:
            r2,s=assess(m,settled,c,12);r['long_check']=r2
            if r2['passed']:accepted.append((q,c,i))
        search.append(r)
        if i%32==0:print('REFINE',i,len(accepted),flush=True)
        if len(accepted)>=24:break
    (DEST/'refinement_search.json').write_text(json.dumps(search,indent=2))
    np.savez(DEST/'refined_candidates.npz',qpos=np.array([v[0] for v in accepted]),ctrl=np.array([v[1] for v in accepted]))
    for q,c,i in accepted:
        trials=[]
        for j in range(5):
            r,_=assess(m,q,c,60,None if j==0 else rng);r.update(candidate=i,perturbation=j);trials.append(r);rows.append(r)
        ok=all(r['passed'] for r in trials);print('CERTIFY',i,ok,flush=True)
        if ok:chosen.append((q,c,i))
        if len(chosen)>=4:break
    report.write_text(json.dumps(dict(rows=rows,selected=[v[2] for v in chosen]),indent=2))
    if not chosen:raise RuntimeError('No certified refinement')
    np.savez(DEST/'grasp_bank_60s.npz',qpos=np.array([v[0] for v in chosen]),ctrl=np.array([v[1] for v in chosen]))
    p=json.loads((ROOT/'assets/wuji_hand2_palm_up/provenance.json').read_text());p.update(grasp_count=len(chosen),active_policy_joints=12,folded_control_deg=np.degrees(FOLD).tolist(),calibration='Policy-derived tripod refinement, five independent 60s static trials')
    (DEST/'provenance.json').write_text(json.dumps(p,indent=2))
    (DEST/'certified_bank_manifest.json').write_text(json.dumps(dict(certification_report=report.name,scene_sha256=hashlib.sha256((DEST/'scene.xml').read_bytes()).hexdigest(),bank_sha256=hashlib.sha256((DEST/'grasp_bank_60s.npz').read_bytes()).hexdigest()),indent=2))
    print('CERTIFIED',len(chosen),flush=True)


if __name__=='__main__':main()
