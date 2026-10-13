"""Replay the previous policy and extract tripod seed poses; no scripted pen motion."""
import hashlib,json,shutil
import numpy as np,mujoco,torch
from train import ActorCritic
from experiments.wuji_hand2.evaluate import observation,contact_forces
from experiments.wuji_hand2.ring_control import natural_ring_command,RING_STEP
from experiments.wuji_hand2.palm_up_tripod_v3.build import ROOT,DEST,FOLD,assess


def main():
    if (DEST/'policy_seed_candidates.npz').exists():raise RuntimeError('Refuse to replace policy seeds')
    DEST.mkdir(exist_ok=True)
    if not (DEST/'scene.xml').exists():shutil.copy2(ROOT/'assets/wuji_hand2_palm_up/scene.xml',DEST/'scene.xml')
    checkpoint=ROOT/'runs/wuji_hand2/palm_up_pinch_v2/wide_6s/checkpoint_001000.pt'
    c=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=c['config'];net=ActorCritic(105,20);net.load_state_dict(c['model']);torch.set_num_threads(1)
    m=mujoco.MjModel.from_xml_path(str(DEST/'scene.xml'));m.opt.iterations=8;m.opt.ls_iterations=8
    b=np.load(ROOT/'assets/wuji_hand2_palm_up/grasp_bank_60s.npz');d=mujoco.MjData(m);d.qpos[:]=b['qpos'][0];d.ctrl[:]=b['ctrl'][0]
    home=d.ctrl.copy();center=d.qpos[20:23].copy();mujoco.mj_forward(m,d);ax=d.xmat[m.body('pen').id].reshape(3,3)[:,2];yaw0=np.arctan2(ax[1],ax[0])
    lo,hi=m.actuator_ctrlrange.T;margin=np.minimum((hi-lo)*.05,np.radians(2));scale=np.clip((hi-lo)*.03,.012,.035);scale[12:]*=.5;scale[12:16]=np.minimum(scale[12:16],RING_STEP)
    last=np.zeros(20);saved=[]
    with torch.no_grad():
        for t in range(480):
            obs=observation(m,d,home,center,yaw0,t*.025,last,contact_forces(m,d),cfg)
            a=net(torch.tensor(obs,dtype=torch.float32)[None])[0][0].numpy().clip(-1,1)
            d.ctrl[:]=natural_ring_command(d.ctrl,np.clip(d.ctrl+a*scale,lo+margin,hi-margin));last=a
            for k in range(5):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            if t>=120 and t%20==0:saved.append((d.qpos.copy(),d.ctrl.copy()))
    rows=[];good=[]
    for i,(q,ctrl) in enumerate(saved):
        q[12:20]=FOLD;ctrl[12:20]=FOLD;row,_=assess(m,q,ctrl,12);row['frame_index']=i;rows.append(row)
        if row['passed']:good.append((q,ctrl))
    (DEST/'policy_seed_probe.json').write_text(json.dumps(rows,indent=2))
    np.savez(DEST/'policy_seed_candidates.npz',qpos=np.array([v[0] for v in good]),ctrl=np.array([v[1] for v in good]))
    (DEST/'policy_seed_lineage.json').write_text(json.dumps(dict(parent=str(checkpoint.relative_to(ROOT)),parent_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        phase_samples='t>=120, t%20==0 for 480 control steps; both unused fingers folded before independent 12s physical test'),indent=2))
    if not good:raise RuntimeError('No stable tripod seeds')
    print('SEEDS',len(good))


if __name__=='__main__':main()
