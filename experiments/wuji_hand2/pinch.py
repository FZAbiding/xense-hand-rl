"""Opposing normal-force diagnostic, distinct from resting on two tips."""
import numpy as np
import torch
import mujoco


def opposing_strength(vectors):
    """Best opposing pair of DIFFERENT fingers, each net normal force > .02 N."""
    tensor=isinstance(vectors,torch.Tensor)
    norm=torch.linalg.vector_norm(vectors,dim=-1) if tensor else np.linalg.norm(vectors,axis=-1)
    unit=vectors/(norm[...,None]+1e-8)
    scores=[]
    for i in range(5):
        for j in range(i+1,5):
            cos=(unit[...,i,:]*unit[...,j,:]).sum(-1)
            active=(norm[...,i]>.02)&(norm[...,j]>.02)
            scores.append((-cos).clip(0,1)*active)
    return torch.stack(scores,-1).max(-1).values if tensor else np.stack(scores,-1).max(-1)


def normal_vectors(model,data):
    pen=model.body('pen').id
    tips={model.body(f'r_{finger}_{suffix}').id:i for i,finger in enumerate(
        ['thumb','index_finger','middle_finger','ring_finger','pinky']) for suffix in ['distal','tip_sensor_frame']}
    vectors=np.zeros((5,3))
    for k,c in enumerate(data.contact):
        a,b=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
        partner=b if a==pen else a if b==pen else -1
        if partner not in tips:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,k,force)
        # MuJoCo frame row 0 points from geom1 to geom2.
        vectors[tips[partner]]+=max(0.,force[0])*c.frame[:3]*(1 if b==pen else -1)
    return vectors
