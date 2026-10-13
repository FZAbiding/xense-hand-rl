"""Check complete index/middle bone chains at every physics substep."""
import warp as wp


@wp.func
def point(xpos: wp.array2d(dtype=wp.vec3), sites: wp.array2d(dtype=wp.vec3),
          bodies: wp.array(dtype=wp.int32), tips: wp.array(dtype=wp.int32),
          world: int, finger: int, vertex: int):
    result=wp.vec3(0.0)
    if vertex<4:result=xpos[world,bodies[finger*4+vertex]]
    else:result=sites[world,tips[finger]]
    return result


@wp.func
def cross2(a: wp.vec3,b: wp.vec3):
    return a[0]*b[2]-a[2]*b[0]


@wp.kernel
def check_posture(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                  sites: wp.array2d(dtype=wp.vec3), qpos: wp.array2d(dtype=wp.float32),
                  bodies: wp.array(dtype=wp.int32), tips: wp.array(dtype=wp.int32), wrist: int,
                  bad: wp.array(dtype=wp.int32), gap: wp.array(dtype=wp.float32)):
    w=wp.tid();rotation=wp.transpose(xmat[w,wrist]);origin=xpos[w,wrist]
    minimum=float(1.0);crossing=int(0)
    for i in range(5):
        a=rotation*(point(xpos,sites,bodies,tips,w,0,i)-origin)
        b=rotation*(point(xpos,sites,bodies,tips,w,1,i)-origin)
        minimum=wp.min(minimum,a[0]-b[0])
    for i in range(4):
        a=rotation*(point(xpos,sites,bodies,tips,w,0,i)-origin)
        aa=rotation*(point(xpos,sites,bodies,tips,w,0,i+1)-origin)
        for j in range(4):
            b=rotation*(point(xpos,sites,bodies,tips,w,1,j)-origin)
            bb=rotation*(point(xpos,sites,bodies,tips,w,1,j+1)-origin)
            u=aa-a;v=bb-b;den=cross2(u,v)
            if wp.abs(den)>1.0e-10:
                t=cross2(b-a,v)/den;s=cross2(b-a,u)/den
                if t>1.0e-5 and t<0.99999 and s>1.0e-5 and s<0.99999:crossing=1
    gap[w]=wp.min(gap[w],minimum)
    reverse=qpos[w,6]<0.0 or qpos[w,7]<0.0 or qpos[w,10]<0.0 or qpos[w,11]<0.0
    if minimum<0.006 or crossing==1 or reverse:bad[w]=1
