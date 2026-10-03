# VuAnimatedModelAsset parser
#############################
# STRUCTURE:
#   f32 lod[3]; 
#   u8 flag;
#   VuSkeleton: u32 boneCount;
#               {char name[32]; u32 hash} * boneCount;
#               int32 parent * boneCount;                          (-1 = root)
#               VuAnimationTransform{f32 T[3]; i16 Q[4]/32768; f32 S[3]} * boneCount;   (local bind pose)
#               f32 aabb[6];
#   u8 flag; VuGfxScene (skinned mesh)   [; optional shadow scene]
#   Skinned vertex: stride = chunk.id; weights+indices are the LAST 8 bytes, UV the 8 before them.
#       mobile  32  pos f32[3]@0 | normal s8[4]@12  | uv@16 | weights u8[4]@24 | indices u8[4]@28
#       pc      36  pos f32[3]@0 | normal i16[3]@12 +pad | uv@20 | weights@28 | indices@32
#       pc      44  = 36 + a tangent i16[3]+sign@20, so uv@28 | weights@36 | indices@40
#############################

import json
import numpy as np
import struct
import math
import sys
import os
import vumodelasset as VM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_skeleton(b, o):
    bc = struct.unpack_from('<I',b,o)[0]
    o += 4
    names = []
    for i in range(bc):
        rec = b[o:o+36]
        e = rec.find(b'\x00')
        names.append(rec[:e].decode('latin1'))
        o += 36
    parents = list( struct.unpack_from('<%di'%bc,b,o) )
    o += bc*4

    bones=[]
    for i in range(bc):
        T = struct.unpack_from('<3f',b,o)
        Q = [x/32767.0 for x in struct.unpack_from('<4h',b,o+12)]
        S = struct.unpack_from('<3f',b,o+20)
        bones.append({
            'name': names[i],
            'parent': parents[i],
            'T': list(T),
            'Q': Q,
            'S': list(S),
        })
        o += 32

    aabb = struct.unpack_from('<6f',b,o)
    o += 24

    return { 'bones':bones,'aabb':aabb }, o

def parse_lod_scenes(b):
    # exact way handled by game
    lod = struct.unpack_from('<3f',b,0)
    flag = b[12]
    _skel,o = parse_skeleton(b,13)

    slots={}

    def opt(name):
        nonlocal o

        if o>=len(b): 
            return
        
        f = b[o]
        o+=1

        if f:
            st=o
            sc=VM.parse_scene(b,st)
            slots[name]=(st,sc)
            o=sc['end']

    def req(name):
        nonlocal o
        st = o
        sc = VM.parse_scene(b,st)
        slots[name] = (st,sc)
        o = sc['end']

    opt('0x40') # slot +0x40   optional (1 presence byte)  -> render LOD
    req('0x38') # slot +0x38   unconditional               -> base / highest-detail render mesh (LOD0)
    opt('0x48') # slot +0x48   optional (1 presence byte)  -> extra scene (e.g. shadow) if present

    return slots, o

def parse_animated(path):
    b = open(path,'rb').read()

    lod = struct.unpack_from('<3f',b,0)
    flag = b[12]
    o = 13
    skel,o = parse_skeleton(b,o)

    # a flag byte then the VuGfxScene
    scene = None
    for st in range(o, o+8):
        try:
            sc = VM.parse_scene(b,st)
            if sc['chunks']: 
                scene=sc
                break
        except Exception: 
            pass
        
    return {
        'lod': lod,
        'skeleton': skel,
        'scene': scene,
        'raw': b
    }

def _find_skin(a, s, nb):
    best = None
    for w in range(12, s-7):
        ssum = a[:,w:w+4].astype(np.int32).sum(1)

        if np.median(ssum)!=255: 
            continue
        
        ji = a[:,w+4:w+8]
        if int(ji.max())<max(nb, 1):
            # score: prefer clean 255 sums
            frac = float(np.mean(ssum==255))
            cand = (frac, w)

            if best is None or cand>best: 
                best = cand

    return (best[1] if best else s-8)

def decode_skinned_chunk(vb, ib, aabb, nb=256, chunk_id=None):
    idx_all = np.frombuffer(ib[:(len(ib)//2)*2],'<u2').astype(np.uint32)
    maxidx = int(idx_all.max()) if idx_all.size else 0

    # chunk.id IS the stride ... the engine's loader divides by it
    # detect_stride is only a fallback for an absurd id
    s = chunk_id if (chunk_id and 12<=chunk_id<=128 and len(vb) % chunk_id==0 and len(vb)//chunk_id>maxidx) else None

    if not s: 
        s = VM.detect_stride(vb,aabb,maxidx) or VM.detect_stride(vb,aabb)

    nv = len(vb)//s

    a = np.frombuffer(vb[:nv*s],np.uint8).reshape(nv,s)
    pos = np.frombuffer(a[:,0:12].tobytes(),'<f4').reshape(nv,3)

    # Normals are s8 on mobile but 3x int16 SNORM on the PC BBR2IA build, exactly as for static models
    n8 = np.frombuffer(a[:,12:16].tobytes(),np.int8).reshape(nv,4)[:,:3].astype(np.float32)/127.0
    normal = n8

    if s >= 20:
        n16 = np.frombuffer(a[:,12:18].tobytes(),'<i2').reshape(nv,3).astype(np.float32)/32767.0
        e8 = float(np.abs(np.linalg.norm(n8,axis=1)-1.0).mean())
        e16 = float(np.abs(np.linalg.norm(n16,axis=1)-1.0).mean())

        tris0 = idx_all[:(len(idx_all)//3)*3].reshape(-1,3)
        tris0 = tris0[(tris0<nv).all(axis=1)]

        if len(tris0) >= 4:
            fn = np.cross(pos[tris0[:,1]]-pos[tris0[:,0]], pos[tris0[:,2]]-pos[tris0[:,0]]).astype(np.float64)
            flat = tris0.reshape(-1)
            wgt = np.repeat(fn,3, axis=0)

            gn = np.stack([np.bincount(flat,wgt[:,k],minlength=nv) for k in range(3)],axis=1)
            gl = np.linalg.norm(gn, axis=1, keepdims=True)
            gn = np.divide(gn, gl, out=np.zeros_like(gn), where=gl > 1e-9)

            def _al(v):
                v = v.astype(np.float64)
                vl = np.linalg.norm(v, axis=1, keepdims=True)

                vu = np.divide(v, vl, out=np.zeros_like(v), where=vl > 1e-9)

                return abs(float(np.mean(np.sum(vu*gn,axis=1))))
            
            if _al(n16) > _al(n8)+1e-6: 
                normal = n16
            
        elif e16 < e8:
            normal = n16

    w = _find_skin(a, s, nb)
    weights = a[:,w:w+4].astype(np.float32)
    ws=weights.sum(1,keepdims=True)
    ws[ws==0] = 1 
    weights = weights/ws

    joints = np.clip(a[:,w+4:w+8],0,max(nb-1,0)).astype(np.uint16)

    # UV = the 2 floats immediately before the weights block
    uvoff = w-8

    if uvoff >= 12 and uvoff+8 <= s:
        uv = np.frombuffer(a[:,uvoff:uvoff+8].tobytes(),'<f4').reshape(nv,2).copy()
        uv[~np.isfinite(uv)]=0.0

    else:
        uv=np.zeros((nv,2),np.float32)

    idx = idx_all[:(len(idx_all)//3)*3]
    tris = idx.reshape(-1,3)
    tris = tris[(tris<nv).all(1)]

    return {
        'pos': pos,
        'normal': normal,
        'uv': uv,
        'weights': weights,
        'joints': joints,
        'idx': tris,
        'nv': nv,
        'stride': s,
        'skin_off': w
    }


###### SKELETON MATH
def quat_to_mat(q):
    x,y,z,w = q
    n = x*x+y*y+z*z+w*w
    if n < 1e-12: 
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w),   s*(x*z+y*w)],
        [s*(x*y+z*w),   1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w),   s*(y*z+x*w),   1-s*(x*x+y*y)]])
def engine_rot(q):
    # VuQuaternion::toRotationMatrix
    return quat_to_mat(q).T

def local_matrix(bn):
    M = np.eye(4)
    R = engine_rot(bn['Q'])
    S = np.diag(bn['S'])
    M[:3,:3] = R@S
    M[:3,3] = bn['T']
    return M

def world_matrices(bones):
    W = [None]*len(bones)

    for i,bn in enumerate(bones):
        L = local_matrix(bn)
        W[i] = L if bn['parent']<0 else W[bn['parent']]@L

    return W

def bind_model_matrices(bones):
    return [local_matrix(bn) for bn in bones]

def bind_local_trs(bones):
    M = bind_model_matrices(bones)
    out = []
    for i, bn in enumerate(bones):
        L = M[i] if bn['parent'] < 0 else np.linalg.inv(M[bn['parent']]) @ M[i]
        out.append(mat_to_trs(L))
    return out

def mat_to_trs(M):
    T = [float(x) for x in M[:3, 3]]
    A = np.array(M[:3, :3], dtype=np.float64)
    S = [float(np.linalg.norm(A[:, k])) or 1.0 for k in range(3)]
    R = np.stack([A[:, k] / S[k] for k in range(3)], axis=1)

    if np.linalg.det(R) < 0: # a mirrored basis: fold the flip into X
        R[:, 0] *= -1.0
        S[0] = -S[0]

    t = R[0, 0] + R[1, 1] + R[2, 2]

    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]

    return T, [float(x) for x in q], [float(x) for x in S]

