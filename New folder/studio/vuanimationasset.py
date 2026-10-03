import struct
import json
import math
import argparse
import sys
import os

FPS=30.0

def parse(path):
    b=open(path,'rb').read(); o=0
    nb,nf=struct.unpack_from('<ii',b,o); o+=8
    frames=[]
    for fi in range(nf):
        bones=[]
        for bi in range(nb):
            T=list(struct.unpack_from('<3f',b,o)); o+=12
            Q=list(struct.unpack_from('<4h',b,o)); o+=8   # raw int16, x,y,z,w
            S=list(struct.unpack_from('<3f',b,o)); o+=12
            bones.append({'T':T,'Q':Q,'S':S})
        frames.append(bones)
    flag=b[o]; o+=1
    assert o==len(b), f"not exact EOF: {o} != {len(b)}"
    return {
        'numBones':nb,
        'numFrames':nf,
        'fps':FPS,
        'duration':round((nf-1)/FPS,6),
        'loopFlag':flag,
        'quatScale':32767,'frames':frames
    }

def q_f(Q):   # raw int16 quat -> float quaternion (x,y,z,w)
    return [q/32767.0 for q in Q]

def _quat_norm(q):
    n=math.sqrt(sum(c*c for c in q)) or 1.0
    return [c/n for c in q]

def cmd_info(x):
    a=parse(x.file)
    print(f"bones={a['numBones']} frames={a['numFrames']} fps={a['fps']} duration={a['duration']}s loopFlag={a['loopFlag']}")
def cmd_dump(x):
    a=parse(x.file); s=json.dumps(a,indent=None if x.compact else 2)
    if x.o: 
        open(x.o,'w').write(s); print("wrote",x.o)
    else: 
        print(s)

def main():
    ap=argparse.ArgumentParser(description="VuAnimationAsset codec")
    s=ap.add_subparsers(dest='cmd',required=True)
    p=s.add_parser('info'); p.add_argument('file'); p.set_defaults(fn=cmd_info)
    p=s.add_parser('dump'); p.add_argument('file'); p.add_argument('-o'); p.add_argument('--compact',action='store_true'); p.set_defaults(fn=cmd_dump)
    a=ap.parse_args(); a.fn(a)

if __name__=='__main__': 
    main()



