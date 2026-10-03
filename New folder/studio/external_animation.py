from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


@dataclass
class ExternalJoint:
    name: str
    parent: int = -1
    offset: Tuple[float,float,float] = (0.0,0.0,0.0)
    channels: List[str] = field(default_factory=list)


@dataclass
class ExternalMotion:
    source: str
    kind: str
    joints: List[ExternalJoint]
    frame_count: int = 0
    frame_time: float = 0.0
    channel_count: int = 0
    values: Optional[List[List[float]]] = None
    fbx_clip: object = None

    @property
    def fps(self) -> float:
        return 1.0/self.frame_time if self.frame_time>1e-9 else 0.0


BBR24 = [
    "Root","Hips","Spine1","Spine2","Spine3","Head",
    "L_Shoulder","L_Elbow","L_Wrist","L_Hand","L_Thumb",
    "R_Shoulder","R_Elbow","R_Wrist","R_Hand","R_Thumb",
    "R_Leg","R_Knee","R_Ankle","R_Foot",
    "L_Leg","L_Knee","L_Ankle","L_Foot",
]


def _norm(name: str) -> str:
    s=str(name or "").strip().lower()
    s=s.replace("mixamorig:","")
    return re.sub(r"[^a-z0-9]", "", s)


ALIASES = {
    "Root": ["root","armature","reference","hips","pelvis"],
    "Hips": ["hips","pelvis"],
    "Spine1": ["spine","spine0","spine1"],
    "Spine2": ["spine1","spine2","chest"],
    "Spine3": ["spine2","spine3","upperchest"],
    "Head": ["head","neck","neck1"],
    "L_Shoulder": ["leftshoulder","leftarm","lshoulder","lupperarm"],
    "L_Elbow": ["leftforearm","leftelbow","lelbow","lforearm"],
    "L_Wrist": ["lefthand","leftwrist","lwrist"],
    "L_Hand": ["lefthand","lhand"],
    "L_Thumb": ["lefthandthumb1","leftthumb","lthumb"],
    "R_Shoulder": ["rightshoulder","rightarm","rshoulder","rupperarm"],
    "R_Elbow": ["rightforearm","rightelbow","relbow","rforearm"],
    "R_Wrist": ["righthand","rightwrist","rwrist"],
    "R_Hand": ["righthand","rhand"],
    "R_Thumb": ["righthandthumb1","rightthumb","rthumb"],
    "L_Leg": ["leftupleg","leftthigh","lleg","lthigh"],
    "L_Knee": ["leftleg","leftknee","lknee","lshin"],
    "L_Ankle": ["leftfoot","leftankle","lankle"],
    "L_Foot": ["lefttoebase","lefttoe","lfoot"],
    "R_Leg": ["rightupleg","rightthigh","rleg","rthigh"],
    "R_Knee": ["rightleg","rightknee","rknee","rshin"],
    "R_Ankle": ["rightfoot","rightankle","rankle"],
    "R_Foot": ["righttoebase","righttoe","rfoot"],
}

# BBR's shoulder bone is the upper-arm pivot; its wrist/hand pair has no
# one-to-one equivalent in Mixamo, so both use the Mixamo hand rotation.
MIXAMO_TO_BBR = {
    "Root":"Hips", "Hips":"Hips", "Spine1":"Spine", "Spine2":"Spine1",
    "Spine3":"Spine2", "Head":"Head",
    "L_Shoulder":"LeftArm", "L_Elbow":"LeftForeArm",
    "L_Wrist":"LeftHand", "L_Hand":"LeftHand", "L_Thumb":"LeftHandThumb1",
    "R_Shoulder":"RightArm", "R_Elbow":"RightForeArm",
    "R_Wrist":"RightHand", "R_Hand":"RightHand", "R_Thumb":"RightHandThumb1",
    "L_Leg":"LeftUpLeg", "L_Knee":"LeftLeg",
    "L_Ankle":"LeftFoot", "L_Foot":"LeftToeBase",
    "R_Leg":"RightUpLeg", "R_Knee":"RightLeg",
    "R_Ankle":"RightFoot", "R_Foot":"RightToeBase",
}


def parse_bvh(path: str) -> ExternalMotion:
    lines=Path(path).read_text(encoding="utf-8",errors="replace").splitlines()
    joints: List[ExternalJoint]=[]
    stack: List[int]=[]
    current=-1
    motion_index=-1
    for i,raw in enumerate(lines):
        line=raw.strip()
        if line.upper()=="MOTION":
            motion_index=i
            break
        m=re.match(r"^(ROOT|JOINT)\s+(.+)$",line,re.I)
        if m:
            parent=stack[-1] if stack else -1
            joints.append(ExternalJoint(m.group(2).strip(),parent))
            current=len(joints)-1
            continue
        if line.startswith("End Site"):
            current=-2
            continue
        if line=="{":
            if current>=0:stack.append(current)
            elif current==-2:stack.append(-2)
            continue
        if line=="}":
            if stack:stack.pop()
            current=stack[-1] if stack and stack[-1]>=0 else -1
            continue
        if current>=0 and line.upper().startswith("OFFSET "):
            parts=line.split()
            if len(parts)>=4:
                try:joints[current].offset=(float(parts[1]),float(parts[2]),float(parts[3]))
                except Exception:pass
        if current>=0 and line.upper().startswith("CHANNELS "):
            parts=line.split()
            try:n=int(parts[1]); joints[current].channels=parts[2:2+n]
            except Exception:pass
    if motion_index<0:
        raise ValueError("BVH MOTION section not found")
    frame_count=0; frame_time=0.0; data_start=None
    for i in range(motion_index+1,min(len(lines),motion_index+8)):
        line=lines[i].strip()
        m=re.match(r"^Frames:\s*(\d+)",line,re.I)
        if m:frame_count=int(m.group(1))
        m=re.match(r"^Frame\s+Time:\s*([0-9eE+\-.]+)",line,re.I)
        if m:
            frame_time=float(m.group(1)); data_start=i+1
            break
    channel_count=sum(len(j.channels) for j in joints)
    values=[]
    if data_start is not None:
        for raw in lines[data_start:data_start+frame_count]:
            try:row=[float(x) for x in raw.strip().split()]
            except Exception:continue
            if row:values.append(row)
    return ExternalMotion(str(path),"BVH",joints,frame_count,frame_time,channel_count,values)


def inspect_fbx_skeleton(path: str) -> ExternalMotion:
    from fbx_motion import load_fbx_clip
    clip=load_fbx_clip(path)
    joints=[ExternalJoint(j.name,j.parent) for j in clip.joints]
    return ExternalMotion(str(path),"FBX animation",joints,clip.frame_count,
                          1.0/clip.fps,len(clip.channels),None,clip)


def load_external_motion(path: str) -> ExternalMotion:
    ext=Path(path).suffix.lower()
    if ext==".bvh":return parse_bvh(path)
    if ext==".fbx":return inspect_fbx_skeleton(path)
    raise ValueError("Supported external animation formats: .bvh and binary .fbx")


def auto_map_bones(source_names: List[str], target_names: List[str]) -> Dict[str,str]:
    source_by_norm={_norm(x):x for x in source_names}
    result={}
    for target in target_names:
        preferred=MIXAMO_TO_BBR.get(target)
        if preferred and _norm(preferred) in source_by_norm:
            result[target]=source_by_norm[_norm(preferred)]
            continue
        choices=[target]+ALIASES.get(target,[])
        hit=None
        for c in choices:
            n=_norm(c)
            if n in source_by_norm:
                hit=source_by_norm[n]; break
        if hit is None:
            # Conservative fuzzy fallback: suffix match after namespace cleanup.
            for n,orig in source_by_norm.items():
                if any(n.endswith(_norm(c)) or _norm(c).endswith(n) for c in choices if len(_norm(c))>=4):
                    hit=orig; break
        if hit is not None:result[target]=hit
    return result


def mapping_report(motion: ExternalMotion, target_names: Optional[List[str]]=None,
                   mapping: Optional[Dict[str,str]]=None) -> str:
    target=list(target_names or BBR24)
    src=[j.name for j in motion.joints]
    mp=mapping if mapping is not None else auto_map_bones(src,target)
    lines=[
        f"External animation: {Path(motion.source).name}",
        f"Format: {motion.kind}",
        f"Source joints: {len(src)}",
        f"Frames: {motion.frame_count}",
        f"FPS: {motion.fps:.3f}" if motion.fps else "FPS: —",
        f"Channels: {motion.channel_count}",
        "",
        f"Mapped target bones: {len(mp)} / {len(target)}",
    ]
    if motion.fbx_clip is not None:
        lines.append("Setup pose at frame 0: skipped" if motion.fbx_clip.skipped_setup_frame
                     else "Starting frame: first keyed FBX pose")
        lines.append("Playback: FBX curves retargeted to the selected character")
    for t in target:
        lines.append(f"  {t:12s} <- {mp.get(t,'(unmapped)')}")
    return "\n".join(lines)
