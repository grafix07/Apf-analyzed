from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import random
import zlib
import numpy as np

from bbr50_assets import AssetDatabase




@dataclass
class AbilityPfxCue:
    ref: str
    role: str
    position: np.ndarray
    rotation: np.ndarray
    bone: str = ""
    mount: str = ""
    start_time: float = 0.0
    duration: float = 2.0
    velocity: np.ndarray = None
    source_key: str = ""

    def __post_init__(self):
        self.position=np.asarray(self.position if self.position is not None else (0,0,0),dtype=np.float32)
        self.rotation=np.asarray(self.rotation if self.rotation is not None else (0,0,0),dtype=np.float32)
        self.velocity=np.asarray(self.velocity if self.velocity is not None else (0,0,0),dtype=np.float32)

@dataclass
class BallSpawn:
    model_ref: str
    looping_pfx: str
    radius: float
    mass: float
    life_time: float
    position: np.ndarray
    velocity: np.ndarray
    source_index: int
    mode: str
    delay: float = 0.0


class EffectLibrary50:
    """VehicleEffectDB reader. Values always come from the loaded package."""
    def __init__(self, db: AssetDatabase):
        self.db = db
        obj = db.json("VuDBAsset/VehicleEffectDB")
        self.effects: Dict[str, Dict[str, Any]] = obj if isinstance(obj, dict) else {}

    def names(self) -> List[str]:
        return sorted(self.effects)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        x = self.effects.get(name)
        return x if isinstance(x, dict) else None

    @staticmethod
    def _round_positive(v: float) -> int:
        # Matches the native DropBalls setup: positive floats are rounded by
        # adding 0.5 and truncating.
        return max(0, int(float(v) + 0.5))

    def drop_ball_spawns(self, name: str, origin=(0.0, 0.0, 1.0), seed=0) -> List[BallSpawn]:
        """Build the native DropBalls launch plan from VehicleEffectDB.

        Reverse-engineering of VuVehicleDropBallsEffect::onApply/onTick gives us
        several details that older Studio builds guessed incorrectly:

        * drop/shoot counts are rounded once from BallCount and DropFactor;
          they are not selected independently with a per-ball probability;
        * speed values are converted from km/h-like tuning units with /3.6;
        * drop angles are centered behind the vehicle (+pi), shoot angles ahead;
        * balls are released progressively across the effect Duration rather
          than all appearing on the same frame.

        PhysX contact/bounce behaviour still belongs to the native runtime, but
        the launch plan itself is source-derived.
        """
        e = self.get(name)
        if not e or e.get("Type") != "VuVehicleDropBallsEffect":
            return []
        bd = e.get("BallData") if isinstance(e.get("BallData"), dict) else {}

        count = self._round_positive(float(e.get("BallCount") or 0.0))
        drop_factor = max(0.0, min(1.0, float(e.get("DropFactor") or 0.0)))
        drop_count = min(count, self._round_positive(count * drop_factor))
        shoot_count = max(0, count - drop_count)

        model_ref = str(bd.get("Model") or "")
        pfx = bd.get("LoopingPfx")
        if isinstance(pfx, dict):
            pfx = pfx.get("Name") or pfx.get("Mount") or ""
        pfx = str(pfx or "")
        radius = float(bd.get("Radius") or 0.0)
        mass = float(bd.get("Mass") or 0.0)
        life = float(bd.get("LifeTime") or e.get("Duration") or 5.0)
        duration = max(0.0, float(e.get("Duration") or 0.0))

        base_seed = zlib.crc32(str(name).encode("utf-8", "replace")) & 0xFFFFFFFF
        rng = random.Random(base_seed ^ (int(seed) & 0xFFFFFFFF))
        origin = np.asarray(origin, dtype=np.float32)
        plan: List[BallSpawn] = []

        def one(i: int, mode: str):
            if mode == "drop":
                speed = float(e.get("DropSpeed") or 0.0)
                vz = float(e.get("DropSpeedVert") or 0.0)
                spread = float(e.get("DropSpread") or 0.0)
                ang = math.radians(rng.uniform(-spread * 0.5, spread * 0.5)) + math.pi
            else:
                speed = float(e.get("ShootSpeed") or 0.0)
                vz = float(e.get("ShootSpeedVert") or 0.0)
                spread = float(e.get("ShootSpread") or 0.0)
                ang = math.radians(rng.uniform(-spread * 0.5, spread * 0.5))

            # Native code multiplies both horizontal and vertical launch values
            # by 0.2777778, i.e. km/h -> m/s.
            s = speed / 3.6
            z = vz / 3.6
            vel = np.asarray((math.sin(ang) * s, math.cos(ang) * s, z), dtype=np.float32)
            pos = origin.copy()
            pos[2] += max(0.0, radius)
            return BallSpawn(model_ref, pfx, radius, mass, life, pos, vel, i, mode, 0.0)

        for i in range(drop_count):
            plan.append(one(i, "drop"))
        for j in range(shoot_count):
            plan.append(one(drop_count + j, "shoot"))

        # Native onTick chooses a random remaining launch record whenever the
        # desired remaining count decreases. Shuffle once to reproduce that
        # unordered removal while keeping editor playback deterministic.
        rng.shuffle(plan)
        if plan and duration > 0.0:
            step = duration / float(len(plan))
            for i, b in enumerate(plan):
                # Native onTick computes int((1-elapsed/duration)*count).
                # At the first positive tick that is already below count, so
                # one ball is released immediately, then roughly every
                # duration/count seconds.
                b.delay = step * float(i)
        return plan

    @staticmethod
    def _vec3(value, default=(0.0,0.0,0.0)) -> np.ndarray:
        if isinstance(value, dict):
            return np.asarray((float(value.get("X",value.get("x",default[0])) or 0.0),
                               float(value.get("Y",value.get("y",default[1])) or 0.0),
                               float(value.get("Z",value.get("z",default[2])) or 0.0)),dtype=np.float32)
        if isinstance(value,(list,tuple)) and len(value)>=3:
            try:return np.asarray((float(value[0]),float(value[1]),float(value[2])),dtype=np.float32)
            except Exception:pass
        return np.asarray(default,dtype=np.float32)

    def pfx_cues(self, name: str) -> List[AbilityPfxCue]:
        """Extract source-authored PFX references from one VehicleEffectDB row.

        BBR effect types expose visuals through a family of fields such as
        StartPfx, LoopingPfx, EndPfx, SplatPfx and nested MissileData records.
        Older Studio builds only simulated VuVehicleDropBallsEffect and ignored
        all of these references. This reader keeps the preview source-driven:
        it discovers only PFX names/mounts that are actually present in the DB.
        """
        root=self.get(name)
        if not root:return []
        main_duration=max(0.0,float(root.get("Duration") or 0.0))
        cues: List[AbilityPfxCue]=[]
        seen=set()

        def classify(key: str):
            k=str(key or "").lower()
            if "endpfx" in k or "falloffpfx" in k:
                return "end"
            if "loopingpfx" in k or "looppfx" in k:
                return "loop"
            if "startpfx" in k or "successpfx" in k or "cast" in k:
                return "start"
            if any(x in k for x in ("splatpfx","hitpfx","reflectpfx","munchpfx","ragdollpfx")):
                return "burst"
            if "pfx" in k:
                return "burst"
            return ""

        def add_value(key, value, context, inherited_speed=0.0, inherited_life=0.0):
            role=classify(key)
            if not role:return
            values=value if isinstance(value,list) else [value]
            for item in values:
                ref=""; mount=""; bone=""; pos=np.zeros(3,dtype=np.float32); rot=np.zeros(3,dtype=np.float32)
                if isinstance(item,str):
                    ref=item.strip()
                elif isinstance(item,dict):
                    ref=str(item.get("Name") or item.get("Pfx") or "").strip()
                    mount=str(item.get("Mount") or "").strip()
                    bone=str(item.get("Bone") or "").strip()
                    pos=self._vec3(item.get("Pos") or item.get("Position"))
                    rot=self._vec3(item.get("Rot") or item.get("Rotation"))
                else:
                    continue
                if not ref and not mount:
                    continue
                item_role=role
                if str(key).lower()=="cockpitpfx":
                    rl=ref.lower()
                    item_role="start" if "transition" in rl else "loop"
                if item_role=="end":
                    start=max(0.0, main_duration if main_duration>0 else inherited_life)
                    duration=1.25
                elif item_role=="loop":
                    start=0.0
                    duration=max(1.5,min(8.0,main_duration or inherited_life or 3.0))
                else:
                    start=0.0
                    duration=1.35 if item_role=="start" else 1.0
                vel=np.zeros(3,dtype=np.float32)
                # Missile PFX is authored to travel along vehicle +Y. Keep the
                # motion modest in the editor so the effect remains inspectable.
                if inherited_speed>0.0 and "missile" in str(context).lower():
                    vel[1]=min(35.0,float(inherited_speed)/3.6)
                sig=(ref,mount,bone,item_role,tuple(float(x) for x in pos),tuple(float(x) for x in rot),round(start,3))
                if sig in seen:continue
                seen.add(sig)
                cues.append(AbilityPfxCue(ref,item_role,pos,rot,bone,mount,start,duration,vel,str(context)+"/"+str(key)))

        def walk(obj, context="Effect", inherited_speed=0.0, inherited_life=0.0):
            if not isinstance(obj,dict):return
            speed=float(obj.get("Speed") or inherited_speed or 0.0)
            life=float(obj.get("LifeTime") or obj.get("Duration") or inherited_life or 0.0)
            for key,value in obj.items():
                if "pfx" in str(key).lower():
                    add_value(key,value,context,speed,life)
                if isinstance(value,dict):
                    walk(value,str(context)+"/"+str(key),speed,life)
                elif isinstance(value,list):
                    for i,x in enumerate(value):
                        if isinstance(x,dict):walk(x,f"{context}/{key}[{i}]",speed,life)
        walk(root,name)
        return cues

