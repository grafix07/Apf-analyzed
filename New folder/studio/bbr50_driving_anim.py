from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import struct

from bbr50_assets import AssetDatabase
from vector_formats import AnimationData, parse_animation_embedded


@dataclass
class DrivingAnimInfo:
    set_name: str
    set_asset: str = ""
    config_template: str = ""
    idle_animation: str = ""
    idle_looping: bool = True
    idle_pose_frame: Optional[float] = None
    steering_animation: str = ""
    source: str = ""
    # BBR's VuDrivingAnimationSetAsset embeds the runtime controls themselves.
    # The relevant controls are named Turn and, for motorcycles, Lean.
    controls: Dict[str, AnimationData] = field(default_factory=dict)
    header_values: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def turn(self) -> Optional[AnimationData]:
        return self.controls.get("Turn")

    @property
    def lean(self) -> Optional[AnimationData]:
        return self.controls.get("Lean")


class DrivingAnimationResolver50:
    """Read the actual runtime driving-animation set.

    Reverse-engineered VuVehicle::update shows that BBR attaches animation
    controls named ``Turn`` and, on motorcycles, ``Lean``.  Both controls are
    initialized at half duration.  At runtime Turn time is driven by normalized
    steering.  Motorcycle Lean time is driven by steering * speedRatio where
    speedRatio = clamp(speedKPH / 50, 0, 1), and the Turn/Lean weights are
    (1-speedRatio, speedRatio).  During powerslide the game blends the weights
    toward Turn=1, Lean=0 at 4 units/second.

    v0.50 ignored the embedded Lean control and replaced the entire pose with an
    external TurnLR clip, which explains the unnatural motorcycle/drift motion.
    """

    def __init__(self, db: AssetDatabase):
        self.db = db
        self._cache: Dict[str, DrivingAnimInfo] = {}

    @staticmethod
    def _timeline_tracks(obj: Any):
        if isinstance(obj, dict):
            comp = obj.get("VuTimelineComponent")
            if isinstance(comp, dict):
                for layer in comp.get("Layers") or []:
                    if isinstance(layer, dict):
                        for track in layer.get("Tracks") or []:
                            if isinstance(track, dict):
                                yield track
            for v in obj.values():
                yield from DrivingAnimationResolver50._timeline_tracks(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from DrivingAnimationResolver50._timeline_tracks(v)

    @staticmethod
    def _parse_set_blob(data: bytes):
        if len(data) < 20:
            return (0.0,0.0,0.0), "", {}
        header=struct.unpack_from("<3f",data,0)
        off=12
        end=data.find(b"\0",off,min(len(data),2048))
        if end<0:
            return header,"",{}
        try: config=data[off:end].decode("utf-8")
        except Exception: config=""
        off=end+1
        if off+4>len(data):
            return header,config,{}
        count=struct.unpack_from("<I",data,off)[0]; off+=4
        controls={}
        # The bake stream places a NUL separator between animation blocks.
        for _ in range(min(int(count),64)):
            while off<len(data) and data[off]==0:
                off+=1
            if off>=len(data): break
            end=data.find(b"\0",off,min(len(data),off+256))
            if end<0: break
            raw_name=data[off:end]
            # A few builds prefix Burn with control byte 0x01.
            raw_name=raw_name.lstrip(b"\x01")
            try: name=raw_name.decode("utf-8")
            except Exception: name=""
            off=end+1
            if off+8>len(data): break
            nb,nf=struct.unpack_from("<II",data,off)
            if not (1<=nb<=512 and 1<=nf<=100000): break
            size=8+int(nb)*int(nf)*32
            if off+size>len(data): break
            blob=data[off:off+size]
            off+=size
            try:
                controls[name]=parse_animation_embedded(blob,name)
            except Exception:
                pass
        return header,config,controls

    def resolve(self, set_name: str) -> DrivingAnimInfo:
        set_name=str(set_name or "Standard")
        if set_name in self._cache:
            return self._cache[set_name]
        set_path="VuDrivingAnimationSetAsset/"+set_name
        info=DrivingAnimInfo(set_name=set_name,set_asset=set_path)
        rec=self.db.get(set_path)
        if not rec:
            info.source="driving set not found"
            self._cache[set_name]=info
            return info

        data=self.db.decode(rec).data
        header,config,controls=self._parse_set_blob(data)
        info.header_values=tuple(float(x) for x in header)
        info.controls=controls
        if config.startswith("CharacterAnimations/"):
            info.config_template="VuTemplateAsset/"+config
            obj=self.db.json(info.config_template)
            for track in self._timeline_tracks(obj):
                anim=track.get("Animation")
                if not isinstance(anim,str) or not anim:
                    continue
                name=str(track.get("Name") or "").lower()
                if not info.idle_animation and (name=="idle" or "idle" in name):
                    info.idle_animation=anim
                    keys=track.get("Keys") if isinstance(track.get("Keys"),list) else []
                    for k in keys:
                        if not isinstance(k,dict): continue
                        if k.get("Looping") is True:
                            info.idle_looping=True
                        if "Time Factor" in k and float(k.get("Time Factor") or 0.0)==0.0:
                            info.idle_looping=False
                            info.idle_pose_frame=float(k.get("Start Time") or 0.0)*30.0

        # Keep the logical external path for inspection/export.  Rendering uses
        # the embedded Turn control because that is what VuVehicle attaches.
        if set_name=="Standard":
            logical="Character/Animations/Driving/Standard/TurnLR"
            if self.db.resolve_animation(logical): info.steering_animation=logical
        elif set_name.startswith("Motorcycle_"):
            subtype=set_name[len("Motorcycle_"):]
            logical=f"Character/Animations/Driving/Motorcycle/{subtype}_TurnLR"
            if self.db.resolve_animation(logical): info.steering_animation=logical

        info.source=set_path+(f" -> {info.config_template}" if info.config_template else "")
        self._cache[set_name]=info
        return info
