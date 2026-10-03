from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re
import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord
from vector_formats import ModelData, AnimationData


@dataclass
class CharacterSpec:
    name: str
    skin: str
    model_ref: str
    model_record: AssetRecord
    model: ModelData
    source: str
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    animations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def report(self) -> str:
        mn, mx = self.model.bounds()
        lines = [
            f"Character: {self.name}",
            f"Skin/model: {self.skin}",
            f"Asset: {self.model_record.path}",
            f"Source archive: {self.source}",
            f"Bones: {len(self.model.bones)}",
            f"Meshes: {len(self.model.meshes)}",
            f"Bounds min: {mn.tolist()}",
            f"Bounds max: {mx.tolist()}",
            "",
            "Mesh vertex layout:",
        ]
        for m in self.model.meshes:
            lines.append(
                f"  {m.name}: vertices={len(m.positions)} tris={len(m.indices)} "
                f"stride={m.vertex_stride} uv@{m.uv_offset} skin@{m.skin_offset}"
            )
        lines += ["", "Normal source: geometry-rebuilt when packed frame is unreliable."]
        if self.warnings:
            lines += ["", "Warnings:"] + ["  " + x for x in self.warnings]
        return "\n".join(lines)


class CharacterResolver50:
    def __init__(self, db: AssetDatabase):
        self.db = db
        self.driver_rows = db.spreadsheet_by_name("Drivers")
        self._models: Dict[str, List[str]] = {}
        prefix = "VuAnimatedModelAsset/Character/"
        for rec in db.all_records():
            if not rec.path.startswith(prefix):
                continue
            tail = rec.path[len(prefix):]
            if tail.startswith("Animations/") or "/Animations/" in tail:
                continue
            parts = tail.split("/")
            if len(parts) != 2:
                continue
            char, model_name = parts
            self._models.setdefault(char, []).append(model_name)
        for v in self._models.values():
            v.sort(key=lambda s: ("_Skin" in s, s.lower()))

    def characters(self) -> List[str]:
        # Spreadsheet order is authored UI order; append any hidden/model-only
        # characters afterward.
        out = [n for n in self.driver_rows if n in self._models]
        out += [n for n in sorted(self._models) if n not in out]
        return out

    def skins(self, character: str) -> List[str]:
        choices = list(self._models.get(character, ()))
        # BeachBro's BeachBall / BeachBall_SkinA assets are synchronized
        # cinematic companion rigs, not selectable character skins. Showing
        # them in the Skin combo made the ball replace BeachBro entirely.
        if str(character or "") == "BeachBro":
            choices = [x for x in choices if not str(x).startswith("BeachBall")]
        return choices

    def resolve(self, character: str, skin: Optional[str] = None) -> CharacterSpec:
        choices = self.skins(character)
        if not choices:
            raise KeyError(f"No character model for {character}")
        if not skin or skin not in choices:
            # Prefer exact base model name.
            skin = character if character in choices else choices[0]
        ref = f"Character/{character}/{skin}"
        rec = self.db.resolve_model(ref)
        if not rec:
            raise KeyError(f"Model not found: {ref}")
        model = self.db.model(rec)
        if not model:
            raise ValueError(f"Unable to decode model: {rec.path}")
        spec = CharacterSpec(character, skin, ref, rec, model, rec.source_name)
        spec.animations = self.compatible_animations(model, character)
        return spec

    @staticmethod
    def _animation_strings(value, key_hint=""):
        out=[]
        if isinstance(value, dict):
            for k,v in value.items():
                out.extend(CharacterResolver50._animation_strings(v, str(k)))
        elif isinstance(value, (list, tuple)):
            for v in value:
                out.extend(CharacterResolver50._animation_strings(v, key_hint))
        elif isinstance(value, str) and "anim" in key_hint.lower():
            out.append(value)
        return out

    def compatible_animations(self, model: ModelData, character: str) -> List[str]:
        """Return only animations authored for the selected character.

        v0.52 filtered only by bone count, so every unrelated character clip
        with the same skeleton size appeared in the dropdown.  Character UI
        animation folders and animation references in that character's ability
        effect are now the authority.
        """
        out=[]; seen=set(); char=str(character or "").lower()
        prefix="VuAnimationAsset/Character/Animations/"
        for rec in self.db.all_records():
            if not rec.path.startswith(prefix):
                continue
            rel=rec.path[len(prefix):]
            rl=rel.lower()
            # Character-specific clips plus the shared driving tree.  All BBR2
            # playable drivers use the same 24-bone runtime driving controls,
            # so hiding Character/Animations/Driving made it impossible to
            # inspect the seated/steering poses from the Characters tab.
            authored=(
                f"lounge_{char}/" in rl
                or f"portraits/{char}" in rl
                or rl.startswith(char+"/")
                or rl.startswith("driving/")
            )
            if not authored:
                continue
            try:
                anim=self.db.animation(rec)
                if anim is not None and anim.bone_count==len(model.bones):
                    logical=rec.path[len("VuAnimationAsset/"):]
                    if logical not in seen: seen.add(logical); out.append(logical)
            except Exception:
                # Exact VuAnimationAsset validation intentionally rejects files
                # whose size does not match 8 + bones*frames*32 + loopFlag.
                pass

        # Character front-end templates are the authoritative source for
        # aliased lounge animation sets.  Two important examples in this
        # package are MonsterHat -> Lounge_Mikka and Roller -> Lounge_Roxy.
        # The model names do not reveal those aliases, but the matching
        # VuTemplateAsset/CharacterAnimations/* timelines explicitly reference
        # the correct clips.  Include those authored references when their
        # bone count matches the selected model instead of hard-coding aliases.
        template_prefix="VuTemplateAsset/CharacterAnimations/"
        needles=[str(character or "").lower()]
        for pth in self.db.records_by_path:
            if not pth.startswith(template_prefix):
                continue
            leaf=pth[len(template_prefix):].lower()
            if not any(n and n in leaf for n in needles):
                continue
            try:
                obj=self.db.json(pth)
            except Exception:
                obj=None
            if obj is None:
                continue
            for logical in self._animation_strings(obj):
                rec=self.db.resolve_animation(logical)
                if not rec:
                    continue
                try:
                    anim=self.db.animation(rec)
                    if anim is not None and anim.bone_count==len(model.bones):
                        norm=rec.path[len("VuAnimationAsset/"):] if rec.path.startswith("VuAnimationAsset/") else logical
                        if norm not in seen:
                            seen.add(norm); out.append(norm)
                except Exception:
                    pass

        # Character ability records sometimes point to Driver/<Character>/...
        # animations rather than the lounge/portrait tree. Include only those
        # explicit references belonging to the selected character.
        try:
            effect_db=self.db.json("VuDBAsset/VehicleEffectDB")
        except Exception:
            effect_db={}
        if isinstance(effect_db, dict):
            for effect_name in (character, character+"_SkinA"):
                effect=effect_db.get(effect_name)
                if not isinstance(effect, dict):
                    continue
                for logical in self._animation_strings(effect):
                    rec=self.db.resolve_animation(logical)
                    if not rec:
                        continue
                    try:
                        anim=self.db.animation(rec)
                        if anim is not None and anim.bone_count==len(model.bones):
                            norm=rec.path[len("VuAnimationAsset/"):] if rec.path.startswith("VuAnimationAsset/") else logical
                            if norm not in seen: seen.add(norm); out.append(norm)
                    except Exception:
                        pass
        return sorted(out)

    def authored_preview_fps(self, character: str, skin: str, logical_ref: str,
                             default_fps: float = 30.0) -> float:
        """Infer an authored preview sample rate from the matching lounge timeline.

        Most BBR2 clips are sampled at 30 Hz. Skeleton/IdleB is a concrete
        exception: the asset contains 640 keys but the authored non-looping
        timeline allocates only ~5.3 seconds, which corresponds to ~120 Hz.
        We only accept canonical 30/60/120 rates when the timeline ratio is
        within 3%, avoiding guesses from blend/hold windows.
        """
        ref = str(logical_ref or "").strip()
        if not ref:
            return float(default_fps)
        candidates = []
        prefix = "VuTemplateAsset/CharacterAnimations/"
        needles = [str(x or "").lower() for x in (skin, character) if x]
        for path in self.db.records_by_path:
            if not path.startswith(prefix):
                continue
            leaf = path[len(prefix):].lower()
            if not any(n in leaf for n in needles):
                continue
            try:
                obj = self.db.json(path)
            except Exception:
                obj = None
            if not isinstance(obj, dict):
                continue
            stack=[obj]; tl=None
            while stack and tl is None:
                x=stack.pop()
                if isinstance(x,dict):
                    data=x.get("data")
                    if isinstance(data,dict):
                        comps=data.get("Components") or {}
                        if isinstance(comps,dict) and isinstance(comps.get("VuTimelineComponent"),dict):
                            tl=comps.get("VuTimelineComponent"); break
                    stack.extend(x.values())
                elif isinstance(x,list):
                    stack.extend(x)
            if not tl:
                continue
            total=float(tl.get("Total Time") or 0.0)
            for layer in tl.get("Layers") or []:
                if not isinstance(layer,dict):
                    continue
                for track in layer.get("Tracks") or []:
                    if not isinstance(track,dict) or track.get("TrackType")!="VuCinematicAnimationTrack":
                        continue
                    if str(track.get("Animation") or "") != ref:
                        continue
                    keys=[k for k in (track.get("Keys") or []) if isinstance(k,dict)]
                    keys.sort(key=lambda k: float(k.get("Time") or 0.0))
                    start=None; looping=False
                    for k in keys:
                        kt=str(k.get("KeyType") or "")
                        t=float(k.get("Time") or 0.0)
                        if kt=="VuCinematicAnimationStartKey":
                            start=t; looping=bool(k.get("Looping",False))
                        elif kt=="VuCinematicAnimationStopKey" and start is not None:
                            if not looping and t > start + 0.1:
                                candidates.append(t-start)
                            start=None
                    if start is not None and not looping and total > start + 0.1:
                        candidates.append(total-start)
        rec=self.db.resolve_animation(ref)
        anim=None
        try:
            anim=self.db.animation(rec) if rec else None
        except Exception:
            anim=None
        if anim is None or anim.frame_count <= 1:
            return float(default_fps)
        for span in sorted(candidates):
            rate=(float(anim.frame_count)-1.0)/max(1e-6,float(span))
            for canonical in (30.0,60.0,120.0):
                if abs(rate-canonical)/canonical <= 0.03:
                    return canonical
        return float(default_fps)

    def animation(self, logical_ref: str) -> Optional[AnimationData]:
        if not logical_ref:
            return None
        rec = self.db.resolve_animation(logical_ref)
        if not rec:
            return None
        try:
            return self.db.animation(rec)
        except Exception:
            return None
