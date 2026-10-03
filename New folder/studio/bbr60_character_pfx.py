from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from bbr50_assets import AssetDatabase


def _f(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _v3(v, default=(0.0, 0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(_f(v[i]) for i in range(3))
    return tuple(float(x) for x in default)


def _color(v):
    # Timeline serializers use both RGBA arrays and named-channel objects.
    # Scalar values on a Pfx Color track are interpreted as alpha.
    if isinstance(v, dict):
        keys = {str(k).lower(): val for k, val in v.items()}
        if any(k in keys for k in ("r", "g", "b", "a", "red", "green", "blue", "alpha")):
            return (
                _f(keys.get("r", keys.get("red", 1.0)), 1.0),
                _f(keys.get("g", keys.get("green", 1.0)), 1.0),
                _f(keys.get("b", keys.get("blue", 1.0)), 1.0),
                _f(keys.get("a", keys.get("alpha", 1.0)), 1.0),
            )
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        return tuple(_f(v[i], 1.0) for i in range(4))
    if isinstance(v, (int, float)):
        return (1.0, 1.0, 1.0, _f(v, 1.0))
    return (1.0, 1.0, 1.0, 1.0)


def _entity_nodes(x: Any):
    if isinstance(x, dict):
        if isinstance(x.get("data"), dict) and (x.get("type") or x.get("name")):
            yield x
        for v in x.values():
            yield from _entity_nodes(v)
    elif isinstance(x, list):
        for v in x:
            yield from _entity_nodes(v)


def _timeline_component(obj: Any) -> Optional[Dict[str, Any]]:
    for e in _entity_nodes(obj):
        comps = (e.get("data") or {}).get("Components") or {}
        tl = comps.get("VuTimelineComponent")
        if isinstance(tl, dict):
            return tl
    return None




def _entity_leaf(name: str) -> str:
    s=str(name or '').replace('\\','/').strip()
    while s.startswith('../'):
        s=s[3:]
    return s.rsplit('/',1)[-1] if '/' in s else s

def _key_time(k: Dict[str, Any]) -> float:
    return _f(k.get("Time"), 0.0)


def _clip_occurrences(track: Dict[str, Any], total_time: float) -> List[Tuple[float, float]]:
    keys = [k for k in (track.get("Keys") or []) if isinstance(k, dict)]
    keys.sort(key=_key_time)
    out: List[Tuple[float, float]] = []
    start = None
    for k in keys:
        kt = str(k.get("KeyType") or "")
        t = _key_time(k)
        if kt == "VuCinematicAnimationStartKey":
            if start is not None:
                out.append((start, max(start, t)))
            start = t
        elif kt == "VuCinematicAnimationStopKey" and start is not None:
            out.append((start, max(start, t)))
            start = None
    if start is not None:
        out.append((start, max(start, total_time)))
    return out


def _interp_vec(keys: List[Tuple[float, Tuple[float, float, float]]], t: float, default):
    if not keys:
        return tuple(default)
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(1, len(keys)):
        t1, v1 = keys[i]
        if t <= t1:
            t0, v0 = keys[i - 1]
            if t1 <= t0 + 1e-8:
                return v1
            a = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            return tuple(float(v0[j] + (v1[j] - v0[j]) * a) for j in range(3))
    return keys[-1][1]


def _interp_color(keys: List[Tuple[float, Tuple[float, float, float, float]]], t: float, default):
    if not keys:
        return tuple(default)
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(1, len(keys)):
        t1, v1 = keys[i]
        if t <= t1:
            t0, v0 = keys[i - 1]
            if t1 <= t0 + 1e-8:
                return v1
            a = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            return tuple(float(v0[j] + (v1[j] - v0[j]) * a) for j in range(4))
    return keys[-1][1]


@dataclass
class CharacterPfxTimelineSpec:
    entity_name: str
    effect_ref: str
    source_template: str
    clip_start: float
    clip_stop: float
    initially_active: bool = False
    pfx_scale: float = 1.0
    pfx_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Absolute template time + action (Start/Stop/Kill).
    note_events: List[Tuple[float, str]] = field(default_factory=list)
    position_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    rotation_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    scale_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    color_keys: List[Tuple[float, Tuple[float, float, float, float]]] = field(default_factory=list)
    # Attachments are only applied when Parent is the selected character actor.
    attachment_keys: List[Tuple[float, str, str, Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=list)
    character_actor_name: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, float(self.clip_stop - self.clip_start))

    def state_before(self, absolute_time: float) -> bool:
        active = bool(self.initially_active)
        for t, action in self.note_events:
            if t > absolute_time + 1e-7:
                break
            a = action.lower()
            if a == "start":
                active = True
            elif a in ("stop", "kill"):
                active = False
        return active

    def transform_at(self, local_time: float):
        t = float(self.clip_start + max(0.0, local_time))
        pos = _interp_vec(self.position_keys, t, self.base_position)
        rot = _interp_vec(self.rotation_keys, t, self.base_rotation)
        scl = _interp_vec(self.scale_keys, t, self.base_scale)
        return pos, rot, tuple(float(x) * float(self.pfx_scale) for x in scl)

    def color_at(self, local_time: float):
        t = float(self.clip_start + max(0.0, local_time))
        return _interp_color(self.color_keys, t, self.pfx_color)

    def attachment_binding_at(self, local_time: float):
        """Return the currently authored attachment binding.

        The lounge timelines can attach PFX not only to the selected character
        actor, but also to companion prop rigs.  Cyber's screen-frame PFX is a
        concrete example: it attaches to ``PropAnimated_Screen`` / keyboard /
        spinner bones.  v0.61 ignored those parents and fell back to the PFX
        entity's base transform, which produced the green block below Cyber.
        """
        if not self.attachment_keys:
            return None
        t = float(self.clip_start + max(0.0, local_time))
        current = None
        for kt, kind, parent, bone, rel_pos, rel_rot in self.attachment_keys:
            if kt > t + 1e-7:
                break
            if kind == "attach":
                current = (str(parent or ""), str(bone or ""), rel_pos, rel_rot)
            elif kind == "detach":
                current = None
        return current

    def attachment_at(self, local_time: float):
        """Backward-compatible character-only attachment view."""
        binding = self.attachment_binding_at(local_time)
        if not binding or not self.character_actor_name:
            return None
        parent, bone, rel_pos, rel_rot = binding
        if parent == self.character_actor_name:
            return (bone, rel_pos, rel_rot)
        return None


class CharacterPfxTimelineResolver60:
    """Resolve character lounge PFX from authored VuTemplateAsset timelines.

    The game does not store character animation smoke/sparks inside the
    VuAnimationAsset itself. They live as VuStaticPfxEntity actors in the
    matching CharacterAnimations/Lounge_* template, and VuTimelineNoteTrack
    Start/Stop/Kill plugs control their lifetime. This resolver keeps those
    authored timings attached to a single animation clip preview.
    """

    def __init__(self, db: AssetDatabase):
        self.db = db
        self._cache: Dict[Tuple[str, str, str], List[CharacterPfxTimelineSpec]] = {}

    def _template_candidates(self, character: str, skin: str, anim_ref: str) -> List[str]:
        """Return character-affine cinematic templates for an animation.

        Most lounge clips live in Lounge_<character>, but several authored
        clips with PFX (Cyber Fidget_D, Pumpkin head effects, podium/standing
        clips, etc.) only appear in Reward_/Podium_/Config_ templates.  v0.60
        therefore starts with the obvious lounge templates and then searches
        all CharacterAnimations templates whose name belongs to this character
        or selected skin.  The exact animation reference is still required
        later, so an unrelated template cannot inject particles merely because
        it contains a similarly named actor.
        """
        out: List[str] = []
        skin = str(skin or "").strip()
        character = str(character or "").strip()
        if skin:
            out.append("VuTemplateAsset/CharacterAnimations/Lounge_" + skin)
        if character:
            out.append("VuTemplateAsset/CharacterAnimations/Lounge_" + character)

        p = str(anim_ref or "").replace("\\", "/")
        marker = "Character/Animations/Lounge_"
        if p.startswith(marker):
            folder = p[len("Character/Animations/"):].split("/", 1)[0]
            out.append("VuTemplateAsset/CharacterAnimations/" + folder)

        # Broader authored fallback, still restricted to templates whose path
        # explicitly names the selected skin/character.  This is what exposes
        # Cyber Fidget_D PFX from Podium_Cyber/Reward_Cyber without borrowing
        # another driver's standing effects.
        prefix = "VuTemplateAsset/CharacterAnimations/"
        needles = [x.lower() for x in (skin, character) if x]
        for path in self.db.records_by_path:
            if not path.startswith(prefix):
                continue
            leaf = path[len(prefix):].lower()
            if any(n in leaf for n in needles):
                out.append(path)

        uniq = []
        for x in out:
            if x not in uniq and self.db.get(x):
                uniq.append(x)
        return uniq

    @staticmethod
    def _pfx_entities(obj: Any) -> Dict[str, Dict[str, Any]]:
        out = {}
        for e in _entity_nodes(obj):
            if e.get("type") == "VuStaticPfxEntity" and e.get("name"):
                out[str(e.get("name"))] = e
        return out

    @staticmethod
    def _animation_occurrences(tl: Dict[str, Any], anim_ref: str):
        total = _f(tl.get("Total Time"), 0.0)
        out = []
        for layer in tl.get("Layers") or []:
            if not isinstance(layer, dict):
                continue
            actor = str(layer.get("Entity Name") or "")
            for track in layer.get("Tracks") or []:
                if not isinstance(track, dict):
                    continue
                if str(track.get("TrackType") or "") != "VuCinematicAnimationTrack":
                    continue
                if str(track.get("Animation") or "") != str(anim_ref or ""):
                    continue
                for start, stop in _clip_occurrences(track, total):
                    out.append((start, stop, actor, layer))
        return out

    @staticmethod
    def _event_score(tl: Dict[str, Any], pfx_names: set[str], start: float, stop: float) -> int:
        score = 0
        for layer in tl.get("Layers") or []:
            if not isinstance(layer, dict) or _entity_leaf(str(layer.get("Entity Name") or "")) not in pfx_names:
                continue
            for track in layer.get("Tracks") or []:
                if not isinstance(track, dict) or str(track.get("TrackType") or "") != "VuTimelineNoteTrack":
                    continue
                for k in track.get("Keys") or []:
                    if not isinstance(k, dict) or k.get("KeyType") != "VuCinematicActorPlug":
                        continue
                    t = _key_time(k)
                    if start - 1e-6 <= t <= stop + 1e-6:
                        score += 1
        return score

    def resolve(self, character: str, skin: str, anim_ref: str) -> List[CharacterPfxTimelineSpec]:
        key = (str(character or ""), str(skin or ""), str(anim_ref or ""))
        if key in self._cache:
            return self._cache[key]
        if not anim_ref:
            self._cache[key] = []
            return []

        best = None
        skin_l=str(skin or "").lower(); char_l=str(character or "").lower()
        for template_path in self._template_candidates(character, skin, anim_ref):
            obj = self.db.json(template_path)
            tl = _timeline_component(obj)
            pfx = self._pfx_entities(obj)
            if not tl or not pfx:
                continue
            occ = self._animation_occurrences(tl, anim_ref)
            if not occ:
                continue
            pfx_names=set(pfx)
            ranked = sorted(
                occ,
                key=lambda x: (self._event_score(tl, pfx_names, x[0], x[1]), x[1] - x[0]),
                reverse=True,
            )
            chosen=ranked[0]
            event_score=self._event_score(tl,pfx_names,chosen[0],chosen[1])
            if event_score <= 0:
                continue
            leaf=template_path.rsplit('/',1)[-1].lower()
            exact_skin=("lounge_"+skin_l) if skin_l else ""
            exact_char=("lounge_"+char_l) if char_l else ""
            # Base skins commonly have skin==character (Cyber/Cyber).  The old
            # substring score could then prefer Lounge_Cyber_SkinA because it
            # also contains "cyber" and happens to have more events.  That
            # shifted the whole cinematic by ~0.53 s and applied SkinA PFX to
            # base Cyber.  Exact authored lounge template identity wins.
            if skin_l and skin_l != char_l and leaf == exact_skin:
                affinity=20
            elif char_l and leaf == exact_char:
                affinity=18
            elif skin_l and skin_l != char_l and skin_l in leaf:
                affinity=8
            elif char_l and char_l in leaf:
                affinity=6
            else:
                affinity=0
            rank=(affinity,event_score,chosen[1]-chosen[0])
            candidate=(chosen,template_path,obj,tl,pfx)
            if best is None or rank > best[0]:
                best=(rank,candidate)

        if best is None:
            self._cache[key] = []
            return []

        (_, (occ, template_path, obj, tl, pfx_entities)) = best
        clip_start, clip_stop, character_actor, _ = occ
        specs: List[CharacterPfxTimelineSpec] = []

        layers_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        for layer in tl.get("Layers") or []:
            if isinstance(layer, dict):
                layers_by_entity.setdefault(_entity_leaf(str(layer.get("Entity Name") or "")), []).append(layer)

        for entity_name, ent in pfx_entities.items():
            data = ent.get("data") or {}
            props = data.get("Properties") or {}
            effect_ref = str(props.get("Effect Name") or "").strip()
            if not effect_ref:
                continue
            tc = ((data.get("Components") or {}).get("VuTransformComponent") or {}).get("Properties") or {}
            spec = CharacterPfxTimelineSpec(
                entity_name=entity_name,
                effect_ref=effect_ref,
                source_template=template_path,
                clip_start=float(clip_start), clip_stop=float(clip_stop),
                initially_active=bool(props.get("Initially Active", False)),
                pfx_scale=max(0.0, _f(props.get("Pfx Scale"), 1.0)),
                pfx_color=_color(props.get("Pfx Color")),
                base_position=_v3(tc.get("Position")),
                base_rotation=_v3(tc.get("Rotation")),
                base_scale=_v3(tc.get("Scale"), (1.0, 1.0, 1.0)),
                character_actor_name=str(character_actor or ""),
            )
            for layer in layers_by_entity.get(entity_name, []):
                for track in layer.get("Tracks") or []:
                    if not isinstance(track, dict):
                        continue
                    tt = str(track.get("TrackType") or "")
                    keys = [k for k in (track.get("Keys") or []) if isinstance(k, dict)]
                    if tt == "VuTimelineNoteTrack":
                        for k in keys:
                            if k.get("KeyType") == "VuCinematicActorPlug":
                                spec.note_events.append((_key_time(k), str(k.get("Plug Name") or "")))
                    elif tt == "VuTimelinePositionTrack":
                        for k in keys:
                            if "Position" in k:
                                spec.position_keys.append((_key_time(k), _v3(k.get("Position"))))
                    elif tt == "VuTimelineRotationTrack":
                        for k in keys:
                            if "Rotation" in k:
                                spec.rotation_keys.append((_key_time(k), _v3(k.get("Rotation"))))
                    elif tt == "VuTimelineScaleTrack":
                        for k in keys:
                            if "Scale" in k:
                                spec.scale_keys.append((_key_time(k), _v3(k.get("Scale"), (1,1,1))))
                    elif tt == "VuCinematicActorColorPropertyTrack" and str(track.get("Property Name") or "") == "Pfx Color":
                        for k in keys:
                            # In these BBR2 timelines an omitted serialized
                            # Value is the default opaque white key.  This is
                            # used heavily by Cyber to fade ScreenFrame alpha
                            # 0 -> 1 -> 0; v0.64 ignored the track completely.
                            spec.color_keys.append((_key_time(k), _color(k.get("Value") if "Value" in k else (1,1,1,1))))
                    elif tt == "VuCinematicAttachmentTrack":
                        for k in keys:
                            kt = str(k.get("KeyType") or "")
                            if kt == "VuCinematicAttachKey":
                                spec.attachment_keys.append((
                                    _key_time(k), "attach", str(k.get("Parent") or ""), str(k.get("Bone Name") or ""),
                                    _v3(k.get("Relative Position")), _v3(k.get("Relative Rotation")),
                                ))
                            elif kt == "VuCinematicDetachKey":
                                spec.attachment_keys.append((_key_time(k), "detach", "", "", (0,0,0), (0,0,0)))
            spec.note_events.sort(key=lambda x: x[0])
            spec.position_keys.sort(key=lambda x: x[0])
            spec.rotation_keys.sort(key=lambda x: x[0])
            spec.scale_keys.sort(key=lambda x: x[0])
            spec.color_keys.sort(key=lambda x: x[0])
            spec.attachment_keys.sort(key=lambda x: x[0])

            active_at_start = spec.state_before(clip_start - 1e-6)
            has_event = any(clip_start - 1e-6 <= t <= clip_stop + 1e-6 for t, _ in spec.note_events)
            # Keep only effects which actually participate in this occurrence.
            if active_at_start or has_event:
                specs.append(spec)

        self._cache[key] = specs
        return specs
