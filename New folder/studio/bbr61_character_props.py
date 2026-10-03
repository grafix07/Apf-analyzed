from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bbr50_assets import AssetDatabase, AssetRecord
from vector_formats import ModelData


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _v3(v, default=(0.0, 0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(_f(v[i]) for i in range(3))
    return tuple(float(x) for x in default)


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
    s = str(name or "").replace("\\", "/").strip()
    while s.startswith("../"):
        s = s[3:]
    return s.rsplit("/", 1)[-1] if "/" in s else s


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


def _interp_vec(keys, t, default):
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


@dataclass
class CharacterPropTimelineSpec:
    entity_name: str
    model_ref: str
    model_record: AssetRecord
    model: ModelData
    source_template: str
    clip_start: float
    clip_stop: float
    character_actor_name: str
    initially_visible: bool = True
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    note_events: List[Tuple[float, str]] = field(default_factory=list)
    position_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    rotation_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    scale_keys: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)
    attachment_keys: List[Tuple[float, str, str, str, Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.clip_stop - self.clip_start)

    def visible_at(self, local_time: float) -> bool:
        t = self.clip_start + max(0.0, float(local_time))
        visible = bool(self.initially_visible)
        for et, action in self.note_events:
            if et > t + 1e-7:
                break
            a = str(action or "").strip().lower()
            if a in ("show", "start", "enable", "visible", "on"):
                visible = True
            elif a in ("hide", "stop", "kill", "disable", "invisible", "off"):
                visible = False
        return visible

    def transform_at(self, local_time: float):
        t = self.clip_start + max(0.0, float(local_time))
        return (
            _interp_vec(self.position_keys, t, self.base_position),
            _interp_vec(self.rotation_keys, t, self.base_rotation),
            _interp_vec(self.scale_keys, t, self.base_scale),
        )

    def attachment_at(self, local_time: float):
        t = self.clip_start + max(0.0, float(local_time))
        current = None
        for kt, kind, parent, bone, rel_pos, rel_rot in self.attachment_keys:
            if kt > t + 1e-7:
                break
            if kind == "attach":
                current = (parent, bone, rel_pos, rel_rot)
            elif kind == "detach":
                current = None
        return current


class CharacterPropTimelineResolver61:
    """Resolve character cinematic props from the authored lounge template.

    Several drivers are not a single model during their front-end animations.
    Skeleton's cutlass is a separate Character/Skeleton/Sword prop attached to
    R_Wrist, for example.  This resolver only accepts prop actors that actually
    participate in the selected character's animation occurrence, which avoids
    accidentally previewing garage scenery/NPCs from the same template.
    """

    def __init__(self, db: AssetDatabase):
        self.db = db
        self._cache: Dict[Tuple[str, str, str], List[CharacterPropTimelineSpec]] = {}

    def _template_candidates(self, character: str, skin: str, anim_ref: str) -> List[str]:
        out = []
        c = str(character or "").strip()
        s = str(skin or "").strip()
        for n in (s, c):
            if n:
                out.append("VuTemplateAsset/CharacterAnimations/Lounge_" + n)
        # Also allow authored Reward/Podium/Clothing templates for aliased
        # drivers such as Roller/Roxy and MonsterHat/Mikka.
        prefix = "VuTemplateAsset/CharacterAnimations/"
        needles = [x.lower() for x in (s, c) if x]
        for p in self.db.records_by_path:
            if p.startswith(prefix):
                leaf = p[len(prefix):].lower()
                if any(n in leaf for n in needles):
                    out.append(p)
        uniq = []
        for p in out:
            if p not in uniq and self.db.get(p):
                uniq.append(p)
        return uniq

    @staticmethod
    def _animation_occurrences(tl: Dict[str, Any], anim_ref: str):
        total = _f(tl.get("Total Time"), 0.0)
        out = []
        for layer in tl.get("Layers") or []:
            if not isinstance(layer, dict):
                continue
            actor = str(layer.get("Entity Name") or "")
            for track in layer.get("Tracks") or []:
                if not isinstance(track, dict) or track.get("TrackType") != "VuCinematicAnimationTrack":
                    continue
                if str(track.get("Animation") or "") != str(anim_ref or ""):
                    continue
                for start, stop in _clip_occurrences(track, total):
                    out.append((start, stop, actor))
        return out

    @staticmethod
    def _prop_entities(obj: Any):
        out = {}
        for e in _entity_nodes(obj):
            typ = str(e.get("type") or "")
            if typ not in ("VuPropStaticEntity", "VuPropAnimatedEntity"):
                continue
            data = e.get("data") or {}
            comps = data.get("Components") or {}
            draw = comps.get("Vu3dDrawStaticModelComponent") or comps.get("Vu3dDrawAnimatedModelComponent")
            if not isinstance(draw, dict):
                continue
            model_ref = str((draw.get("Properties") or {}).get("Model Asset") or "").strip()
            if model_ref and e.get("name"):
                out[str(e.get("name"))] = (e, model_ref)
        return out

    def resolve(self, character: str, skin: str, anim_ref: str) -> List[CharacterPropTimelineSpec]:
        key = (str(character or ""), str(skin or ""), str(anim_ref or ""))
        if key in self._cache:
            return self._cache[key]
        if not anim_ref:
            self._cache[key] = []
            return []

        best = None
        for p in self._template_candidates(character, skin, anim_ref):
            obj = self.db.json(p)
            tl = _timeline_component(obj)
            if not tl:
                continue
            occ = self._animation_occurrences(tl, anim_ref)
            if not occ:
                continue
            # Prefer the occurrence with the greatest number of prop-affine
            # attachment/note tracks inside its time span.
            score = 0
            for l in tl.get("Layers") or []:
                for tr in (l.get("Tracks") or []) if isinstance(l, dict) else []:
                    if isinstance(tr, dict) and tr.get("TrackType") in ("VuCinematicAttachmentTrack", "VuTimelineNoteTrack"):
                        score += 1
            cand = (score, p, obj, tl, occ[0])
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            self._cache[key] = []
            return []

        _, template_path, obj, tl, occ = best
        clip_start, clip_stop, character_actor = occ
        props = self._prop_entities(obj)
        layers: Dict[str, List[Dict[str, Any]]] = {}
        for layer in tl.get("Layers") or []:
            if isinstance(layer, dict):
                layers.setdefault(_entity_leaf(layer.get("Entity Name")), []).append(layer)

        result: List[CharacterPropTimelineSpec] = []
        for entity_name, (ent, model_ref) in props.items():
            # Cyber's Screen/Keyboard/Spinner actors are already evaluated as
            # a synchronized 15-bone companion rig by the viewport.  Feeding
            # the same VuPropAnimatedEntity through this generic prop path drew
            # a second *unanimated bind-pose* screen on top of Cyber (the large
            # duplicate panels seen in v0.62/v0.64).  Static LockOpen/LockClose
            # props are deliberately retained below.
            if str(character or "") == "Cyber" and str(model_ref).startswith("Character/Animations/Lounge_Cyber/Screen"):
                continue
            prop_layers = layers.get(_entity_leaf(entity_name), [])
            if not prop_layers:
                continue
            data = ent.get("data") or {}
            comps = data.get("Components") or {}
            script_props = ((comps.get("VuScriptComponent") or {}).get("Properties") or {})
            entity_props = data.get("Properties") if isinstance(data.get("Properties"), dict) else {}
            tc = ((comps.get("VuTransformComponent") or {}).get("Properties") or {})

            note_events = []
            pos_keys = []
            rot_keys = []
            scale_keys = []
            att_keys = []
            attached_to_character = False
            event_in_clip = False
            for layer in prop_layers:
                for tr in layer.get("Tracks") or []:
                    if not isinstance(tr, dict):
                        continue
                    tt = str(tr.get("TrackType") or "")
                    for k in [x for x in (tr.get("Keys") or []) if isinstance(x, dict)]:
                        kt = str(k.get("KeyType") or "")
                        t = _key_time(k)
                        if tt == "VuTimelineNoteTrack" and kt == "VuCinematicActorPlug":
                            note_events.append((t, str(k.get("Plug Name") or "")))
                            if clip_start - 1e-6 <= t <= clip_stop + 1e-6:
                                event_in_clip = True
                        elif tt == "VuTimelinePositionTrack" and "Position" in k:
                            pos_keys.append((t, _v3(k.get("Position"))))
                        elif tt == "VuTimelineRotationTrack" and "Rotation" in k:
                            rot_keys.append((t, _v3(k.get("Rotation"))))
                        elif tt == "VuTimelineScaleTrack" and "Scale" in k:
                            scale_keys.append((t, _v3(k.get("Scale"), (1, 1, 1))))
                        elif tt == "VuCinematicAttachmentTrack":
                            if kt == "VuCinematicAttachKey":
                                parent = str(k.get("Parent") or "")
                                bone = str(k.get("Bone Name") or "")
                                att_keys.append((t, "attach", parent, bone, _v3(k.get("Relative Position")), _v3(k.get("Relative Rotation"))))
                                if _entity_leaf(parent) == _entity_leaf(character_actor):
                                    attached_to_character = True
                            elif kt == "VuCinematicDetachKey":
                                att_keys.append((t, "detach", "", "", (0, 0, 0), (0, 0, 0)))
            # Only props tied to this character occurrence are safe to preview.
            if not (attached_to_character or event_in_clip):
                continue
            rec = self.db.resolve_model(model_ref)
            if not rec:
                continue
            try:
                model = self.db.model(rec)
            except Exception:
                model = None
            if model is None:
                continue
            spec = CharacterPropTimelineSpec(
                entity_name=entity_name,
                model_ref=model_ref,
                model_record=rec,
                model=model,
                source_template=template_path,
                clip_start=float(clip_start),
                clip_stop=float(clip_stop),
                character_actor_name=str(character_actor or ""),
                initially_visible=bool(entity_props.get("Initially Visible", script_props.get("Visible", True))),
                base_position=_v3(tc.get("Position")),
                base_rotation=_v3(tc.get("Rotation")),
                base_scale=_v3(tc.get("Scale"), (1, 1, 1)),
                note_events=sorted(note_events),
                position_keys=sorted(pos_keys),
                rotation_keys=sorted(rot_keys),
                scale_keys=sorted(scale_keys),
                attachment_keys=sorted(att_keys),
            )
            result.append(spec)

        self._cache[key] = result
        return result
