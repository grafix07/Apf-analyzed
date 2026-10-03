from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord


def _dict(x):
    return x if isinstance(x, dict) else {}


def child_items(value: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize Vector ChildEntities list/dict encodings.

    BBR2 templates use both forms.  Dict children are often override fragments
    and omit explicit type/name because the dictionary key identifies the child.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(value, list):
        for i, c in enumerate(value):
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or f"@{i}")
            out.append((name, c))
    elif isinstance(value, dict):
        for k, c in value.items():
            if not isinstance(c, dict):
                continue
            cc = copy.deepcopy(c)
            cc.setdefault("name", str(k))
            out.append((str(k), cc))
    return out


def _merge_child_entities(base: Any, override: Any) -> List[Dict[str, Any]]:
    base_items = child_items(base)
    override_items = child_items(override)
    result: List[Dict[str, Any]] = [copy.deepcopy(c) for _, c in base_items]
    index: Dict[str, int] = {}
    for i, c in enumerate(result):
        nm = str(c.get("name") or "")
        if nm:
            index[nm] = i

    for key, ov in override_items:
        target_name = str(ov.get("name") or key)
        if target_name in index:
            i = index[target_name]
            result[i] = deep_merge_entity(result[i], ov)
        elif key in index:
            i = index[key]
            result[i] = deep_merge_entity(result[i], ov)
        else:
            cc = copy.deepcopy(ov)
            cc.setdefault("name", target_name)
            index[target_name] = len(result)
            result.append(cc)
    return result


def deep_merge_entity(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a template entity with an authored instance/override entity."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "data":
            out["data"] = deep_merge_data(_dict(out.get("data")), _dict(value))
        elif key in ("Properties", "Components") and isinstance(value, dict):
            # Template override fragments stored inside a ChildEntities dict put
            # Properties/Components directly on the fragment, while the full
            # serialized entity keeps them under data.  Merge into data when
            # the base entity uses that canonical form.
            if isinstance(out.get("data"), dict):
                data = copy.deepcopy(out["data"])
                existing = _dict(data.get(key))
                merged = copy.deepcopy(existing)
                for k, v in value.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = deep_merge_plain(merged[k], v)
                    else:
                        merged[k] = copy.deepcopy(v)
                data[key] = merged
                out["data"] = data
            else:
                existing = _dict(out.get(key))
                merged = copy.deepcopy(existing)
                for k, v in value.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = deep_merge_plain(merged[k], v)
                    else:
                        merged[k] = copy.deepcopy(v)
                out[key] = merged
        elif key == "ChildEntities":
            if isinstance(out.get("data"), dict):
                data = copy.deepcopy(out["data"])
                data[key] = _merge_child_entities(data.get(key), value)
                out["data"] = data
            else:
                out[key] = _merge_child_entities(out.get(key), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def deep_merge_plain(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k == "ChildEntities":
            out[k] = _merge_child_entities(out.get(k), v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_plain(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def deep_merge_data(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k == "ChildEntities":
            out[k] = _merge_child_entities(out.get(k), v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_plain(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class TemplateExpander:
    def __init__(self, db: AssetDatabase):
        self.db = db
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _template_root(self, type_name: str) -> Optional[Dict[str, Any]]:
        if not type_name.startswith("#"):
            return None
        logical = type_name[1:]
        path = "VuTemplateAsset/" + logical
        if path in self._cache:
            return copy.deepcopy(self._cache[path])
        obj = self.db.json(path)
        root = _dict(_dict(obj).get("RootEntity"))
        if not root:
            return None
        self._cache[path] = copy.deepcopy(root)
        return copy.deepcopy(root)

    def expand_entity(self, entity: Dict[str, Any], stack: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
        stack = stack or ()
        ent = copy.deepcopy(entity)
        typ = str(ent.get("type") or "")
        if typ.startswith("#"):
            if typ not in stack:
                root = self._template_root(typ)
                if root:
                    # Preserve instance identity/type while inheriting the
                    # template root's data tree.
                    merged = deep_merge_entity(root, ent)
                    merged["type"] = typ
                    if ent.get("name") is not None:
                        merged["name"] = ent.get("name")
                    ent = merged
                    stack = stack + (typ,)

        data = _dict(ent.get("data"))
        children = child_items(data.get("ChildEntities"))
        if children:
            expanded = []
            for _, c in children:
                expanded.append(self.expand_entity(c, stack))
            data["ChildEntities"] = expanded
            ent["data"] = data
        return ent

    def expand_project(self, project_ref: str | AssetRecord) -> Dict[str, Any]:
        rec = project_ref if isinstance(project_ref, AssetRecord) else self.db.resolve_project(project_ref)
        if rec is None:
            raise KeyError(project_ref)
        obj = self.db.json(rec)
        root = _dict(_dict(obj).get("RootEntity"))
        if not root:
            raise ValueError(f"{rec.path} has no RootEntity")
        return self.expand_entity(root)


@dataclass
class Transform:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))

    @classmethod
    def from_entity(cls, entity: Dict[str, Any]) -> "Transform":
        data = _dict(entity.get("data"))
        comps = _dict(data.get("Components"))
        tc = _dict(comps.get("VuTransformComponent"))
        props = _dict(tc.get("Properties"))
        # Dict children in template overrides can place components at top level.
        if not props:
            tc = _dict(_dict(entity.get("Components")).get("VuTransformComponent"))
            props = _dict(tc.get("Properties"))
        p = props.get("Position", (0.0, 0.0, 0.0))
        r = props.get("Rotation", (0.0, 0.0, 0.0))
        s = props.get("Scale", (1.0, 1.0, 1.0))
        return cls(_vec3(p, 0.0), _vec3(r, 0.0), _vec3(s, 1.0))

    def matrix(self) -> np.ndarray:
        # Vector project Euler values are degrees.  Compose X then Y then Z in
        # local space, matching the established editor transform convention.
        rx, ry, rz = np.radians(self.rotation_deg.astype(np.float64))
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float32)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], np.float32)
        M = np.eye(4, dtype=np.float32)
        M[:3,:3] = (Rz @ Ry @ Rx) * self.scale[np.newaxis,:]
        M[:3,3] = self.position
        return M


def _vec3(v, default: float) -> np.ndarray:
    if isinstance(v, dict):
        return np.asarray([
            float(v.get("X", default)), float(v.get("Y", default)), float(v.get("Z", default))
        ], dtype=np.float32)
    if isinstance(v, (list, tuple)):
        vals = list(v) + [default, default, default]
        return np.asarray(vals[:3], dtype=np.float32)
    return np.asarray([default, default, default], dtype=np.float32)


@dataclass
class EntityView:
    entity: Dict[str, Any]
    path: str
    parent_path: str
    local_transform: Transform
    world_matrix: np.ndarray

    @property
    def type(self) -> str:
        return str(self.entity.get("type") or "")

    @property
    def name(self) -> str:
        return str(self.entity.get("name") or "")

    @property
    def data(self) -> Dict[str, Any]:
        return _dict(self.entity.get("data"))

    @property
    def properties(self) -> Dict[str, Any]:
        p = self.data.get("Properties")
        if not isinstance(p, dict):
            p = self.entity.get("Properties")
        return _dict(p)


def walk_entities(root: Dict[str, Any]) -> Iterator[EntityView]:
    def rec(ent: Dict[str, Any], parent_path: str, parent_world: np.ndarray, ordinal: int):
        name = str(ent.get("name") or ent.get("type") or f"Entity{ordinal}")
        path = parent_path + "/" + name if parent_path else name
        tr = Transform.from_entity(ent)
        world = parent_world @ tr.matrix()
        view = EntityView(ent, path, parent_path, tr, world)
        yield view
        data = _dict(ent.get("data"))
        for i, (_, child) in enumerate(child_items(data.get("ChildEntities"))):
            yield from rec(child, path, world, i)

    yield from rec(root, "", np.eye(4, dtype=np.float32), 0)
