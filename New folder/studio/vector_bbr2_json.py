from __future__ import annotations

"""Generic BBR2 little-endian binary-JSON decoder.

The BBR APF Studio package supplied by the user exposed the game's compact
binary-JSON layout directly.  Vector Studio already had several specialised
walkers for vehicle/effect/PFX data; this module centralises the same format so
those systems can consume the authored object tree instead of rediscovering
nearby strings or hand-decoding only a subset of property types.

BBR2 binary JSON layout (root value at +8):
  0 null
  1 int32
  2 float32
  3 bool/u32
  4 relative C-string reference
  5 array of relative value offsets
  6 object/map with FNV-1a-64 key descriptors
  8 byte buffer; 4/8/12/16-byte buffers are float vectors in these assets

A useful refinement over the supplied standalone parser is that map descriptor
``aux`` is first interpreted as the serialized key-name relative offset.  This
resolves keys whose text does not otherwise occur as a generic C-string scan
candidate (for example VehicleEffectDB/EarthquakePermanent).  FNV lookup is the
fallback and the hash itself is preserved when neither method is available.
"""

import base64
import math
import struct
from typing import Any, Dict, Iterable, List, Optional, Tuple


def fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for b in str(text).encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _valid_text(text: str) -> bool:
    return all(ord(c) >= 32 or c in "\t\n\r" for c in text)


def _c_string(data: bytes, offset: int, max_len: int = 65536) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset, min(len(data), offset + max_len))
    if end < 0:
        return ""
    raw = data[offset:end]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return text if _valid_text(text) else ""


def _cstring_candidates(data: bytes) -> Dict[int, str]:
    """Return FNV hash -> readable C string candidates from one asset.

    Map descriptors normally carry the exact key string offset, so this is only
    a fallback.  Keeping it makes the decoder compatible with assets whose aux
    field is absent/zero or points outside the current root object.
    """
    out: Dict[int, str] = {}
    # Splitting is fast for the BBR2 asset sizes used here and matches the
    # supplied APF Studio parser.  Reject very long/binary fragments.
    for part in data.split(b"\x00"):
        if not part or len(part) > 512:
            continue
        try:
            text = part.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and _valid_text(text):
            out.setdefault(fnv1a64(text), text)
    return out


def looks_like_bbr2_json(data: bytes) -> bool:
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) < 12:
        return False
    try:
        version, declared_size, root_type = struct.unpack_from("<III", data, 0)
    except struct.error:
        return False
    return (
        version == 1
        and 12 <= declared_size <= len(data)
        and root_type in (0, 1, 2, 3, 4, 5, 6, 8)
    )


class BBR2JsonReader:
    MAX_ITEMS = 1_000_000
    MAX_DEPTH = 128

    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.names = _cstring_candidates(self.data)

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def _i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.data, off)[0]

    def _f32(self, off: int) -> float:
        return struct.unpack_from("<f", self.data, off)[0]

    def _type(self, off: int) -> int:
        if off < 0 or off + 4 > len(self.data):
            return -1
        try:
            return self._u32(off)
        except struct.error:
            return -1

    def _key_name(self, map_off: int, key_hash: int, aux: int) -> str:
        # In BBR2 Project/Template/DB maps this field is the relative key-name
        # string offset.  Validate the FNV before accepting it, so an unrelated
        # aux field can never silently rename a key.
        if aux:
            name = _c_string(self.data, map_off + int(aux), 1024)
            if name and fnv1a64(name) == int(key_hash):
                return name
        name = self.names.get(int(key_hash), "")
        if name and fnv1a64(name) == int(key_hash):
            return name
        return f"#{int(key_hash):016x}"

    def parse(self, off: int = 8, depth: int = 0, stack: Optional[set] = None) -> Any:
        if stack is None:
            stack = set()
        if depth > self.MAX_DEPTH or off < 0 or off + 4 > len(self.data):
            return None
        if off in stack:
            # Serialized values are trees in the observed files; retain a clear
            # marker rather than recursing forever if a malformed asset loops.
            return {"__bbr2_ref_loop__": int(off)}

        typ = self._type(off)
        try:
            if typ == 0:
                return None
            if typ == 1:
                return self._i32(off + 4)
            if typ == 2:
                value = self._f32(off + 4)
                return float(value) if math.isfinite(value) else 0.0
            if typ == 3:
                return bool(self._u32(off + 4))
            if typ == 4:
                if off + 8 > len(self.data):
                    return ""
                rel = self._u32(off + 4)
                # The value is relative to the type-4 record itself. rel==8 is
                # therefore the common inline-string form.
                text = _c_string(self.data, off + int(rel))
                if text or (0 <= off + int(rel) < len(self.data) and self.data[off + int(rel):off + int(rel) + 1] == b"\x00"):
                    return text
                return f"__string_ref_{int(rel)}__"
            if typ == 5:
                if off + 8 > len(self.data):
                    return []
                count = self._u32(off + 4)
                if count > self.MAX_ITEMS or off + 8 + count * 4 > len(self.data):
                    return []
                new_stack = set(stack)
                new_stack.add(off)
                out = []
                for i in range(count):
                    rel = self._u32(off + 8 + i * 4)
                    child = off + int(rel)
                    if child < 0 or child + 4 > len(self.data):
                        out.append(None)
                    else:
                        out.append(self.parse(child, depth + 1, new_stack))
                return out
            if typ == 6:
                if off + 8 > len(self.data):
                    return {}
                count = self._u32(off + 4)
                table_end = off + 8 + count * 16
                if count > self.MAX_ITEMS or table_end > len(self.data):
                    return {}
                new_stack = set(stack)
                new_stack.add(off)
                obj: Dict[str, Any] = {}
                for i in range(count):
                    d = off + 8 + i * 16
                    key_hash, aux, rel = struct.unpack_from("<QII", self.data, d)
                    key = self._key_name(off, key_hash, aux)
                    child = off + int(rel)
                    value = self.parse(child, depth + 1, new_stack) if 0 <= child < len(self.data) else None
                    # Duplicate property names are not expected in the authored
                    # BBR2 maps. If one appears, retain every value rather than
                    # silently losing data.
                    if key in obj:
                        previous = obj[key]
                        if isinstance(previous, list) and previous and isinstance(previous[-1], dict) and "__duplicate_value__" in previous[-1]:
                            previous.append({"__duplicate_value__": value})
                        else:
                            obj[key] = [previous, {"__duplicate_value__": value}]
                    else:
                        obj[key] = value
                return obj
            if typ == 8:
                if off + 8 > len(self.data):
                    return None
                size = self._u32(off + 4)
                if size > len(self.data) - off - 8:
                    return None
                raw = self.data[off + 8:off + 8 + size]
                if size in (4, 8, 12, 16):
                    vals = struct.unpack_from("<%df" % (size // 4), raw, 0)
                    if all(math.isfinite(float(v)) for v in vals):
                        return [float(v) for v in vals]
                return {"__bbr2_bytes__": base64.b64encode(raw).decode("ascii")}
        except (struct.error, ValueError, OverflowError):
            return None
        return {"__bbr2_unknown_type__": int(typ), "offset": int(off)}


def decode_bbr2_json(data: bytes) -> Any:
    if not looks_like_bbr2_json(data):
        raise ValueError("not BBR2 little-endian binary JSON")
    return BBR2JsonReader(data).parse(8)


def walk_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def named_entities(root: Any) -> List[Dict[str, Any]]:
    """Return authored entity wrappers from a decoded Project/Template tree.

    Only wrappers that serialize all three normal entity fields (data/type/name)
    are returned.  The root wrapper often has no ``name`` and is deliberately
    omitted; older Vector Studio builds invented the asset filename as its name,
    creating one extra pseudo-entity per file.
    """
    out: List[Dict[str, Any]] = []
    for wrapper in walk_dicts(root):
        data = wrapper.get("data")
        entity_type = wrapper.get("type")
        entity_name = wrapper.get("name")
        if not isinstance(data, dict) or not isinstance(entity_type, str) or not isinstance(entity_name, str) or not entity_name:
            continue

        props = data.get("Properties")
        if not isinstance(props, dict):
            props = {}
        comps = data.get("Components")
        if not isinstance(comps, dict):
            comps = {}
        tf = comps.get("VuTransformComponent")
        if not isinstance(tf, dict):
            tf = {}
        tf_props = tf.get("Properties")
        if not isinstance(tf_props, dict):
            tf_props = {}

        def vec3(name: str, default=None):
            value = tf_props.get(name)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return tuple(float(value[i]) for i in range(3))
                except Exception:
                    pass
            return default

        out.append({
            "name": entity_name,
            "type": entity_type,
            "data": data,
            "properties": dict(props),
            "position": vec3("Position", None),
            "rotation": vec3("Rotation", (0.0, 0.0, 0.0)),
            "scale": vec3("Scale", (1.0, 1.0, 1.0)),
        })
    return out
