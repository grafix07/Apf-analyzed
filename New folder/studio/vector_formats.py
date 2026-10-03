#!/usr/bin/env python3
"""Core Vector Unit / BBR format support used by BBR Vector Studio.

Reverse engineered from the supplied game assets and the working Noesis parser.
The module primarily provides Vector asset decoding; APF rebuilding is handled
by the dedicated APF writer used by custom-skin export.
"""
from __future__ import annotations

import io
import lzma
import math
import mmap
import os
import re
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


HEADER_SIZE = 0x3C


@dataclass
class APFEntry:
    index: int
    path: str
    offset: int
    unpacked_size: int
    stored_size: int
    checksum: int
    flags: int

    @property
    def asset_type(self) -> str:
        return self.path.split("/", 1)[0] if "/" in self.path else "(root)"

    @property
    def asset_name(self) -> str:
        return self.path.split("/", 1)[1] if "/" in self.path else self.path


class APFArchive:
    def __init__(self, path):
        self.path = Path(path)
        self.fp = None
        self.mm = None
        self.version = 0
        self.index_offset = 0
        self.entry_count = 0
        self.index_size = 0
        self.archive_tag = 0
        self.platform = ""
        self.entries: List[APFEntry] = []
        self._by_path = {}
        self.open()

    def open(self):
        self.fp = open(self.path, "rb")
        self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.mm) < HEADER_SIZE or self.mm[:4] != b"FPUV":
            raise ValueError("Not a supported Vector FPUV/APF archive.")
        (self.version, self.index_offset, self.entry_count,
         self.index_size, self.archive_tag) = struct.unpack_from("<IIIII", self.mm, 4)
        # BBR2 mobile uses this v8 FPUV layout.  Some related Vector desktop/
        # legacy builds use version 7 with the same entry record shape, so v0.34
        # accepts both and validates every index range/record.  This broadens
        # Island Adventure compatibility without pretending untested variants
        # are identical.
        if self.version not in (7, 8):
            raise ValueError(
                "Unsupported Vector APF/FPUV version %d (supported: 7, 8)."
                % self.version
            )
        raw = self.mm[0x18:0x38]
        self.platform = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
        end = self.index_offset + self.index_size
        if self.index_offset < HEADER_SIZE or end > len(self.mm):
            raise ValueError("Invalid APF index range.")
        p = self.index_offset
        self.entries.clear()
        for i in range(self.entry_count):
            nul = self.mm.find(b"\0", p, end)
            if nul < 0:
                raise ValueError("Unterminated APF index path.")
            name = self.mm[p:nul].decode("utf-8", "replace")
            p = nul + 1
            if p + 20 > end:
                raise ValueError("Truncated APF index metadata.")
            vals = struct.unpack_from("<IIIII", self.mm, p)
            p += 20
            e = APFEntry(i, name, *vals)
            self.entries.append(e)

        # Some modified/desktop-exported APFs reserve a small metadata/padding
        # tail inside index_size after the final entry.  It is not another asset
        # record and must not make an otherwise valid archive unloadable.  Keep
        # the old strict validation for large gaps, but accept a bounded tail
        # and expose it for diagnostics.
        self.index_tail_size = max(0, end - p)
        if self.index_tail_size > 0x400:
            raise ValueError(
                "APF index parsed %d entries but ended at 0x%X; expected 0x%X."
                % (self.entry_count, p, end)
            )
        self._by_path = {e.path: e for e in self.entries}

    def close(self):
        if self.mm is not None:
            self.mm.close(); self.mm = None
        if self.fp is not None:
            self.fp.close(); self.fp = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def get(self, path: str) -> Optional[APFEntry]:
        return self._by_path.get(path)

    def stored_bytes(self, entry: APFEntry) -> bytes:
        if entry.offset < HEADER_SIZE or entry.offset + entry.stored_size > self.index_offset:
            raise ValueError("Invalid asset payload range.")
        return self.mm[entry.offset:entry.offset + entry.stored_size]

    @staticmethod
    def _lzma_props(first_byte):
        if first_byte >= 9 * 5 * 5:
            raise ValueError("Invalid LZMA properties byte.")
        lc = first_byte % 9
        x = first_byte // 9
        lp = x % 5
        pb = x // 5
        return lc, lp, pb

    def codec(self, entry: APFEntry) -> str:
        """Return the compression method encoded by the APF entry flags.

        BBR APF Studio confirms that the upper 16 bits are the authoritative
        codec selector (0 raw, 1 zlib, 2 raw-LZMA, 3 Snappy, 4 LZ4). Older
        Studio builds guessed from payload bytes, which could misclassify a
        valid stream whose first bytes happened to resemble another codec.
        """
        hi = (int(entry.flags) >> 16) & 0xFFFF
        return {0: "raw", 1: "zlib", 2: "lzma-raw", 3: "snappy", 4: "lz4"}.get(
            hi, "unknown"
        )

    @staticmethod
    def _snappy_copy(out: bytearray, offset: int, length: int):
        if offset <= 0 or offset > len(out):
            raise ValueError("Invalid Snappy copy offset %d." % offset)
        start = len(out) - offset
        for i in range(int(length)):
            out.append(out[start + i])

    @classmethod
    def _snappy_decompress(cls, blob: bytes, expected_size: int) -> bytes:
        # Raw Snappy block decoder adapted from the supplied BBR APF Studio.
        i = 0
        shift = 0
        declared = 0
        while True:
            if i >= len(blob):
                raise ValueError("Truncated Snappy length.")
            b = blob[i]; i += 1
            declared |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
            if shift > 35:
                raise ValueError("Invalid Snappy length varint.")
        if declared != int(expected_size):
            raise ValueError(
                "Snappy declared size %d, expected %d." % (declared, expected_size)
            )

        out = bytearray()
        n = len(blob)
        while i < n:
            tag = blob[i]; i += 1
            kind = tag & 3
            if kind == 0:
                length = tag >> 2
                if length < 60:
                    length += 1
                else:
                    nbytes = length - 59
                    if i + nbytes > n:
                        raise ValueError("Truncated Snappy literal length.")
                    length = int.from_bytes(blob[i:i+nbytes], "little") + 1
                    i += nbytes
                if i + length > n:
                    raise ValueError("Truncated Snappy literal.")
                out.extend(blob[i:i+length]); i += length
            elif kind == 1:
                length = ((tag >> 2) & 7) + 4
                if i >= n:
                    raise ValueError("Truncated Snappy copy-1.")
                offset = ((tag & 0xE0) << 3) | blob[i]; i += 1
                cls._snappy_copy(out, offset, length)
            elif kind == 2:
                length = (tag >> 2) + 1
                if i + 2 > n:
                    raise ValueError("Truncated Snappy copy-2.")
                offset = blob[i] | (blob[i+1] << 8); i += 2
                cls._snappy_copy(out, offset, length)
            else:
                length = (tag >> 2) + 1
                if i + 4 > n:
                    raise ValueError("Truncated Snappy copy-4.")
                offset = int.from_bytes(blob[i:i+4], "little"); i += 4
                cls._snappy_copy(out, offset, length)

        if len(out) != int(expected_size):
            raise ValueError(
                "Snappy output size %d, expected %d." % (len(out), expected_size)
            )
        return bytes(out)

    @staticmethod
    def _lz4_block_decompress(blob: bytes, expected_size: int) -> bytes:
        """Decode a raw LZ4 block without requiring the optional lz4 package.

        Vector APF codec 4 stores the normal LZ4 block token stream and keeps
        the uncompressed size in the APF index, so no LZ4 frame header is
        required.  This fallback implements the literal/match sequence defined
        by the LZ4 block format and is used when python-lz4 is unavailable.
        """
        src = memoryview(blob)
        i = 0
        n = len(src)
        out = bytearray()

        def read_ext_length(base: int) -> int:
            nonlocal i
            length = base
            if base == 15:
                while True:
                    if i >= n:
                        raise ValueError("Truncated LZ4 extended length.")
                    b = int(src[i]); i += 1
                    length += b
                    if b != 255:
                        break
            return length

        while i < n:
            token = int(src[i]); i += 1
            literal_len = read_ext_length(token >> 4)
            if i + literal_len > n:
                raise ValueError("Truncated LZ4 literal run.")
            if literal_len:
                out.extend(src[i:i + literal_len])
                i += literal_len

            # LZ4 permits the final sequence to contain literals only.
            if i >= n:
                break

            if i + 2 > n:
                raise ValueError("Truncated LZ4 match offset.")
            offset = int(src[i]) | (int(src[i + 1]) << 8)
            i += 2
            if offset <= 0 or offset > len(out):
                raise ValueError("Invalid LZ4 match offset %d." % offset)

            match_len = read_ext_length(token & 0x0F) + 4
            start = len(out) - offset
            # Match copies may overlap their own output, so append byte-wise.
            for j in range(match_len):
                out.append(out[start + j])
                if len(out) > int(expected_size):
                    raise ValueError("LZ4 output exceeds expected size %d." % expected_size)

        if len(out) != int(expected_size):
            raise ValueError(
                "LZ4 output size %d, expected %d." % (len(out), expected_size)
            )
        return bytes(out)

    def decode(self, entry: APFEntry) -> bytes:
        blob = bytes(self.stored_bytes(entry))
        codec = self.codec(entry)

        if codec == "raw":
            if len(blob) != int(entry.unpacked_size):
                raise ValueError(
                    "Raw APF entry %s is %d bytes, expected %d."
                    % (entry.path, len(blob), entry.unpacked_size)
                )
            return blob

        if codec == "zlib":
            out = zlib.decompress(blob)
        elif codec == "lzma-raw":
            if len(blob) < 5:
                raise ValueError("Truncated Vector raw-LZMA stream: %s" % entry.path)
            # Vector stores the standard five LZMA1 property/dictionary bytes but
            # omits the FORMAT_ALONE eight-byte uncompressed-size field. Restore
            # that field exactly as the supplied APF Studio does.
            self._lzma_props(blob[0])
            if int.from_bytes(blob[1:5], "little") <= 0:
                raise ValueError("Invalid Vector raw-LZMA dictionary size.")
            fixed = blob[:5] + struct.pack("<Q", int(entry.unpacked_size)) + blob[5:]
            out = lzma.decompress(fixed, format=lzma.FORMAT_ALONE)
        elif codec == "snappy":
            out = self._snappy_decompress(blob, int(entry.unpacked_size))
        elif codec == "lz4":
            try:
                import lz4.block
                out = lz4.block.decompress(
                    blob, uncompressed_size=int(entry.unpacked_size)
                )
            except Exception:
                # Island Adventure's Steam APF uses codec 4 heavily.  Do not
                # make opening the archive depend on a separately installed
                # wheel: use the built-in raw-LZ4 block decoder as fallback.
                out = self._lz4_block_decompress(blob, int(entry.unpacked_size))
        else:
            raise ValueError(
                "Unsupported APF compression flags 0x%08X for %s."
                % (int(entry.flags), entry.path)
            )

        if len(out) != int(entry.unpacked_size):
            raise ValueError(
                "Decoded %s to %d bytes, expected %d."
                % (entry.path, len(out), entry.unpacked_size)
            )
        return bytes(out)


def open_apf_or_package(path: str, choose_name=None):
    """Return ``(APFArchive, temp_path_or_None)``.

    Besides direct APF files, Vector releases can package archives in APK/ZIP
    containers.  Modern Android bundles (XAPK/APKS) commonly put the real APF
    inside a nested asset-pack APK, so scan one nested ZIP level as well.  The
    chooser receives readable names such as ``asset_pack.apk::assets/Assets.apf``.
    This also makes the Studio usable with BBR2 mobile bundles while retaining
    ordinary desktop/Island Adventure ``.apf`` loading.
    """
    p = Path(path)
    if p.suffix.lower() == ".apf":
        return APFArchive(p), None
    if p.suffix.lower() not in (".apk", ".zip", ".xapk", ".apks"):
        raise ValueError("Choose an .apf, .apk, .zip, .xapk or .apks file.")

    candidates = []   # (display_name, raw_bytes_or_None, outer_member_or_None)
    with zipfile.ZipFile(p, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".apf"):
                candidates.append((name, None, name))

        # XAPK/APKS bundles usually store game data in one nested asset APK.
        # Keep this bounded to one level so malformed packages cannot recurse.
        for nested_name in z.namelist():
            if not nested_name.lower().endswith((".apk", ".zip")):
                continue
            try:
                nested_blob = z.read(nested_name)
                with zipfile.ZipFile(io.BytesIO(nested_blob), "r") as nz:
                    for inner in nz.namelist():
                        if inner.lower().endswith(".apf"):
                            candidates.append((
                                nested_name + "::" + inner,
                                nz.read(inner),
                                None,
                            ))
            except (zipfile.BadZipFile, KeyError, OSError):
                continue

        if not candidates:
            raise ValueError("The package contains no APF archives (including nested asset APKs).")

        displays = [x[0] for x in candidates]
        display = displays[0] if len(displays) == 1 else (choose_name(displays) if choose_name else displays[0])
        if not display:
            raise ValueError("No APF selected.")
        idx = displays.index(display)
        _, blob, outer_member = candidates[idx]

        tmp = tempfile.NamedTemporaryFile(prefix="bbr_vector_", suffix=".apf", delete=False)
        if blob is not None:
            tmp.write(blob)
        else:
            with z.open(outer_member) as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
        tmp.close()

    return APFArchive(tmp.name), tmp.name


@dataclass
class Bone:
    index: int
    name: str
    parent: int
    translation: np.ndarray
    rotation: np.ndarray       # x,y,z,w
    scale: np.ndarray


@dataclass
class MeshData:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    colors: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    joints: Optional[np.ndarray] = None
    buffer_index: int = 0
    vertex_stride: int = 0
    uv_offset: int = 0
    skin_offset: int = -1
    custom_color: Optional[np.ndarray] = None
    custom_texture: Optional[object] = None




def recompute_mesh_normals(positions, indices):
    """Compute area-weighted smooth vertex normals from authored geometry.

    BBR2 mobile model buffers contain an 8-byte packed tangent-frame block at
    offset 12.  Older Studio builds interpreted the first six bytes as three
    signed 16-bit normal components.  That happens to look plausible for many
    vertices but produces near-zero/invalid normals on a large portion of
    character meshes and makes faces look faceted/blocky.

    The triangle topology and positions are authoritative, so when the packed
    frame cannot be decoded reliably we derive normals from the actual mesh
    geometry instead of guessing a byte layout.
    """
    p = np.asarray(positions, dtype=np.float32)
    tri = np.asarray(indices, dtype=np.int64)
    out = np.zeros_like(p, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or tri.size == 0:
        return out
    tri = tri.reshape((-1, 3))
    valid = (tri >= 0).all(axis=1) & (tri < len(p)).all(axis=1)
    tri = tri[valid]
    if not len(tri):
        return out
    a = p[tri[:, 0]]
    b = p[tri[:, 1]]
    c = p[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    # Cross-product magnitude is twice triangle area. Keeping that magnitude
    # naturally gives larger authored faces proportionally more influence.
    for corner in range(3):
        np.add.at(out, tri[:, corner], fn)
    mag = np.linalg.norm(out, axis=1)
    good = mag > 1e-12
    out[good] /= mag[good, None]
    out[~good] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return out


def normals_need_geometry_rebuild(normals):
    n = np.asarray(normals, dtype=np.float32)
    if n.ndim != 2 or n.shape[1] != 3 or len(n) == 0:
        return True
    finite = np.isfinite(n).all(axis=1)
    if float(np.mean(finite)) < 0.995:
        return True
    mag = np.linalg.norm(n[finite], axis=1)
    if len(mag) == 0:
        return True
    # Packed-frame misreads in the supplied BBR2 character meshes typically
    # leave 10-25% of normals below 0.5 length.  Authored unit normals should
    # not do that.  Also catch the BBR2 static fallback where every normal was
    # the same (0,1,0).
    bad = np.mean((mag < 0.70) | (mag > 1.30))
    unique = len(np.unique(np.round(n[finite], 4), axis=0))
    return bad > 0.02 or unique <= 2

@dataclass
class ModelData:
    name: str
    meshes: List[MeshData]
    bones: List[Bone] = field(default_factory=list)
    source_path: str = ""

    @property
    def bone_count(self):
        return len(self.bones)

    def bounds(self):
        pts = [m.positions for m in self.meshes if len(m.positions)]
        if not pts:
            return np.zeros(3, np.float32), np.ones(3, np.float32)
        a = np.concatenate(pts, axis=0)
        return a.min(axis=0), a.max(axis=0)


@dataclass
class MaterialData:
    name: str
    shader: str = ""
    diffuse_color: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    )
    diffuse_texture: str = ""
    paint_texture: str = ""
    decal_texture: str = ""
    decal_color_texture: str = ""
    normal_texture: str = ""
    mask_texture: str = ""
    env_texture: str = ""
    additive_env_texture: str = ""
    fresnel_texture: str = ""
    detail_texture: str = ""
    one_bit_alpha_texture: str = ""
    parallax_texture: str = ""
    detail_uv_scale: float = 1.0
    emissive_amount: float = 0.0


@dataclass
class TextureData:
    name: str
    width: int
    height: int
    mip_count: int
    format_code: int
    rgba: np.ndarray
    format_name: str = ""


@dataclass
class CubeTextureData:
    name: str
    width: int
    height: int
    mip_count: int
    format_code: int
    faces: List[TextureData] = field(default_factory=list)
    format_name: str = ""

@dataclass
class AnimationData:
    name: str
    bone_count: int
    frame_count: int
    fps: float
    translations: np.ndarray  # [frame,bone,3]
    rotations: np.ndarray     # [frame,bone,4], native int16 / 32767.0 (x,y,z,w)
    scales: np.ndarray        # [frame,bone,3]
    loop_flag: int = 0
    trailing: bytes = b""
    key_size: int = 32
    exact_eof: bool = True

    @property
    def duration(self):
        return max(0.0, (self.frame_count - 1) / float(self.fps))

    @property
    def looping(self):
        return bool(self.loop_flag)

    @property
    def expected_size(self):
        return 8 + int(self.bone_count) * int(self.frame_count) * int(self.key_size) + 1

    def codec_report(self):
        flag=(str(self.loop_flag) if self.loop_flag >= 0 else "embedded:none")
        suffix=("exact" if self.exact_eof else "embedded")
        return (
            f"{self.bone_count} bones • {self.frame_count} frames • {self.fps:g} FPS • "
            f"{self.duration:.3f}s • loopFlag={flag} • "
            f"VuAnimation {self.key_size}-byte {suffix}"
        )


def quat_matrix_engine(q):
    """Match the transposed quaternion matrix used by the verified Noesis plugin."""
    x, y, z, w = [float(v) for v in q]
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        R = np.eye(3, dtype=np.float32)
    else:
        s = 2.0 / n
        R = np.array([
            [1-s*(y*y+z*z), s*(x*y-z*w),   s*(x*z+y*w)],
            [s*(x*y+z*w),   1-s*(x*x+z*z), s*(y*z-x*w)],
            [s*(x*z-y*w),   s*(y*z+x*w),   1-s*(x*x+y*y)],
        ], dtype=np.float32)
    return R.T


def trs_matrix(t, q, s):
    M = np.eye(4, dtype=np.float32)
    R = quat_matrix_engine(q)
    S = np.asarray(s, dtype=np.float32)
    M[:3, :3] = R * S[np.newaxis, :]
    M[:3, 3] = np.asarray(t, dtype=np.float32)
    return M


def parse_skeleton(data: bytes) -> List[Bone]:
    if len(data) <= 24 or not (b"Root\x00" in data[8:64] or b"root\x00" in data[8:64]):
        return []
    bc = struct.unpack_from("<I", data, 13)[0]
    if bc < 1 or bc > 512:
        return []
    o = 17
    names = []
    for _ in range(bc):
        if o + 36 > len(data):
            return []
        rec = data[o:o+36]
        e = rec.find(b"\x00")
        if e < 0: e = 32
        names.append(rec[:e].decode("latin1", "ignore"))
        o += 36
    if o + bc * 4 + bc * 32 > len(data):
        return []
    parents = list(struct.unpack_from("<%di" % bc, data, o)); o += bc * 4
    bones = []
    for i in range(bc):
        t = np.array(struct.unpack_from("<3f", data, o), dtype=np.float32)
        qi = struct.unpack_from("<4h", data, o+12)
        # Native VuAnimationTransform::deserialize multiplies signed int16
        # quaternion components by exactly 1/32767.0. Keep the decoded values
        # un-normalized here; quat_matrix_engine()/SLERP normalize as needed,
        # matching the engine's toMatrix path without mutating authored keys.
        q = np.array([x / 32767.0 for x in qi], dtype=np.float32)
        if float(np.linalg.norm(q)) < 1e-8:
            q = np.array([0,0,0,1], dtype=np.float32)
        s = np.array(struct.unpack_from("<3f", data, o+20), dtype=np.float32)
        bones.append(Bone(i, names[i], parents[i], t, q, s))
        o += 32
    return bones


def _find_uv_offset(data, start_off, stride, vertex_count=None):
    """Find UV0 from a Vector Unit static/animated vertex buffer.

    Vehicle static buffers in the supplied BBR2 build are not one single
    layout.  In particular, common 24/28-byte vehicle buffers store UV0 at
    byte 16, while some 28-byte attachment buffers use byte 20.  v0.53
    hard-coded byte 20 for the primary BBR2 buffer, which read the second UV
    component from the next packed field (often NaN) and made most paint/decal
    materials collapse to a flat colour.

    Score candidate float pairs across many vertices.  Packed tangent/normal
    bytes can look like tiny finite floats, so candidates with denormal or
    effectively constant components are explicitly penalized.
    """
    if stride < 20:
        return -1

    candidates = [o for o in (20, 16, 12, 24, 28, 32, 36, 40, 44, 48, 52, 56)
                  if o + 8 <= stride]
    if not candidates:
        return -1

    available = max(0, len(data) - start_off) // max(1, stride)
    if vertex_count is not None:
        available = min(available, max(0, int(vertex_count)))
    sample_count = min(256, available)
    if sample_count <= 0:
        return -1

    best_off = -1
    best_score = -1.0e30
    for cand in candidates:
        vals = []
        for i in range(sample_count):
            try:
                u, v = struct.unpack_from('<2f', data, start_off + i * stride + cand)
            except (struct.error, ValueError):
                break
            if math.isfinite(u) and math.isfinite(v):
                vals.append((float(u), float(v)))

        # A UV field should be valid for essentially the whole vertex buffer.
        if len(vals) < max(4, int(sample_count * 0.75)):
            continue

        arr = np.asarray(vals, dtype=np.float64)
        abs_arr = np.abs(arr)
        sane = np.all(abs_arr <= 8.0, axis=1)
        broad = np.all(abs_arr <= 64.0, axis=1)
        extreme = np.any(abs_arr > 1.0e4, axis=1)
        denorm = np.logical_and(abs_arr > 0.0, abs_arr < 1.0e-20)
        ranges = np.ptp(arr, axis=0)

        score = 0.0
        score += float(np.mean(sane)) * 12.0
        score += float(np.mean(broad)) * 3.0
        score -= float(np.mean(extreme)) * 30.0
        score -= float(np.mean(denorm)) * 18.0

        # Real model UV maps normally vary on both axes.  Packed tangent data
        # at offset 12 often produces one near-zero/denormal component and one
        # actual UV component, so this separates it cleanly from offset 16.
        for r in ranges:
            if r > 1.0e-4:
                score += 2.0
            elif r > 1.0e-8:
                score -= 1.0
            else:
                score -= 5.0

        # In close calls prefer the later authored field rather than the packed
        # tangent bytes that precede it.
        score += cand * 0.001
        if score > best_score:
            best_score = score
            best_off = cand

    return best_off



def _find_static_color_offset(data, start_off, stride, vertex_count, uv_off):
    """Locate an authored COLOR0 field in Vector static vertex buffers.

    COLOR0 is stored *after* UV0 in the layouts used by the supplied BBR2 and
    Island Adventure assets.  v0.68.1 still searched fields before UV0 as a
    fallback.  On the common Android 24-byte layout (pos + packed frame + UV0)
    that made the packed normal/tangent bytes look like RGB, producing the
    neon/rainbow colours seen on vehicle attachments and some track props.

    Only score aligned fields at/after the end of UV0.  Reject the very common
    all-zero-alpha pattern, which is packed frame data rather than UNORM8 RGBA.
    This keeps the known authored layouts intact (28/32-byte mobile, 40-byte
    Steam with UV0@28/COLOR0@36) while treating layouts with no colour channel
    as plain white vertex colour.
    """
    stride=int(stride); vertex_count=max(0,int(vertex_count or 0)); uv_off=int(uv_off or -1)
    if stride < 20 or vertex_count <= 0 or uv_off < 12:
        return -1
    first=uv_off+8
    if first+4 > stride:
        return -1
    candidates=[off for off in range(first, stride-3, 4) if off+4 <= stride]
    sample=min(512, vertex_count)
    best=-1; best_score=-1e30
    for off in candidates:
        try:
            arr=np.empty((sample,4),dtype=np.uint8)
            for i in range(sample):
                arr[i]=np.frombuffer(data,dtype=np.uint8,count=4,offset=start_off+i*stride+off)
        except Exception:
            continue
        a=arr[:,3].astype(np.int16)
        rgb=arr[:,:3].astype(np.float32)
        opaque=float(np.mean(a>=248))
        transparent=float(np.mean(a<=7))
        binary_alpha=float(np.mean((a<=7)|(a>=248)))
        gray=float(np.mean((np.max(rgb,axis=1)-np.min(rgb,axis=1))<=4.0))
        unique_ratio=float(len(np.unique(arr,axis=0)))/float(max(1,sample))
        # Packed tangent/frame bytes in the false Android cases have alpha=0
        # for essentially every vertex.  Authored COLOR0 may contain some
        # transparency, but not a whole buffer of zero-alpha random RGB.
        if transparent > 0.97 and opaque < 0.01:
            continue
        score=opaque*12.0 + binary_alpha*3.0 + gray*2.0 - unique_ratio*1.5
        if off == first:
            score += 2.0
        if off == stride-4:
            score += 1.0
        if score > best_score:
            best_score=score; best=off
    return best

VALID_STRIDES_ANIM = {20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64}

def parse_model(data):
    lod_idx = 0
    is_animated = False
    try:
        if len(data) > 24 and (b'Root\x00' in data[8:40] or b'root\x00' in data[8:40]):
            is_animated = True
    except:
        pass

    meshes = []

    if is_animated:
        import re
        matches = list(re.finditer(b'([\x01-\x20]\x00\x00\x00)([A-Za-z0-9_/.-]{4,}\x00)', data))

        best_score = 0
        best_block = None

        for m in matches:
            off = m.start()
            try:
                num_names = struct.unpack_from("<I", data, off)[0]
                if num_names < 1 or num_names > 20: continue
                tmp = off + 4
                names = []
                for _ in range(num_names):
                    end = data.index(b"\x00", tmp)
                    names.append(data[tmp:end].decode("ascii", "ignore"))
                    tmp = end + 1
                if not any("/" in n for n in names): continue

                struct_count = struct.unpack_from("<I", data, tmp)[0]
                if struct_count < 1 or struct_count > 30: continue
                tmp += 4
                hdr16 = struct.unpack_from("<4I", data, tmp); tmp += 16

                buf_hdrs = []
                mesh_hdrs = []
                secondary_hints = []

                for _ in range(struct_count):
                    i0, i1, i2, i3 = struct.unpack_from("<4I", data, tmp + 24)
                    if i2 in VALID_STRIDES_ANIM and i3 > 0 and i2 < 100 and i3 % i2 == 0:
                        buf_hdrs.append({"stride": i2, "size": i3})
                    elif i0 == 0:
                        mesh_hdrs.append({"idx_start": i2, "idx_count": i3, "name_idx": i1})
                    else:
                        secondary_hints.append({"tri_count": i3, "name_idx": i1})
                    tmp += 40

                if not buf_hdrs: continue

                total_verts_primary = buf_hdrs[0]["size"] // buf_hdrs[0]["stride"]
                secondary_tris = sum(h["tri_count"] for h in secondary_hints)
                score = total_verts_primary + secondary_tris

                if score > best_score:
                    best_score = score
                    best_block = {
                        "names": names,
                        "buf_hdrs": buf_hdrs,
                        "mesh_hdrs": mesh_hdrs,
                        "secondary_hints": secondary_hints,
                        "data_start": tmp,
                        "total_verts_primary": total_verts_primary,
                    }
            except:
                continue

        if not best_block:
            return {"meshes": meshes}

        names     = best_block["names"]
        buf_hdrs  = best_block["buf_hdrs"]
        mesh_hdrs = best_block["mesh_hdrs"]
        sec_hints = best_block["secondary_hints"]
        off       = best_block["data_start"]
        total_verts = best_block["total_verts_primary"]
        stride    = buf_hdrs[0]["stride"]
        uv_off    = _find_uv_offset(data, off, stride)

        # Primary vertex buffer
        verts = []
        for i in range(total_verts):
            v_off = off + i * stride
            px, py, pz = struct.unpack_from("<3f", data, v_off)
            nx_raw, ny_raw, nz_raw = struct.unpack_from("<3h", data, v_off + 12)
            nx, ny, nz = nx_raw / 32767.0, ny_raw / 32767.0, nz_raw / 32767.0
            u, v = struct.unpack_from("<2f", data, v_off + uv_off) if uv_off >= 12 else (0.0, 0.0)
            
            # Skin data follows the UV pair in the Vector animated-vertex layouts.
            # Do NOT assume it is always the final 8 bytes: Cyber Screens/Cockpit
            # use a 36-byte layout with an additional 4-byte field after joints.
            w = uv_off + 8
            if w + 8 > stride:
                w = stride - 8
            w0, w1, w2, w3 = struct.unpack_from("<4B", data, v_off + w)
            j0, j1, j2, j3 = struct.unpack_from("<4B", data, v_off + w + 4)
            w_sum = float(w0 + w1 + w2 + w3) or 1.0
            weights = (w0 / w_sum, w1 / w_sum, w2 / w_sum, w3 / w_sum)
            joints = (j0, j1, j2, j3)
            
            verts.append((px, py, pz, nx, ny, nz, u, v, weights, joints))
        off += buf_hdrs[0]["size"]

        total_indices = struct.unpack_from("<I", data, off)[0]; off += 4
        raw_indices = struct.unpack_from("<{0}H".format(total_indices), data, off); off += total_indices * 2

        if not mesh_hdrs:
            tris = [(raw_indices[t], raw_indices[t+1], raw_indices[t+2])
                    for t in range(0, total_indices - 2, 3)
                    if raw_indices[t] < total_verts and raw_indices[t+1] < total_verts and raw_indices[t+2] < total_verts]
            meshes.append({"name": (names[0] if names else "mesh_0") + "_LOD{0}".format(lod_idx), "vertices": verts, "indices": tris})
        else:
            used_name_indices = {h["name_idx"] for h in mesh_hdrs} | {h["name_idx"] for h in sec_hints}
            avail_names = [i for i in range(len(names)) if i not in used_name_indices]
            valid_hdrs = [h for h in mesh_hdrs if h["idx_start"] + h["idx_count"] * 3 <= total_indices]
            valid_hdrs.sort(key=lambda x: x["idx_start"])
            filled = []
            cur = 0
            for hdr in valid_hdrs:
                s, cnt = hdr["idx_start"], hdr["idx_count"] * 3
                if s > cur:
                    gidx = avail_names.pop(0) if avail_names else 0
                    filled.append({"idx_start": cur, "idx_count": (s - cur) // 3, "name_idx": gidx})
                filled.append(hdr)
                cur = s + cnt
            if cur < total_indices:
                gidx = avail_names.pop(0) if avail_names else 0
                filled.append({"idx_start": cur, "idx_count": (total_indices - cur) // 3, "name_idx": gidx})

            for hdr in filled:
                nidx = hdr["name_idx"]
                nm = (names[nidx] if nidx < len(names) else "mesh_{0}".format(nidx)) + "_LOD{0}".format(lod_idx)
                seg = raw_indices[hdr["idx_start"]: hdr["idx_start"] + hdr["idx_count"] * 3]
                tris = [(seg[t], seg[t+1], seg[t+2])
                        for t in range(0, len(seg) - 2, 3)
                        if seg[t] < total_verts and seg[t+1] < total_verts and seg[t+2] < total_verts]
                meshes.append({"name": nm, "vertices": verts, "indices": tris})

        # Secondary standalone body buffers (body mesh stored after the primary accessory block)
        for hint in sec_hints:
            try:
                if off + 8 > len(data): break
                sec_stride = struct.unpack_from("<I", data, off)[0]
                sec_size   = struct.unpack_from("<I", data, off + 4)[0]
                if sec_stride not in VALID_STRIDES_ANIM: break
                if sec_size == 0 or sec_size % sec_stride != 0: break
                sec_vc = sec_size // sec_stride
                off += 8
                # BBR2 static secondary buffers use the same authored UV0 offset.
                sec_uv_off = _find_uv_offset(data, off, sec_stride, sec_vc)
                sec_verts = []
                for i in range(sec_vc):
                    v_off = off + i * sec_stride
                    px, py, pz = struct.unpack_from("<3f", data, v_off)
                    nx_raw, ny_raw, nz_raw = struct.unpack_from("<3h", data, v_off + 12)
                    nx, ny, nz = nx_raw / 32767.0, ny_raw / 32767.0, nz_raw / 32767.0
                    u, v = struct.unpack_from("<2f", data, v_off + sec_uv_off) if sec_uv_off >= 12 else (0.0, 0.0)
                    
                    w = sec_uv_off + 8
                    if w + 8 > sec_stride:
                        w = sec_stride - 8
                    w0, w1, w2, w3 = struct.unpack_from("<4B", data, v_off + w)
                    j0, j1, j2, j3 = struct.unpack_from("<4B", data, v_off + w + 4)
                    w_sum = float(w0 + w1 + w2 + w3) or 1.0
                    weights = (w0 / w_sum, w1 / w_sum, w2 / w_sum, w3 / w_sum)
                    joints = (j0, j1, j2, j3)
                    
                    sec_verts.append((px, py, pz, nx, ny, nz, u, v, weights, joints))
                off += sec_size
                sec_idx_count = struct.unpack_from("<I", data, off)[0]; off += 4
                sec_raw = struct.unpack_from("<{0}H".format(sec_idx_count), data, off); off += sec_idx_count * 2
                tris = [(sec_raw[t], sec_raw[t+1], sec_raw[t+2])
                        for t in range(0, sec_idx_count - 2, 3)
                        if sec_raw[t] < sec_vc and sec_raw[t+1] < sec_vc and sec_raw[t+2] < sec_vc]
                nidx = hint["name_idx"]
                nm = (names[nidx] if nidx < len(names) else "mesh_{0}".format(nidx)) + "_LOD{0}".format(lod_idx)
                meshes.append({"name": nm, "vertices": sec_verts, "indices": tris})
            except:
                break

        return {"meshes": meshes}


    is_bbr2 = False
    off = 0
    num_names = struct.unpack_from("<I", data, 0)[0]
    if not (0 < num_names <= 128):
        is_bbr2 = True
        
        # Scan for all string blocks in the file
        import re
        matches = list(re.finditer(b'([\x01-\x20]\x00\x00\x00)([A-Za-z0-9_/.-]{4,}\x00)', data))
        
        best_off = 0
        best_num = 0
        for m in matches:
            test_off = m.start()
            try:
                nn = struct.unpack_from("<I", data, test_off)[0]
                if 0 < nn <= 128:
                    first_nul = data.index(b"\x00", test_off + 4)
                    s = data[test_off+4:first_nul].decode("ascii")
                    if "/" in s:
                        best_off = test_off
                        best_num = nn
            except:
                pass
                
        if best_off > 0:
            off = best_off
            num_names = best_num
    off += 4
    names, off = _read_cstrings(data, off, num_names)

    mesh_hdrs = []
    buf_hdrs = []
    buffers = {}
    shapes = []
    
    if is_bbr2:
        struct_count = struct.unpack_from("<I", data, off)[0]; off += 4
        # 16-byte block: last int is the tri count for single-mesh new-format files
        hdr16 = struct.unpack_from("<4I", data, off)
        tri_count_hint = hdr16[3]  # used as fallback if no mesh header
        off += 16

        for _ in range(struct_count):
            bmin = struct.unpack_from("<3f", data, off)
            bmax = struct.unpack_from("<3f", data, off + 12)
            i0, i1, i2, i3 = struct.unpack_from("<4I", data, off + 24)
            off += 40

            is_buf = False
            if i2 in (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64) and i3 > 0 and i3 % i2 == 0:
                if i2 < 100: is_buf = True

            if is_buf:
                buf_hdrs.append({"stride": i2, "size": i3})
            else:
                mesh_hdrs.append({
                    "bucket": i0, "flags1": i1, "vert_start": 0, "vert_end": 0,
                    "idx_start": i2, "idx_count": i3,
                    "bbox_min": bmin, "bbox_max": bmax,
                })

        if len(buf_hdrs) > 0:
            vert_stride = buf_hdrs[0]["stride"]
            vert_block_size = buf_hdrs[0]["size"]
        else:
            vert_stride = 28
            vert_block_size = 0

        total_verts = vert_block_size // vert_stride
        # Static BBR2 vehicles use multiple vertex layouts.  Detect UV0 from
        # the complete primary buffer instead of assuming byte 20.
        uv_off = _find_uv_offset(data, off, vert_stride, total_verts) if vert_block_size > 0 else -1
        color_off = _find_static_color_offset(data, off, vert_stride, total_verts, uv_off)
        verts = []
        for i in range(total_verts):
            px, py, pz = struct.unpack_from("<3f", data, off)
            u, v = struct.unpack_from("<2f", data, off + uv_off) if uv_off >= 12 else (0.0, 0.0)
            if color_off >= 0:
                cr, cg, cb, ca = struct.unpack_from("<4B", data, off + color_off)
                color = (cr/255.0, cg/255.0, cb/255.0, ca/255.0)
            else:
                color = (1.0, 1.0, 1.0, 1.0)
            verts.append((px, py, pz, 0.0, 1.0, 0.0, u, v, color))
            off += vert_stride

        try:
            total_indices = struct.unpack_from("<I", data, off)[0]
            off += 4
            raw_indices = struct.unpack_from("<{0}H".format(total_indices), data, off)
            off += total_indices * 2
        except:
            total_indices = 0
            raw_indices = ()

        buffers[0] = {"verts": verts, "indices": raw_indices}

        while off + 8 <= len(data):
            try:
                sec_stride = struct.unpack_from("<I", data, off)[0]
                sec_size   = struct.unpack_from("<I", data, off + 4)[0]
                if sec_stride not in VALID_STRIDES_ANIM: break
                if sec_size == 0 or sec_size % sec_stride != 0: break
                off += 8
                sec_vc = sec_size // sec_stride
                sec_uv_off = _find_uv_offset(data, off, sec_stride, sec_vc)
                sec_color_off = _find_static_color_offset(data, off, sec_stride, sec_vc, sec_uv_off)
                sec_verts = []
                for i in range(sec_vc):
                    px, py, pz = struct.unpack_from("<3f", data, off + i * sec_stride)
                    u, v = struct.unpack_from("<2f", data, off + i * sec_stride + sec_uv_off) if sec_uv_off >= 12 else (0.0, 0.0)
                    if sec_color_off >= 0:
                        cr, cg, cb, ca = struct.unpack_from("<4B", data, off + i * sec_stride + sec_color_off)
                        color = (cr/255.0, cg/255.0, cb/255.0, ca/255.0)
                    else:
                        color = (1.0, 1.0, 1.0, 1.0)
                    sec_verts.append((px, py, pz, 0.0, 1.0, 0.0, u, v, color))
                off += sec_size
                sec_idx_count = struct.unpack_from("<I", data, off)[0]
                off += 4
                sec_raw = struct.unpack_from("<{0}H".format(sec_idx_count), data, off)
                off += sec_idx_count * 2
                
                buf_idx = len(buffers)
                buffers[buf_idx] = {"verts": sec_verts, "indices": sec_raw}
            except:
                break

        used_name_indices = {h["flags1"] for h in mesh_hdrs}
        avail_names = [i for i in range(len(names)) if i not in used_name_indices]

        final_mesh_hdrs = []
        for b_idx in range(len(buffers)):
            b_hdrs = [h for h in mesh_hdrs if h["bucket"] == b_idx and h["idx_start"] + h["idx_count"] * 3 <= len(buffers[b_idx]["indices"])]
            b_hdrs.sort(key=lambda x: x["idx_start"])
            
            cur = 0
            for hdr in b_hdrs:
                s, cnt = hdr["idx_start"], hdr["idx_count"] * 3
                if s > cur:
                    gidx = avail_names.pop(0) if avail_names else 0
                    final_mesh_hdrs.append(dict(hdr, idx_start=cur, idx_count=(s-cur)//3, flags1=gidx))
                final_mesh_hdrs.append(hdr)
                cur = s + cnt
                
            if cur < len(buffers[b_idx]["indices"]):
                gidx = avail_names.pop(0) if avail_names else 0
                final_mesh_hdrs.append({"bucket":b_idx, "flags1":gidx, "vert_start":0, "vert_end":0,
                                        "idx_start":cur, "idx_count":(len(buffers[b_idx]["indices"])-cur)//3,
                                        "bbox_min":(0,0,0), "bbox_max":(0,0,0)})

        if len(final_mesh_hdrs) == 0 and len(buffers[0]["indices"]) > 0:
            actual_tri_count = tri_count_hint if tri_count_hint > 0 and tri_count_hint * 3 <= len(buffers[0]["indices"]) else len(buffers[0]["indices"]) // 3
            final_mesh_hdrs.append({
                "bucket": 0, "flags1": 0, "vert_start": 0, "vert_end": 0,
                "idx_start": 0, "idx_count": actual_tri_count,
                "bbox_min": (0,0,0), "bbox_max": (0,0,0)
            })

        mesh_hdrs = final_mesh_hdrs
    else:
        num_shapes = struct.unpack_from("<I", data, off)[0]; off += 4
        shapes, off = _read_cstrings(data, off, num_shapes)
        mesh_count = struct.unpack_from("<I", data, off)[0]; off += 4

        for _ in range(mesh_count):
            flags0, flags1, v_start, v_end, idx_start, idx_count = struct.unpack_from("<6i", data, off)
            bmin = struct.unpack_from("<3f", data, off + 24)
            bmax = struct.unpack_from("<3f", data, off + 40)
            mesh_hdrs.append({
                "bucket": flags0, "flags1": flags1, "vert_start": v_start, "vert_end": v_end,
                "idx_start": idx_start, "idx_count": idx_count,
                "bbox_min": bmin, "bbox_max": bmax,
            })
            off += 56

        off += 32
        num_bufs = struct.unpack_from("<I", data, off)[0]; off += 4
        
        for b in range(num_bufs):
            vert_stride, vert_block_size = struct.unpack_from("<2i", data, off)
            off += 8
            buf_start = off
            total_verts = vert_block_size // vert_stride
            
            verts = []
            for i in range(total_verts):
                px, py, pz = struct.unpack_from("<3f", data, off)
                nx_raw, ny_raw, nz_raw = struct.unpack_from("<3h", data, off + 12)
                nx, ny, nz = nx_raw / 32767.0, ny_raw / 32767.0, nz_raw / 32767.0
                u, v = struct.unpack_from("<2f", data, off + 20)
                if vert_stride >= 32:
                    cr, cg, cb, ca = struct.unpack_from("<4B", data, off + 28)
                    color = (cr/255.0, cg/255.0, cb/255.0, ca/255.0)
                else:
                    color = (1.0, 1.0, 1.0, 1.0)
                verts.append((px, py, pz, nx, ny, nz, u, v, color))
                off += vert_stride
            
            total_indices = struct.unpack_from("<H", data, off)[0]
            off += 2
            raw_indices = struct.unpack_from("<{0}H".format(total_indices), data, off)
            off += total_indices * 2
            
            buf_size = off - buf_start
            if buf_size % 4 != 0:
                off += 4 - (buf_size % 4)
                
            buffers[b] = {
                "verts": verts,
                "indices": raw_indices
            }

    for i, hdr in enumerate(mesh_hdrs):
        nidx = hdr["flags1"]
        name = names[nidx] if nidx < len(names) else "mesh_{0}".format(nidx)
        buf_idx = hdr["bucket"]
        if buf_idx not in buffers:
            buf_idx = 0
            
        buf = buffers[buf_idx]
        v_start = hdr["vert_start"]
        v_end   = hdr["vert_end"]
        if v_end == 0: v_end = len(buf["verts"])
        mesh_verts = list(buf["verts"][v_start:v_end])

        i_start = hdr["idx_start"]
        i_count = hdr["idx_count"] * 3
        mesh_raw_idx = buf["indices"][i_start : i_start + i_count]

        tris = []
        if is_bbr2:
            for t in range(0, len(mesh_raw_idx), 3):
                if t + 2 >= len(mesh_raw_idx): break
                a = mesh_raw_idx[t] - v_start
                b = mesh_raw_idx[t+1] - v_start
                c = mesh_raw_idx[t+2] - v_start
                if 0 <= a < len(mesh_verts) and 0 <= b < len(mesh_verts) and 0 <= c < len(mesh_verts):
                    tris.append((a, b, c))
        else:
            for t in range(len(mesh_raw_idx) - 2):
                raw_a, raw_b, raw_c = mesh_raw_idx[t], mesh_raw_idx[t+1], mesh_raw_idx[t+2]
                if raw_a == 65535 or raw_b == 65535 or raw_c == 65535:
                    continue
                if t % 2 == 0:
                    a, b, c = raw_a - v_start, raw_b - v_start, raw_c - v_start
                else:
                    a, b, c = raw_b - v_start, raw_a - v_start, raw_c - v_start
                if a == b or b == c or a == c:
                    continue
                if 0 <= a < len(mesh_verts) and 0 <= b < len(mesh_verts) and 0 <= c < len(mesh_verts):
                    tris.append((a, b, c))

        meshes.append({
            "name": name,
            "vertices": mesh_verts,
            "indices": tris,
            "vertex_stride": int(vert_stride) if 'vert_stride' in locals() else 0,
            "uv_offset": int(uv_off) if 'uv_off' in locals() else 0,
            "skin_offset": -1,
        })

    return {"meshes": meshes}





# Keep the original parser as a fallback for non-animated/static formats.
_legacy_parse_model = parse_model


def _detect_skin_offset(data: bytes, start: int, stride: int,
                        vertex_count: int, bone_count: int) -> int:
    """
    Locate the 8-byte [4 weights][4 joints] block from the data itself.

    Confirmed BBR layouts include:
      stride 32 -> skin @ 24
      stride 36 -> skin @ 28 (normal character meshes)
      stride 36 -> skin @ 24 (Cyber Screens / Cockpit)

    The key invariant is that normalized uint8 weights overwhelmingly sum to
    255 and meaningful joint indices fit inside the model skeleton.
    """
    if vertex_count <= 0 or stride < 20:
        return max(0, stride - 8)

    best = None
    for off in range(12, stride - 7):
        exact255 = 0
        valid = 0
        invalid = 0
        nonzero = 0

        for vi in range(vertex_count):
            p = start + vi * stride + off
            if p + 8 > len(data):
                break
            w = data[p:p+4]
            j = data[p+4:p+8]
            sw = int(sum(w))
            if sw > 0:
                nonzero += 1
            if sw == 255:
                exact255 += 1

            if sw > 0:
                if bone_count <= 0 or all(int(x) < bone_count for x in j):
                    valid += 1
                else:
                    invalid += 1

        score = exact255 * 10 + valid * 2 - invalid * 20
        candidate = (score, exact255, valid, -invalid, -off, off)
        if best is None or candidate > best:
            best = candidate

    if best is not None:
        _, exact255, valid, _, _, off = best
        # Every supplied animated character buffer passes these thresholds.
        if exact255 >= max(1, int(vertex_count * 0.75)) and \
           valid >= max(1, int(vertex_count * 0.90)):
            return int(off)

    return max(0, stride - 8)


def _parse_animated_model_v08(data: bytes, bone_count: int):
    """
    Correct BBR2 animated-model buffer parser.

    Header records contain TWO different kinds of records:
      - one primary vertex-buffer descriptor
      - mesh descriptors whose first integer is the buffer/bucket index

    Additional vertex buffers are serialized after the previous buffer's
    index data as:
        uint32 stride
        uint32 buffer_size
        vertex bytes
        uint32 index_count
        uint16 indices[index_count]

    v0.7 incorrectly treated every non-zero bucket mesh descriptor as though
    it represented a new buffer. Cyber exposed the bug:
      bucket 0: Boots + main body
      bucket 1: Gadgets + Lens
      bucket 2: Shirt
    """
    matches = list(re.finditer(
        b'([\x01-\x20]\x00\x00\x00)([A-Za-z0-9_/.-]{4,}\x00)',
        data
    ))

    best = None
    best_score = -1

    for m in matches:
        off = m.start()
        try:
            num_names = struct.unpack_from("<I", data, off)[0]
            if not (1 <= num_names <= 128):
                continue

            p = off + 4
            names = []
            for _ in range(num_names):
                end = data.index(b"\x00", p)
                names.append(data[p:end].decode("ascii", "ignore"))
                p = end + 1

            if not any("/" in n for n in names):
                continue

            struct_count = struct.unpack_from("<I", data, p)[0]
            if not (1 <= struct_count <= 64):
                continue
            p += 4

            hdr16 = struct.unpack_from("<4I", data, p)
            p += 16

            desc = []
            for _ in range(struct_count):
                if p + 40 > len(data):
                    raise ValueError("Truncated animated model descriptors")
                i0, i1, i2, i3 = struct.unpack_from("<4I", data, p + 24)
                desc.append((i0, i1, i2, i3))
                p += 40

            buffer_descs = [
                x for x in desc
                if x[2] in VALID_STRIDES_ANIM
                and x[3] > 0
                and x[2] < 100
                and x[3] % x[2] == 0
            ]
            if len(buffer_descs) != 1:
                continue

            primary = buffer_descs[0]
            score = primary[3] // primary[2]
            if score > best_score:
                best_score = score
                best = (names, hdr16, desc, primary, p)
        except Exception:
            continue

    if best is None:
        return {"meshes": []}

    names, hdr16, desc, primary_desc, p = best
    mesh_descs = [x for x in desc if x != primary_desc]

    max_bucket = max([int(x[0]) for x in mesh_descs] + [0])
    buffers = []

    for bucket in range(max_bucket + 1):
        if bucket == 0:
            stride = int(primary_desc[2])
            size = int(primary_desc[3])
        else:
            if p + 8 > len(data):
                raise ValueError("Missing secondary animated vertex-buffer header")
            stride, size = struct.unpack_from("<II", data, p)
            p += 8
            stride = int(stride)
            size = int(size)
            if stride not in VALID_STRIDES_ANIM or size <= 0 or size % stride:
                raise ValueError(
                    "Invalid secondary animated buffer %d: stride=%d size=%d"
                    % (bucket, stride, size)
                )

        vertex_count = size // stride
        start = p
        if start + size > len(data):
            raise ValueError("Animated vertex buffer exceeds file size")

        skin_off = _detect_skin_offset(
            data, start, stride, vertex_count, bone_count
        )
        # Animated BBR2 also has more than one 36-byte layout.  Characters such
        # as Tribal use UV0@20 with skin@28, while several motorcycle/vehicle
        # paint buffers use UV0@16, a 4-byte field at 24, then skin@28.
        # Detect the actual float pair and require it to end before skin data.
        uv_off = _find_uv_offset(data, start, stride, vertex_count)
        if uv_off < 12 or uv_off + 8 > skin_off:
            uv_off = skin_off - 8
        if uv_off < 12 or uv_off + 8 > stride:
            uv_off = 20 if stride >= 28 else max(12, stride - 8)

        verts = []
        for vi in range(vertex_count):
            vo = start + vi * stride
            px, py, pz = struct.unpack_from("<3f", data, vo)

            # VuAnimatedModelAsset uses two packed normal layouts in the
            # supplied builds. The native/mobile 32-byte skinned layout stores
            # s8[4] at +12; wider PC/tangent layouts use 3x int16 SNORM.
            if stride == 32 and vo + 16 <= len(data):
                nxr, nyr, nzr, _nw = struct.unpack_from("<4b", data, vo + 12)
                nx, ny, nz = (nxr / 127.0, nyr / 127.0, nzr / 127.0)
            elif vo + 18 <= len(data):
                nxr, nyr, nzr = struct.unpack_from("<3h", data, vo + 12)
                nx, ny, nz = (
                    nxr / 32767.0,
                    nyr / 32767.0,
                    nzr / 32767.0,
                )
            else:
                nx, ny, nz = 0.0, 1.0, 0.0

            try:
                u, v = struct.unpack_from("<2f", data, vo + uv_off)
            except Exception:
                u, v = 0.0, 0.0

            w0, w1, w2, w3 = struct.unpack_from(
                "<4B", data, vo + skin_off
            )
            j0, j1, j2, j3 = struct.unpack_from(
                "<4B", data, vo + skin_off + 4
            )

            wsum = float(w0 + w1 + w2 + w3)
            if wsum <= 0.0:
                weights = (1.0, 0.0, 0.0, 0.0)
                joints = (0, 0, 0, 0)
            else:
                weights = (
                    w0 / wsum, w1 / wsum,
                    w2 / wsum, w3 / wsum
                )
                joints = (j0, j1, j2, j3)

            verts.append((
                px, py, pz, nx, ny, nz, u, v,
                weights, joints
            ))

        p += size

        if p + 4 > len(data):
            raise ValueError("Missing animated index count")
        index_count = struct.unpack_from("<I", data, p)[0]
        p += 4

        if index_count > 10000000 or p + index_count * 2 > len(data):
            raise ValueError("Invalid animated index buffer size")

        raw_indices = struct.unpack_from(
            "<%dH" % index_count, data, p
        )
        p += index_count * 2

        buffers.append({
            "bucket": bucket,
            "stride": stride,
            "vertex_count": vertex_count,
            "uv_offset": uv_off,
            "skin_offset": skin_off,
            "vertices": verts,
            "indices": raw_indices,
        })

    # Explicit material/mesh names already used by descriptors. Any material
    # name not used explicitly belongs to an index-range gap.
    used_names = {
        int(d[1]) for d in mesh_descs
        if 0 <= int(d[1]) < len(names)
    }
    available_names = [
        i for i in range(len(names)) if i not in used_names
    ]

    meshes = []

    for bucket, buf in enumerate(buffers):
        total_indices = len(buf["indices"])
        headers = []

        for d in mesh_descs:
            b, name_idx, idx_start, tri_count = [int(x) for x in d]
            if b != bucket:
                continue
            if not (0 <= name_idx < len(names)):
                continue
            if idx_start + tri_count * 3 > total_indices:
                continue
            headers.append({
                "name_idx": name_idx,
                "idx_start": idx_start,
                "tri_count": tri_count,
                "implicit": False,
            })

        headers.sort(key=lambda h: h["idx_start"])

        filled = []
        cur = 0

        for h in headers:
            if h["idx_start"] > cur:
                nidx = available_names.pop(0) if available_names else 0
                filled.append({
                    "name_idx": nidx,
                    "idx_start": cur,
                    "tri_count": (h["idx_start"] - cur) // 3,
                    "implicit": True,
                })

            filled.append(h)
            cur = h["idx_start"] + h["tri_count"] * 3

        if cur < total_indices:
            nidx = (
                available_names.pop(0)
                if available_names
                else (headers[-1]["name_idx"] if headers else min(bucket, len(names)-1))
            )
            filled.append({
                "name_idx": nidx,
                "idx_start": cur,
                "tri_count": (total_indices - cur) // 3,
                "implicit": True,
            })

        if not filled and total_indices:
            nidx = available_names.pop(0) if available_names else min(bucket, len(names)-1)
            filled = [{
                "name_idx": nidx,
                "idx_start": 0,
                "tri_count": total_indices // 3,
                "implicit": True,
            }]

        for h in filled:
            count = h["tri_count"] * 3
            seg = buf["indices"][h["idx_start"]:h["idx_start"] + count]
            tris = []
            for t in range(0, len(seg) - 2, 3):
                a, b, c = int(seg[t]), int(seg[t+1]), int(seg[t+2])
                if a < buf["vertex_count"] and b < buf["vertex_count"] and c < buf["vertex_count"]:
                    tris.append((a, b, c))

            if not tris:
                continue

            name_idx = h["name_idx"]
            mesh_name = (
                names[name_idx]
                if 0 <= name_idx < len(names)
                else "mesh_%d" % name_idx
            ) + "_LOD0"

            meshes.append({
                "name": mesh_name,
                "vertices": buf["vertices"],
                "indices": tris,
                "buffer_index": bucket,
                "vertex_stride": buf["stride"],
                "uv_offset": buf["uv_offset"],
                "skin_offset": buf["skin_offset"],
            })

    return {"meshes": meshes}


def parse_model(data, bone_count=0):
    is_animated = False
    try:
        is_animated = (
            len(data) > 24
            and (
                b"Root\x00" in data[8:64]
                or b"root\x00" in data[8:64]
            )
        )
    except Exception:
        pass

    if is_animated:
        return _parse_animated_model_v08(data, int(bone_count or 0))
    return _legacy_parse_model(data)


def parse_model_asset(data: bytes, name="Model", source_path="") -> ModelData:
    bones = parse_skeleton(data)
    raw = parse_model(data, len(bones))
    meshes = []
    for m in raw.get("meshes", []):
        vs = m.get("vertices", [])
        tris = m.get("indices", [])
        if not vs or not tris:
            continue
        pos = np.asarray([[v[0],v[1],v[2]] for v in vs], dtype=np.float32)
        nrm = np.asarray([[v[3],v[4],v[5]] for v in vs], dtype=np.float32)
        uv = np.asarray([[v[6],v[7]] for v in vs], dtype=np.float32)
        idx = np.asarray(tris, dtype=np.uint32)
        if normals_need_geometry_rebuild(nrm):
            nrm = recompute_mesh_normals(pos, idx)
        else:
            # Normalize authored normals before skinning/lighting.
            _mag = np.linalg.norm(nrm, axis=1)
            _ok = _mag > 1e-12
            nrm[_ok] /= _mag[_ok, None]
        colors = None
        if not bones and len(vs[0]) > 8 and isinstance(vs[0][8], (tuple, list, np.ndarray)) and len(vs[0][8]) == 4:
            colors = np.asarray([v[8] for v in vs], dtype=np.float32)
        weights = joints = None
        if bones and len(vs[0]) > 9:
            weights = np.zeros((len(vs),4), dtype=np.float32)
            joints = np.zeros((len(vs),4), dtype=np.int32)
            for vi,v in enumerate(vs):
                w = list(v[8]); j = list(v[9])
                valid=[]
                for ji,wi in zip(j,w):
                    if wi <= 1e-6: continue
                    if not bones or 0 <= ji < len(bones): valid.append((int(ji),float(wi)))
                if not valid:
                    valid=[(0,1.0)]
                sw=sum(x[1] for x in valid) or 1.0
                for k,(ji,wi) in enumerate(valid[:4]):
                    joints[vi,k]=ji; weights[vi,k]=wi/sw
        meshes.append(MeshData(
            m.get("name","mesh"), pos, nrm, uv, idx, colors, weights, joints,
            buffer_index=int(m.get("buffer_index", 0)),
            vertex_stride=int(m.get("vertex_stride", 0)),
            uv_offset=int(m.get("uv_offset", 0)),
            skin_offset=int(m.get("skin_offset", -1)),
        ))
    return ModelData(name=name, meshes=meshes, bones=bones, source_path=source_path)


def parse_animation(data: bytes, name="Animation", fps=30.0) -> AnimationData:
    """Decode the exact BBR2 ``VuAnimationAsset`` codec.

    Layout, verified against the supplied codec and every animation in the
    target XAPK:
      int32 bone_count, int32 frame_count
      frame-major/bone-major keys, 32 bytes each:
        3f translation + 4h quaternion(x,y,z,w) + 3f scale
      uint8 loopFlag at exact EOF

    Quaternion components follow the native VuAnimationTransform decoder: int16 / 32767.0.  Sampling
    normalizes/interpolates them as needed; the parser itself does not mutate
    the authored keys.
    """
    if len(data) < 9:
        raise ValueError("Animation file is too small.")
    nb,nf = struct.unpack_from("<ii",data,0)
    if not (1 <= nb <= 512 and 1 <= nf <= 100000):
        raise ValueError("Invalid Vector animation header: bones=%d frames=%d" % (nb,nf))
    key_size=32
    payload_end = 8 + nb*nf*key_size
    expected = payload_end + 1
    if len(data) != expected:
        raise ValueError(
            "VuAnimationAsset size mismatch: expected exactly %d bytes "
            "(8 + %d*%d*32 + 1), got %d" % (expected,nb,nf,len(data))
        )
    t=np.zeros((nf,nb,3),np.float32)
    q=np.zeros((nf,nb,4),np.float32)
    s=np.zeros((nf,nb,3),np.float32)
    for fi in range(nf):
        for bi in range(nb):
            o=8+((fi*nb+bi)*key_size)
            t[fi,bi]=struct.unpack_from("<3f",data,o)
            q[fi,bi]=np.asarray(struct.unpack_from("<4h",data,o+12),dtype=np.float32)/32767.0
            s[fi,bi]=struct.unpack_from("<3f",data,o+20)
    loop_flag=int(data[payload_end])
    return AnimationData(
        name=name, bone_count=nb, frame_count=nf, fps=float(fps),
        translations=t, rotations=q, scales=s, loop_flag=loop_flag,
        trailing=b"", key_size=key_size, exact_eof=True
    )


def parse_animation_embedded(data: bytes, name="Animation", fps=30.0) -> AnimationData:
    """Decode a 32-byte animation block embedded inside another asset.

    ``VuDrivingAnimationSetAsset`` stores Turn/Lean control blocks with the same
    frame/bone key layout as ``VuAnimationAsset`` but without the final standalone
    asset loopFlag byte.  This helper keeps the external asset codec strict while
    still decoding those embedded controls correctly.
    """
    if len(data) < 8:
        raise ValueError("Embedded animation block is too small.")
    nb,nf=struct.unpack_from("<ii",data,0)
    if not (1 <= nb <= 512 and 1 <= nf <= 100000):
        raise ValueError("Invalid embedded animation header: bones=%d frames=%d" % (nb,nf))
    key_size=32
    expected=8+nb*nf*key_size
    if len(data) != expected:
        raise ValueError("Embedded animation size mismatch: expected exactly %d bytes, got %d" % (expected,len(data)))
    t=np.zeros((nf,nb,3),np.float32)
    q=np.zeros((nf,nb,4),np.float32)
    sc=np.zeros((nf,nb,3),np.float32)
    for fi in range(nf):
        for bi in range(nb):
            o=8+((fi*nb+bi)*key_size)
            t[fi,bi]=struct.unpack_from("<3f",data,o)
            q[fi,bi]=np.asarray(struct.unpack_from("<4h",data,o+12),dtype=np.float32)/32767.0
            sc[fi,bi]=struct.unpack_from("<3f",data,o+20)
    return AnimationData(
        name=name,bone_count=nb,frame_count=nf,fps=float(fps),
        translations=t,rotations=q,scales=sc,loop_flag=-1,
        trailing=b"",key_size=key_size,exact_eof=False
    )


def local_matrices_from_bones(bones: List[Bone]):
    return [trs_matrix(b.translation,b.rotation,b.scale) for b in bones]


def global_matrices(local_mats, bones: List[Bone]):
    out=[]
    for i,M in enumerate(local_mats):
        p=bones[i].parent
        out.append(out[p] @ M if 0 <= p < i else M.copy())
    return out


def bind_globals(model: ModelData):
    if not model.bones: return []
    # Native VuSkeleton::load deserializes the stored pose and then
    # buildDerivedData() calls transformModelPoseToLocalPose. Therefore the
    # serialized skeleton pose is model-space, despite the parent table being
    # present. Animation tracks are local-to-parent. Do not parent-compose the
    # serialized bind records here.
    return local_matrices_from_bones(model.bones)


def _quat_slerp(q0, q1, t):
    q0=np.asarray(q0,dtype=np.float64); q1=np.asarray(q1,dtype=np.float64)
    n0=float(np.linalg.norm(q0)); n1=float(np.linalg.norm(q1))
    if n0 < 1e-10: q0=np.asarray([0,0,0,1],dtype=np.float64)
    else: q0/=n0
    if n1 < 1e-10: q1=np.asarray([0,0,0,1],dtype=np.float64)
    else: q1/=n1
    dot=float(np.dot(q0,q1))
    if dot < 0.0:
        q1=-q1; dot=-dot
    dot=max(-1.0,min(1.0,dot))
    if dot > 0.9995:
        q=q0+(q1-q0)*float(t)
        n=float(np.linalg.norm(q))
        return (q/n if n>1e-10 else np.asarray([0,0,0,1],dtype=np.float64)).astype(np.float32)
    theta=math.acos(dot)
    st=math.sin(theta)
    a=math.sin((1.0-float(t))*theta)/st
    b=math.sin(float(t)*theta)/st
    return (q0*a+q1*b).astype(np.float32)


def _animation_frame_pair(anim: AnimationData, frame: float, loop_override=None):
    """Return (f0, f1, frac) while respecting the asset loop flag.

    ``loop_override`` is reserved for explicit preview controls.  Runtime/control
    sampling defaults to the exact ``loopFlag`` stored in the asset.
    """
    fc=max(1,int(anim.frame_count))
    looping=bool(anim.looping if loop_override is None else loop_override)
    if fc<=1:
        return 0,0,0.0
    raw=float(frame)
    if looping:
        ff=raw%float(fc)
        base=math.floor(ff)
        f0=int(base)%fc
        frac=ff-base
        f1=(f0+1)%fc
        return f0,f1,frac
    ff=max(0.0,min(raw,float(fc-1)))
    base=math.floor(ff)
    f0=int(base)
    frac=ff-base
    if f0>=fc-1:
        return fc-1,fc-1,0.0
    return f0,f0+1,frac


def animation_globals(model: ModelData, anim: Optional[AnimationData], frame, loop_override=None):
    """Build animated global matrices, including sub-frame interpolation.

    v0.34 keeps Vector's authored 30 FPS keys but samples them at the viewport's
    ~60 Hz refresh. Translation/scale are linearly interpolated and quaternion
    rotation uses shortest-path SLERP, removing the visible 30 FPS stepping that
    made otherwise-correct character animation look mechanical.
    """
    if not model.bones: return []
    if anim is None:
        return bind_globals(model)
    if anim.bone_count != len(model.bones):
        raise ValueError("Animation/model bone-count mismatch (%d vs %d)." % (anim.bone_count,len(model.bones)))
    f0,f1,frac=_animation_frame_pair(anim,frame,loop_override)
    local=[]
    for i in range(anim.bone_count):
        t0=np.asarray(anim.translations[f0,i],dtype=np.float32).copy()
        q0=np.asarray(anim.rotations[f0,i],dtype=np.float32).copy()
        s0=np.asarray(anim.scales[f0,i],dtype=np.float32).copy()
        t1=np.asarray(anim.translations[f1,i],dtype=np.float32).copy()
        q1=np.asarray(anim.rotations[f1,i],dtype=np.float32).copy()
        s1=np.asarray(anim.scales[f1,i],dtype=np.float32).copy()

        # Some companion tracks use NaN vectors as an inactive/hidden sentinel.
        # Treat visibility changes as discrete key events instead of interpolating
        # through invalid transforms.
        valid0=np.isfinite(t0).all() and np.isfinite(s0).all()
        valid1=np.isfinite(t1).all() and np.isfinite(s1).all()
        if not valid0 or not valid1:
            if valid0:
                t=t0; sc=s0; q=q0
            else:
                t=np.asarray(model.bones[i].translation,dtype=np.float32).copy()
                sc=np.zeros(3,dtype=np.float32)
                q=np.asarray([0,0,0,1],dtype=np.float32)
        else:
            t=(t0*(1.0-frac)+t1*frac).astype(np.float32)
            sc=(s0*(1.0-frac)+s1*frac).astype(np.float32)
            if not np.isfinite(q0).all(): q0=np.asarray([0,0,0,1],dtype=np.float32)
            if not np.isfinite(q1).all(): q1=np.asarray([0,0,0,1],dtype=np.float32)
            q=_quat_slerp(q0,q1,frac)
        if not np.isfinite(q).all():
            q=np.asarray([0,0,0,1],dtype=np.float32)
        local.append(trs_matrix(t,q,sc))
    return global_matrices(local,model.bones)



def animation_bone_visibility(model: ModelData, anim: Optional[AnimationData], frame, loop_override=None):
    """Return per-bone visibility for animation-driven hide/show tracks.

    A few BBR companion rigs (notably Cyber's lounge screen rig) serialize
    inactive bones with NaN transforms or an exact zero scale.  The ordinary
    skinning path converts those transforms into collapsed matrices.  If a mesh
    is still submitted, triangles can bridge between collapsed and live
    vertices and appear as huge rings/panels.  Treat the inactive transform as
    a visibility state so the viewport can skip meshes dominated by hidden
    bones instead of drawing the collapsed geometry.
    """
    if not model.bones:
        return []
    if anim is None:
        return [True] * len(model.bones)
    if anim.bone_count != len(model.bones):
        return [True] * len(model.bones)

    f0, f1, frac = _animation_frame_pair(anim, frame, loop_override)
    out = []
    for i in range(anim.bone_count):
        t0 = np.asarray(anim.translations[f0, i], dtype=np.float32)
        s0 = np.asarray(anim.scales[f0, i], dtype=np.float32)
        t1 = np.asarray(anim.translations[f1, i], dtype=np.float32)
        s1 = np.asarray(anim.scales[f1, i], dtype=np.float32)

        valid0 = np.isfinite(t0).all() and np.isfinite(s0).all()
        valid1 = np.isfinite(t1).all() and np.isfinite(s1).all()

        # Match animation_globals()' discrete NaN-sentinel behavior: the
        # current key owns visibility until the next key is reached.
        if not valid0:
            out.append(False)
            continue

        if valid1:
            sc = (s0 * (1.0 - frac) + s1 * frac).astype(np.float32)
        else:
            sc = s0

        # Exact/near-exact zero scale is also used as a hide key on some prop
        # bones.  Ordinary small squash/stretch remains visible.
        visible = bool(np.isfinite(sc).all() and float(np.max(np.abs(sc))) > 1.0e-5)
        out.append(visible)
    return out


def bind_local_matrices(model: ModelData):
    """Convert BBR model-space bind records into local-to-parent bind matrices.

    Animated-model skeleton records in BBR are stored in model/global space.
    Vehicle suspension steering clips, however, are authored as *delta* local
    transforms: their zero translations and identity scales mean "no change",
    not "move the bone to the origin".  Deriving the bind locals lets those
    delta clips be evaluated without collapsing the suspension skeleton.
    """
    if not model.bones:
        return []
    bg=bind_globals(model)
    out=[]
    for i,G in enumerate(bg):
        parent=model.bones[i].parent
        if 0 <= parent < i:
            try:
                out.append((np.linalg.inv(bg[parent]) @ G).astype(np.float32))
            except np.linalg.LinAlgError:
                out.append(G.copy())
        else:
            out.append(G.copy())
    return out


def animation_globals_additive(model: ModelData, anim: Optional[AnimationData], frame, loop_override=None):
    """Evaluate a BBR delta animation on top of the model's bind pose.

    This path is intentionally separate from :func:`animation_globals`.
    Character driving clips contain full local transforms, while vehicle
    suspension Steering Animation assets contain rotation/translation deltas.
    Treating the latter as absolute transforms was the cause of the exploded
    motorcycle/Pumpkin suspension seen in v0.51.
    """
    if not model.bones:
        return []
    if anim is None:
        return bind_globals(model)
    if anim.bone_count != len(model.bones):
        raise ValueError("Animation/model bone-count mismatch (%d vs %d)." % (anim.bone_count,len(model.bones)))
    f0,f1,frac=_animation_frame_pair(anim,frame,loop_override)
    bind_local=bind_local_matrices(model)
    local=[]
    for i in range(anim.bone_count):
        t0=np.asarray(anim.translations[f0,i],dtype=np.float32)
        t1=np.asarray(anim.translations[f1,i],dtype=np.float32)
        q0=np.asarray(anim.rotations[f0,i],dtype=np.float32)
        q1=np.asarray(anim.rotations[f1,i],dtype=np.float32)
        s0=np.asarray(anim.scales[f0,i],dtype=np.float32)
        s1=np.asarray(anim.scales[f1,i],dtype=np.float32)
        if not np.isfinite(t0).all(): t0=np.zeros(3,dtype=np.float32)
        if not np.isfinite(t1).all(): t1=t0
        if not np.isfinite(q0).all(): q0=np.asarray([0,0,0,1],dtype=np.float32)
        if not np.isfinite(q1).all(): q1=q0
        if not np.isfinite(s0).all(): s0=np.ones(3,dtype=np.float32)
        if not np.isfinite(s1).all(): s1=s0
        t=(t0*(1.0-frac)+t1*frac).astype(np.float32)
        sc=(s0*(1.0-frac)+s1*frac).astype(np.float32)
        q=_quat_slerp(q0,q1,frac)
        delta=trs_matrix(t,q,sc)
        local.append((bind_local[i] @ delta).astype(np.float32))
    return global_matrices(local,model.bones)


def skin_mesh(mesh: MeshData, model: ModelData, current_globals):
    if mesh.weights is None or mesh.joints is None or not model.bones:
        return mesh.positions
    bind=bind_globals(model)
    if not bind or len(current_globals)!=len(bind):
        return mesh.positions
    skin=[]
    for cur,b in zip(current_globals,bind):
        try: skin.append(cur @ np.linalg.inv(b))
        except np.linalg.LinAlgError: skin.append(np.eye(4,dtype=np.float32))
    p4=np.concatenate([mesh.positions,np.ones((len(mesh.positions),1),np.float32)],axis=1)
    out=np.zeros((len(mesh.positions),3),np.float32)
    for k in range(4):
        w=mesh.weights[:,k]
        active=np.nonzero(w>1e-7)[0]
        for vi in active:
            ji=int(mesh.joints[vi,k])
            if 0 <= ji < len(skin):
                out[vi] += (skin[ji] @ p4[vi])[:3] * w[vi]
    return out


def skin_mesh_and_normals(mesh: MeshData, model: ModelData, current_globals, bone_visible=None):
    """Skin positions *and* normals with the same bone weights.

    Older Studio builds animated vertex positions but kept bind-pose normals.
    The geometry was real, but lighting across elbows, shoulders, faces and
    other moving parts could look faceted/blocky because normals no longer
    matched the deformed surface.  v0.33 transforms normals through the skin
    matrices as well and renormalizes them for smooth animated shading.
    """
    base_n = np.asarray(mesh.normals, dtype=np.float32)
    if mesh.weights is None or mesh.joints is None or not model.bones:
        n = base_n.copy()
        if len(n):
            mag = np.linalg.norm(n, axis=1)
            ok = mag > 1e-8
            n[ok] /= mag[ok, None]
        return mesh.positions, n

    bind = bind_globals(model)
    if not bind or len(current_globals) != len(bind):
        return mesh.positions, base_n

    skin = []
    normal_mats = []
    for cur, b in zip(current_globals, bind):
        try:
            sm = np.asarray(cur @ np.linalg.inv(b), dtype=np.float32)
        except np.linalg.LinAlgError:
            sm = np.eye(4, dtype=np.float32)
        skin.append(sm)
        try:
            nm = np.linalg.inv(sm[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            nm = sm[:3, :3].astype(np.float32)
        normal_mats.append(nm)

    p4 = np.concatenate([
        mesh.positions, np.ones((len(mesh.positions), 1), np.float32)
    ], axis=1)
    out_p = np.zeros((len(mesh.positions), 3), np.float32)
    out_n = np.zeros((len(mesh.positions), 3), np.float32)
    used_weight = np.zeros(len(mesh.positions), dtype=np.float32)
    authored_weight = np.zeros(len(mesh.positions), dtype=np.float32)

    # Track authored vs actually-used weights separately. Hidden companion bones
    # are skipped; their vertices will be culled by the viewport's triangle
    # visibility pass. More importantly, genuinely unweighted vertices retain
    # their bind position instead of collapsing to (0,0,0), which previously
    # created huge spikes/rings on Cyber's screen/spinner models.
    for k in range(min(4, mesh.weights.shape[1])):
        w = np.asarray(mesh.weights[:, k], dtype=np.float32)
        active = np.nonzero(w > 1e-7)[0]
        authored_weight[active] += w[active]
        for vi in active:
            ji = int(mesh.joints[vi, k])
            if not (0 <= ji < len(skin)):
                continue
            if bone_visible is not None and ji < len(bone_visible) and not bool(bone_visible[ji]):
                continue
            wk = float(w[vi])
            out_p[vi] += (skin[ji] @ p4[vi])[:3] * wk
            if vi < len(base_n):
                out_n[vi] += (normal_mats[ji] @ base_n[vi]) * wk
            used_weight[vi] += wk

    weighted = used_weight > 1e-8
    if np.any(weighted):
        # A few assets do not sum packed skin weights to exactly one. Normalize
        # the accumulated result rather than shrinking those vertices toward the
        # origin. This is harmless for ordinary normalized character meshes.
        out_p[weighted] /= used_weight[weighted, None]
        out_n[weighted] /= used_weight[weighted, None]

    unweighted = authored_weight <= 1e-8
    if np.any(unweighted):
        out_p[unweighted] = np.asarray(mesh.positions, dtype=np.float32)[unweighted]
        if len(base_n) == len(out_n):
            out_n[unweighted] = base_n[unweighted]

    # Fully-hidden weighted vertices are deliberately left collapsed; triangles
    # containing them are removed before drawing. This preserves the game's
    # show/hide semantics without contaminating adjacent visible geometry.
    if len(out_n):
        mag = np.linalg.norm(out_n, axis=1)
        ok = mag > 1e-8
        out_n[ok] /= mag[ok, None]
        fallback=(~ok) & unweighted
        if np.any(fallback) and len(base_n) == len(out_n):
            out_n[fallback] = base_n[fallback]
    return out_p, out_n



def _material_string_property(data: bytes, key: str) -> str:
    """Read one exact serialized Vector material string property.

    v0.32 avoids matching a key inside a longer property name.  The old
    ``EnvTexture`` lookup could stop inside ``AdditiveEnvTexture`` and report
    ``Proxy_additive`` instead of the real cubemap (for example Cyber's
    ``Colors/Gold_cube``).
    """
    marker = key.encode("ascii") + b"\x00"
    start = 0
    i = -1
    while True:
        i = data.find(marker, start)
        if i < 0:
            return ""
        if i == 0:
            break
        prev = data[i - 1]
        # Reject substring hits inside ASCII identifiers such as
        # AdditiveEnvTexture.  Binary/NUL boundaries are valid property starts.
        if not (48 <= prev <= 57 or 65 <= prev <= 90 or 97 <= prev <= 122 or prev == 95):
            break
        start = i + 1

    p = i + len(marker)
    if p + 4 > len(data):
        return ""

    # Vector material properties store a 32-bit kind before string data.
    p += 4
    while p < len(data) and data[p] == 0:
        p += 1

    if p >= len(data):
        return ""

    e = data.find(b"\x00", p)
    if e < 0:
        return ""

    raw = data[p:e]
    if not raw:
        return ""

    # Reject binary/vector payloads accidentally queried as strings.
    if any(b < 0x20 or b > 0x7E for b in raw):
        return ""

    return raw.decode("utf-8", "replace")



def _material_vec4_property(data: bytes, key: str):
    marker = key.encode("ascii") + b"\x00"
    i = data.find(marker)
    if i < 0:
        return None
    p = i + len(marker)
    if p + 20 > len(data):
        return None
    try:
        value_type = struct.unpack_from("<I", data, p)[0]
        if value_type != 4:
            return None
        return np.asarray(struct.unpack_from("<4f", data, p + 4), dtype=np.float32)
    except Exception:
        return None


def _material_float_property(data: bytes, key: str):
    marker = key.encode("ascii") + b"\x00"
    i = data.find(marker)
    if i < 0:
        return None
    p = i + len(marker)
    for delta in (4, 0, 8):
        if p + delta + 4 <= len(data):
            try:
                f = struct.unpack_from("<f", data, p + delta)[0]
                if math.isfinite(f) and -10000.0 <= f <= 10000.0:
                    return float(f)
            except Exception:
                pass
    return None


def parse_material_asset(data: bytes, name="Material") -> MaterialData:
    nul = data.find(b"\x00")
    shader = data[:nul].decode("utf-8", "replace") if nul > 0 else ""

    color = _material_vec4_property(data, "DiffuseColor")
    if color is None:
        color = _material_vec4_property(data, "VehiclePaintColor")
    if color is None:
        color = _material_vec4_property(data, "PaintColor")
    if color is None:
        color = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    diffuse = (
        _material_string_property(data, "DiffuseTexture")
        or _material_string_property(data, "ColorTexture")
        or _material_string_property(data, "ColorMap")
        # Island Adventure/Steam skybox materials use SkyTexture rather than
        # DiffuseTexture. Treat it as the fixed-function color source so the
        # map viewer shows the authored sky atlas instead of a white material.
        or _material_string_property(data, "SkyTexture")
    )
    paint = (
        _material_string_property(data, "VehiclePaintColor")
        or _material_string_property(data, "PaintColor")
    )
    decal = (
        _material_string_property(data, "VehicleDecalTexture")
        or _material_string_property(data, "DecalTexture")
    )
    decal_color = (
        _material_string_property(data, "VehicleDecalColor")
        or _material_string_property(data, "DecalColor")
    )
    mask = _material_string_property(data, "MaskTexture")
    normal = _material_string_property(data, "NormalTexture")
    detail = _material_string_property(data, "DetailTexture")
    one_bit_alpha = _material_string_property(data, "OneBitAlphaTexture")
    parallax = _material_string_property(data, "ParallaxTexture")
    env = (
        _material_string_property(data, "EnvTexture")
        or _material_string_property(data, "VehicleEnvTexture")
    )
    additive_env = (
        _material_string_property(data, "AdditiveEnvTexture")
        or _material_string_property(data, "VehicleAdditiveEnvTexture")
    )
    fresnel = (
        _material_string_property(data, "VehicleFresnelTexture")
        or _material_string_property(data, "FresnelTexture")
    )
    detail_uv_scale = _material_float_property(data, "DetailUvScale")
    if detail_uv_scale is None or not math.isfinite(float(detail_uv_scale)) or float(detail_uv_scale) <= 0:
        detail_uv_scale = 1.0

    return MaterialData(
        name=name,
        shader=shader,
        diffuse_color=color,
        diffuse_texture=diffuse or paint,
        paint_texture=paint,
        decal_texture=decal,
        decal_color_texture=decal_color,
        normal_texture=normal,
        mask_texture=mask,
        env_texture=env,
        additive_env_texture=additive_env,
        fresnel_texture=fresnel,
        detail_texture=detail,
        one_bit_alpha_texture=one_bit_alpha,
        parallax_texture=parallax,
        detail_uv_scale=float(detail_uv_scale),
        emissive_amount=_material_float_property(data, "EmissiveAmount") or 0.0,
    )



def _rgb565(v):
    return np.asarray([
        ((v >> 11) & 31) * 255 // 31,
        ((v >> 5) & 63) * 255 // 63,
        (v & 31) * 255 // 31,
    ], dtype=np.uint8)


def _decode_bc1(data: bytes, width: int, height: int, force_four_color=False):
    out = np.zeros((height, width, 4), dtype=np.uint8)
    p = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            c0, c1, bits = struct.unpack_from("<HHI", data, p)
            p += 8
            a = _rgb565(c0).astype(np.int32)
            b = _rgb565(c1).astype(np.int32)
            colors = np.zeros((4, 4), dtype=np.uint8)
            colors[0, :3], colors[1, :3] = a, b
            colors[0, 3] = colors[1, 3] = 255
            if c0 > c1 or force_four_color:
                colors[2, :3] = ((2*a+b)//3).astype(np.uint8)
                colors[3, :3] = ((a+2*b)//3).astype(np.uint8)
                colors[2:, 3] = 255
            else:
                colors[2, :3] = ((a+b)//2).astype(np.uint8)
                colors[2, 3] = 255
            for py in range(4):
                for px in range(4):
                    x, y = bx+px, by+py
                    if x < width and y < height:
                        out[y, x] = colors[(bits >> (2*(py*4+px))) & 3]
    return out


def _decode_bc3(data: bytes, width: int, height: int):
    out = np.zeros((height, width, 4), dtype=np.uint8)
    p = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            a0, a1 = data[p], data[p+1]
            abits = int.from_bytes(data[p+2:p+8], "little")
            p += 8
            alphas = [0]*8
            alphas[0], alphas[1] = a0, a1
            if a0 > a1:
                for i in range(1, 7):
                    alphas[i+1] = ((7-i)*a0 + i*a1)//7
            else:
                for i in range(1, 5):
                    alphas[i+1] = ((5-i)*a0 + i*a1)//5
                alphas[6], alphas[7] = 0, 255

            c0, c1, bits = struct.unpack_from("<HHI", data, p)
            p += 8
            ca, cb = _rgb565(c0).astype(np.int32), _rgb565(c1).astype(np.int32)
            colors = np.zeros((4, 3), dtype=np.uint8)
            colors[0], colors[1] = ca, cb
            colors[2] = ((2*ca+cb)//3).astype(np.uint8)
            colors[3] = ((ca+2*cb)//3).astype(np.uint8)

            for py in range(4):
                for px in range(4):
                    x, y = bx+px, by+py
                    if x < width and y < height:
                        n = py*4+px
                        out[y, x, :3] = colors[(bits >> (2*n)) & 3]
                        out[y, x, 3] = alphas[(abits >> (3*n)) & 7]
    return out


def _decoder_rgba_from_bgra(raw: bytes, width: int, height: int):
    """texture2ddecoder returns BGRA. Convert it to RGBA numpy."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4)).copy()
    arr = arr[:, :, [2, 1, 0, 3]]
    return arr


def _decode_bc1_preferred(data: bytes, width: int, height: int, force_four_color=False):
    """Use V10's native decoder path when available, with a correctness patch.

    ``texture2ddecoder`` is substantially faster than the old per-pixel Python
    loop.  Steam format 17 is opaque BC1 and must use four-colour interpolation
    even for c0<=c1 blocks, so those blocks are repaired after native decode.
    """
    try:
        import texture2ddecoder as td
        rgba = _decoder_rgba_from_bgra(td.decode_bc1(data, width, height), width, height)
    except Exception:
        return _decode_bc1(data, width, height, force_four_color=force_four_color)
    if not force_four_color:
        return rgba
    p = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            if p + 8 > len(data):
                return rgba
            c0, c1, bits = struct.unpack_from("<HHI", data, p); p += 8
            if c0 > c1:
                continue
            a = _rgb565(c0).astype(np.int32); b = _rgb565(c1).astype(np.int32)
            c2 = ((2*a+b)//3).astype(np.uint8); c3 = ((a+2*b)//3).astype(np.uint8)
            for n in range(16):
                idx=(bits>>(2*n))&3
                if idx < 2: continue
                x=bx+(n&3); y=by+(n>>2)
                if x < width and y < height:
                    rgba[y,x,:3] = c2 if idx==2 else c3
                    rgba[y,x,3] = 255
    rgba[...,3] = 255
    return rgba


def _decode_bc3_preferred(data: bytes, width: int, height: int):
    try:
        import texture2ddecoder as td
        return _decoder_rgba_from_bgra(td.decode_bc3(data, width, height), width, height)
    except Exception:
        return _decode_bc3(data, width, height)


def _decode_bc4(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode BC4/ATI1 into a single 8-bit channel."""
    out = np.zeros((height, width), dtype=np.uint8)
    p = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            if p + 8 > len(data):
                raise ValueError("BC4 payload is truncated.")
            a0, a1 = data[p], data[p + 1]
            bits = int.from_bytes(data[p + 2:p + 8], "little")
            p += 8
            vals = [0] * 8
            vals[0], vals[1] = int(a0), int(a1)
            if a0 > a1:
                for i in range(1, 7):
                    vals[i + 1] = ((7 - i) * a0 + i * a1) // 7
            else:
                for i in range(1, 5):
                    vals[i + 1] = ((5 - i) * a0 + i * a1) // 5
                vals[6], vals[7] = 0, 255
            for py in range(4):
                for px in range(4):
                    x, y = bx + px, by + py
                    if x < width and y < height:
                        n = py * 4 + px
                        out[y, x] = vals[(bits >> (3 * n)) & 7]
    return out


def _decode_bc5(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode BC5/ATI2 and reconstruct a displayable normal-map blue channel."""
    r = np.zeros((height, width), dtype=np.uint8)
    g = np.zeros((height, width), dtype=np.uint8)
    p = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            if p + 16 > len(data):
                raise ValueError("BC5 payload is truncated.")
            blocks = (data[p:p + 8], data[p + 8:p + 16]); p += 16
            chans = []
            for block in blocks:
                a0, a1 = block[0], block[1]
                bits = int.from_bytes(block[2:8], "little")
                vals = [0] * 8
                vals[0], vals[1] = int(a0), int(a1)
                if a0 > a1:
                    for i in range(1, 7): vals[i + 1] = ((7 - i) * a0 + i * a1) // 7
                else:
                    for i in range(1, 5): vals[i + 1] = ((5 - i) * a0 + i * a1) // 5
                    vals[6], vals[7] = 0, 255
                chans.append((bits, vals))
            for py in range(4):
                for px in range(4):
                    x, y = bx + px, by + py
                    if x < width and y < height:
                        n = py * 4 + px
                        r[y, x] = chans[0][1][(chans[0][0] >> (3 * n)) & 7]
                        g[y, x] = chans[1][1][(chans[1][0] >> (3 * n)) & 7]
    xf = r.astype(np.float32) / 127.5 - 1.0
    yf = g.astype(np.float32) / 127.5 - 1.0
    zf = np.sqrt(np.maximum(0.0, 1.0 - xf * xf - yf * yf))
    b = np.clip(np.round((zf * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = r, g, b, 255
    return rgba


def _decode_vector_jpeg(data: bytes, name: str) -> Optional[TextureData]:
    """Decode Vector preview textures stored as a 26-byte wrapper + JPEG.

    V10 identified this wrapper.  It is validated by 69 track-preview assets in
    the supplied Island Adventure Steam APF, all of which have JPEG SOI at +26.
    """
    if len(data) < 30 or data[26:28] != b"\xff\xd8":
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data[26:])) as im:
            arr = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    except Exception as exc:
        raise ValueError("JPEG-backed Vector texture could not be decoded: %s" % exc) from exc
    h, w = arr.shape[:2]
    return TextureData(name=name, width=int(w), height=int(h), mip_count=1,
                       format_code=-2, rgba=arr,
                       format_name="JPEG (Vector 26-byte wrapper)")


def _decode_legacy_vector_texture(data: bytes, name="Texture", platform="") -> TextureData:
    """V10-compatible fallback for older/generic Vector texture headers.

    This is intentionally a fallback.  BBR2/Island Adventure's validated
    42-byte header remains authoritative because its small format IDs have
    different meanings from DXGI IDs in older desktop assets.
    """
    if len(data) < 65:
        raise ValueError("Texture asset is too small for the legacy Vector header.")
    try:
        fields = struct.unpack_from("<16I", data, 1)
    except struct.error as exc:
        raise ValueError("Legacy Vector texture header is truncated.") from exc
    width, height, fmt, mips = int(fields[5]), int(fields[6]), int(fields[8]), int(fields[9])
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError("Invalid legacy Vector texture dimensions: %dx%d" % (width, height))

    android_fmts = {6407, 6408, 36196, 37492, 37494, 37496, 37497}
    payload_off = 70 if fmt in android_fmts else 65
    if len(data) <= payload_off:
        raise ValueError("Legacy Vector texture has no payload.")
    payload = bytes(data[payload_off:])
    bw, bh = max(1, (width + 3)//4), max(1, (height + 3)//4)

    if fmt == 28:  # DXGI_R8G8B8A8_UNORM
        need = width * height * 4
        if len(payload) < need: raise ValueError("RGBA8 legacy texture payload is truncated.")
        rgba = np.frombuffer(payload[:need], np.uint8).reshape((height, width, 4)).copy()
        fmt_name = "DXGI RGBA8"
    elif fmt == 51:  # DXGI_R16_FLOAT
        need = width * height * 2
        if len(payload) < need: raise ValueError("R16F legacy texture payload is truncated.")
        a = np.frombuffer(payload[:need], dtype=np.float16).astype(np.float32).reshape((height, width))
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        lo, hi = float(a.min()), float(a.max())
        a = (a-lo)/(hi-lo) if hi > lo else np.zeros_like(a)
        c = np.clip(np.round(a*255),0,255).astype(np.uint8)
        rgba = np.empty((height,width,4),np.uint8); rgba[...,0:3]=c[...,None]; rgba[...,3]=255
        fmt_name = "DXGI R16_FLOAT"
    elif fmt in (71,72):
        need=bw*bh*8
        if len(payload)<need: raise ValueError("BC1 legacy texture payload is truncated.")
        rgba=_decode_bc1_preferred(payload[:need],width,height,force_four_color=(fmt==71))
        fmt_name="BC1 / DXT1 (DXGI)"
    elif fmt in (77,78):
        need=bw*bh*16
        if len(payload)<need: raise ValueError("BC3 legacy texture payload is truncated.")
        rgba=_decode_bc3_preferred(payload[:need],width,height); fmt_name="BC3 / DXT5 (DXGI)"
    elif fmt == 80:
        need=bw*bh*8
        ch=_decode_bc4(payload[:need],width,height)
        rgba=np.empty((height,width,4),np.uint8); rgba[...,0:3]=ch[...,None]; rgba[...,3]=255
        fmt_name="BC4 / ATI1 (DXGI)"
    elif fmt == 83:
        need=bw*bh*16; rgba=_decode_bc5(payload[:need],width,height); fmt_name="BC5 / ATI2 (DXGI)"
    elif fmt in android_fmts:
        try:
            import texture2ddecoder as td
        except ImportError as exc:
            raise RuntimeError("ETC/EAC texture decoding requires the 'texture2ddecoder' package.") from exc
        if fmt == 6408:
            need=width*height*4
            rgba=np.frombuffer(payload[:need],np.uint8).reshape((height,width,4)).copy(); fmt_name="GL_RGBA"
        elif fmt == 6407:
            need=width*height*3
            rgb=np.frombuffer(payload[:need],np.uint8).reshape((height,width,3)).copy()
            rgba=np.empty((height,width,4),np.uint8); rgba[...,:3]=rgb; rgba[...,3]=255; fmt_name="GL_RGB"
        else:
            if fmt == 36196: raw=td.decode_etc1(payload,width,height); fmt_name="ETC1 RGB8"
            elif fmt == 37492: raw=td.decode_etc2(payload,width,height); fmt_name="ETC2 RGB8"
            elif fmt == 37494:
                if not hasattr(td,'decode_etc2a1'):
                    raise RuntimeError("ETC2 RGB8A1 needs a texture2ddecoder build with decode_etc2a1; update the package.")
                raw=td.decode_etc2a1(payload,width,height); fmt_name="ETC2 RGB8A1"
            else: raw=td.decode_etc2a8(payload,width,height); fmt_name="ETC2 RGBA8/EAC"
            rgba=_decoder_rgba_from_bgra(raw,width,height)
        # V10's generic GL wrapper uses lower-left texture origin.
        rgba=np.flipud(rgba).copy()
    else:
        raise ValueError("Unsupported legacy Vector texture format %d." % fmt)
    return TextureData(name=name,width=width,height=height,mip_count=max(1,mips),
                       format_code=fmt,rgba=rgba,format_name=fmt_name)


def parse_texture_asset(data: bytes, name="Texture", platform="") -> TextureData:
    """Decode the top mip of a Vector Unit ``VuTextureAsset``.

    The BBR2/Island Adventure 42-byte header is preferred and has been validated
    against the user's real Steam and GooglePlay archives.  V10's JPEG and
    generic DXGI/GL layouts are used only when that header is not applicable.
    """
    jpeg = _decode_vector_jpeg(data, name)
    if jpeg is not None:
        return jpeg
    if len(data) < 42:
        return _decode_legacy_vector_texture(data, name, platform)

    fmt = struct.unpack_from("<H", data, 0x16)[0]
    width = struct.unpack_from("<H", data, 0x1A)[0]
    height = struct.unpack_from("<H", data, 0x1E)[0]
    mips = struct.unpack_from("<H", data, 0x22)[0]
    payload_size = struct.unpack_from("<I", data, 0x26)[0]
    known_custom = {1,3,5,16,17,18,19,35,36}
    custom_valid = (
        fmt in known_custom and 1 <= width <= 16384 and 1 <= height <= 16384
        and 0 < payload_size <= len(data) - 42
    )
    if not custom_valid:
        return _decode_legacy_vector_texture(data, name, platform)

    payload = bytes(data[42:42 + payload_size])
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    platform_l = (platform or "").lower()
    is_android = any(x in platform_l for x in ("googleplay","android","amazon"))

    # Validated BBR2 custom codes shared across Steam/mobile.
    if fmt == 5:
        top_size = width * height * 4
        if len(payload) < top_size: raise ValueError("RGBA8 texture payload is truncated.")
        rgba = np.frombuffer(payload[:top_size], dtype=np.uint8).reshape((height,width,4)).copy()
        return TextureData(name,width,height,mips,fmt,rgba,"RGBA8")
    if fmt == 1:
        top_size = width * height
        if len(payload) < top_size: raise ValueError("R8 texture payload is truncated.")
        ch=np.frombuffer(payload[:top_size],np.uint8).reshape((height,width)).copy()
        rgba=np.empty((height,width,4),np.uint8); rgba[...,0:3]=ch[...,None]; rgba[...,3]=255
        return TextureData(name,width,height,mips,fmt,rgba,"R8 / grayscale")
    if fmt == 3:
        top_size = width * height * 2
        if len(payload) < top_size: raise ValueError("RG8 texture payload is truncated.")
        rg=np.frombuffer(payload[:top_size],np.uint8).reshape((height,width,2)).copy()
        rgba=np.empty((height,width,4),np.uint8); rgba[...,0:2]=rg; rgba[...,2:3]=255; rgba[...,3]=255
        return TextureData(name,width,height,mips,fmt,rgba,"RG8")

    if is_android:
        try:
            import texture2ddecoder
        except ImportError as exc:
            raise RuntimeError("Android ETC2 texture decoding requires the 'texture2ddecoder' package. Run RUN_BBR_VECTOR_STUDIO.bat again.") from exc
        if fmt == 16:
            top_size=bw*bh*8; fmt_name="ETC2 RGB8 (GooglePlay)"
            decoder=texture2ddecoder.decode_etc2
        elif fmt == 35:
            top_size=bw*bh*8; fmt_name="ETC2 RGB8A1 (GooglePlay)"
            decoder=getattr(texture2ddecoder,'decode_etc2a1',None)
            if decoder is None:
                raise RuntimeError("ETC2 RGB8A1 needs a texture2ddecoder build with decode_etc2a1; update the package.")
        elif fmt == 36:
            top_size=bw*bh*16; fmt_name="ETC2 RGBA8 / EAC (GooglePlay)"
            decoder=texture2ddecoder.decode_etc2a8
        else:
            raise ValueError("Unsupported GooglePlay Vector texture format code %d." % fmt)
        if len(payload)<top_size:
            raise ValueError("%s texture payload is truncated (%d < %d bytes)." % (fmt_name,len(payload),top_size))
        raw=decoder(payload[:top_size],width,height)
        rgba=_decoder_rgba_from_bgra(raw,width,height)
    else:
        if fmt == 17:
            top_size=bw*bh*8
            if len(payload)<top_size: raise ValueError("BC1 texture payload is truncated.")
            rgba=_decode_bc1_preferred(payload[:top_size],width,height,force_four_color=True); fmt_name="BC1 / DXT1 opaque (Steam)"
        elif fmt == 18:
            top_size=bw*bh*8
            if len(payload)<top_size: raise ValueError("BC1A texture payload is truncated.")
            rgba=_decode_bc1_preferred(payload[:top_size],width,height); fmt_name="BC1 / DXT1 1-bit alpha (Steam)"
        elif fmt == 19:
            top_size=bw*bh*16
            if len(payload)<top_size: raise ValueError("BC3 texture payload is truncated.")
            rgba=_decode_bc3_preferred(payload[:top_size],width,height); fmt_name="BC3 / DXT5 (Steam)"
        elif fmt == 16:
            top_size=bw*bh*8; rgba=_decode_bc1_preferred(payload[:top_size],width,height); fmt_name="BC1 / DXT1 (desktop fallback)"
        elif fmt == 36:
            top_size=bw*bh*16; rgba=_decode_bc3_preferred(payload[:top_size],width,height); fmt_name="BC3 / DXT5 (desktop fallback)"
        else:
            raise ValueError("Unsupported Vector texture format code %d on platform '%s'." % (fmt,platform or 'unknown'))
    return TextureData(name=name,width=width,height=height,mip_count=mips,format_code=fmt,rgba=rgba,format_name=fmt_name)

def parse_water_map_asset(data: bytes, name="WaterMap") -> TextureData:
    """Decode the RGBA8 top mip of a VuWaterMapAsset.

    This asset has the same five texture fields as a VuCubeTextureAsset at
    offset 21, with its first texel at byte 41.  The channels are shader data:
    shadow, foam, decal, and water coverage respectively.
    """
    if len(data) < 41:
        raise ValueError("Water map header is truncated")
    fmt, width, height, mips, payload_size = struct.unpack_from("<5I", data, 21)
    top_size = width * height * 4
    if (fmt != 5 or not 1 <= width <= 16384 or not 1 <= height <= 16384
            or not 1 <= mips <= 16 or payload_size < top_size
            or len(data) < 41 + top_size or len(data) < 41 + payload_size):
        raise ValueError("Invalid or unsupported RGBA8 water map")
    rgba = np.frombuffer(data, dtype=np.uint8, count=top_size, offset=41).reshape(height, width, 4).copy()
    return TextureData(name, width, height, mips, fmt, rgba, "WaterMap RGBA8 (shadow/foam/decal/coverage)")


def parse_cube_texture_asset(data: bytes, name="CubeTexture", platform="") -> CubeTextureData:
    """Decode Vector Unit VuCubeTextureAsset top mips.

    Six face mip chains each have their own 20-byte texture descriptor.  The
    first descriptor begins at offset 21 after the common cube header; every
    subsequent descriptor follows the preceding face's mip chain.  Treating
    the first payload size as a fixed stride shifts alternate ETC blocks by
    four bytes and produces the neon checkerboard seen on AlienA's domes.
    """
    if len(data) < 41:
        raise ValueError("Cube texture asset is too small.")
    faces: List[TextureData] = []
    cursor = 21
    fmt = width = height = mips = None
    for i in range(6):
        if cursor + 20 > len(data):
            raise ValueError("Cube texture face %d descriptor is truncated." % i)
        face_fmt, face_width, face_height, face_mips, face_size = struct.unpack_from("<5I", data, cursor)
        if (not 1 <= face_width <= 16384 or not 1 <= face_height <= 16384
                or not 1 <= face_mips <= 16 or face_size <= 0
                or cursor + 20 + face_size > len(data)):
            raise ValueError("Cube texture face %d is invalid or truncated." % i)
        if i == 0:
            fmt, width, height, mips = face_fmt, face_width, face_height, face_mips
        elif (face_fmt, face_width, face_height, face_mips) != (fmt, width, height, mips):
            raise ValueError("Cube texture face %d has mismatched dimensions or format." % i)
        # The ordinary texture decoder expects a one-byte longer header.
        header = b"\x01" + bytes(data[:21]) + bytes(data[cursor:cursor+20])
        face_blob = bytes(data[cursor+20:cursor+20+face_size])
        faces.append(parse_texture_asset(header + face_blob, f"{name}/face{i}", platform))
        cursor += 20 + face_size
    return CubeTextureData(name=name, width=width, height=height, mip_count=mips,
                           format_code=fmt, faces=faces,
                           format_name=(faces[0].format_name if faces else ""))


def material_path_for_mesh(mesh_name: str) -> str:
    base = mesh_name.split("_LOD", 1)[0] if "_LOD" in mesh_name else mesh_name
    return "VuMaterialAsset/" + base


def _usable_texture_name(name: str) -> bool:
    if not name:
        return False
    nl = str(name).strip().lower()
    if not nl or nl == "none":
        return False
    if nl.endswith("/none"):
        return False
    if nl.startswith(("proxy_", "colors/")):
        return False
    return True


def _load_archive_texture(archive: APFArchive, tex_name: str):
    if not _usable_texture_name(tex_name):
        return None
    tp = "VuTextureAsset/" + tex_name
    te = archive.get(tp)
    if te is None:
        return None
    try:
        return parse_texture_asset(
            archive.decode(te), tex_name, archive.platform
        )
    except Exception:
        return None


def _resize_rgba_nearest(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgba is None:
        return None
    src_h, src_w = rgba.shape[:2]
    if src_w == width and src_h == height:
        return np.asarray(rgba, dtype=np.uint8)
    x = np.clip(
        np.round(np.linspace(0, src_w - 1, width)).astype(np.int32),
        0, max(0, src_w - 1)
    )
    y = np.clip(
        np.round(np.linspace(0, src_h - 1, height)).astype(np.int32),
        0, max(0, src_h - 1)
    )
    return np.asarray(rgba[y][:, x], dtype=np.uint8)


def _solid_texture_from_color(name: str, color, size: int = 4) -> TextureData:
    c = np.asarray(color, dtype=np.float32).reshape(-1)
    if c.size < 4:
        c = np.pad(c, (0, 4 - c.size), mode="constant", constant_values=1.0)
    c = np.clip(c[:4], 0.0, 1.0)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:] = np.round(c * 255.0).astype(np.uint8)
    return TextureData(
        name=name,
        width=size,
        height=size,
        mip_count=1,
        format_code=-1,
        rgba=rgba,
        format_name="SolidColor",
    )


def _alpha_composite(base_rgba: np.ndarray, over_rgba: np.ndarray) -> np.ndarray:
    if base_rgba is None:
        return np.asarray(over_rgba, dtype=np.uint8).copy()
    if over_rgba is None:
        return np.asarray(base_rgba, dtype=np.uint8).copy()

    bh, bw = base_rgba.shape[:2]
    oh, ow = over_rgba.shape[:2]
    if (bw, bh) != (ow, oh):
        over_rgba = _resize_rgba_nearest(over_rgba, bw, bh)

    base = np.asarray(base_rgba, dtype=np.float32) / 255.0
    over = np.asarray(over_rgba, dtype=np.float32) / 255.0

    oa = over[..., 3:4]
    ba = base[..., 3:4]
    out_a = oa + ba * (1.0 - oa)

    # Standard source-over alpha composite.
    out_rgb = over[..., :3] * oa + base[..., :3] * (1.0 - oa)

    out = np.concatenate((out_rgb, out_a), axis=2)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _build_material_texture(
    archive: APFArchive, mat: MaterialData, mesh_name: str = ""
):
    # For Beach Buggy Racing vehicle paint materials the design is often split
    # across a base paint texture plus a decal texture. Build a simple
    # composite preview texture so the viewer shows the vehicle's intended skin.
    ordered_names = []
    # The base material preview uses diffuse/paint only. Vector car decals
    # are usually grayscale MASKS, not ready-to-alpha-composite RGBA images.
    # Vehicle-specific decal coloring is handled by build_vehicle_skin_texture.
    for tname in (
        mat.diffuse_texture,
        mat.paint_texture,
    ):
        if _usable_texture_name(tname) and tname not in ordered_names:
            ordered_names.append(tname)

    loaded = []
    loaded_names = []
    for tname in ordered_names:
        tex = _load_archive_texture(archive, tname)
        if tex is not None:
            loaded.append(tex)
            loaded_names.append(tname)

    if loaded:
        # Vector CarPaint shaders use 1-D ``Ramp_*`` textures as a lighting
        # lookup, not as ordinary model UV images.  Mapping a 128x1 ramp over
        # Cyber's shirt (and similar materials) made the mesh sample the dark
        # end of the ramp and appear charcoal/green instead of white/grey.
        # Convert a paint ramp to a bright representative material swatch and
        # let the viewport lighting shade the real 3-D geometry.
        shader_l = str(getattr(mat, "shader", "") or "").lower()
        ramp_tex = None
        ramp_name = ""
        if "carpaint" in shader_l:
            for n, t in zip(loaded_names, loaded):
                nl = str(n).lower()
                if "/ramp_" in nl or nl.rsplit("/", 1)[-1].startswith("ramp_"):
                    if int(t.width) >= max(8, int(t.height) * 8):
                        ramp_tex, ramp_name = t, n
                        break
        if ramp_tex is not None and ramp_tex.rgba is not None:
            rgb = np.asarray(ramp_tex.rgba[..., :3], dtype=np.float32).reshape(-1, 3)
            if len(rgb):
                lum = rgb.mean(axis=1)
                order = np.argsort(lum)
                # ~80th percentile reproduces the authored paint's normally-lit
                # base while still leaving GL lighting room for grey shadows.
                pick = order[int(round((len(order) - 1) * 0.80))]
                base = np.clip(rgb[pick], 0, 255).astype(np.uint8)
                rgba = np.empty((4, 4, 4), dtype=np.uint8)
                rgba[..., :3] = base.reshape(1, 1, 3)
                rgba[..., 3] = 255
                return TextureData(
                    name=mat.name or mesh_name or "CarPaintRampPreview",
                    width=4, height=4, mip_count=1, format_code=-32,
                    rgba=rgba, format_name="CarPaintRampPreview",
                ), "VuTextureAsset/" + ramp_name + " (shader ramp preview)"

        width = max(int(t.width) for t in loaded)
        height = max(int(t.height) for t in loaded)
        out = None
        for tex in loaded:
            rgba = _resize_rgba_nearest(tex.rgba, width, height)
            out = _alpha_composite(out, rgba)

        comp_name = mat.name or mesh_name or "CompositeTexture"
        tp = " | ".join("VuTextureAsset/" + n for n in loaded_names)
        return TextureData(
            name=comp_name,
            width=width,
            height=height,
            mip_count=max(int(t.mip_count) for t in loaded),
            format_code=-1,
            rgba=np.asarray(out, dtype=np.uint8),
            format_name="Composite",
        ), tp

    # If there is no decodable texture, fall back to the material color instead
    # of the rig placeholder color. This avoids large red/orange placeholder
    # meshes on vehicles whose material is color-only or mostly transparent.
    dc = getattr(mat, "diffuse_color", None)
    if dc is not None:
        tex = _solid_texture_from_color(mat.name or mesh_name or "SolidColor", dc)
        return tex, "(material-color)"

    return None, ""


def _representative_texture_color(tex: TextureData, fallback=(1.0, 1.0, 1.0)):
    if tex is None or tex.rgba is None or tex.rgba.size == 0:
        return np.asarray(fallback, dtype=np.float32)
    rgba = np.asarray(tex.rgba, dtype=np.float32) / 255.0
    rgb = rgba[..., :3].reshape(-1, 3)
    if rgba.shape[2] >= 4:
        alpha = rgba[..., 3].reshape(-1)
        keep = alpha > 0.05
        if np.any(keep):
            rgb = rgb[keep]
    if len(rgb) == 0:
        return np.asarray(fallback, dtype=np.float32)
    # Paint ramps contain lighting/shading variants. The median gives a stable
    # base color without mapping that 1D ramp across the decal UVs.
    return np.clip(np.median(rgb, axis=0), 0.0, 1.0).astype(np.float32)


def _vehicle_decal_mask(tex: TextureData):
    if tex is None or tex.rgba is None:
        return None
    src = np.asarray(tex.rgba, dtype=np.float32) / 255.0
    mask = src[..., 0]
    if src.shape[2] >= 4 and np.any(src[..., 3] < 0.999):
        mask = mask * src[..., 3]
    return np.clip(mask, 0.0, 1.0)


def _apply_vehicle_decal_color(rgb, decal_tex, decal_rgb):
    if decal_tex is None:
        return rgb
    mask = _vehicle_decal_mask(decal_tex)
    if mask is None:
        return rgb
    if rgb.shape[:2] != mask.shape:
        h, w = mask.shape
        solid = np.empty((h, w, 3), dtype=np.float32)
        solid[:] = np.asarray(rgb[0, 0], dtype=np.float32)
        rgb = solid
    m = mask[..., None]
    return rgb * (1.0 - m) + np.asarray(decal_rgb, dtype=np.float32).reshape(1, 1, 3) * m


def build_vehicle_skin_texture(
    archive: APFArchive,
    paint_color: str,
    decal_color: str,
    decal_name: str,
    name: str = "VehicleSkin",
    extra_decals=None,
):
    """Build a paint/decal preview texture from Vector vehicle skin data.

    `extra_decals` accepts optional (decal_name, color_name) layers. It is used
    when a vehicle has a permanent/signature design in addition to the selected
    Vehicle Skins row. CandyCane_Default is the key example: CandyCane stripes
    + Snow overlay.
    """
    extra_decals = list(extra_decals or [])

    paint_tex = _load_archive_texture(
        archive, "Car/Paint/" + str(paint_color)
    ) if paint_color and str(paint_color).lower() != "none" else None
    decal_color_tex = _load_archive_texture(
        archive, "Car/Paint/" + str(decal_color)
    ) if decal_color and str(decal_color).lower() != "none" else None

    base_rgb = _representative_texture_color(paint_tex, fallback=(0.55,0.55,0.55))
    primary_decal_rgb = _representative_texture_color(decal_color_tex, fallback=(1.0,1.0,1.0))

    layers=[]
    for extra_name, extra_color_name in extra_decals:
        if not extra_name:
            continue
        tex=_load_archive_texture(archive,"Car/Decal/"+str(extra_name))
        if tex is None:
            continue
        ctex=_load_archive_texture(archive,"Car/Paint/"+str(extra_color_name)) if extra_color_name and str(extra_color_name).lower()!='none' else None
        crgb=_representative_texture_color(ctex,fallback=primary_decal_rgb)
        layers.append((str(extra_name),tex,crgb))

    no_primary = (not decal_name or str(decal_name).lower() in ('none','car/decals/none','car/decal/none'))
    if not no_primary:
        tex=_load_archive_texture(archive,"Car/Decal/"+str(decal_name))
        if tex is not None:
            layers.append((str(decal_name),tex,primary_decal_rgb))

    if not layers:
        rgba=np.empty((4,4,4),dtype=np.uint8)
        rgba[...,:3]=np.round(base_rgb*255.0).astype(np.uint8)
        rgba[...,3]=255
        return TextureData(name=name,width=4,height=4,mip_count=1,format_code=-2,rgba=rgba,format_name='VehicleSkinSolid'), f'VehicleSkins: paint={paint_color}, decal=None'

    width=max(int(t.width) for _,t,_ in layers)
    height=max(int(t.height) for _,t,_ in layers)
    rgb=np.empty((height,width,3),dtype=np.float32); rgb[:]=base_rgb.reshape(1,1,3)
    layer_names=[]
    for lname,ltex,lrgb in layers:
        if ltex.width!=width or ltex.height!=height:
            resized=_resize_rgba_nearest(ltex.rgba,width,height)
            ltex=TextureData(name=ltex.name,width=width,height=height,mip_count=1,format_code=ltex.format_code,rgba=resized,format_name=ltex.format_name)
        rgb=_apply_vehicle_decal_color(rgb,ltex,lrgb)
        layer_names.append(lname)

    rgba=np.empty((height,width,4),dtype=np.uint8)
    rgba[...,:3]=np.clip(np.round(rgb*255.0),0,255).astype(np.uint8)
    rgba[...,3]=255
    return TextureData(name=name,width=width,height=height,mip_count=1,format_code=-2,rgba=rgba,format_name='VehicleSkinComposite'), f"VehicleSkins: paint={paint_color}, decalColor={decal_color}, layers={'+'.join(layer_names)}"


def _rgb_to_hue_sector(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    d = mx - mn

    h = np.zeros_like(mx, dtype=np.float32)
    nz = d > 1e-6

    mr = nz & (mx == r)
    mg = nz & (mx == g)
    mb = nz & (mx == b)

    h[mr] = ((g[mr] - b[mr]) / d[mr]) % 6.0
    h[mg] = ((b[mg] - r[mg]) / d[mg]) + 2.0
    h[mb] = ((r[mb] - g[mb]) / d[mb]) + 4.0

    h /= 6.0

    sector = np.floor(h * 6.0 + 0.5).astype(np.int32) % 6

    sat = np.zeros_like(mx, dtype=np.float32)
    valid = mx > 1e-6
    sat[valid] = d[valid] / mx[valid]

    return sector, sat, mx


def build_candycane_original_texture(
    archive: APFArchive,
    name: str = "CandyCane_Default_Christmas",
):
    """
    Reconstruct the original Candy Coupe Christmas livery for the preview.

    CandyCane_Rainbow is a dedicated fixed-livery texture using the exact
    stripe UV layout around the vehicle shell. Convert those rainbow bands to
    alternating peppermint red/white, preserve baked shading, then overlay the
    real Snow decal.
    """
    rainbow = _load_archive_texture(
        archive,
        "Car/Vehicles/CandyCane/CandyCane_Rainbow",
    )
    if rainbow is None:
        return None, ""

    src = np.asarray(rainbow.rgba, dtype=np.float32) / 255.0
    rgb = src[..., :3]

    sector, sat, value = _rgb_to_hue_sector(rgb)

    # One rainbow cycle should become ONE red band + ONE white band,
    # not three alternating pairs.  Keep red around the red/orange/magenta
    # half of the hue wheel and white across the opposite half.
    white_band = np.isin(sector, (2, 3, 4))
    white_band |= sat < 0.14

    red = np.asarray(
        [0.93, 0.055, 0.025],
        dtype=np.float32,
    )
    white = np.asarray(
        [0.97, 0.97, 0.95],
        dtype=np.float32,
    )

    out_rgb = np.empty_like(rgb)
    out_rgb[:] = red
    out_rgb[white_band] = white

    # Preserve baked highlights/shadows from the source livery.
    shade = np.clip(0.45 + value * 0.55, 0.35, 1.0)
    out_rgb *= shade[..., None]

    # Preserve true dark detail pixels.
    dark = value < 0.16
    if np.any(dark):
        out_rgb[dark] = rgb[dark]

    # Real Christmas detail layer.
    snow = _load_archive_texture(
        archive,
        "Car/Decal/Snow",
    )
    if snow is not None:
        if (
            snow.width != rainbow.width
            or snow.height != rainbow.height
        ):
            snow_rgba = _resize_rgba_nearest(
                snow.rgba,
                rainbow.width,
                rainbow.height,
            )
            snow = TextureData(
                name=snow.name,
                width=rainbow.width,
                height=rainbow.height,
                mip_count=1,
                format_code=snow.format_code,
                rgba=snow_rgba,
                format_name=snow.format_name,
            )

        mask = _vehicle_decal_mask(snow)
        if mask is not None:
            m = np.clip(mask[..., None], 0.0, 1.0)
            out_rgb = (
                out_rgb * (1.0 - m)
                + white.reshape(1, 1, 3) * m
            )

    rgba = np.empty_like(rainbow.rgba, dtype=np.uint8)
    rgba[..., :3] = np.clip(
        np.round(out_rgb * 255.0),
        0,
        255,
    ).astype(np.uint8)
    rgba[..., 3] = rainbow.rgba[..., 3]

    return TextureData(
        name=name,
        width=rainbow.width,
        height=rainbow.height,
        mip_count=1,
        format_code=-3,
        rgba=rgba,
        format_name="CandyCaneOriginalChristmas",
    ), (
        "CandyCane original: Rainbow UV layout -> "
        "red/white peppermint + Snow"
    )



def resolve_material_and_texture(archive: APFArchive, mesh_name: str):
    mp = material_path_for_mesh(mesh_name)
    me = archive.get(mp)
    if me is None:
        return MaterialData(mesh_name), None, mp, ""
    mat = parse_material_asset(archive.decode(me), mesh_name)
    tex, tp = _build_material_texture(archive, mat, mesh_name)
    return mat, tex, mp, tp


def _read_cstrings(data: bytes, off: int, count: int):
    """Read count NUL-terminated strings and return (strings, new_offset)."""
    out = []
    for _ in range(int(count)):
        if off >= len(data):
            raise ValueError("String table exceeds file size.")
        end = data.find(b"\x00", off)
        if end < 0:
            raise ValueError("Unterminated string in model string table.")
        out.append(data[off:end].decode("ascii", "replace"))
        off = end + 1
    return out, off


def printable_strings(data: bytes, minimum=4, limit=300):
    vals=re.findall(rb"[\x20-\x7e]{%d,}" % minimum,data)
    return [x.decode("utf-8","replace") for x in vals[:limit]]

# ---------------------------------------------------------------------------
# v0.51 multi-control driving animation support

def _sample_animation_local_trs(model: ModelData, anim: Optional[AnimationData], frame: float, loop_override=None):
    """Sample one animation into local TRS tuples without composing parents."""
    if not model.bones:
        return []
    if anim is None:
        return [
            (np.asarray(b.translation, dtype=np.float32).copy(),
             np.asarray(b.rotation, dtype=np.float32).copy(),
             np.asarray(b.scale, dtype=np.float32).copy())
            for b in model.bones
        ]
    if anim.bone_count != len(model.bones):
        raise ValueError("Animation/model bone-count mismatch (%d vs %d)." % (anim.bone_count, len(model.bones)))
    f0,f1,frac=_animation_frame_pair(anim,frame,loop_override)
    out=[]
    for i in range(anim.bone_count):
        t0=np.asarray(anim.translations[f0,i],dtype=np.float32); t1=np.asarray(anim.translations[f1,i],dtype=np.float32)
        q0=np.asarray(anim.rotations[f0,i],dtype=np.float32); q1=np.asarray(anim.rotations[f1,i],dtype=np.float32)
        s0=np.asarray(anim.scales[f0,i],dtype=np.float32); s1=np.asarray(anim.scales[f1,i],dtype=np.float32)
        valid0=np.isfinite(t0).all() and np.isfinite(s0).all()
        valid1=np.isfinite(t1).all() and np.isfinite(s1).all()
        if not valid0 or not valid1:
            if valid0:
                t=t0.copy(); sc=s0.copy(); q=q0.copy()
            else:
                b=model.bones[i]
                t=np.asarray(b.translation,dtype=np.float32).copy()
                sc=np.zeros(3,dtype=np.float32)
                q=np.asarray([0,0,0,1],dtype=np.float32)
        else:
            t=(t0*(1.0-frac)+t1*frac).astype(np.float32)
            sc=(s0*(1.0-frac)+s1*frac).astype(np.float32)
            q=_quat_slerp(q0,q1,frac)
        out.append((t,q,sc))
    return out


def animation_globals_weighted(model: ModelData, layers):
    """Blend authored animation controls by weight, then compose the skeleton.

    `layers` is an iterable of `(AnimationData, frame, weight)`.  This mirrors
    BBR's driving-animation set behavior more closely than replacing the whole
    driver pose with a single TurnLR clip.  In particular motorcycles blend the
    authored Turn and Lean controls instead of snapping between them.
    """
    if not model.bones:
        return []
    active=[]
    for anim,frame,weight in layers:
        w=float(weight)
        if anim is not None and w>1e-7:
            active.append((anim,float(frame),w))
    if not active:
        return bind_globals(model)
    total=sum(x[2] for x in active)
    if total<=1e-8:
        return bind_globals(model)
    active=[(a,f,w/total) for a,f,w in active]
    poses=[(_sample_animation_local_trs(model,a,f),w) for a,f,w in active]
    local=[]
    for bi in range(len(model.bones)):
        t=np.zeros(3,dtype=np.float64); sc=np.zeros(3,dtype=np.float64)
        qref=None; qsum=np.zeros(4,dtype=np.float64)
        for pose,w in poses:
            ti,qi,si=pose[bi]
            t += np.asarray(ti,dtype=np.float64)*w
            sc += np.asarray(si,dtype=np.float64)*w
            q=np.asarray(qi,dtype=np.float64)
            n=float(np.linalg.norm(q))
            if n<1e-10: q=np.asarray([0,0,0,1],dtype=np.float64)
            else: q/=n
            if qref is None: qref=q.copy()
            elif float(np.dot(q,qref))<0.0: q=-q
            qsum += q*w
        qn=float(np.linalg.norm(qsum))
        q=(qsum/qn if qn>1e-10 else np.asarray([0,0,0,1],dtype=np.float64)).astype(np.float32)
        local.append(trs_matrix(t.astype(np.float32),q,sc.astype(np.float32)))
    return global_matrices(local,model.bones)
