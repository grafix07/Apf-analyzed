from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from vector_formats import APFArchive, TextureData

# Vector APF/FPUV uses a 64-byte header.  In particular, bytes 60..63 carry
# the header integrity hash in the BBR2 packer.  Do not reduce this to 0x3C.
HEADER_SIZE = 0x40


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _rgb565(r, g, b):
    return ((int(r) >> 3) << 11) | ((int(g) >> 2) << 5) | (int(b) >> 3)

def _encode_bc1_block(px):
    # Conservative BC1 encoder: choose min/max RGB endpoints and nearest palette index.
    cols = np.asarray(px, dtype=np.uint8).reshape(16, 4)[:, :3]
    c0 = _rgb565(*cols.max(axis=0)); c1 = _rgb565(*cols.min(axis=0))
    if c0 < c1: c0, c1 = c1, c0
    def unpack(c):
        return np.array([((c>>11)&31)*255//31, ((c>>5)&63)*255//63, (c&31)*255//31], dtype=np.int32)
    a, b = unpack(c0), unpack(c1)
    palette = np.stack([a, b, (2*a+b)//3, (a+2*b)//3])
    d=((cols[:,None,:].astype(np.int32)-palette[None,:,:])**2).sum(axis=2)
    idx=d.argmin(axis=1).astype(np.uint32)
    bits=0
    for i,v in enumerate(idx): bits |= int(v) << (2*i)
    return struct.pack('<HHI', c0, c1, bits)

def _encode_bc1(rgba, w, h):
    arr=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)
    pw=((w+3)//4)*4; ph=((h+3)//4)*4
    pad=np.zeros((ph,pw,4),dtype=np.uint8); pad[:h,:w]=arr
    out=bytearray()
    for y in range(0,ph,4):
        for x in range(0,pw,4): out += _encode_bc1_block(pad[y:y+4,x:x+4])
    return bytes(out)

def _encode_bc3_block(px):
    a=np.asarray(px,dtype=np.uint8).reshape(16,4)[:,3]
    a0=int(a.max()); a1=int(a.min())
    vals=[a0,a1]
    if a0>a1: vals += [(6*a0+a1)//7,(5*a0+2*a1)//7,(4*a0+3*a1)//7,(3*a0+4*a1)//7,(2*a0+5*a1)//7,(a0+6*a1)//7]
    else: vals += [(4*a0+a1)//5,(3*a0+2*a1)//5,(2*a0+3*a1)//5,(a0+4*a1)//5,0,255]
    inds=[min(range(8),key=lambda j:abs(int(v)-vals[j])) for v in a]
    abits=0
    for i,v in enumerate(inds): abits |= int(v) << (3*i)
    return bytes([a0,a1])+abits.to_bytes(6,'little')+_encode_bc1_block(px)

def _encode_bc3(rgba,w,h):
    arr=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)
    pw=((w+3)//4)*4; ph=((h+3)//4)*4
    pad=np.zeros((ph,pw,4),dtype=np.uint8); pad[:h,:w]=arr
    out=bytearray()
    for y in range(0,ph,4):
        for x in range(0,pw,4): out += _encode_bc3_block(pad[y:y+4,x:x+4])
    return bytes(out)

def _encode_texture_preserving_format(texture: TextureData, template: bytes) -> bytes:
    rgba=np.asarray(texture.rgba,dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4: raise ValueError('Custom skin texture must be RGBA8.')
    if len(template) < 42: raise ValueError('Original texture BIN is too small.')
    hdr=bytearray(template)
    fmt=struct.unpack_from('<H',hdr,0x16)[0]
    old_w=struct.unpack_from('<H',hdr,0x1A)[0]; old_h=struct.unpack_from('<H',hdr,0x1E)[0]
    mips=max(1,struct.unpack_from('<H',hdr,0x22)[0])
    # Keep the original dimensions unless the user supplied a genuinely different image.
    h,w=rgba.shape[:2]
    if (w,h)!=(old_w,old_h):
        # Skin replacement should match the shipped texture dimensions; resizing avoids corrupt mip/layout metadata.
        from PIL import Image
        im=Image.fromarray(rgba,'RGBA').resize((old_w,old_h),Image.Resampling.LANCZOS)
        rgba=np.asarray(im,dtype=np.uint8)
        w,h=old_w,old_h
    chunks=[]
    from PIL import Image
    base=Image.fromarray(rgba,'RGBA')
    for level in range(mips):
        mw=max(1,w>>level); mh=max(1,h>>level)
        im=base if level==0 else base.resize((mw,mh),Image.Resampling.LANCZOS)
        raw=np.asarray(im,dtype=np.uint8)
        if fmt==5: chunks.append(raw.tobytes())
        elif fmt in (17,18): chunks.append(_encode_bc1(raw,mw,mh))
        elif fmt==19: chunks.append(_encode_bc3(raw,mw,mh))
        else:
            raise ValueError(f'Character skin texture format {fmt} cannot be safely re-encoded by this Studio build.')
    payload=b''.join(chunks)
    struct.pack_into('<H',hdr,0x1A,w); struct.pack_into('<H',hdr,0x1E,h)
    struct.pack_into('<H',hdr,0x22,mips); struct.pack_into('<I',hdr,0x26,len(payload))
    return bytes(hdr[:42])+payload

def encode_rgba8_texture(texture: TextureData, template: Optional[bytes] = None) -> bytes:
    if template is None:
        raise ValueError('A matching original texture BIN is required for safe APF skin export.')
    return _encode_texture_preserving_format(texture, template)

def _compress(data: bytes, flags: int) -> Tuple[bytes, int]:
    """Match the proven BBR packer's edited-entry policy.

    Codec 0 remains raw. Codec 1 remains zlib. Legacy codecs 2/3 are emitted
    as zlib with codec 1 because the working packer deliberately normalizes
    edited entries this way. Codec 4 is handled as zlib too rather than
    writing a bogus codec id.
    """
    hi = (int(flags) >> 16) & 0xFFFF
    lo = int(flags) & 0xFFFF
    if hi == 0:
        return data, int(flags)
    if hi == 1:
        return zlib.compress(data, 9), int(flags)
    # The established BBR packer normalizes non-zlib compressed edits to zlib.
    return zlib.compress(data, 9), (1 << 16) | lo


def write_apf_from_archive(source_apf: str | Path, output_apf: str | Path,
                           replacements: Dict[str, bytes]) -> Dict[str, int]:
    """Clone an APF using the runtime-compatible Vector packing layout.

    Critical invariants:
      * preserve the complete 64-byte original header and unknown metadata;
      * preserve original body order (directory order is not body order);
      * copy every untouched payload byte-for-byte;
      * use FNV-1a32 for replacement payload hashes;
      * rebuild the directory and update its hash at header[20:24];
      * update the trailing 4-byte header hash at bytes 60..63.
    """
    src = Path(source_apf)
    out = Path(output_apf)
    arc = APFArchive(src)
    try:
        if len(arc.mm) < HEADER_SIZE or bytes(arc.mm[:4]) != b"FPUV":
            raise ValueError("Not a supported Vector FPUV/APF archive.")

        # Read the original directory directly so we preserve exact names,
        # ordering and every field not intentionally changed.
        raw = bytes(arc.mm)
        _version, dir_off, entry_count, dir_size, _dir_hash = struct.unpack_from("<IIIII", raw, 4)
        if dir_off < HEADER_SIZE or dir_off + dir_size > len(raw):
            raise ValueError("Invalid APF directory range.")

        entries = []
        p = dir_off
        end = dir_off + dir_size
        for i in range(entry_count):
            nul = raw.find(b"\0", p, end)
            if nul < 0 or p + 20 > end:
                raise ValueError("Invalid APF directory entry.")
            name = raw[p:nul].decode("utf-8", "replace")
            p = nul + 1
            off, usize, csize, checksum, flags = struct.unpack_from("<IIIII", raw, p)
            p += 20
            if off < HEADER_SIZE or off + csize > dir_off:
                raise ValueError(f"Invalid payload range for {name}.")
            entries.append({
                "name": name, "orig_off": off, "usize": usize,
                "csize": csize, "hash": checksum, "flags": flags,
            })

        # Vector APFs store payloads in offset order, not necessarily directory order.
        body_order = sorted(entries, key=lambda e: e["orig_off"])
        body_start = min(e["orig_off"] for e in entries) if entries else HEADER_SIZE
        if body_start < HEADER_SIZE:
            raise ValueError("APF payload begins inside the header.")

        out_bytes = bytearray(raw[:body_start])
        replaced = 0
        changed_names = []

        for e in body_order:
            name = e["name"]
            replacement = replacements.get(name)
            if replacement is None:
                e["new_off"] = len(out_bytes)
                original = raw[e["orig_off"]:e["orig_off"] + e["csize"]]
                out_bytes.extend(original)
                e["new_usize"] = e["usize"]
                e["new_csize"] = e["csize"]
                e["new_hash"] = e["hash"]
                e["new_flags"] = e["flags"]
            else:
                plain = bytes(replacement)
                stored, new_flags = _compress(plain, e["flags"])
                e["new_off"] = len(out_bytes)
                out_bytes.extend(stored)
                e["new_usize"] = len(plain)
                e["new_csize"] = len(stored)
                e["new_hash"] = fnv1a32(plain)
                e["new_flags"] = new_flags
                replaced += 1
                changed_names.append(name)

        # Rebuild only the directory records. Directory order remains exactly
        # the original order.
        directory = bytearray()
        for e in entries:
            directory.extend(e["name"].encode("utf-8") + b"\0")
            directory.extend(struct.pack(
                "<IIIII",
                e["new_off"], e["new_usize"], e["new_csize"],
                e["new_hash"], e["new_flags"],
            ))

        directory_offset = len(out_bytes)
        out_bytes.extend(directory)

        # Preserve all unknown header bytes. Only the fields known to change
        # with a repack are rewritten.
        header = bytearray(out_bytes[:HEADER_SIZE])
        struct.pack_into("<I", header, 0x08, directory_offset)
        struct.pack_into("<I", header, 0x0C, len(entries))
        struct.pack_into("<I", header, 0x10, len(directory))
        struct.pack_into("<I", header, 0x14, fnv1a32(bytes(directory)))

        # The working BBR packer computes the trailing header hash over all
        # header bytes except the hash field itself.
        struct.pack_into("<I", header, 0x3C, fnv1a32(bytes(header[:0x3C])))
        out_bytes[:HEADER_SIZE] = header

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(out_bytes)
        tmp.replace(out)

        return {
            "entries": len(entries),
            "replaced": replaced,
            "size": out.stat().st_size,
            "directory_size": len(directory),
            "header_size": HEADER_SIZE,
        }
    finally:
        arc.close()


def _asset_leaf(path: str) -> str:
    return Path(str(path).replace("\\", "/")).name.lower()


def find_skin_entries(arc: APFArchive, stem: str, asset_type: str) -> list[str]:
    target = Path(str(stem).replace("\\", "/")).stem.lower()
    prefix = asset_type.rstrip("/").lower() + "/"
    entries = [e for e in arc.entries if e.path.lower().startswith(prefix)]
    exact = [e.path for e in entries if _asset_leaf(e.path) == target]
    if exact:
        return exact
    candidates = [e.path for e in entries
                  if _asset_leaf(e.path).startswith(target) or target.startswith(_asset_leaf(e.path))]
    return sorted(candidates, key=lambda p: (len(_asset_leaf(p)), len(p), p.lower()))[:1]


def build_skin_replacements(source_apf: str | Path, textures: Dict[str, TextureData],
                            texture_templates: Optional[Dict[str, bytes]] = None,
                            model_bytes: Optional[Dict[str, bytes]] = None) -> Tuple[Dict[str, bytes], list[str], list[str]]:
    arc = APFArchive(source_apf)
    replacements: Dict[str, bytes] = {}
    matched: list[str] = []
    missing: list[str] = []
    try:
        for key, tex in textures.items():
            paths = find_skin_entries(arc, key, "VuTextureAsset")
            if not paths:
                missing.append("VuTextureAsset/" + key)
                continue
            for path in paths:
                template = None
                if texture_templates:
                    template = texture_templates.get(key) or texture_templates.get(Path(path).name)
                if template is None:
                    entry = arc.get(path)
                    if entry is not None:
                        template = bytes(arc.mm[entry.offset:entry.offset + entry.stored_size]) if entry.flags >> 16 == 0 else None
                        if template is None:
                            # APF compression must be decoded before the texture header can be used.
                            template = arc.decode(entry)
                replacements[path] = encode_rgba8_texture(tex, template)
                matched.append(path)
        for key, blob in (model_bytes or {}).items():
            paths = find_skin_entries(arc, key, "VuAnimatedModelAsset")
            if not paths:
                missing.append("VuAnimatedModelAsset/" + key)
                continue
            for path in paths:
                replacements[path] = bytes(blob)
                matched.append(path)
    finally:
        arc.close()
    return replacements, matched, missing
