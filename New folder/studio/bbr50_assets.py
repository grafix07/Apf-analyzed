from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vector_formats import (
    APFArchive, APFEntry, AnimationData, MaterialData, ModelData, TextureData, CubeTextureData,
    parse_animation, parse_material_asset, parse_model_asset, parse_texture_asset, parse_cube_texture_asset, parse_water_map_asset,
)
from vector_bbr2_json import decode_bbr2_json, looks_like_bbr2_json


@dataclass(frozen=True)
class AssetRecord:
    path: str
    source_name: str
    archive_index: int
    entry_index: int
    unpacked_size: int
    stored_size: int


@dataclass
class DecodedAsset:
    record: AssetRecord
    data: bytes
    json_value: Any = None


class AssetDatabase:
    """Unified, read-only view over every APF in a BBR package.

    v0.50 intentionally keeps archive identity with every record.  Older Studio
    paths flattened data too early, which made it hard to know whether a value
    came from Assets.apf, HW.apf, a spreadsheet, or a fallback.  The database
    is now the authority layer for every scene resolver.
    """

    MODEL_TYPES = ("VuAnimatedModelAsset", "VuStaticModelAsset")

    def __init__(self):
        self.archives: List[Tuple[str, APFArchive]] = []
        self.records_by_path: Dict[str, List[AssetRecord]] = {}
        self._temp_dir: Optional[str] = None
        self._json_cache: Dict[Tuple[int, int], Any] = {}
        self._model_cache: Dict[Tuple[int, int], ModelData] = {}
        self._material_cache: Dict[Tuple[int, int], MaterialData] = {}
        self._texture_cache: Dict[Tuple[int, int], TextureData] = {}
        self._water_map_cache: Dict[Tuple[int, int], TextureData] = {}
        self._cube_texture_cache: Dict[Tuple[int, int], CubeTextureData] = {}
        self._animation_cache: Dict[Tuple[int, int], AnimationData] = {}
        self._spreadsheet_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._pfx_index: Optional[Dict[str, Dict[str, Any]]] = None
        self.game_variant: str = "Unknown Vector package"
        self.is_island_adventure: bool = False
        self.has_map_tracks: bool = False
        self.is_mobile_bbr2: bool = False

    # ------------------------------------------------------------------ load
    @classmethod
    def open(cls, path: str | os.PathLike) -> "AssetDatabase":
        db = cls()
        db._load_path(Path(path))
        db._detect_game_variant()
        return db

    def close(self):
        for _, arc in self.archives:
            try:
                arc.close()
            except Exception:
                pass
        self.archives.clear()
        self.records_by_path.clear()
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _add_archive(self, display_name: str, path: Path):
        arc = APFArchive(path)
        ai = len(self.archives)
        self.archives.append((display_name, arc))
        for e in arc.entries:
            rec = AssetRecord(
                path=e.path,
                source_name=display_name,
                archive_index=ai,
                entry_index=e.index,
                unpacked_size=e.unpacked_size,
                stored_size=e.stored_size,
            )
            self.records_by_path.setdefault(e.path, []).append(rec)

    @staticmethod
    def _file_signature(path: Path, display_name: str):
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(32)
            if size >= 32:
                f.seek(max(0, size - 32))
            tail = f.read(32)
        return (display_name, size, head, tail)

    def _load_path(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".apf":
            self._add_archive(path.name, path)
            # Many mobile installs keep Assets.apf and optional content packs
            # (including the HW track materials) side by side. Selecting just
            # Assets.apf must not leave those referenced models/textures out.
            selected=path.resolve()
            siblings=sorted((p for p in path.parent.iterdir()
                             if p.is_file() and p.suffix.lower()==".apf" and p.resolve()!=selected),
                            key=lambda p:p.name.lower())
            for sibling in siblings:
                try: self._add_archive(sibling.name,sibling)
                except (ValueError,OSError):
                    # A stray or incomplete sibling must not prevent loading
                    # the APF the user explicitly selected.
                    continue
            return
        if path.suffix.lower() not in (".apk", ".zip", ".xapk", ".apks"):
            raise ValueError("Choose an APF, APK, XAPK, APKS or ZIP package.")

        # v0.67 streams APFs to temporary files instead of z.read()-ing the
        # whole archive into RAM.  Island Adventure's Steam Assets.apf is about
        # 553 MB unpacked, so the older all-in-memory path could needlessly
        # duplicate hundreds of megabytes before mmap even opened the APF.
        self._temp_dir = tempfile.mkdtemp(prefix="bbr67_apfs_")
        found: List[Tuple[str, Path]] = []
        seq = 0

        def extract_member(zf: zipfile.ZipFile, member: str, display_name: str) -> Path:
            nonlocal seq
            safe = Path(member.replace("::", "__")).name or f"archive_{seq}.apf"
            out = Path(self._temp_dir) / f"{seq:03d}_{safe}"
            seq += 1
            with zf.open(member, "r") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            found.append((display_name, out))
            return out

        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            for name in names:
                if name.lower().endswith(".apf"):
                    extract_member(z, name, name)

            # XAPK/APKS commonly place APFs inside an asset-pack APK.  Keep the
            # traversal bounded to one nested ZIP level, but stream the nested
            # container too so large packages do not require a giant bytes copy.
            for nested in names:
                if not nested.lower().endswith((".apk", ".zip")):
                    continue
                try:
                    nested_path = Path(self._temp_dir) / f"nested_{seq:03d}_{Path(nested).name}"
                    seq += 1
                    with z.open(nested, "r") as src, open(nested_path, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    with zipfile.ZipFile(nested_path, "r") as nz:
                        for inner in nz.namelist():
                            if inner.lower().endswith(".apf"):
                                extract_member(nz, inner, nested + "::" + inner)
                except (zipfile.BadZipFile, KeyError, OSError):
                    continue

        if not found:
            raise ValueError("Package contains no APF archives.")

        seen = set()
        for name, apf_path in found:
            sig = self._file_signature(apf_path, name)
            if sig in seen:
                try:
                    apf_path.unlink()
                except OSError:
                    pass
                continue
            seen.add(sig)
            self._add_archive(name, apf_path)

    def _detect_game_variant(self):
        platforms = {str(arc.platform or "").strip().lower() for _, arc in self.archives}
        has_career_map = "VuProjectAsset/Screens_Premium/Career_Map" in self.records_by_path
        has_career_nodes = "VuSpreadsheetAsset/Career Nodes" in self.records_by_path
        has_tracks = any(p.startswith("VuProjectAsset/Tracks/") for p in self.records_by_path)

        desktop_like = any(p and p not in ("googleplay", "android", "amazon") for p in platforms)
        mobile_like = any(p in ("googleplay", "android", "amazon") for p in platforms)
        self.has_map_tracks = bool(has_tracks)
        self.is_island_adventure = bool(desktop_like and has_career_map and has_career_nodes and has_tracks)
        self.is_mobile_bbr2 = bool(mobile_like and has_tracks)
        if self.is_island_adventure:
            self.game_variant = "Beach Buggy Racing 2: Island Adventure"
        elif self.is_mobile_bbr2:
            self.game_variant = "Beach Buggy Racing 2 (mobile)"
        elif has_tracks:
            self.game_variant = "Beach Buggy Racing / Vector track package"
        else:
            self.game_variant = "Vector FPUV/APF package"

    # -------------------------------------------------------------- records
    def all_records(self) -> List[AssetRecord]:
        out = []
        for paths in self.records_by_path.values():
            out.extend(paths)
        return sorted(out, key=lambda r: (r.path.lower(), r.source_name.lower()))

    def _entry(self, rec: AssetRecord) -> APFEntry:
        _, arc = self.archives[rec.archive_index]
        return arc.entries[rec.entry_index]

    def decode(self, rec_or_path: AssetRecord | str) -> DecodedAsset:
        rec = rec_or_path if isinstance(rec_or_path, AssetRecord) else self.get(rec_or_path)
        if rec is None:
            raise KeyError(rec_or_path)
        _, arc = self.archives[rec.archive_index]
        data = arc.decode(self._entry(rec))
        obj = None
        if looks_like_bbr2_json(data):
            try:
                obj = decode_bbr2_json(data)
            except Exception:
                obj = None
        return DecodedAsset(rec, data, obj)

    def get_all(self, path: str) -> List[AssetRecord]:
        return list(self.records_by_path.get(path, ()))

    def get(self, path: str) -> Optional[AssetRecord]:
        candidates = self.records_by_path.get(path)
        if not candidates:
            return None
        # If a path exists in multiple APFs, prefer the most specialized pack
        # (HW, DLC, etc.) over the generic Assets archive.  Keep the source in
        # the returned record so this choice remains inspectable.
        def priority(r: AssetRecord):
            s = r.source_name.lower()
            generic = 1 if "assets.apf" in s else 0
            return (generic, r.archive_index)
        return sorted(candidates, key=priority)[0]

    @staticmethod
    def logical_variants(ref: str) -> List[str]:
        ref = str(ref or "").strip().strip("/")
        if not ref:
            return []
        out = [ref]
        if ref.startswith("Car/") and not ref.startswith("Car/Vehicles/"):
            out.append("Car/Vehicles/" + ref[len("Car/"):])
        if ref.startswith("Car/Vehicles/"):
            out.append("Car/" + ref[len("Car/Vehicles/"):])
        # Legacy JungleA track projects refer to this authored treeline through
        # a level-local alias, while the actual model lives under Plant/.
        if ref == "Level/JungleA/Treeline":
            out.append("Plant/Treeline")
        # Some exported JSON uses #Template/Path notation.
        if ref.startswith("#"):
            out.append(ref[1:])
        # Stable unique order.
        uniq = []
        for x in out:
            if x not in uniq:
                uniq.append(x)
        return uniq

    def resolve(self, ref: str, asset_types: Sequence[str]) -> Optional[AssetRecord]:
        if not ref:
            return None
        ref = str(ref).strip()
        # Already fully qualified.
        if ref in self.records_by_path:
            return self.get(ref)
        for logical in self.logical_variants(ref):
            for typ in asset_types:
                p = f"{typ}/{logical}"
                rec = self.get(p)
                if rec:
                    return rec
        return None

    def resolve_model(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, self.MODEL_TYPES)

    def resolve_material(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuMaterialAsset",))

    def resolve_texture(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuTextureAsset",))

    def resolve_cube_texture(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuCubeTextureAsset",))

    def resolve_water_map(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuWaterMapAsset",))

    def resolve_animation(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuAnimationAsset",))

    def resolve_project(self, ref: str) -> Optional[AssetRecord]:
        return self.resolve(ref, ("VuProjectAsset", "VuTemplateAsset"))

    # ------------------------------------------------------------- typed load
    @staticmethod
    def _cache_key(rec: AssetRecord):
        return rec.archive_index, rec.entry_index

    def json(self, rec_or_path: AssetRecord | str) -> Any:
        rec = rec_or_path if isinstance(rec_or_path, AssetRecord) else self.get(rec_or_path)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._json_cache:
            dec = self.decode(rec)
            self._json_cache[key] = dec.json_value
        return self._json_cache[key]

    def model(self, rec_or_ref: AssetRecord | str) -> Optional[ModelData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_model(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._model_cache:
            dec = self.decode(rec)
            self._model_cache[key] = parse_model_asset(dec.data, rec.path, rec.path)
        return self._model_cache[key]

    def material(self, rec_or_ref: AssetRecord | str) -> Optional[MaterialData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_material(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._material_cache:
            dec = self.decode(rec)
            self._material_cache[key] = parse_material_asset(dec.data, rec.path)
        return self._material_cache[key]

    def texture(self, rec_or_ref: AssetRecord | str) -> Optional[TextureData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_texture(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._texture_cache:
            _, arc = self.archives[rec.archive_index]
            dec = self.decode(rec)
            self._texture_cache[key] = parse_texture_asset(
                dec.data, rec.path, arc.platform
            )
        return self._texture_cache[key]

    def water_map(self, rec_or_ref: AssetRecord | str) -> Optional[TextureData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_water_map(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._water_map_cache:
            self._water_map_cache[key] = parse_water_map_asset(self.decode(rec).data, rec.path)
        return self._water_map_cache[key]

    def cube_texture(self, rec_or_ref: AssetRecord | str) -> Optional[CubeTextureData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_cube_texture(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._cube_texture_cache:
            _, arc = self.archives[rec.archive_index]
            dec = self.decode(rec)
            self._cube_texture_cache[key] = parse_cube_texture_asset(dec.data, rec.path, arc.platform)
        return self._cube_texture_cache[key]

    def animation(self, rec_or_ref: AssetRecord | str) -> Optional[AnimationData]:
        rec = rec_or_ref if isinstance(rec_or_ref, AssetRecord) else self.resolve_animation(rec_or_ref)
        if rec is None:
            return None
        key = self._cache_key(rec)
        if key not in self._animation_cache:
            dec = self.decode(rec)
            self._animation_cache[key] = parse_animation(dec.data, rec.path)
        return self._animation_cache[key]

    # ------------------------------------------------------------ spreadsheets
    def spreadsheet(self, logical_name: str) -> List[Dict[str, Any]]:
        if logical_name in self._spreadsheet_cache:
            return self._spreadsheet_cache[logical_name]
        path = logical_name if logical_name.startswith("VuSpreadsheetAsset/") else "VuSpreadsheetAsset/" + logical_name
        obj = self.json(path)
        rows: List[Dict[str, Any]] = []
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            header = [str(x) if x is not None else "" for x in obj[0]]
            for raw in obj[1:]:
                if isinstance(raw, list) and len(raw) == len(header):
                    rows.append(dict(zip(header, raw)))
        self._spreadsheet_cache[logical_name] = rows
        return rows

    def spreadsheet_by_name(self, logical_name: str, key="Name") -> Dict[str, Dict[str, Any]]:
        out = {}
        for row in self.spreadsheet(logical_name):
            v = row.get(key)
            if isinstance(v, str) and v:
                out[v] = row
        return out

    # ------------------------------------------------------------- PFX generic
    def pfx_systems(self) -> Dict[str, Dict[str, Any]]:
        if self._pfx_index is not None:
            return self._pfx_index
        self._pfx_index = {}
        root = self.json("VuPfxAsset/Generic")

        def walk(x):
            if isinstance(x, dict):
                if x.get("BaseType") == "system" and isinstance(x.get("Name"), str):
                    self._pfx_index.setdefault(x["Name"], x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(root)
        return self._pfx_index

    def pfx(self, ref: str) -> Optional[Dict[str, Any]]:
        if not ref:
            return None
        name = str(ref).strip().split("/")[-1]
        return self.pfx_systems().get(name)

    # ------------------------------------------------------------- references
    def references_from_json(self, obj: Any) -> List[Tuple[str, AssetRecord]]:
        out: List[Tuple[str, AssetRecord]] = []
        seen = set()

        def consider(s: str):
            if not isinstance(s, str) or not s or len(s) > 512:
                return
            tests = [
                (self.MODEL_TYPES, "model"),
                (("VuMaterialAsset",), "material"),
                (("VuTextureAsset",), "texture"),
                (("VuCubeTextureAsset",), "cube texture"),
                (("VuAnimationAsset",), "animation"),
                (("VuProjectAsset", "VuTemplateAsset"), "project/template"),
            ]
            for types, kind in tests:
                rec = self.resolve(s, types)
                if rec:
                    key = (kind, rec.path, rec.source_name)
                    if key not in seen:
                        seen.add(key)
                        out.append((kind, rec))
                    return
            # PFX live inside Generic rather than separate APF entries.
            if s.split("/")[-1] in self.pfx_systems():
                key = ("pfx", s, "VuPfxAsset/Generic")
                if key not in seen:
                    seen.add(key)
                    generic = self.get("VuPfxAsset/Generic")
                    if generic:
                        out.append(("pfx:" + s, generic))

        def walk(x):
            if isinstance(x, str):
                consider(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj)
        return out

    def report(self) -> str:
        lines = [f"Game profile: {self.game_variant}", f"APFs: {len(self.archives)}", f"Unique asset paths: {len(self.records_by_path)}"]
        for name, arc in self.archives:
            lines.append(f"  {name}: {len(arc.entries)} entries • platform={arc.platform} • FPUV v{arc.version}")
        return "\n".join(lines)
