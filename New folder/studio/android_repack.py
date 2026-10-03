from __future__ import annotations

"""Android APK/XAPK container repacking helpers for BBR Vector Studio.

This module deliberately works at the Android ZIP-container level only.  It can
replace an existing APF inside a normal APK or inside a nested split/asset-pack
APK in an XAPK/APKS archive.  It does not encode Vector Unit asset formats or
bypass Android signature verification.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import os
import tempfile
import zipfile
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ApfTarget:
    label: str
    outer_entry: str
    apf_entry: str

    @property
    def nested(self) -> bool:
        return bool(self.outer_entry)


def _is_zip(path: str | Path) -> bool:
    try:
        return zipfile.is_zipfile(str(path))
    except Exception:
        return False


def discover_apf_targets(source_path: str | Path) -> List[ApfTarget]:
    """Find APFs that can be replaced without extracting the whole package."""
    source = Path(source_path)
    if not source.is_file() or not _is_zip(source):
        raise ValueError("Choose a valid APK, XAPK, APKS or ZIP package.")

    suffix = source.suffix.lower()
    out: List[ApfTarget] = []
    with zipfile.ZipFile(source, "r") as outer:
        names = outer.namelist()
        # A normal APK directly contains assets/*.apf.
        for name in names:
            if name.lower().endswith(".apf"):
                out.append(ApfTarget(name, "", name))

        # XAPK/APKS packages commonly contain a dedicated asset_pack.apk.
        if suffix in (".xapk", ".apks", ".zip") or not out:
            for nested_name in names:
                if not nested_name.lower().endswith((".apk", ".zip")):
                    continue
                try:
                    payload = outer.read(nested_name)
                    if not zipfile.is_zipfile(BytesIO(payload)):
                        continue
                    with zipfile.ZipFile(BytesIO(payload), "r") as nested:
                        for inner in nested.namelist():
                            if inner.lower().endswith(".apf"):
                                out.append(ApfTarget(f"{nested_name}::{inner}", nested_name, inner))
                except (KeyError, OSError, zipfile.BadZipFile):
                    continue

    # Stable unique ordering; Assets.apf first because it is the normal target
    # for future model/texture replacement work.
    unique = {x.label: x for x in out}
    return sorted(unique.values(), key=lambda x: ("assets.apf" not in x.apf_entry.lower(), x.label.lower()))


def _clone_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Copy metadata while allowing zipfile to recompress the data."""
    z = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    z.comment = info.comment
    z.extra = info.extra
    z.create_system = info.create_system
    z.create_version = info.create_version
    z.extract_version = info.extract_version
    z.flag_bits = info.flag_bits & ~0x08
    z.volume = info.volume
    z.internal_attr = info.internal_attr
    z.external_attr = info.external_attr
    z.compress_type = info.compress_type
    return z


def _write_member(dst: zipfile.ZipFile, info: zipfile.ZipInfo, data: bytes) -> None:
    z = _clone_info(info)
    # Android packages often store already-compressed assets.  Preserve the
    # original member's compression method to avoid needlessly inflating APKs.
    dst.writestr(z, data, compress_type=info.compress_type)


def _rebuild_nested_apk(original_bytes: bytes, apf_entry: str, replacement: Path, output_path: Path) -> None:
    with zipfile.ZipFile(BytesIO(original_bytes), "r") as src, zipfile.ZipFile(output_path, "w", allowZip64=True) as dst:
        found = False
        for info in src.infolist():
            if info.filename == apf_entry:
                found = True
                _write_member(dst, info, replacement.read_bytes())
            else:
                _write_member(dst, info, src.read(info.filename))
        if not found:
            raise KeyError(f"APF entry not found in nested APK: {apf_entry}")


def _rebuild_direct_apk(source: Path, apf_entry: str, replacement: Path, output: Path) -> None:
    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(output, "w", allowZip64=True) as dst:
        found = False
        for info in src.infolist():
            if info.filename == apf_entry:
                found = True
                _write_member(dst, info, replacement.read_bytes())
            else:
                _write_member(dst, info, src.read(info.filename))
        if not found:
            raise KeyError(f"APF entry not found in APK: {apf_entry}")


def _updated_xapk_manifest(raw: bytes, apk_sizes: dict[str, int]) -> bytes:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw
    if isinstance(doc, dict) and isinstance(doc.get("split_apks"), list):
        total = 0
        for item in doc["split_apks"]:
            if isinstance(item, dict):
                name = item.get("file")
                if isinstance(name, str) and name in apk_sizes:
                    total += int(apk_sizes[name])
        if total:
            doc["total_size"] = total
            return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return raw


def repack_with_apf(source_path: str | Path, target: ApfTarget, replacement_apf: str | Path,
                    output_path: str | Path) -> str:
    """Replace one embedded APF and rebuild the Android ZIP container.

    The result is intentionally *not* claimed to be installable. Any modified
    APK loses its original signature.  For XAPK/APKS, every split APK must be
    signed with the same certificate before installation.
    """
    source = Path(source_path)
    replacement = Path(replacement_apf)
    output = Path(output_path)
    if not source.is_file() or not _is_zip(source):
        raise ValueError("Source package is not a valid ZIP/APK/XAPK/APKS archive.")
    if not replacement.is_file() or replacement.suffix.lower() != ".apf":
        raise ValueError("Replacement file must be an .apf archive.")
    if source.resolve() == output.resolve():
        raise ValueError("Choose a different output path; the source package is never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)

    if not target.nested:
        _rebuild_direct_apk(source, target.apf_entry, replacement, output)
        return str(output)

    nested_tmp: Optional[Path] = None
    try:
        with zipfile.ZipFile(source, "r") as outer:
            try:
                nested_original = outer.read(target.outer_entry)
            except KeyError:
                raise KeyError(f"Nested APK not found: {target.outer_entry}")

            fd, tmp_name = tempfile.mkstemp(prefix="bbr_repack_", suffix=".apk")
            os.close(fd)
            nested_tmp = Path(tmp_name)
            _rebuild_nested_apk(nested_original, target.apf_entry, replacement, nested_tmp)
            nested_size = nested_tmp.stat().st_size

            # Compute post-repack APK byte sizes so APKPure-style manifest.json
            # keeps its total_size field consistent with the rebuilt archive.
            apk_sizes = {}
            for info in outer.infolist():
                if info.filename.lower().endswith(".apk"):
                    apk_sizes[info.filename] = nested_size if info.filename == target.outer_entry else info.file_size

            with zipfile.ZipFile(output, "w", allowZip64=True) as dst:
                for info in outer.infolist():
                    if info.filename == target.outer_entry:
                        _write_member(dst, info, nested_tmp.read_bytes())
                    elif info.filename == "manifest.json":
                        raw = outer.read(info.filename)
                        _write_member(dst, info, _updated_xapk_manifest(raw, apk_sizes))
                    else:
                        _write_member(dst, info, outer.read(info.filename))
    finally:
        if nested_tmp is not None:
            try: nested_tmp.unlink()
            except OSError: pass
    return str(output)



def verify_repacked_apf(package_path: str | Path, target: ApfTarget,
                        expected_apf: str | Path) -> bool:
    """Confirm the rebuilt package contains the exact modified APF bytes."""
    package=Path(package_path)
    expected=Path(expected_apf).read_bytes()
    with zipfile.ZipFile(package,"r") as outer:
        if not target.nested:
            try:
                actual=outer.read(target.apf_entry)
            except KeyError:
                return False
            return actual == expected
        try:
            nested_bytes=outer.read(target.outer_entry)
        except KeyError:
            return False
        try:
            with zipfile.ZipFile(BytesIO(nested_bytes),"r") as nested:
                actual=nested.read(target.apf_entry)
        except (KeyError,zipfile.BadZipFile):
            return False
        return actual == expected


def signing_guidance(source_path: str | Path) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix in (".xapk", ".apks", ".zip"):
        return (
            "The rebuilt package contains a modified nested APK, so the original Android signature is no longer valid. "
            "Before installing, sign every APK split in the bundle with the same certificate (normally after zipalign), "
            "then rebuild the outer XAPK/APKS container. The Studio does not bypass Android signature verification."
        )
    return (
        "The rebuilt APK must be zipaligned and signed with an Android signing key before installation. "
        "The Studio does not bypass Android signature verification."
    )
