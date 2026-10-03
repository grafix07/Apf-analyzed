from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from io import BytesIO
import os
import struct
import zipfile
import zlib


FSB5_MODE_NAMES = {
    0: "NONE", 1: "PCM8", 2: "PCM16", 3: "PCM24", 4: "PCM32",
    5: "PCMFLOAT", 6: "GCADPCM", 7: "IMAADPCM", 8: "VAG",
    9: "HEVAG", 10: "XMA", 11: "MPEG", 12: "CELT", 13: "AT9",
    14: "XWMA", 15: "VORBIS",
}

# FSB5's compact frequency code. 0 means an explicit FREQUENCY metadata chunk.
FSB5_FREQUENCIES = {
    1: 8000, 2: 11000, 3: 11025, 4: 16000, 5: 22050,
    6: 24000, 7: 32000, 8: 44100, 9: 48000,
}


@dataclass
class FsbSampleInfo:
    index: int
    name: str
    frequency: int
    channels: int
    sample_count: int
    data_offset: int
    data_size: int = 0
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None
    vorbis_crc32: Optional[int] = None
    metadata_types: List[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.frequency <= 0:
            return 0.0
        return float(self.sample_count) / float(self.frequency)


@dataclass
class FsbBankInfo:
    offset: int
    version: int
    mode: int
    header_size: int
    sample_header_size: int
    name_table_size: int
    data_size: int
    total_size: int
    samples: List[FsbSampleInfo] = field(default_factory=list)

    @property
    def mode_name(self) -> str:
        return FSB5_MODE_NAMES.get(self.mode, f"MODE_{self.mode}")

    @property
    def data_relative_offset(self) -> int:
        return int(self.header_size + self.sample_header_size + self.name_table_size)


class Fsb5Parser:
    """Dependency-free FSB5 metadata/sample reader used by Vector Studio.

    BBR2 VuAudioBankAsset files contain FMOD event data followed by an embedded
    FSB5 sample bank.  Android BBR2 stores its samples as FSB5 Vorbis.  The game
    does not embed a complete Ogg stream in each FSB sample: the Vorbis setup
    packet is selected by a 32-bit setup CRC and lives in FMOD's native binary.

    v0.36 therefore reads the real FSB sample packets and combines them with the
    matching setup packet recovered from the game's own libfmod.so.  No audio is
    synthesized or substituted.
    """

    HEADER_SIZE_V1 = 60

    @classmethod
    def find(cls, data: bytes) -> Optional[FsbBankInfo]:
        off = data.find(b"FSB5")
        if off < 0:
            return None
        return cls.parse(data, off)

    @classmethod
    def parse(cls, data: bytes, offset: int = 0) -> FsbBankInfo:
        if offset < 0 or offset + cls.HEADER_SIZE_V1 > len(data):
            raise ValueError("FSB5 header is truncated.")
        magic, version, count, sh_size, name_size, data_size, mode = struct.unpack_from(
            "<4s6I", data, offset
        )
        if magic != b"FSB5":
            raise ValueError("FSB5 signature not found at requested offset.")
        if count > 1_000_000 or sh_size > len(data) or name_size > len(data) or data_size > len(data):
            raise ValueError("Unreasonable FSB5 header values.")

        header_size = cls.HEADER_SIZE_V1 + (4 if version == 0 else 0)
        p = offset + header_size
        sample_headers_end = p + int(sh_size)
        names_start = sample_headers_end
        data_start = names_start + int(name_size)
        total = header_size + int(sh_size) + int(name_size) + int(data_size)
        if data_start + int(data_size) > len(data):
            raise ValueError("Embedded FSB5 payload is truncated.")

        samples: List[FsbSampleInfo] = []
        for idx in range(int(count)):
            if p + 8 > sample_headers_end:
                raise ValueError("FSB5 sample-header table is truncated.")
            raw = struct.unpack_from("<Q", data, p)[0]
            p += 8
            has_extra = bool(raw & 1)
            freq_code = (raw >> 1) & 0xF
            channels = 2 if ((raw >> 5) & 1) else 1
            rel_data = (raw >> 6) & 0x0FFFFFFF
            sample_count = (raw >> 34) & 0x3FFFFFFF
            frequency = FSB5_FREQUENCIES.get(int(freq_code), 0)
            loop_start = loop_end = None
            vorbis_crc = None
            metadata_types: List[int] = []

            next_chunk = has_extra
            while next_chunk:
                if p + 4 > sample_headers_end:
                    raise ValueError("FSB5 metadata header is truncated.")
                chunk = struct.unpack_from("<I", data, p)[0]
                p += 4
                next_chunk = bool(chunk & 1)
                size = int((chunk >> 1) & 0xFFFFFF)
                typ = int((chunk >> 25) & 0x7F)
                if p + size > sample_headers_end:
                    raise ValueError("FSB5 metadata payload is truncated.")
                payload = data[p:p + size]
                p += size
                metadata_types.append(typ)
                if typ == 1 and size >= 1:  # CHANNELS
                    channels = max(1, int(payload[0]))
                elif typ == 2 and size >= 4:  # FREQUENCY
                    frequency = int(struct.unpack_from("<I", payload, 0)[0])
                elif typ == 3 and size >= 8:  # LOOP
                    loop_start, loop_end = struct.unpack_from("<II", payload, 0)
                elif typ == 11 and size >= 4:  # VORBISDATA
                    vorbis_crc = int(struct.unpack_from("<I", payload, 0)[0])

            samples.append(FsbSampleInfo(
                index=idx,
                name=f"sample_{idx:04d}",
                frequency=frequency,
                channels=channels,
                sample_count=int(sample_count),
                data_offset=int(rel_data) * 16,
                loop_start=loop_start,
                loop_end=loop_end,
                vorbis_crc32=vorbis_crc,
                metadata_types=metadata_types,
            ))

        # Name table begins with one uint32 relative string offset per sample.
        if name_size and count:
            if names_start + 4 * int(count) <= data_start:
                offsets = struct.unpack_from("<%dI" % int(count), data, names_start)
                for sample, rel in zip(samples, offsets):
                    s = names_start + int(rel)
                    if names_start <= s < data_start:
                        e = data.find(b"\x00", s, data_start)
                        if e >= 0:
                            sample.name = data[s:e].decode("utf-8", "replace") or sample.name

        for i, sample in enumerate(samples):
            start = sample.data_offset
            end = int(data_size)
            if i + 1 < len(samples):
                end = min(end, samples[i + 1].data_offset)
            sample.data_size = max(0, end - start)

        return FsbBankInfo(
            offset=int(offset),
            version=int(version),
            mode=int(mode),
            header_size=int(header_size),
            sample_header_size=int(sh_size),
            name_table_size=int(name_size),
            data_size=int(data_size),
            total_size=int(total),
            samples=samples,
        )

    @staticmethod
    def embedded_bytes(data: bytes, bank: FsbBankInfo) -> bytes:
        start = int(bank.offset)
        end = start + int(bank.total_size)
        return bytes(data[start:end])

    @staticmethod
    def sample_bytes(data: bytes, bank: FsbBankInfo, sample: FsbSampleInfo) -> bytes:
        data_start = int(bank.offset) + int(bank.data_relative_offset)
        start = data_start + int(sample.data_offset)
        end = start + int(sample.data_size)
        if start < 0 or end > len(data) or end < start:
            raise ValueError("FSB5 sample payload is outside the audio bank.")
        return bytes(data[start:end])

    @staticmethod
    def vorbis_packets(data: bytes, bank: FsbBankInfo, sample: FsbSampleInfo) -> List[bytes]:
        """Read FMOD FSB5 Vorbis packet framing (uint16 size + packet)."""
        raw = Fsb5Parser.sample_bytes(data, bank, sample)
        packets: List[bytes] = []
        p = 0
        while p + 2 <= len(raw):
            size = int(struct.unpack_from("<H", raw, p)[0])
            p += 2
            if size == 0:
                break
            if p + size > len(raw):
                # FSB sample ranges are 16-byte aligned.  A real packet may not
                # cross the advertised sample range; treat a crossing as corrupt.
                raise ValueError("Vorbis packet exceeds the FSB5 sample payload.")
            packets.append(bytes(raw[p:p + size]))
            p += size
        if not packets:
            raise ValueError("No Vorbis packets were found in this FSB5 sample.")
        return packets


@dataclass
class AudioPayloadInfo:
    kind: str
    payload_offset: int = 0
    payload_size: int = 0
    bank: Optional[FsbBankInfo] = None
    note: str = ""

    def report(self) -> str:
        lines=[f"Audio payload: {self.kind}", f"Payload offset: {self.payload_offset}", f"Payload bytes: {self.payload_size}"]
        if self.bank is not None:
            b=self.bank
            lines += [f"FSB5 version: {b.version}", f"Codec/mode: {b.mode_name}", f"Samples: {len(b.samples)}"]
            for sm in b.samples[:200]:
                loop=(f" loop={sm.loop_start}-{sm.loop_end}" if sm.loop_start is not None else "")
                crc=(f" setupCRC=0x{sm.vorbis_crc32:08X}" if sm.vorbis_crc32 is not None else "")
                lines.append(f"  [{sm.index:03d}] {sm.name} • {sm.frequency} Hz • {sm.channels} ch • {sm.duration:.2f}s{loop}{crc}")
            if len(b.samples)>200: lines.append(f"  ... {len(b.samples)-200} more samples")
        if self.note: lines.append(self.note)
        return "\n".join(lines)


def android_mp3_offset(data: bytes) -> int:
    """V10-compatible detection for Vector mobile audio streams.

    Mobile ``VuAudioStreamAsset`` payloads may be a four-byte Vector prefix
    followed directly by MP3/ID3 rather than an FSB5 bank.
    """
    for off in (4,0):
        if len(data)>off+3 and data[off:off+3]==b"ID3": return off
    for off in (4,0):
        if len(data)>off+1 and data[off]==0xFF and (data[off+1]&0xE0)==0xE0: return off
    return -1


def inspect_audio_payload(data: bytes) -> AudioPayloadInfo:
    bank=Fsb5Parser.find(data)
    if bank is not None:
        return AudioPayloadInfo("FSB5",bank.offset,bank.total_size,bank,
                                "FSB5 can be exported losslessly. Vorbis playback/decoding needs the matching FMOD setup packet or vgmstream.")
    off=android_mp3_offset(data)
    if off>=0:
        return AudioPayloadInfo("MP3",off,len(data)-off,None,
                                "Mobile Vector stream: raw MP3 after a small wrapper; no transcoding is required.")
    return AudioPayloadInfo("Unknown/unsupported",0,len(data),None,
                            "No FSB5 or MP3 payload signature was detected.")


def extract_audio_payload(data: bytes) -> Tuple[str, bytes, AudioPayloadInfo]:
    info=inspect_audio_payload(data)
    if info.kind=="FSB5" and info.bank is not None:
        return ".fsb", Fsb5Parser.embedded_bytes(data,info.bank), info
    if info.kind=="MP3":
        return ".mp3", bytes(data[info.payload_offset:info.payload_offset+info.payload_size]), info
    return ".bin", bytes(data), info


# ---------------------------------------------------------------------------
# FMOD Vorbis setup recovery from the game's own native binary
# ---------------------------------------------------------------------------

@dataclass
class VorbisSetupPacket:
    crc32: int
    packet: bytes
    binary_name: str = ""
    binary_offset: int = 0


class FmodVorbisSetupLibrary:
    """Recover FMOD Vorbis setup packets from BBR's libfmod.so.

    FMOD stores a table keyed by the same setup CRC serialized in FSB5 metadata.
    On Android/arm64 the table's setup pointer is relocated at load time, so the
    ELF RELA addend is the authoritative address of the packet.  Following that
    relocation is substantially safer than hard-coding addresses for one build.
    """

    def __init__(self, packets: Optional[Dict[int, VorbisSetupPacket]] = None):
        self.packets: Dict[int, VorbisSetupPacket] = dict(packets or {})
        self.source = ""

    def get(self, crc32: Optional[int]) -> Optional[bytes]:
        if crc32 is None:
            return None
        rec = self.packets.get(int(crc32) & 0xFFFFFFFF)
        return rec.packet if rec else None

    @staticmethod
    def _elf_sections(data: bytes):
        if len(data) < 64 or data[:4] != b"\x7fELF" or data[5] != 1:
            return [], 0
        elf_class = data[4]
        sections = []
        try:
            if elf_class == 2:  # ELF64 little-endian
                shoff = struct.unpack_from("<Q", data, 0x28)[0]
                shentsize = struct.unpack_from("<H", data, 0x3A)[0]
                shnum = struct.unpack_from("<H", data, 0x3C)[0]
                if shentsize < 64:
                    return [], elf_class
                for i in range(int(shnum)):
                    p = int(shoff) + i * int(shentsize)
                    if p + 64 > len(data):
                        break
                    name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                        "<IIQQQQIIQQ", data, p
                    )
                    sections.append({"type": typ, "flags": flags, "addr": addr,
                                     "offset": off, "size": size, "entsize": entsize})
            elif elf_class == 1:  # ELF32 little-endian
                shoff = struct.unpack_from("<I", data, 0x20)[0]
                shentsize = struct.unpack_from("<H", data, 0x2E)[0]
                shnum = struct.unpack_from("<H", data, 0x30)[0]
                if shentsize < 40:
                    return [], elf_class
                for i in range(int(shnum)):
                    p = int(shoff) + i * int(shentsize)
                    if p + 40 > len(data):
                        break
                    name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                        "<IIIIIIIIII", data, p
                    )
                    sections.append({"type": typ, "flags": flags, "addr": addr,
                                     "offset": off, "size": size, "entsize": entsize})
        except (struct.error, ValueError):
            return [], elf_class
        return sections, elf_class

    @staticmethod
    def _file_to_va(sections, file_offset: int) -> Optional[int]:
        for sec in sections:
            off = int(sec["offset"]); size = int(sec["size"])
            if size > 0 and off <= file_offset < off + size:
                return int(sec["addr"]) + (int(file_offset) - off)
        return None

    @staticmethod
    def _va_to_file(sections, va: int) -> Optional[int]:
        for sec in sections:
            addr = int(sec["addr"]); size = int(sec["size"])
            if size > 0 and addr <= va < addr + size:
                return int(sec["offset"]) + (int(va) - addr)
        return None

    @classmethod
    def from_binary_bytes(cls, data: bytes, name: str = "libfmod.so"):
        lib = cls()
        lib.source = name
        sections, elf_class = cls._elf_sections(data)
        if not sections:
            raise ValueError("The selected FMOD binary is not a supported little-endian ELF file.")

        rela: Dict[int, int] = {}
        for sec in sections:
            if int(sec["type"]) != 4:  # SHT_RELA
                continue
            off = int(sec["offset"]); size = int(sec["size"])
            ent = int(sec.get("entsize") or (24 if elf_class == 2 else 12))
            if ent <= 0:
                ent = 24 if elf_class == 2 else 12
            end = min(len(data), off + size)
            p = off
            while p + ent <= end:
                try:
                    if elf_class == 2:
                        r_offset, _r_info, r_addend = struct.unpack_from("<QQq", data, p)
                    else:
                        r_offset, _r_info, r_addend = struct.unpack_from("<IIi", data, p)
                    rela[int(r_offset)] = int(r_addend)
                except struct.error:
                    break
                p += ent

        # Search likely table entries rather than assuming CRC values in advance.
        # A setup table entry contains a uint32 CRC, followed nearby by a pointer
        # relocation to a packet beginning with 0x05 + "vorbis", then a uint32
        # packet byte count immediately after the pointer field.
        seen = set()
        for p in range(0, max(0, len(data) - 36), 4):
            try:
                crc = int(struct.unpack_from("<I", data, p)[0])
            except struct.error:
                break
            if crc == 0 or crc in seen:
                continue
            p_va = cls._file_to_va(sections, p)
            if p_va is None:
                continue
            found = None
            # Current arm64 FMOD uses +20, but keep the scan structural.
            for delta in range(4, 33, 4):
                target_va = int(p_va) + delta
                setup_va = rela.get(target_va)
                if setup_va is None:
                    continue
                setup_off = cls._va_to_file(sections, setup_va)
                if setup_off is None or setup_off < 0 or setup_off + 7 > len(data):
                    continue
                if data[setup_off:setup_off + 7] != b"\x05vorbis":
                    continue
                ptr_size = 8 if elf_class == 2 else 4
                size_off = p + delta + ptr_size
                if size_off + 4 > len(data):
                    continue
                packet_size = int(struct.unpack_from("<I", data, size_off)[0])
                if packet_size < 32 or packet_size > 1_000_000:
                    continue
                if setup_off + packet_size > len(data):
                    continue
                packet = bytes(data[setup_off:setup_off + packet_size])
                if not packet.startswith(b"\x05vorbis"):
                    continue
                found = VorbisSetupPacket(
                    crc32=crc, packet=packet, binary_name=name,
                    binary_offset=int(setup_off)
                )
                break
            if found is not None:
                lib.packets[crc] = found
                seen.add(crc)

        if not lib.packets:
            raise ValueError("No FMOD Vorbis setup table was recovered from the native binary.")
        return lib

    @classmethod
    def from_binary_file(cls, path: str):
        p = Path(path)
        return cls.from_binary_bytes(p.read_bytes(), p.name)

    @staticmethod
    def _zip_fmod_bytes(zf: zipfile.ZipFile, depth: int = 0):
        names = zf.namelist()
        # Prefer the 64-bit Android binary. It contains the setup table used by
        # the arm64 build and gives us RELA addends directly.
        direct = sorted(
            [n for n in names if n.lower().endswith(("/libfmod.so", "/fmod.so"))],
            key=lambda n: (0 if "arm64-v8a" in n.lower() else 1, len(n))
        )
        for n in direct:
            try:
                return zf.read(n), n
            except Exception:
                pass
        if depth >= 2:
            return None
        nested = sorted(
            [n for n in names if n.lower().endswith((".apk", ".zip"))],
            key=lambda n: (0 if "arm64" in n.lower() else 1, len(n))
        )
        for n in nested:
            try:
                blob = zf.read(n)
                with zipfile.ZipFile(BytesIO(blob), "r") as nz:
                    result = FmodVorbisSetupLibrary._zip_fmod_bytes(nz, depth + 1)
                    if result:
                        data, inner = result
                        return data, n + "!" + inner
            except Exception:
                continue
        return None

    @classmethod
    def from_game_package(cls, path: str):
        p = Path(path)
        if p.suffix.lower() == ".so":
            return cls.from_binary_file(str(p))
        if not zipfile.is_zipfile(str(p)):
            raise ValueError("Game package is not a ZIP/APK/XAPK/APKS archive and is not libfmod.so.")
        with zipfile.ZipFile(str(p), "r") as zf:
            result = cls._zip_fmod_bytes(zf)
        if not result:
            raise ValueError("libfmod.so was not found in the game package.")
        data, name = result
        return cls.from_binary_bytes(data, name)


# ---------------------------------------------------------------------------
# Ogg/Vorbis reconstruction
# ---------------------------------------------------------------------------

_OGG_POLY = 0x04C11DB7
_OGG_CRC_TABLE = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = (((_r << 1) ^ _OGG_POLY) & 0xFFFFFFFF) if (_r & 0x80000000) else ((_r << 1) & 0xFFFFFFFF)
    _OGG_CRC_TABLE.append(_r)


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[((crc >> 24) & 0xFF) ^ b]
    return crc


def _ogg_page(packet: bytes, serial: int, sequence: int, *, bos=False, eos=False,
              granule: int = 0) -> bytes:
    segments = []
    remaining = len(packet)
    while remaining >= 255:
        segments.append(255)
        remaining -= 255
    segments.append(remaining)
    if len(segments) > 255:
        raise ValueError("Vorbis packet is too large for the simple one-packet Ogg page writer.")
    flags = (2 if bos else 0) | (4 if eos else 0)
    header = bytearray(
        b"OggS" + bytes([0, flags])
        + struct.pack("<QII", int(granule) & 0xFFFFFFFFFFFFFFFF,
                      int(serial) & 0xFFFFFFFF, int(sequence) & 0xFFFFFFFF)
        + b"\x00\x00\x00\x00"
        + bytes([len(segments)]) + bytes(segments)
    )
    blob = header + packet
    struct.pack_into("<I", blob, 22, _ogg_crc(blob))
    return bytes(blob)


def _vorbis_identification_packet(channels: int, sample_rate: int) -> bytes:
    # BBR2's FMOD Vorbis banks use the standard 256/2048 Vorbis block sizes.
    # Those are the block exponents used by the recovered setup packets.
    blocksize = (11 << 4) | 8
    return b"\x01vorbis" + struct.pack(
        "<IBIiiiBB", 0, max(1, int(channels)), max(1, int(sample_rate)),
        0, 0, 0, blocksize, 1
    )


def _vorbis_comment_packet() -> bytes:
    vendor = b"BBR Vector Studio"
    return (
        b"\x03vorbis" + struct.pack("<I", len(vendor)) + vendor
        + struct.pack("<I", 0) + b"\x01"
    )


def rebuild_fsb5_vorbis_ogg(data: bytes, bank: FsbBankInfo,
                            sample: FsbSampleInfo, setup_packet: bytes) -> bytes:
    if int(bank.mode) != 15:
        raise ValueError(f"Ogg reconstruction currently expects FSB5 Vorbis, not {bank.mode_name}.")
    if not setup_packet or not setup_packet.startswith(b"\x05vorbis"):
        raise ValueError("The matching FMOD Vorbis setup packet is missing or invalid.")
    if sample.frequency <= 0:
        raise ValueError("The sample rate is missing from the FSB5 metadata.")

    packets = Fsb5Parser.vorbis_packets(data, bank, sample)
    # Stable serial avoids Python's randomized hash() and keeps saved output
    # reproducible for the same source sample.
    serial_seed = f"{sample.name}|{sample.index}|{sample.sample_count}".encode("utf-8", "replace")
    serial = zlib.crc32(serial_seed) & 0xFFFFFFFF
    if serial == 0:
        serial = 0x42525232

    out = bytearray()
    out += _ogg_page(_vorbis_identification_packet(sample.channels, sample.frequency), serial, 0, bos=True, granule=0)
    out += _ogg_page(_vorbis_comment_packet(), serial, 1, granule=0)
    out += _ogg_page(setup_packet, serial, 2, granule=0)
    for i, packet in enumerate(packets):
        last = i == len(packets) - 1
        granule = int(sample.sample_count) if last else 0xFFFFFFFFFFFFFFFF
        out += _ogg_page(packet, serial, 3 + i, eos=last, granule=granule)
    return bytes(out)
