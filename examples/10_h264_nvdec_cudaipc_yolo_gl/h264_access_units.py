from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


VCL_NAL_TYPES = {1, 2, 3, 4, 5}


@dataclass(frozen=True)
class NalUnit:
    nal_type: int
    raw: bytes          # Includes Annex-B start code.
    ebsp: bytes         # Excludes start code and NAL header byte.

    @property
    def is_vcl(self) -> bool:
        return self.nal_type in VCL_NAL_TYPES

    @property
    def is_idr(self) -> bool:
        return self.nal_type == 5


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    def read_bit(self) -> int:
        bytepos = self.bitpos // 8
        if bytepos >= len(self.data):
            raise EOFError("end of RBSP")
        shift = 7 - (self.bitpos % 8)
        self.bitpos += 1
        return (self.data[bytepos] >> shift) & 1

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_ue(self) -> int:
        zeros = 0
        while self.read_bit() == 0:
            zeros += 1
        if zeros == 0:
            return 0
        return (1 << zeros) - 1 + self.read_bits(zeros)


def ebsp_to_rbsp(ebsp: bytes) -> bytes:
    """Remove H264 emulation-prevention bytes from a NAL payload."""
    out = bytearray()
    zeros = 0
    for b in ebsp:
        if zeros >= 2 and b == 0x03:
            zeros = 0
            continue
        out.append(b)
        if b == 0:
            zeros += 1
        else:
            zeros = 0
    return bytes(out)


def first_mb_in_slice(nal: NalUnit) -> int | None:
    """Return first_mb_in_slice for a VCL NAL, or None when it cannot be parsed."""
    if not nal.is_vcl:
        return None
    try:
        return BitReader(ebsp_to_rbsp(nal.ebsp)).read_ue()
    except Exception:
        return None


def start_code_positions(data: bytes) -> Iterator[tuple[int, int]]:
    """Yield (position, start_code_length) for Annex-B 3-byte or 4-byte start codes."""
    i = 0
    n = len(data)
    while i < n - 3:
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            yield i, 4
            i += 4
        elif data[i : i + 3] == b"\x00\x00\x01":
            yield i, 3
            i += 3
        else:
            i += 1


def parse_annexb_nals(data: bytes) -> list[NalUnit]:
    positions = list(start_code_positions(data))
    if not positions:
        raise ValueError("input is not Annex-B H264: no 00 00 01 / 00 00 00 01 start code found")

    nals: list[NalUnit] = []
    for idx, (pos, sc_len) in enumerate(positions):
        header = pos + sc_len
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(data)
        if header >= end:
            continue
        nal_header = data[header]
        nal_type = nal_header & 0x1F
        raw = data[pos:end]
        ebsp = data[header + 1 : end]
        nals.append(NalUnit(nal_type=nal_type, raw=raw, ebsp=ebsp))

    if not nals:
        raise ValueError("input contains start codes but no NAL units")
    return nals


def has_vcl(nals: list[NalUnit]) -> bool:
    return any(n.is_vcl for n in nals)


def pack_access_unit(nals: list[NalUnit]) -> bytes:
    return b"".join(n.raw for n in nals)


def split_by_aud(nals: list[NalUnit]) -> list[bytes]:
    """Split access units using AUD NAL type 9. This is the simplest/cleanest path."""
    units: list[bytes] = []
    prefix: list[NalUnit] = []
    current: list[NalUnit] = []

    for nal in nals:
        if nal.nal_type == 9:  # AUD = Access Unit Delimiter
            if current and has_vcl(current):
                units.append(pack_access_unit(current))
            elif current:
                prefix.extend(current)
            current = prefix + [nal]
            prefix = []
        elif current:
            current.append(nal)
        else:
            # Keep leading SPS/PPS/SEI and prepend it to the first AUD unit.
            prefix.append(nal)

    if current and has_vcl(current):
        units.append(pack_access_unit(current))

    return units


def split_by_slice_headers(nals: list[NalUnit]) -> list[bytes]:
    """Fallback splitter for Annex-B streams without AUD.

    This is intentionally small, not a full H264 parser. It works for common
    x264-style elementary streams where a new picture starts with a VCL NAL
    whose first_mb_in_slice is zero. For the most reliable demo, encode with
    aud=1 and use split_by_aud instead.
    """
    units: list[bytes] = []
    current: list[NalUnit] = []
    seen_vcl = False

    for nal in nals:
        if nal.is_vcl:
            first_mb = first_mb_in_slice(nal)
            starts_new_picture = seen_vcl and first_mb == 0
            if starts_new_picture:
                if has_vcl(current):
                    units.append(pack_access_unit(current))
                current = []
                seen_vcl = False
            current.append(nal)
            seen_vcl = True
            continue

        # Repeated SPS/PPS/SEI after a VCL usually belongs to the next AU.
        if seen_vcl and nal.nal_type in {6, 7, 8, 9, 10, 11, 12}:
            if has_vcl(current):
                units.append(pack_access_unit(current))
            current = []
            seen_vcl = False

        current.append(nal)

    if current and has_vcl(current):
        units.append(pack_access_unit(current))

    return units


def load_h264_access_units(path: str, *, require_aud: bool = False) -> list[bytes]:
    with open(path, "rb") as f:
        data = f.read()

    nals = parse_annexb_nals(data)
    has_aud = any(n.nal_type == 9 for n in nals)

    if has_aud:
        units = split_by_aud(nals)
        splitter_name = "AUD"
    elif require_aud:
        raise ValueError(
            "no AUD NAL units found. Re-create .h264 with x264 option aud=1, "
            "or run without --require-aud to use the simple fallback splitter."
        )
    else:
        units = split_by_slice_headers(nals)
        splitter_name = "slice-header fallback"

    if not units:
        raise ValueError("could not split any H264 access units from the file")

    keyframes = sum(1 for u in units if h264_is_keyframe(u))
    print(
        f"input: {path!r}, {len(nals)} NALs, {len(units)} access units, "
        f"{keyframes} IDR/keyframes, splitter={splitter_name}",
        flush=True,
    )
    return units


def h264_is_keyframe(access_unit: bytes) -> bool:
    try:
        return any(n.nal_type == 5 for n in parse_annexb_nals(access_unit))
    except Exception:
        return False
