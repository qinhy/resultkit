#!/usr/bin/env python3
"""
Minimal H264 file -> NVDEC -> CUDA IPC -> OpenGL demo.

Runtime pipeline:

    .h264 Annex-B file
        -> access-unit splitter, one encoded frame at a time
        -> PyNvVideoCodec / NVDEC, decoded in CUDA memory
        -> resultkit Model4Mat.ImageMatCUDAPubSub CUDA IPC image
        -> resultkit OpenGL/PBO viewer

Run in two terminals:

    # Terminal 1: subscribe CUDA IPC image and show it with OpenGL
    python h264_nvdec_cudaipc_gl.py show --width 1280 --height 720 --fps 30

    # Terminal 2: read .h264, NVDEC decode, publish CUDA IPC image
    python h264_nvdec_cudaipc_gl.py decode-pub --input demo.h264 --width 1280 --height 720 --fps 30

Notes:
    - No FFmpeg is used at runtime.
    - The input should be raw H264 Annex-B, not MP4.
    - Best input is encoded with AUD NALs, repeat SPS/PPS headers, and no B frames:
        -x264-params repeat-headers=1:aud=1 -bf 0

PS:
ffmpeg -i input.mp4 \
  -an \
  -vf "fps=30,format=yuv420p" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -profile:v baseline \
  -bf 0 \
  -g 30 \
  -keyint_min 30 \
  -sc_threshold 0 \
  -x264-params "repeat-headers=1:aud=1" \
  -f h264 \
  demo_iponly_30fps.h264
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

# Keep this if the demo lives in an examples/tests folder next to resultkit.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
from resultkit.mat import DataType, MatDevice


DEFAULT_IMAGE_TOPIC = "ImageMatCUDAPubSub:h264FileDemo"
VCL_NAL_TYPES = {1, 2, 3, 4, 5}


@dataclass(frozen=True)
class Config:
    input_path: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    device: int = 0
    image_topic: str = DEFAULT_IMAGE_TOPIC
    num_slots: int = 3
    max_frames: int | None = None
    stats_every: int = 100
    loop: bool = False
    flip_y: bool = True
    require_aud: bool = False


class FpsMeter:
    def __init__(self, name: str, every: int = 100):
        self.name = name
        self.every = max(1, int(every))
        self.count = 0
        self.t0 = time.perf_counter()

    def tick(self) -> None:
        self.count += 1
        if self.count % self.every:
            return
        dt = time.perf_counter() - self.t0
        if dt > 0:
            print(f"{self.name}: {self.count / dt:.2f} fps", flush=True)


class FramePacer:
    def __init__(self, fps: int):
        self.period = 1.0 / max(int(fps), 1)
        self.next_t = time.perf_counter()

    def sleep(self) -> None:
        self.next_t += self.period
        time.sleep(max(0.0, self.next_t - time.perf_counter()))


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


def require_pynvvideocodec():
    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        try:
            import pynvvideocodec as nvc
        except ImportError as exc:
            raise RuntimeError("PyNvVideoCodec/pynvvideocodec is required for decode-pub mode") from exc
    return nvc


class NvdecH264Decoder:
    """Tiny PyNvVideoCodec wrapper. Output is expected to stay in CUDA memory."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nvc = require_pynvvideocodec()
        self.packet_index = 0
        self._packet_buffers: list[np.ndarray] = []
        self.decoder = self._create_decoder()

    def _create_decoder(self):
        codec = getattr(getattr(self.nvc, "cudaVideoCodec", None), "H264", None)
        if codec is None:
            raise RuntimeError("PyNvVideoCodec cudaVideoCodec.H264 was not found")

        kwargs = dict(
            gpuid=int(self.cfg.device),
            codec=codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
            maxwidth=int(self.cfg.width),
            maxheight=int(self.cfg.height),
        )

        # Ask for a single RGB image plane so it can be published directly to GL.
        output_color_type = getattr(self.nvc, "OutputColorType", None)
        if output_color_type is not None and hasattr(output_color_type, "RGB"):
            kwargs["outputColorType"] = output_color_type.RGB

        latency_type = getattr(self.nvc, "DisplayDecodeLatencyType", None)
        if latency_type is not None:
            latency = getattr(latency_type, "ZERO", getattr(latency_type, "LOW", None))
            if latency is not None:
                kwargs["latency"] = latency

        try:
            return self.nvc.CreateDecoder(**kwargs)
        except TypeError:
            # Some builds expose fewer kwargs.
            kwargs.pop("outputColorType", None)
            kwargs.pop("latency", None)
            return self.nvc.CreateDecoder(**kwargs)

    def _end_of_picture_flag(self):
        flag_enum = getattr(self.nvc, "VideoPacketFlag", None)
        if flag_enum is None:
            return 0
        for name in ("ENDOFPICTURE", "END_OF_PICTURE", "ENDOFPICTURE_FLAG"):
            if hasattr(flag_enum, name):
                return getattr(flag_enum, name)
        return 0

    @staticmethod
    def _try_set(obj, name: str, value) -> bool:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

    def _make_packet(self, payload: bytes, *, keyframe: bool):
        packet_cls = getattr(self.nvc, "PacketData", None)
        if packet_cls is None:
            return payload

        packet = packet_cls()

        # Keep a numpy-owned buffer alive because some PyNvVideoCodec builds
        # read bsl_data as a native pointer.
        packet_buf = np.frombuffer(payload, dtype=np.uint8).copy()
        self._packet_buffers.append(packet_buf)
        if len(self._packet_buffers) > 16:
            del self._packet_buffers[:-16]

        flag = self._end_of_picture_flag()

        # Set both internal/native and documented field names for compatibility.
        self._try_set(packet, "bsl_data", int(packet_buf.ctypes.data))
        self._try_set(packet, "bsl", int(packet_buf.nbytes))
        self._try_set(packet, "bitstream", payload)
        self._try_set(packet, "size", int(packet_buf.nbytes))
        self._try_set(packet, "pts", int(self.packet_index))
        self._try_set(packet, "dts", int(self.packet_index))
        self._try_set(packet, "duration", 1)
        self._try_set(packet, "key", bool(keyframe))
        self._try_set(packet, "decode_flag", flag)
        self._try_set(packet, "flags", flag)
        return packet

    @staticmethod
    def _frames_as_list(frames) -> list:
        if frames is None:
            return []
        if frames.__class__.__name__ == "DecodedFrame" or hasattr(frames, "__dlpack__"):
            return [frames]
        if isinstance(frames, (list, tuple)):
            return list(frames)
        try:
            return list(frames)
        except TypeError:
            return [frames]

    def decode(self, payload: bytes, *, keyframe: bool = False) -> list:
        packet = self._make_packet(payload, keyframe=keyframe)
        frames = self.decoder.Decode(packet)
        self.packet_index += 1

        if hasattr(self.decoder, "SyncOnCUStream"):
            self.decoder.SyncOnCUStream()

        import torch

        tensors = [torch.from_dlpack(frame) for frame in self._frames_as_list(frames)]
        for tensor in tensors:
            if not tensor.is_cuda:
                raise RuntimeError("NVDEC returned CPU memory; expected CUDA device memory")
        return tensors


def as_hwc_rgb8(tensor, cfg: Config):
    """Normalize PyNv RGB output to contiguous HWC/RGB/uint8 CUDA tensor."""
    import torch

    t = tensor.detach()

    if t.ndim == 3 and int(t.shape[0]) == 3 and int(t.shape[-1]) != 3:
        # CHW/RGBP -> HWC/RGB, crop pitch padding.
        t = t[:, : cfg.height, : cfg.width].permute(1, 2, 0)
    elif t.ndim == 3 and int(t.shape[-1]) >= 3:
        # HWC/RGB, crop pitch padding and optional alpha.
        t = t[: cfg.height, : cfg.width, :3]
    else:
        raise RuntimeError(
            f"unsupported decoded CUDA tensor shape {tuple(t.shape)}. "
            "This demo expects PyNvVideoCodec OutputColorType.RGB."
        )

    if t.dtype != torch.uint8:
        t = t.clamp(0, 255).to(torch.uint8)

    return t.contiguous()


@contextmanager
def pycuda_context(device: int):
    import pycuda.driver as cuda

    cuda.init()
    ctx = cuda.Device(int(device)).make_context()
    try:
        yield
    finally:
        ctx.pop()
        ctx.detach()


def mat_device(device: int):
    return getattr(MatDevice, f"CUDA{int(device)}", MatDevice.CUDA0)


def make_cuda_image_endpoint(cfg: Config, *, is_pub: bool):
    import pycuda.gpuarray as gpuarray

    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError("Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build")

    try:
        data = gpuarray.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
    except Exception as e:
        print(e)
        raise ValueError("PyCUDA context is trying to allocate GPU memory, but probably no current one. There are maybe muti-context exits.")
    
    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.RGB,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=mat_device(cfg.device),
        data=data,
        num_slots=int(cfg.num_slots),
    )
    img.set_id(cfg.image_topic).init()

    # Some resultkit versions use is_pub, some infer it from pub/sub calls.
    try:
        img.is_pub = bool(is_pub)
    except Exception:
        pass

    return img


def close_quietly(obj) -> None:
    try:
        obj.close()
    except Exception:
        pass


def decode_pub_loop(cfg: Config) -> None:
    if not cfg.input_path:
        raise ValueError("decode-pub requires --input demo.h264")

    access_units = load_h264_access_units(cfg.input_path, require_aud=cfg.require_aud)

    with pycuda_context(cfg.device):
        decoder = NvdecH264Decoder(cfg)
        image_pub = make_cuda_image_endpoint(cfg, is_pub=True)
        pacer = FramePacer(cfg.fps)
        meter = FpsMeter("decode-pub", cfg.stats_every)

        print(
            f"decode-pub: {cfg.input_path!r} -> NVDEC CUDA -> {cfg.image_topic!r} "
            f"({cfg.width}x{cfg.height} @ {cfg.fps} fps)",
            flush=True,
        )

        published = 0
        try:
            while True:
                for access_unit in access_units:
                    for tensor in decoder.decode(access_unit, keyframe=h264_is_keyframe(access_unit)):
                        image_pub.pub(data=as_hwc_rgb8(tensor, cfg))
                        published += 1
                        meter.tick()
                        pacer.sleep()

                        if cfg.max_frames is not None and published >= cfg.max_frames:
                            return

                if not cfg.loop:
                    return
        finally:
            close_quietly(image_pub)


def gl_show_loop(cfg: Config) -> None:
    from resultkit.cudavis import ImageMatCudaGlViewer

    # Follow resultkit's GL path: initialize the viewer first, then create the
    # CUDA IPC image endpoint in the CUDA/GL context owned by the viewer.
    viewer = ImageMatCudaGlViewer(
        width=int(cfg.width),
        height=int(cfg.height),
        fps=float(cfg.fps),
        device=int(cfg.device),
        flip_y=bool(cfg.flip_y),
        max_frames=cfg.max_frames,
    )
    viewer.init()

    image_sub = make_cuda_image_endpoint(cfg, is_pub=False)

    print(
        f"show: GL viewer subscribing {cfg.image_topic!r} "
        f"({cfg.width}x{cfg.height} @ {cfg.fps} fps)",
        flush=True,
    )

    try:
        viewer.run(img=image_sub)
    finally:
        close_quietly(image_sub)


def torch_loop(cfg: Config) -> None:
    import time
    import pycuda.driver as cuda
    import torch
    import cv2

    cuda.init()
    # Make PyTorch initialize CUDA on the same device first.
    torch.cuda.set_device(cfg.device)
    torch.empty(1, device=f"cuda:{cfg.device}")
    # Retain the CUDA primary context, which is what PyTorch normally uses.
    ctx = cuda.Device(cfg.device).retain_primary_context()
    ctx.push()

    try:
        image_sub = make_cuda_image_endpoint(cfg, is_pub=False)
        last_sequence = -1

        while True:
            image_sub.sub(copy=False, sync=True)

            if getattr(image_sub, "_remote_mem", None) is None:
                time.sleep(0.001)
                continue

            sequence = int(getattr(image_sub, "sequence", -1))
            if sequence == last_sequence:
                time.sleep(0.001)
                continue

            last_sequence = sequence

            t = image_sub.get_data_torch(copy=False, sync=False)

            print(
                f"seq={sequence} shape={tuple(t.shape)} "
                f"dtype={t.dtype} device={t.device} ptr={hex(t.data_ptr())} ",
                # f"sum={t.float().mean()}",
                flush=True,
            )
            
            # bgr = t.detach().cpu().numpy()[:, :, ::-1]
            # cv2.imshow(cfg.image_topic, bgr)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    finally:
        try:
            image_sub.close()
        except Exception:
            pass

        ctx.pop()
        ctx.detach()

def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(
        description="Demo: raw .h264 Annex-B file -> NVDEC -> CUDA IPC -> OpenGL, no FFmpeg runtime."
    )
    p.add_argument("role", choices=("decode-pub", "show", "pub", "gl", "torch"))
    p.add_argument("--input", dest="input_path", help="raw H264 Annex-B file, required for decode-pub")
    p.add_argument("--width", type=int, default=1280, help="decoded frame width / GL image width")
    p.add_argument("--height", type=int, default=720, help="decoded frame height / GL image height")
    p.add_argument("--fps", type=int, default=30, help="playback/publish pacing FPS")
    p.add_argument("--device", type=int, default=0, help="CUDA device id")
    p.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    p.add_argument("--num-slots", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stats-every", type=int, default=100)
    p.add_argument("--loop", action="store_true", help="loop the .h264 file in decode-pub mode")
    p.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--require-aud",
        action="store_true",
        help="fail if the .h264 stream has no AUD NALs instead of using the simple fallback splitter",
    )
    return p.parse_args(argv)


def config_from_args(args) -> Config:
    return Config(
        input_path=args.input_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        device=args.device,
        image_topic=args.image_topic,
        num_slots=args.num_slots,
        max_frames=args.max_frames,
        stats_every=args.stats_every,
        loop=args.loop,
        flip_y=args.flip_y,
        require_aud=args.require_aud,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    role = {"pub": "decode-pub", "gl": "show"}.get(args.role, args.role)

    try:
        if role == "decode-pub":
            decode_pub_loop(cfg)
        elif role == "show":
            gl_show_loop(cfg)
        elif role == "torch":
            torch_loop(cfg)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
