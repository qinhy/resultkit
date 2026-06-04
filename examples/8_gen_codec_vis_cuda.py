#!/usr/bin/env python3
"""
Small H264 -> CUDA IPC -> OpenGL demo.

Pipeline, normally run in three terminals:

    # 1) Publish synthetic frames as H264 Annex-B access units.
    python cuda_ipc_h264_demo.py pub

    # 2) Subscribe H264, decode with PyNvVideoCodec/NVDEC, publish CUDA IPC images.
    python cuda_ipc_h264_demo.py bridge

    # 3) Subscribe CUDA IPC images and show with the OpenGL/PBO viewer.
    python cuda_ipc_h264_demo.py show

The code intentionally keeps only the demo path:
    BGR numpy test frame
      -> FFmpeg H264 access-unit
      -> Model4Mat.EncodedImageMatPubSub
      -> PyNvVideoCodec decoded CUDA tensor
      -> Model4Mat.ImageMatCUDAPubSub
      -> resultkit.cudavis.ImageMatCudaGlViewer
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Keep this if the script lives in a examples/tests folder next to resultkit.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, ImageShapeType, Model4Mat
from resultkit.mat import DataType, MatDevice


NS_PER_SECOND = 1_000_000_000
DEFAULT_ENCODED_TOPIC = "EncodedImageMatPubSub:h264"
DEFAULT_IMAGE_TOPIC = "ImageMatCUDAPubSub:decoded"


@dataclass(frozen=True)
class Config:
    width: int = 96
    height: int = 64
    fps: int = 30
    crf: int = 18
    device: int = 0
    max_frames: int | None = None
    stats_every: int = 100

    ffmpeg_bin: str = os.environ.get("FFMPEG_BIN", "ffmpeg")
    encoded_topic: str = DEFAULT_ENCODED_TOPIC
    image_topic: str = DEFAULT_IMAGE_TOPIC
    encoded_capacity: int = 64 * 1024
    num_slots: int = 3
    flip_y: bool = True

    @property
    def frame_period_ns(self) -> int:
        return int(NS_PER_SECOND / max(self.fps, 1))


class FpsMeter:
    def __init__(self, name: str, every: int):
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
    """Simple wall-clock limiter: first frame now, later frames at cfg.fps."""

    def __init__(self, fps: int):
        self.period_s = 1.0 / max(int(fps), 1)
        self.next_t = time.perf_counter()

    def wait(self) -> None:
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
            now = time.perf_counter()

        # Advance from the scheduled time, but if we fell behind, recover
        # without trying to replay every missed sleep.
        self.next_t = max(self.next_t + self.period_s, now)


def make_frame(i: int, width: int, height: int) -> np.ndarray:
    """Deterministic HWC/BGR test pattern."""
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    frame[:, :, 1] = (i * 7) % 255
    frame[:, :, 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    return frame


def require_ffmpeg(ffmpeg_bin: str) -> str:
    ffmpeg = shutil.which(ffmpeg_bin)
    if not ffmpeg:
        raise RuntimeError(f"ffmpeg not found: {ffmpeg_bin!r}. Use --ffmpeg-bin or FFMPEG_BIN.")
    return ffmpeg


def run_ffmpeg(ffmpeg: str, args: list[str], input_bytes: bytes) -> bytes:
    proc = subprocess.run(
        [ffmpeg, *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return proc.stdout


def encode_h264_access_unit(frame_bgr: np.ndarray, cfg: Config, ffmpeg: str) -> bytes:
    """Encode one HWC/BGR frame as one independent H264 Annex-B access unit."""
    h, w = frame_bgr.shape[:2]
    args = [
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{w}x{h}",
        "-r", str(cfg.fps),
        "-i", "pipe:0",
        "-frames:v", "1",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", str(cfg.crf),
        "-x264-params", "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",
        "-pix_fmt", "yuv420p",
        "-f", "h264",
        "pipe:1",
    ]
    payload = run_ffmpeg(ffmpeg, args, np.ascontiguousarray(frame_bgr).tobytes())
    if not payload:
        raise RuntimeError("ffmpeg returned an empty H264 payload")
    return payload


def make_encoded_endpoint(cfg: Config, *, is_pub: bool):
    pkt = Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.H264,
        color_format=ColorFormat.BGR,
        frame_index=0,
        pts_ns=0,
        dts_ns=0,
        is_keyframe=True,
        width=int(cfg.width),
        height=int(cfg.height),
        valid_nbytes=0,
        data=np.zeros(int(cfg.encoded_capacity), dtype=np.uint8),
    )
    pkt.set_id(cfg.encoded_topic).init()
    pkt.is_pub = bool(is_pub)
    return pkt


def publish_encoded_frame(pub, payload: bytes, frame_index: int, cfg: Config) -> None:
    if len(payload) > cfg.encoded_capacity:
        raise RuntimeError(
            f"H264 payload is {len(payload)} bytes, larger than --encoded-capacity "
            f"{cfg.encoded_capacity}."
        )

    pts = int(frame_index * cfg.frame_period_ns)
    pub.codec = CodecFormat.H264
    pub.color_format = ColorFormat.BGR
    pub.frame_index = int(frame_index)
    pub.pts_ns = pts
    pub.dts_ns = pts
    pub.is_keyframe = True
    pub.width = int(cfg.width)
    pub.height = int(cfg.height)
    pub.valid_nbytes = int(len(payload))
    pub.pub(data=np.frombuffer(payload, dtype=np.uint8).copy())


def packet_nbytes(pkt) -> int:
    if hasattr(pkt, "nbytes") and callable(pkt.nbytes):
        return int(pkt.nbytes())
    return int(getattr(pkt, "valid_nbytes", 0))


def packet_payload(pkt) -> bytes:
    valid_nbytes = int(getattr(pkt, "valid_nbytes", 0))
    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        out = payload.tobytes() if isinstance(payload, np.ndarray) else bytes(payload)
        return out[:valid_nbytes] if valid_nbytes else out

    data = np.asarray(pkt.data).reshape(-1)
    return data[:valid_nbytes].tobytes()


def close_quietly(obj) -> None:
    try:
        obj.close()
    except Exception:
        pass


def encoded_publisher_loop(cfg: Config) -> None:
    ffmpeg = require_ffmpeg(cfg.ffmpeg_bin)
    pub = make_encoded_endpoint(cfg, is_pub=True)
    fps = FpsMeter("encoded pub", cfg.stats_every)

    print(f"pub  : {cfg.width}x{cfg.height} H264 -> {cfg.encoded_topic!r}", flush=True)
    frame_index = 0
    next_frame_t = time.perf_counter()

    try:
        while cfg.max_frames is None or frame_index < cfg.max_frames:
            frame_index += 1
            frame = make_frame(frame_index, cfg.width, cfg.height)
            payload = encode_h264_access_unit(frame, cfg, ffmpeg)
            publish_encoded_frame(pub, payload, frame_index, cfg)
            fps.tick()

            next_frame_t += 1.0 / max(cfg.fps, 1)
            time.sleep(max(0.0, next_frame_t - time.perf_counter()))
    finally:
        close_quietly(pub)


def require_pynvvideocodec():
    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        try:
            import pynvvideocodec as nvc
        except ImportError as exc:
            raise RuntimeError("PyNvVideoCodec/pynvvideocodec is required for bridge mode.") from exc
    return nvc


class NvdecH264:
    """Tiny PyNvVideoCodec wrapper that returns CUDA torch tensors."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nvc = require_pynvvideocodec()
        self.index = 0
        self._packet_buffers: list[np.ndarray] = []
        self.decoder = self._create_decoder()

    def _create_decoder(self):
        codec = getattr(getattr(self.nvc, "cudaVideoCodec", None), "H264", None)
        if codec is None:
            raise RuntimeError("PyNvVideoCodec cudaVideoCodec.H264 was not found.")

        kwargs = dict(
            gpuid=int(self.cfg.device),
            codec=codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
            maxwidth=int(self.cfg.width),
            maxheight=int(self.cfg.height),
        )

        # Keep the published CUDA IPC image simple: one HWC RGB uint8 frame.
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

    def _packet(self, payload: bytes):
        packet_cls = getattr(self.nvc, "PacketData", None)
        if packet_cls is None:
            return payload

        packet = packet_cls()
        packet_buf = np.frombuffer(payload, dtype=np.uint8).copy()
        self._packet_buffers.append(packet_buf)
        self._packet_buffers = self._packet_buffers[-16:]

        flag = self._end_of_picture_flag()

        # PyNvVideoCodec builds differ: set both native and documented names.
        self._try_set(packet, "bsl_data", int(packet_buf.ctypes.data))
        self._try_set(packet, "bsl", int(packet_buf.nbytes))
        self._try_set(packet, "bitstream", payload)
        self._try_set(packet, "size", int(packet_buf.nbytes))
        self._try_set(packet, "pts", int(self.index))
        self._try_set(packet, "dts", int(self.index))
        self._try_set(packet, "duration", 1)
        self._try_set(packet, "key", True)
        self._try_set(packet, "decode_flag", flag)
        self._try_set(packet, "flags", flag)
        return packet

    @staticmethod
    def _as_list(frames) -> list:
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

    def decode(self, payload: bytes) -> list:
        packet = self._packet(payload)
        frames = self.decoder.Decode(packet)
        self.index += 1
        if hasattr(self.decoder, "SyncOnCUStream"):
            self.decoder.SyncOnCUStream()

        import torch

        tensors = [torch.from_dlpack(frame) for frame in self._as_list(frames)]
        for t in tensors:
            if not t.is_cuda:
                raise RuntimeError("NVDEC returned a CPU tensor; expected CUDA device memory.")
        return tensors


def as_hwc_rgb8(tensor, cfg: Config):
    """Crop possible pitch padding and normalize PyNv output to HWC/RGB/uint8."""
    import torch

    t = tensor.detach()

    # Some RGB outputs are CHW/RGBP. The GL IPC endpoint below is HWC.
    if t.ndim == 3 and int(t.shape[0]) == 3 and int(t.shape[-1]) != 3:
        t = t[:, : cfg.height, : cfg.width].permute(1, 2, 0)
    elif t.ndim == 3 and int(t.shape[-1]) >= 3:
        t = t[: cfg.height, : cfg.width, :3]
    else:
        raise RuntimeError(f"unsupported decoded tensor shape: {tuple(t.shape)}")

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
    try:
        data = gpuarray.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
    except Exception as e:
        print(e)
        raise ValueError("PyCUDA is trying to allocate GPU memory, but probably no current one. There are maybe muti-context exits.")
    
    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.RGB,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=mat_device(cfg.device),
        data=data,
        num_slots=int(cfg.num_slots),
    )
    img.set_id(cfg.image_topic).init()
    try:
        img.is_pub = bool(is_pub)
    except Exception:
        pass
    return img


def decode_bridge_loop(cfg: Config) -> None:
    """EncodedImageMatPubSub -> NVDEC CUDA tensor -> ImageMatCUDAPubSub."""
    with pycuda_context(cfg.device):
        sub = make_encoded_endpoint(cfg, is_pub=False)
        pub = make_cuda_image_endpoint(cfg, is_pub=True)
        dec = NvdecH264(cfg)
        fps = FpsMeter("decode bridge", cfg.stats_every)
        pacer = FramePacer(cfg.fps)

        print(
            f"bridge: {cfg.encoded_topic!r} -> NVDEC -> {cfg.image_topic!r} "
            f"at <= {cfg.fps} fps",
            flush=True,
        )
        count = 0

        try:
            while cfg.max_frames is None or count < cfg.max_frames:
                pkt = sub.sub()
                if packet_nbytes(pkt) <= 0:
                    time.sleep(0.001)
                    continue
                if CodecFormat(pkt.codec) != CodecFormat.H264:
                    raise RuntimeError(f"expected H264 packet, got {pkt.codec}")

                for tensor in dec.decode(packet_payload(pkt)):
                    pacer.wait()
                    pub.pub(data=as_hwc_rgb8(tensor, cfg))
                    count += 1
                    fps.tick()
                    if cfg.max_frames is not None and count >= cfg.max_frames:
                        break
        finally:
            close_quietly(pub)
            close_quietly(sub)


def gl_show_loop(cfg: Config) -> None:
    """ImageMatCUDAPubSub -> OpenGL/PBO viewer."""
    from resultkit.cudavis import ImageMatCudaGlViewer

    vis = ImageMatCudaGlViewer(
        width=int(cfg.width),
        height=int(cfg.height),
        fps=float(cfg.fps),
        device=int(cfg.device),
        flip_y=bool(cfg.flip_y),
        max_frames=cfg.max_frames,
    )
    vis.init()

    img = make_cuda_image_endpoint(cfg, is_pub=False)
    print(f"show : GL viewer subscribing {cfg.image_topic!r}", flush=True)

    try:
        vis.run(img=img)
    finally:
        close_quietly(img)


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Minimal encoded pub -> NVDEC CUDA IPC -> GL show demo.")
    p.add_argument("role", choices=("pub", "bridge", "show", "sub", "gl"))
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stats-every", type=int, default=100)
    p.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    p.add_argument("--topic", dest="encoded_topic", default=DEFAULT_ENCODED_TOPIC)
    p.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    p.add_argument("--encoded-capacity", type=int, default=64 * 1024)
    p.add_argument("--num-slots", type=int, default=3)
    p.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def config_from_args(args) -> Config:
    return Config(
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        device=args.device,
        max_frames=args.max_frames,
        stats_every=args.stats_every,
        ffmpeg_bin=args.ffmpeg_bin,
        encoded_topic=args.encoded_topic,
        image_topic=args.image_topic,
        encoded_capacity=args.encoded_capacity,
        num_slots=args.num_slots,
        flip_y=args.flip_y,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)

    role = {"sub": "bridge", "gl": "show"}.get(args.role, args.role)

    try:
        if role == "pub":
            encoded_publisher_loop(cfg)
        elif role == "bridge":
            decode_bridge_loop(cfg)
        elif role == "show":
            gl_show_loop(cfg)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
