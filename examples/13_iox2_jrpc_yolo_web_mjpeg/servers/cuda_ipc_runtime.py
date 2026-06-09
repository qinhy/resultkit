from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from h264_access_units import h264_is_keyframe, load_h264_access_units

# Keep this if the demo lives in an examples/tests folder next to resultkit.
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent.parent)))

from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
from resultkit.mat import DataType

DEFAULT_IMAGE_TOPIC = "ImageMatCUDAPubSub:h264FileDemo"


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

    # Use the CUDA primary context so PyCUDA, PyTorch, and CUDA IPC agree.
    ctx = cuda.Device(int(device)).retain_primary_context()
    ctx.push()

    try:
        yield
    finally:
        ctx.pop()
        ctx.detach()


def mat_device(device: int):
    from resultkit.mat import MatDevice

    return getattr(MatDevice, f"CUDA{int(device)}", MatDevice.CUDA0)


def make_cuda_image_endpoint(cfg: Config, *, is_pub: bool):
    import pycuda.gpuarray as gpuarray

    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError("Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build")

    try:
        data = gpuarray.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
    except Exception as e:
        print(e)
        raise ValueError(
            "PyCUDA context is trying to allocate GPU memory, but probably no current one. "
            "There are maybe muti-context exits."
        )

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


class StoppableLoop:
    """Small base class for loops that can run blocking or in a worker thread."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._exception: BaseException | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    def start(self, *, blocking: bool = True) -> "StoppableLoop":
        """Start the loop.

        Args:
            blocking: When True, run on the current thread. When False, run in a
                daemon worker thread and return immediately.
        """
        if self._running:
            return self

        self._stop_event.clear()
        self._exception = None

        if blocking:
            self._run_guarded()
        else:
            self._thread = threading.Thread(target=self._run_guarded, daemon=True)
            self._thread.start()

        return self

    def stop(self, *, join: bool = True, timeout: float | None = None) -> None:
        """Request the loop to stop and optionally wait for the worker thread."""
        self._stop_event.set()

        thread = self._thread
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def join(self, timeout: float | None = None) -> None:
        """Wait for a non-blocking start() worker thread to finish."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        if self._exception is not None:
            raise self._exception

    def _run_guarded(self) -> None:
        self._running = True
        try:
            self._run()
        except BaseException as exc:
            self._exception = exc
            raise
        finally:
            self._running = False

    def _run(self) -> None:
        raise NotImplementedError


class DecodePubLoop(StoppableLoop):
    """NVDEC H264 -> CUDA image publisher loop."""

    def _run(self) -> None:
        cfg = self.cfg

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
                while not self._stop_event.is_set():
                    for access_unit in access_units:
                        if self._stop_event.is_set():
                            return

                        for tensor in decoder.decode(access_unit, keyframe=h264_is_keyframe(access_unit)):
                            if self._stop_event.is_set():
                                return

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


class GlShowLoop(StoppableLoop):
    """CUDA IPC image subscriber -> OpenGL viewer loop."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.viewer = None
        self.image_sub = None

    def stop(self, *, join: bool = True, timeout: float | None = None) -> None:
        super().stop(join=False)

        # ImageMatCudaGlViewer is expected to own the GL run loop. Different
        # resultkit builds expose different shutdown method names, so try the
        # common ones if the viewer is already initialized.
        viewer = self.viewer
        if viewer is not None:
            for method_name in ("stop", "close", "shutdown", "destroy", "quit"):
                method = getattr(viewer, method_name, None)
                if callable(method):
                    try:
                        method()
                        break
                    except Exception:
                        pass

        thread = self._thread
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        from resultkit.cudavis import ImageMatCudaGlViewer

        cfg = self.cfg

        # Follow resultkit's GL path: initialize the viewer first, then create
        # the CUDA IPC image endpoint in the CUDA/GL context owned by viewer.
        self.viewer = ImageMatCudaGlViewer(
            width=int(cfg.width),
            height=int(cfg.height),
            fps=float(cfg.fps),
            device=int(cfg.device),
            flip_y=bool(cfg.flip_y),
            max_frames=cfg.max_frames,
        )
        self.viewer.init()

        self.image_sub = make_cuda_image_endpoint(cfg, is_pub=False)

        print(
            f"show: GL viewer subscribing {cfg.image_topic!r} "
            f"({cfg.width}x{cfg.height} @ {cfg.fps} fps)",
            flush=True,
        )

        try:
            self.viewer.run(img=self.image_sub)
        finally:
            close_quietly(self.image_sub)
            self.image_sub = None
            self.viewer = None


def decode_pub_loop(cfg: Config) -> None:
    """Backward-compatible function wrapper for the class-based loop."""
    DecodePubLoop(cfg).start(blocking=True)


def gl_show_loop(cfg: Config) -> None:
    """Backward-compatible function wrapper for the class-based loop."""
    GlShowLoop(cfg).start(blocking=True)

