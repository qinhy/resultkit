import contextlib
import ctypes
import os
import platform
import queue
import shutil
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import torch

from .generator import ImageMatGenerator
from ..MatModel import CodecFormat, ColorFormat
ColorType = ColorFormat
from ..logger import logger


class _LiveBitstreamFeeder:
    """
    Feeds compressed OAK H264/H265 bytes into PyNvVideoCodec CreateDemuxer(callback).

    One feeder is used per encoded stream: RGB, left mono, right mono.
    """

    def __init__(self, bitstream_queue, stop_event):
        self.q = bitstream_queue
        self.stop_event = stop_event
        self.pending = bytearray()
        self.eof = False
        self.total_bytes_fed = 0

    def feed_chunk(self, demuxer_buffer):
        if self.stop_event.is_set():
            self.pending.clear()
            self.eof = True
            return 0

        capacity = len(demuxer_buffer)

        while len(self.pending) == 0 and not self.eof:
            if self.stop_event.is_set():
                self.eof = True
                break

            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.eof = True
                break

            self.pending.extend(item)

        if len(self.pending) == 0 and self.eof:
            return 0

        n = min(capacity, len(self.pending))
        demuxer_buffer[:n] = self.pending[:n]
        del self.pending[:n]
        self.total_bytes_fed += n
        return n



class _StreamDecoderBackend(ABC):
    """Compressed H264/H265 bytes -> decoded torch.Tensor frames for one stream."""

    # True when next_tensor() already returns the public tensor shape/value range.
    # For gst-nvivafilter this is [1, 3, H, W], CUDA, normalized fp16/fp32.
    returns_public_tensor = False
    returns_normalized_tensor = False

    def __init__(
        self,
        owner,
        stream_name: str,
        codec: str,
        width: int,
        height: int,
        output_color: str,
        bitstream_queue_size: int,
        stop_event: threading.Event,
    ):
        self.owner = owner
        self.stream_name = stream_name
        self.codec = str(codec)
        self.width = int(width)
        self.height = int(height)
        self.output_color = str(output_color)
        self.bitstream_queue_size = int(bitstream_queue_size)
        self.stop_event = stop_event
        self.closed = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def push_bitstream(self, data: bytes):
        pass

    @abstractmethod
    def end_of_stream(self):
        pass

    @abstractmethod
    def next_tensor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def close(self):
        pass

    def stats(self) -> Dict[str, float]:
        return {}


class _PyNvVideoCodecStreamBackend(_StreamDecoderBackend):
    """dGPU backend. This keeps the original PyNvVideoCodec + DLPack behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nvc = None
        self.bitstream_q = queue.Queue(maxsize=self.bitstream_queue_size)
        self.feeder: Optional[_LiveBitstreamFeeder] = None
        self.demuxer = None
        self.decoder = None
        self.packet_iter = None
        self.pending_decoded_frames = []
        self.demux_packet_count = 0
        self._decoded_frame_refs = []

    @staticmethod
    def _output_color_type(nvc, name: str):
        if name == "rgbp":
            return nvc.OutputColorType.RGBP
        if name == "rgb":
            return nvc.OutputColorType.RGB
        if name == "native":
            return nvc.OutputColorType.NATIVE
        raise ValueError(f"Unsupported decoder output color: {name}")

    @staticmethod
    def _get_low_latency_enum(nvc):
        if hasattr(nvc, "DisplayDecodeLatencyType"):
            if hasattr(nvc.DisplayDecodeLatencyType, "LOW"):
                return nvc.DisplayDecodeLatencyType.LOW
        if hasattr(nvc, "DisplayDecodeLatency"):
            enum = nvc.DisplayDecodeLatency
            for name in ("DISPLAYDECODELATENCY_LOW", "LOW"):
                if hasattr(enum, name):
                    return getattr(enum, name)
        return None

    @staticmethod
    def _set_end_of_picture(nvc, packet):
        if not hasattr(nvc, "VideoPacketFlag"):
            return
        flag = None
        for name in ("ENDOFPICTURE", "END_OF_PICTURE"):
            if hasattr(nvc.VideoPacketFlag, name):
                flag = getattr(nvc.VideoPacketFlag, name)
                break
        if flag is None:
            return
        for attr in ("decode_flag", "flags"):
            try:
                setattr(packet, attr, flag)
                return
            except Exception:
                pass

    def start(self):
        # Import lazily so Jetson can import this module without PyNvVideoCodec.
        import PyNvVideoCodec as nvc

        self.nvc = nvc
        self.feeder = _LiveBitstreamFeeder(self.bitstream_q, self.stop_event)

    def _ensure_decoder(self):
        if self.decoder is not None:
            return

        nvc = self.nvc
        if nvc is None or self.feeder is None:
            raise RuntimeError("PyNvVideoCodec stream backend was not started")

        logger(f"[PyNvVideoCodecStreamBackend:info] Creating PyNvVideoCodec demuxer/decoder for {self.stream_name} stream...")
        self.demuxer = nvc.CreateDemuxer(self.feeder.feed_chunk)
        kwargs = {
            "gpuid": self.owner.gpu_id,
            "codec": self.demuxer.GetNvCodecId(),
            "usedevicememory": True,
            "maxwidth": self.width,
            "maxheight": self.height,
            "outputColorType": self._output_color_type(nvc, self.output_color),
        }
        if self.owner.low_latency:
            latency = self._get_low_latency_enum(nvc)
            if latency is not None:
                kwargs["latency"] = latency
            else:
                logger("[PyNvVideoCodecStreamBackend:warning] PyNvVideoCodec low-latency enum not found.")
        self.decoder = nvc.CreateDecoder(**kwargs)
        self.packet_iter = iter(self.demuxer)

    def push_bitstream(self, data: bytes):
        if self.closed or self.stop_event.is_set():
            return
        while not self.closed and not self.stop_event.is_set():
            try:
                self.bitstream_q.put(data, timeout=0.1)
                return
            except queue.Full:
                continue

    def end_of_stream(self):
        try:
            self.bitstream_q.put_nowait(None)
            return
        except queue.Full:
            try:
                self.bitstream_q.get_nowait()
            except Exception:
                pass
            try:
                self.bitstream_q.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass

    def _retain_decoded_frame_ref(self, frame):
        self._decoded_frame_refs.append(frame)
        max_refs = max(1, int(getattr(self.owner, "retain_decoded_frame_refs", 16)))
        if len(self._decoded_frame_refs) > max_refs:
            self._decoded_frame_refs = self._decoded_frame_refs[-max_refs:]

    def next_tensor(self) -> torch.Tensor:
        self._ensure_decoder()

        while not self.stop_event.is_set():
            if self.pending_decoded_frames:
                frame = self.pending_decoded_frames.pop(0)
                tensor = torch.from_dlpack(frame)
                self._retain_decoded_frame_ref(frame)
                return tensor

            if self.packet_iter is None or self.decoder is None:
                raise StopIteration

            try:
                packet = next(self.packet_iter)
            except StopIteration:
                raise
            except Exception:
                if self.stop_event.is_set() or self.closed:
                    raise StopIteration
                raise

            self.demux_packet_count += 1

            if self.owner.low_latency:
                self._set_end_of_picture(self.nvc, packet)

            try:
                frames = self.decoder.Decode(packet)
            except Exception:
                if self.stop_event.is_set() or self.closed:
                    raise StopIteration
                raise

            for frame in frames:
                self.pending_decoded_frames.append(frame)

        raise StopIteration

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.end_of_stream()
        self.pending_decoded_frames.clear()
        self._decoded_frame_refs.clear()
        self.packet_iter = None
        self.decoder = None
        self.demuxer = None
        self.feeder = None

    def stats(self) -> Dict[str, float]:
        return {
            "demux_packets": float(self.demux_packet_count),
            "fed_mb": float((self.feeder.total_bytes_fed if self.feeder else 0) / 1_000_000),
        }


class _GstNvVivaFilterStreamBackend(_StreamDecoderBackend):
    """
    Jetson optimized backend for one stream.

    Pipeline:
        appsrc
          -> h264parse/h265parse
          -> nvv4l2decoder
          -> NVMM NV12
          -> nvivafilter custom CUDA library
          -> preallocated torch CUDA tensor
          -> fakesink

    The custom nvivafilter library is expected to expose:
        set_torch_output_buffer(void* ptr, int dtype_code, int n, int c, int h, int w)

    Strongly recommended optional symbols:
        get_torch_output_frame_count() -> int
        set_channel_order(int order)

    This backend returns:
        [1, 3, H, W], CUDA, fp16/fp32, normalized RGB.

    Stereo note:
        The stock single-stream library usually stores its output buffer in global
        C state. For three simultaneous streams, this backend defaults to copying
        the .so to a unique path per stream so RGB/left/right do not share globals.
    """

    returns_public_tensor = True
    returns_normalized_tensor = True

    _CHANNEL_ORDER_MAP = {
        "auto": 0,
        "rgba": 1,
        "bgra": 2,
        "argb": 3,
        "abgr": 4,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Gst = None
        self.pipeline = None
        self.appsrc = None
        self.bus = None
        self.lib = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.device = torch.device(f"cuda:{int(self.owner.gpu_id)}")
        self.frame_index = 0
        self.bytes_pushed = 0
        self.last_returned_frame_count = 0
        self._lock = threading.RLock()
        self._tmpdir = None
        self._so_path = None

    def _configured_so_path(self) -> str:
        specific = getattr(self.owner, f"{self.stream_name}_gst_nvivafilter_so", None)
        if specific:
            return str(specific)
        if self.stream_name in ("left", "right"):
            stereo_specific = getattr(self.owner, "stereo_gst_nvivafilter_so", None)
            if stereo_specific:
                return str(stereo_specific)
        return str(self.owner.gst_nvivafilter_so)

    def _prepare_so_path(self) -> str:
        base_path = os.path.abspath(self._configured_so_path())
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"nvivafilter .so does not exist: {base_path}")

        if not bool(getattr(self.owner, "gst_nvivafilter_copy_so_per_stream", True)):
            self._so_path = base_path
            return base_path

        root = getattr(self.owner, "gst_nvivafilter_work_dir", None)
        self._tmpdir = tempfile.mkdtemp(
            prefix=f"depthai_{self.stream_name}_nvivafilter_",
            dir=str(root) if root else None,
        )
        dst = os.path.join(self._tmpdir, f"{self.stream_name}_{os.path.basename(base_path)}")
        shutil.copy2(base_path, dst)
        self._so_path = dst
        return dst

    def _load_library(self):
        so_path = self._prepare_so_path()
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

        if not hasattr(lib, "set_torch_output_buffer"):
            raise RuntimeError(f"{so_path} does not export set_torch_output_buffer(...)")

        lib.set_torch_output_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.set_torch_output_buffer.restype = None

        if hasattr(lib, "set_channel_order"):
            lib.set_channel_order.argtypes = [ctypes.c_int]
            lib.set_channel_order.restype = None

        if hasattr(lib, "get_torch_output_frame_count"):
            lib.get_torch_output_frame_count.argtypes = []
            lib.get_torch_output_frame_count.restype = ctypes.c_int
        elif bool(self.owner.gst_nvivafilter_require_frame_count):
            raise RuntimeError(
                f"{so_path} does not export get_torch_output_frame_count(). "
                "For live three-stream decode this symbol is needed to know when "
                "each nvivafilter instance has completed a frame."
            )

        self.lib = lib
        return so_path

    def _allocate_output_tensor(self):
        dtype_name = str(self.owner.gst_nvivafilter_dtype).lower()
        if dtype_name == "fp16":
            torch_dtype = torch.float16
            dtype_code = 1
        elif dtype_name == "fp32":
            torch_dtype = torch.float32
            dtype_code = 0
        else:
            raise ValueError("gst_nvivafilter_dtype must be 'fp16' or 'fp32'")

        if not torch.cuda.is_available():
            raise RuntimeError("gst-nvivafilter requires torch CUDA")

        with torch.cuda.device(self.device):
            tensor = torch.empty(
                (1, 3, self.height, self.width),
                device=self.device,
                dtype=torch_dtype,
            )
            tensor.zero_()
            torch.cuda.synchronize(self.device)

        n, c, h, w = tensor.shape
        self.lib.set_torch_output_buffer(
            ctypes.c_void_p(tensor.data_ptr()),
            dtype_code,
            int(n),
            int(c),
            int(h),
            int(w),
        )

        self.output_tensor = tensor
        return tensor

    def _get_frame_count(self) -> Optional[int]:
        if self.lib is None or not hasattr(self.lib, "get_torch_output_frame_count"):
            return None
        return int(self.lib.get_torch_output_frame_count())

    def start(self):
        if self.output_color != "rgbp":
            raise ValueError(
                "gst-nvivafilter backend returns normalized RGB NCHW. "
                "Use decoder_output_color='rgbp' and stereo_decoder_output_color='rgbp'."
            )

        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as e:
            raise RuntimeError(
                "GStreamer Python bindings are required for decoder_backend='gst-nvivafilter'. "
                "On Jetson install python3-gi and the GStreamer plugins."
            ) from e

        Gst.init(None)
        self.Gst = Gst

        so_path = self._load_library()

        channel_order = str(self.owner.gst_nvivafilter_channel_order).lower()
        if channel_order not in self._CHANNEL_ORDER_MAP:
            raise ValueError(
                "gst_nvivafilter_channel_order must be one of: "
                + ", ".join(self._CHANNEL_ORDER_MAP)
            )

        if hasattr(self.lib, "set_channel_order"):
            self.lib.set_channel_order(self._CHANNEL_ORDER_MAP[channel_order])
        elif channel_order != "auto":
            logger(
                "[GstNvVivaFilterStreamBackend:warning] nvivafilter library does not expose set_channel_order(); "
                f"requested channel_order={channel_order!r} will be ignored."
            )

        tensor = self._allocate_output_tensor()

        if self.codec == "h265":
            caps = "video/x-h265,stream-format=(string)byte-stream,alignment=(string)au"
            parser = "h265parse config-interval=-1"
        elif self.codec == "h264":
            caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au"
            parser = "h264parse config-interval=-1"
        else:
            raise ValueError(f"Unsupported codec: {self.codec}")

        decoder_props = ["enable-max-performance=true"]
        if bool(self.owner.gst_nvivafilter_disable_dpb):
            decoder_props.append("disable-dpb=true")
        if bool(self.owner.gst_nvivafilter_enable_full_frame):
            decoder_props.append("enable-full-frame=true")
        decoder_props = " ".join(decoder_props)
        silent = str(bool(self.owner.gst_nvivafilter_silent)).lower()

        pipeline_desc = f"""
            appsrc name=src
                is-live=true
                block=true
                format=time
                do-timestamp=true
                caps=\"{caps}\"
            ! queue max-size-buffers={int(self.owner.gst_queue_size)} max-size-time=0 max-size-bytes=0
            ! {parser}
            ! nvv4l2decoder {decoder_props}
            ! video/x-raw(memory:NVMM),format=NV12,width=(int){self.width},height=(int){self.height}
            ! nvivafilter cuda-process=true customer-lib-name={so_path} silent={silent}
            ! video/x-raw(memory:NVMM),format=RGBA
            ! fakesink sync=false
        """
        pipeline_desc = " ".join(pipeline_desc.split())

        logger(f"[GstNvVivaFilterStreamBackend:info] Creating GStreamer nvivafilter torch pipeline for {self.stream_name} stream:")
        logger(f"[GstNvVivaFilterStreamBackend:info]  {pipeline_desc}")
        logger("[GstNvVivaFilterStreamBackend:info]"
            f"  output tensor: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}, clone_output={self.owner.gst_nvivafilter_clone_output}"
        )

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name("src")
        self.bus = self.pipeline.get_bus()

        if self.appsrc is None:
            raise RuntimeError(f"Could not create appsrc for {self.stream_name} nvivafilter pipeline")

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"GStreamer nvivafilter pipeline for {self.stream_name} failed to enter PLAYING state")

        initial_count = self._get_frame_count()
        self.last_returned_frame_count = int(initial_count or 0)

    def _raise_if_bus_error(self):
        if self.bus is None:
            return
        Gst = self.Gst
        while True:
            msg = self.bus.timed_pop_filtered(
                0,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if msg is None:
                return
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                raise RuntimeError(f"GStreamer {self.stream_name} error: {err}; debug={debug}")
            if msg.type == Gst.MessageType.EOS:
                raise StopIteration

    def push_bitstream(self, data: bytes):
        if self.closed or self.stop_event.is_set():
            return
        Gst = self.Gst
        if Gst is None:
            raise RuntimeError(f"GStreamer {self.stream_name} backend was not started")

        with self._lock:
            if self.closed or self.stop_event.is_set() or self.appsrc is None:
                return
            self._raise_if_bus_error()
            appsrc = self.appsrc
            duration = int(Gst.SECOND / max(float(self.owner.capture_fps), 1e-6))
            pts = self.frame_index * duration
            self.frame_index += 1

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = duration
        buf.pts = pts
        buf.dts = pts

        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            if self.closed or self.stop_event.is_set():
                return
            raise RuntimeError(f"GStreamer {self.stream_name} push-buffer failed: {ret}")

        self.bytes_pushed += len(data)

    def end_of_stream(self):
        with self._lock:
            if self.appsrc is None or self.Gst is None:
                return
            try:
                self.appsrc.emit("end-of-stream")
            except Exception:
                pass

    def next_tensor(self) -> torch.Tensor:
        if self.output_tensor is None:
            raise StopIteration

        timeout_sec = float(self.owner.gst_nvivafilter_wait_timeout_sec)
        deadline = time.monotonic() + max(timeout_sec, 0.001)

        while not self.stop_event.is_set():
            self._raise_if_bus_error()

            frame_count = self._get_frame_count()
            if frame_count is None:
                # Last-resort mode for experimental libraries without frame_count:
                # sleep a frame period and return a snapshot. This can repeat frames.
                time.sleep(1.0 / max(float(self.owner.capture_fps), 1e-6))
                frame_count = self.last_returned_frame_count + 1

            if frame_count > self.last_returned_frame_count:
                self.last_returned_frame_count = frame_count

                torch.cuda.synchronize(self.device)

                if bool(self.owner.gst_nvivafilter_clone_output):
                    out = self.output_tensor.detach().clone()
                    torch.cuda.synchronize(self.device)
                    return out

                return self.output_tensor

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {self.stream_name} nvivafilter frame. "
                    f"last_frame_count={self.last_returned_frame_count}, bytes_pushed={self.bytes_pushed}"
                )

            time.sleep(0.001)

        raise StopIteration

    def close(self):
        if self.closed:
            return
        self.closed = True

        with self._lock:
            try:
                self.end_of_stream()
            except Exception:
                pass
            if self.pipeline is not None and self.Gst is not None:
                try:
                    self.pipeline.set_state(self.Gst.State.NULL)
                except Exception:
                    pass
            self.bus = None
            self.appsrc = None
            self.pipeline = None

        self.output_tensor = None
        self.lib = None

        if self._tmpdir:
            try:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            except Exception:
                pass
            self._tmpdir = None

    def stats(self) -> Dict[str, float]:
        frame_count = self._get_frame_count()
        return {
            "gst_nvivafilter_frames": float(frame_count or 0),
            "fed_mb": float(self.bytes_pushed / 1_000_000),
        }


class _EncodedStreamRuntime:
    """Host-side state for one encoded DepthAI stream."""

    def __init__(self, name: str, bitstream_queue_size: int, stop_event: threading.Event):
        self.name = name
        self.depthai_q = None
        self.bitstream_queue_size = int(bitstream_queue_size)
        self.decoder_backend: Optional[_StreamDecoderBackend] = None
        self.producer_thread = None
        self.decode_thread = None
        self.packet_count = 0
        self.byte_count = 0
        self.decoded_frame_count = 0
        self.latest_tensor = None
        self.latest_tensor_normalized = False
        self.latest_frame_index = 0
        self.latest_at = 0.0

    def push_eof(self):
        if self.decoder_backend is not None:
            self.decoder_backend.end_of_stream()

    def close_decoder(self):
        if self.decoder_backend is not None:
            self.decoder_backend.close()
            self.decoder_backend = None


def _looks_like_jetson() -> bool:
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as f:
            model = f.read().lower()
        if "jetson" in model or "tegra" in model:
            return True
    except Exception:
        pass
    return platform.machine().lower() in ("aarch64", "arm64")


def _create_stream_decoder_backend(
    owner,
    stream: _EncodedStreamRuntime,
    *,
    codec: str,
    width: int,
    height: int,
    output_color: str,
    stop_event: threading.Event,
) -> _StreamDecoderBackend:
    backend = getattr(owner, "decoder_backend", "auto")
    if backend == "auto":
        backend = "gst-nvivafilter" if _looks_like_jetson() else "pynvvideocodec"

    if backend == "pynvvideocodec":
        return _PyNvVideoCodecStreamBackend(
            owner,
            stream.name,
            codec,
            width,
            height,
            output_color,
            stream.bitstream_queue_size,
            stop_event,
        )

    if backend == "gst-nvivafilter":
        return _GstNvVivaFilterStreamBackend(
            owner,
            stream.name,
            codec,
            width,
            height,
            output_color,
            stream.bitstream_queue_size,
            stop_event,
        )

    raise ValueError(f"Unsupported decoder_backend: {backend}")


class _DepthAIPoeRGBStereoH26xBottomTorchTensorCapture:
    """
    RGB + stereo capture using H26x on all three streams.

    v7 keeps bottom-row packing and adds decoder backends for non-contiguous BCHW bottom slices.

    Device side:
        CAM_A RGB  -> NV12 -> VideoEncoder H264/H265
        CAM_B left -> NV12/YUV400p -> VideoEncoder H264/H265
        CAM_C right-> NV12/YUV400p -> VideoEncoder H264/H265

    Host side:
        RGB, left and right encoded streams are decoded by a pluggable backend:
            * PyNvVideoCodec on dGPU
            * GStreamer nvv4l2decoder + nvivafilter on Jetson
        The latest decoded left/right grayscale frames are packed into the bottom
        rows of the RGB tensor. The returned tensor keeps the RGB-only shape:

            [1, 3, rgb_height, rgb_width]

        The bottom packed_stereo_rows rows are overwritten with:
            left.flatten(), right.flatten()
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.stop_event = threading.Event()

        self.rgb = _EncodedStreamRuntime("rgb", owner.rgb_bitstream_queue_size, self.stop_event)
        self.left = _EncodedStreamRuntime("left", owner.stereo_bitstream_queue_size, self.stop_event)
        self.right = _EncodedStreamRuntime("right", owner.stereo_bitstream_queue_size, self.stop_event)

        self.combined_frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_combined_count = 0

        self._latest_stereo_lock = threading.Lock()
        self._stereo_ready = threading.Event()

        self._released = False
        self._exit_stack = contextlib.ExitStack()

        self._start()

    def _open_device(self):
        src = str(self.source).strip()
        if src.startswith("depthai://"):
            src = src.replace("depthai://", "", 1).strip()
        if src in ("", "auto", "default", "none", "None"):
            return dai.Device()
        try:
            return dai.Device(dai.DeviceInfo(src))
        except Exception:            
            self._log("error", f"could not open DepthAI device: {src}")
            raise

    @staticmethod
    def _camera_socket(socket_name: str):
        if hasattr(dai.CameraBoardSocket, socket_name):
            return getattr(dai.CameraBoardSocket, socket_name)
        aliases = {"RGB": "CAM_A", "LEFT": "CAM_B", "RIGHT": "CAM_C"}
        alias = aliases.get(socket_name)
        if alias and hasattr(dai.CameraBoardSocket, alias):
            return getattr(dai.CameraBoardSocket, alias)
        raise ValueError(f"Unsupported camera socket: {socket_name}")

    @staticmethod
    def _depthai_profile(codec: str):
        if codec == "h265":
            return dai.VideoEncoderProperties.Profile.H265_MAIN
        if codec == "h264":
            return dai.VideoEncoderProperties.Profile.H264_MAIN
        raise ValueError(f"Unsupported codec: {codec}")

    @staticmethod
    def _output_color_type(name: str):
        raise RuntimeError("Decoder color conversion is handled by stream decoder backends.")

    @staticmethod
    def _img_frame_type(type_name: str):
        if hasattr(dai.ImgFrame.Type, type_name):
            return getattr(dai.ImgFrame.Type, type_name)

        # Older/newer DepthAI builds expose 8-bit mono using slightly different
        # enum names. Prefer YUV400p for VideoEncoder, because the encoder
        # warning explicitly says it accepts NV12 or YUV400p.
        if type_name == "GRAY8":
            for fallback in ("YUV400p", "YUV400P", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        if type_name == "YUV400p":
            for fallback in ("YUV400P", "GRAY8", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        raise ValueError(f"Unsupported DepthAI ImgFrame.Type: {type_name}")

    @staticmethod
    def _resize_mode(mode_name: str):
        if hasattr(dai.ImgResizeMode, mode_name):
            return getattr(dai.ImgResizeMode, mode_name)
        raise ValueError(f"Unsupported DepthAI ImgResizeMode: {mode_name}")

    @staticmethod
    def _packet_to_bytes(packet):
        data = packet.getData()
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        arr = np.asarray(data, dtype=np.uint8)
        return arr.tobytes()

    def _create_depthai_pipeline(self):
        owner = self.owner

        raw_pipeline = dai.Pipeline(self.device)
        if hasattr(raw_pipeline, "__enter__"):
            pipeline = self._exit_stack.enter_context(raw_pipeline)
        else:
            pipeline = raw_pipeline

        rgb_socket = self._camera_socket(owner.rgb_camera_socket)
        left_socket = self._camera_socket(owner.left_camera_socket)
        right_socket = self._camera_socket(owner.right_camera_socket)

        rgb_cam = pipeline.create(dai.node.Camera).build(rgb_socket)
        left_cam = pipeline.create(dai.node.Camera).build(left_socket)
        right_cam = pipeline.create(dai.node.Camera).build(right_socket)

        rgb_nv12 = rgb_cam.requestOutput(
            (owner.rgb_width, owner.rgb_height),
            dai.ImgFrame.Type.NV12,
            self._resize_mode(owner.rgb_resize_mode),
            owner.capture_fps,
        )

        mono_type = self._img_frame_type(owner.stereo_encoder_input_type)
        stereo_resize_mode = self._resize_mode(owner.stereo_resize_mode)
        left_gray = left_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            mono_type,
            stereo_resize_mode,
            owner.capture_fps,
        )
        right_gray = right_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            mono_type,
            stereo_resize_mode,
            owner.capture_fps,
        )

        rgb_encoder = pipeline.create(dai.node.VideoEncoder).build(
            rgb_nv12,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.rgb_codec),
        )
        left_encoder = pipeline.create(dai.node.VideoEncoder).build(
            left_gray,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.stereo_codec),
        )
        right_encoder = pipeline.create(dai.node.VideoEncoder).build(
            right_gray,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.stereo_codec),
        )

        for enc_name, enc, bitrate in (
            ("RGB", rgb_encoder, owner.rgb_bitrate_kbps),
            ("left", left_encoder, owner.stereo_bitrate_kbps),
            ("right", right_encoder, owner.stereo_bitrate_kbps),
        ):
            try:
                enc.setBitrateKbps(int(bitrate))
            except Exception as e:
                logger(f"[GstNvVivaFilterStreamBackend:warning] could not set OAK {enc_name} encoder bitrate: {e}")
            try:
                enc.setKeyframeFrequency(int(owner.capture_fps))
            except Exception:
                pass

        self.rgb.depthai_q = rgb_encoder.out.createOutputQueue(
            maxSize=owner.rgb_depthai_queue_size,
            blocking=True,
        )
        self.left.depthai_q = left_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )
        self.right.depthai_q = right_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )

        pipeline.start()
        return pipeline
    
    def _log(self, path:str, msg: str):
        level="info"
        if path in ["error","warning","debug","info"]:
            level=path
            
        logger(f"[{self.owner.uuid}:{path}] {msg}", level=level)

    def _start(self):
        owner = self.owner

        for label, width in (("rgb_width", owner.rgb_width), ("stereo_width", owner.stereo_width)):
            if width % 32 != 0:
                raise ValueError(
                    f"DepthAI H264/H265 encoder requires {label} to be a multiple of 32; got {width}."
                )

        stereo_values = 2 * int(owner.stereo_width) * int(owner.stereo_height)
        values_per_row = 3 * int(owner.rgb_width)
        payload_rows = (stereo_values + values_per_row - 1) // values_per_row
        payload_start = int(owner.rgb_height) - payload_rows
        if payload_start < 0:
            raise ValueError(
                "Stereo payload does not fit inside RGB bottom rows: "
                f"payload_rows={payload_rows}, rgb_height={owner.rgb_height}."
            )

        self._log("info", "Opening DepthAI device...")
        self.device = self._open_device()

        decoder_backend_name = getattr(owner, "decoder_backend", "auto")
        if decoder_backend_name == "auto":
            decoder_backend_name = "gst-nvivafilter" if _looks_like_jetson() else "pynvvideocodec"

        self._log("info","Connected DepthAI device:")
        self._log("status",f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        self._log("status",f"  Cameras: {self.device.getConnectedCameras()}")

        self._log("info","Starting DepthAI RGB + H26x stereo bottom-pack tensor pipeline v7:")
        self._log("status",f"  Source: {self.source}")
        self._log("status",f"  RGB socket: {owner.rgb_camera_socket}")
        self._log("status",f"  Left socket: {owner.left_camera_socket}")
        self._log("status",f"  Right socket: {owner.right_camera_socket}")
        self._log("status",f"  RGB size: {owner.rgb_width}x{owner.rgb_height}")
        self._log("status",f"  Stereo size: {owner.stereo_width}x{owner.stereo_height}")
        self._log("status",f"  Capture FPS: {owner.capture_fps}")
        self._log("status",f"  OAK RGB encoder: {owner.rgb_codec.upper()} @ {owner.rgb_bitrate_kbps} kbps")
        self._log("status",f"  OAK stereo encoders: {owner.stereo_codec.upper()} @ {owner.stereo_bitrate_kbps} kbps each")
        self._log("status",f"  Decoder backend: {getattr(owner, 'decoder_backend', 'auto')} -> {decoder_backend_name}")
        self._log("status",f"  RGB decoder output: {owner.decoder_output_color}")
        self._log("status",f"  Stereo decoder output: {owner.stereo_decoder_output_color}")
        self._log("status",f"  Stereo encoder input type: {owner.stereo_encoder_input_type}")
        self._log("status",f"  Output: [1, 3, {owner.rgb_height}, {owner.rgb_width}]")
        self._log("status",f"  Stereo payload rows inside RGB bottom: {payload_rows}")
        self._log("status",f"  Stereo payload start row: {payload_start}")
        self._log("info",  f"  Note: bottom RGB rows are overwritten by stereo payload")
        self._log("status",f"  normalize_rgb: {owner.normalize_rgb}")
        self._log("status",f"  normalize_stereo: {owner.normalize_stereo}")
        self._log("status",f"  GPU ID: {owner.gpu_id}")

        self.pipeline = self._create_depthai_pipeline()

        self._create_stream_decoder(
            self.rgb,
            codec=owner.rgb_codec,
            max_width=owner.rgb_width,
            max_height=owner.rgb_height,
            output_color=owner.decoder_output_color,
        )
        self._create_stream_decoder(
            self.left,
            codec=owner.stereo_codec,
            max_width=owner.stereo_width,
            max_height=owner.stereo_height,
            output_color=owner.stereo_decoder_output_color,
        )
        self._create_stream_decoder(
            self.right,
            codec=owner.stereo_codec,
            max_width=owner.stereo_width,
            max_height=owner.stereo_height,
            output_color=owner.stereo_decoder_output_color,
        )

        for stream in (self.rgb, self.left, self.right):
            stream.decoder_backend.start()

        for stream in (self.rgb, self.left, self.right):
            stream.producer_thread = threading.Thread(
                target=self._encoded_producer_loop,
                args=(stream,),
                daemon=True,
            )
            stream.producer_thread.start()

        for stream in (self.left, self.right):
            stream.decode_thread = threading.Thread(
                target=self._stereo_decode_loop,
                args=(stream,),
                daemon=True,
            )
            stream.decode_thread.start()

        self._log("info","DepthAI RGB + H26x stereo bottom-pack tensor pipeline ready.")

    def _create_stream_decoder(self, stream: _EncodedStreamRuntime, codec: str, max_width: int, max_height: int, output_color: str):
        stream.decoder_backend = _create_stream_decoder_backend(
            self.owner,
            stream,
            codec=codec,
            width=int(max_width),
            height=int(max_height),
            output_color=output_color,
            stop_event=self.stop_event,
        )

    def _encoded_producer_loop(self, stream: _EncodedStreamRuntime):
        try:
            while not self.stop_event.is_set():
                try:
                    if hasattr(stream.depthai_q, "tryGet"):
                        pkt = stream.depthai_q.tryGet()
                        if pkt is None:
                            time.sleep(0.001)
                            continue
                    else:
                        pkt = stream.depthai_q.get()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    raise

                data = self._packet_to_bytes(pkt)
                stream.packet_count += 1
                stream.byte_count += len(data)

                # Do not drop compressed packets during normal operation. Dropping
                # H26x bytes corrupts the stream until a later keyframe.
                if stream.decoder_backend is not None:
                    stream.decoder_backend.push_bitstream(data)

        except Exception:
            if not self.stop_event.is_set() and not self._released:
                self._log("error",f"DepthAI {stream.name} producer thread failed:")
                traceback.print_exc()
        finally:
            stream.push_eof()

    def _decode_next_frame_tensor(self, stream: _EncodedStreamRuntime) -> torch.Tensor:
        if stream.decoder_backend is None:
            raise StopIteration
        tensor = stream.decoder_backend.next_tensor()
        stream.decoded_frame_count += 1
        stream.latest_tensor_normalized = bool(getattr(stream.decoder_backend, "returns_normalized_tensor", False))
        return tensor

    def _stereo_decode_loop(self, stream: _EncodedStreamRuntime):
        try:
            while not self.stop_event.is_set():
                tensor = self._decode_next_frame_tensor(stream)
                gray = self._stereo_decoded_tensor_to_gray(tensor)

                with self._latest_stereo_lock:
                    stream.latest_tensor = gray
                    stream.latest_frame_index = stream.decoded_frame_count
                    stream.latest_at = time.monotonic()
                    if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                        self._stereo_ready.set()

        except StopIteration:
            pass
        except Exception:
            if not self.stop_event.is_set() and not self._released:
                self._log("error",f"DepthAI {stream.name} stereo decode thread failed:")
                traceback.print_exc()
        finally:
            self._stereo_ready.set()

    def _stereo_decoded_tensor_to_gray(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert decoded stereo video frame to a single-channel [H, W] GPU tensor.

        Preferred path:
            stereo_encoder_input_type='NV12'
            stereo_decoder_output_color='rgbp'

        v2 used native NV12 decode and sliced the first H rows as luma. On some
        NVDEC/PyNvVideoCodec builds that surface can be pitched/planar in a way
        that appears as horizontal stripes when reshaped as [H, W]. v3 defaults
        to RGBP output for stereo and uses channel 0 as grayscale.
        """

        owner = self.owner
        stereo_h = int(owner.stereo_height)
        stereo_w = int(owner.stereo_width)
        output_color = str(getattr(owner, "stereo_decoder_output_color", "rgbp"))

        # Optional one-shot debug hook: override/enable this on the generator
        # if you need to inspect the actual PyNvVideoCodec tensor shape.
        if getattr(owner, "debug_stereo_decoded_shape", False):
            printed_attr = f"_debug_printed_{getattr(self, 'idx', 0)}_{output_color}"
            if not getattr(self, printed_attr, False):
                self._log("info",f"Decoded stereo tensor shape={tuple(tensor.shape)}, dtype={tensor.dtype}, color={output_color}")
                setattr(self, printed_attr, True)

        # gst-nvivafilter returns [1, 3, H, W] normalized RGB. Use channel 0
        # for mono/stereo payload streams.
        if tensor.ndim == 4 and tensor.shape[0] >= 1 and tensor.shape[1] >= 1:
            gray = tensor[0, 0]
            return gray.contiguous()

        if output_color == "native":
            # Fallback only. Native NV12 can be pitched/planar depending on the decoder build.
            # If you see horizontal stripes, use stereo_decoder_output_color="rgbp".
            if tensor.ndim == 2:
                if tensor.shape[0] >= stereo_h and tensor.shape[1] >= stereo_w:
                    return tensor[:stereo_h, :stereo_w].contiguous()

            # Some decoder builds may include a leading plane/batch dimension.
            if tensor.ndim == 3:
                if tensor.shape[-2] >= stereo_h and tensor.shape[-1] >= stereo_w:
                    return tensor[..., :stereo_h, :stereo_w].reshape(-1, stereo_h, stereo_w)[0].contiguous()

        # RGB/RGBP fallback. For mono content decoded to RGB, channels should be equal.
        if tensor.ndim == 2:
            gray = tensor
        elif tensor.ndim == 3 and tensor.shape[0] >= 1:
            gray = tensor[0]
        elif tensor.ndim == 3 and tensor.shape[-1] >= 1:
            gray = tensor[..., 0]
        else:
            raise ValueError(f"Cannot extract gray stereo frame from tensor shape {tuple(tensor.shape)}")

        if gray.shape[-2:] != (stereo_h, stereo_w):
            if owner.strict_stereo_shape:
                raise ValueError(
                    f"Decoded stereo frame shape {tuple(gray.shape)} does not match "
                    f"({owner.stereo_height}, {owner.stereo_width})."
                )
            gray = torch.nn.functional.interpolate(
                gray.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
                size=(stereo_h, stereo_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if not owner.normalize_stereo:
                gray = gray.round().clamp(0, 255).to(dtype=torch.uint8)

        return gray.contiguous()

    def _get_latest_stereo_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
            return self.left.latest_tensor, self.right.latest_tensor

        timeout = float(getattr(self.owner, "stereo_startup_timeout_sec", 2.0))
        if timeout > 0:
            self._stereo_ready.wait(timeout=timeout)

        with self._latest_stereo_lock:
            if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                return self.left.latest_tensor, self.right.latest_tensor

        if getattr(self.owner, "allow_missing_stereo", False):
            device = self._torch_device()
            zero = torch.zeros(
                (int(self.owner.stereo_height), int(self.owner.stereo_width)),
                dtype=torch.uint8,
                device=device,
            )
            return zero, zero

        while not self.stop_event.is_set() and not self._released:
            self._stereo_ready.wait(timeout=0.01)
            with self._latest_stereo_lock:
                if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                    return self.left.latest_tensor, self.right.latest_tensor

        raise StopIteration

    def _torch_device(self):
        owner = self.owner
        if owner.torch_device:
            return torch.device(owner.torch_device)
        if torch.cuda.is_available() and owner.gpu_id is not None and owner.gpu_id >= 0:
            return torch.device(f"cuda:{owner.gpu_id}")
        return torch.device("cpu")

    def _decoded_rgb_frame_to_tensor(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        backend = self.rgb.decoder_backend
        if backend is not None and getattr(backend, "returns_public_tensor", False):
            public_tensor = frame_tensor
            preview_tensor = frame_tensor[0] if frame_tensor.ndim == 4 else frame_tensor
            self.owner.on_rgb_tensor(preview_tensor, self.rgb.decoded_frame_count)

            if self.owner.show_rgb_preview:
                self._show_small_rgb_preview(preview_tensor)

            return public_tensor

        self.owner.on_rgb_tensor(frame_tensor, self.rgb.decoded_frame_count)

        if self.owner.show_rgb_preview:
            self._show_small_rgb_preview(frame_tensor)

        tensor = frame_tensor.unsqueeze(0)
        if self.owner.normalize_rgb:
            tensor = tensor / 255.0
        return tensor

    def _decode_next_rgb_tensor(self):
        tensor = self._decode_next_frame_tensor(self.rgb)
        return self._decoded_rgb_frame_to_tensor(tensor)

    def _show_small_rgb_preview(self, tensor: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        if tensor.ndim != 3:
            self._log("error",f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        if tensor.shape[0] == 3:
            small_hwc = tensor[:, ::stride, ::stride].permute(1, 2, 0).contiguous()
        elif tensor.shape[-1] == 3:
            small_hwc = tensor[::stride, ::stride, :].contiguous()
        else:
            self._log("error",f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        small = small_hwc.detach()
        if small.dtype.is_floating_point and float(small.max()) <= 1.5:
            small = small.mul(255.0)
        small_rgb = small.cpu().numpy()
        if small_rgb.dtype != np.uint8:
            small_rgb = np.clip(small_rgb, 0, 255).astype(np.uint8)
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(owner.rgb_window_name, small_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _show_small_stereo_preview(self, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        left = left_gray[::stride, ::stride]
        right = right_gray[::stride, ::stride]
        preview = torch.cat((left, right), dim=1).detach()
        if owner.normalize_stereo and preview.dtype.is_floating_point:
            preview = preview.mul(255.0)
        if preview.dtype != torch.uint8:
            preview = preview.clamp(0, 255).to(dtype=torch.uint8)
        cv2.imshow(owner.stereo_window_name, preview.cpu().numpy())
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _rgb_tensor_to_bchw3(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        if rgb_tensor.ndim != 4:
            raise ValueError(f"Expected RGB tensor with 4 dims, got {tuple(rgb_tensor.shape)}")
        if rgb_tensor.shape[1] == 3:
            return rgb_tensor.contiguous()
        if rgb_tensor.shape[-1] == 3:
            return rgb_tensor.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "Bottom packed RGB+stereo output needs 3 RGB channels. "
            f"Got RGB tensor shape {tuple(rgb_tensor.shape)}. "
            "Use decoder_output_color='rgbp' or 'rgb'."
        )

    def _prepare_stereo_flat_for_rgb(self, gray: torch.Tensor, rgb: torch.Tensor, *, source_normalized: bool = False) -> torch.Tensor:
        flat = gray.reshape(-1)
        if flat.device != rgb.device or flat.dtype != rgb.dtype:
            flat = flat.to(device=rgb.device, dtype=rgb.dtype, non_blocking=self.owner.non_blocking_gpu_copy)
        if self.owner.normalize_stereo and flat.dtype.is_floating_point and not source_normalized:
            # PyNvVideoCodec decode produces 0..255 values. Match normalized RGB by
            # writing 0..1 payload values when requested. gst-nvivafilter already
            # writes normalized values, so do not divide those again.
            flat = flat / 255.0
        return flat

    def _copy_flat_into_payload_channels(self, payload: torch.Tensor, flat: torch.Tensor, start_offset: int = 0):
        """
        Copy a 1-D stereo payload into BCHW bottom rows without using
        payload.reshape(...), because rgb[:, :, start:, :] is not contiguous in
        BCHW memory. Calling reshape() on that non-contiguous slice can create a
        temporary copy, so writes never reach the returned RGB tensor.

        Storage order used by pack/unpack:
            channel 0 bottom rows, then channel 1 bottom rows, then channel 2 bottom rows.
        """

        # payload shape: [1, 3, payload_rows, rgb_width]
        if payload.ndim != 4 or payload.shape[0] != 1 or payload.shape[1] != 3:
            raise ValueError(f"Expected payload [1, 3, rows, width], got {tuple(payload.shape)}")

        remaining = int(flat.numel())
        src_pos = 0
        dst_pos = int(start_offset)
        per_channel = int(payload.shape[2] * payload.shape[3])

        for channel in range(3):
            if remaining <= 0:
                break
            if dst_pos >= per_channel:
                dst_pos -= per_channel
                continue

            dst_view = payload[0, channel].reshape(-1)
            n = min(remaining, per_channel - dst_pos)
            dst_view[dst_pos:dst_pos + n].copy_(flat[src_pos:src_pos + n], non_blocking=True)
            src_pos += n
            remaining -= n
            dst_pos = 0

        if remaining != 0:
            raise ValueError("Stereo payload did not fit into bottom rows.")

    def _fill_payload_tail_channels(self, payload: torch.Tensor, start_offset: int, value: float):
        # Fill unused tail using the same channel-chunk layout as packing.
        per_channel = int(payload.shape[2] * payload.shape[3])
        total = 3 * per_channel
        pos = int(start_offset)
        if pos >= total:
            return
        for channel in range(3):
            if pos >= per_channel:
                pos -= per_channel
                continue
            dst_view = payload[0, channel].reshape(-1)
            dst_view[pos:].fill_(value)
            pos = 0

    def _pack_encoded_stereo(self, rgb_tensor: torch.Tensor, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        rgb = self._rgb_tensor_to_bchw3(rgb_tensor)
        b, _, rgb_h, rgb_w = rgb.shape

        if b != 1:
            raise ValueError(f"Expected batch size 1, got {b}.")

        stereo_h = int(owner.stereo_height)
        stereo_w = int(owner.stereo_width)

        if tuple(left_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(
                f"Left stereo shape {tuple(left_gray.shape)} does not match {(stereo_h, stereo_w)}"
            )

        if tuple(right_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(
                f"Right stereo shape {tuple(right_gray.shape)} does not match {(stereo_h, stereo_w)}"
            )

        stereo_one_values = stereo_h * stereo_w
        values_per_payload_row = 3 * int(rgb_w)

        payload_rows_per_stereo = (
            stereo_one_values + values_per_payload_row - 1
        ) // values_per_payload_row

        total_payload_rows = 2 * payload_rows_per_stereo

        if total_payload_rows > int(rgb_h):
            raise ValueError(
                "Stereo payload does not fit in top + bottom RGB tensor rows: "
                f"payload_rows_per_stereo={payload_rows_per_stereo}, "
                f"total_payload_rows={total_payload_rows}, rgb_height={rgb_h}."
            )

        top_payload = rgb[:, :, :payload_rows_per_stereo, :]
        bottom_payload = rgb[:, :, rgb_h - payload_rows_per_stereo :, :]

        left_flat = self._prepare_stereo_flat_for_rgb(
            left_gray,
            rgb,
            source_normalized=bool(getattr(self.left, "latest_tensor_normalized", False)),
        )[:stereo_one_values]

        right_flat = self._prepare_stereo_flat_for_rgb(
            right_gray,
            rgb,
            source_normalized=bool(getattr(self.right, "latest_tensor_normalized", False)),
        )[:stereo_one_values]

        # Left stereo goes into the top rows.
        self._copy_flat_into_payload_channels(
            top_payload,
            left_flat,
            start_offset=0,
        )

        # Right stereo goes into the bottom rows.
        self._copy_flat_into_payload_channels(
            bottom_payload,
            right_flat,
            start_offset=0,
        )

        if getattr(owner, "clear_unused_payload_tail", False):
            pad_value = float(owner.packed_stereo_pad_value)

            self._fill_payload_tail_channels(
                top_payload,
                start_offset=stereo_one_values,
                value=pad_value,
            )

            self._fill_payload_tail_channels(
                bottom_payload,
                start_offset=stereo_one_values,
                value=pad_value,
            )

        return rgb

    def next_frame(self):
        if self._released:
            raise StopIteration

        try:
            rgb_tensor = self._decode_next_rgb_tensor()
            left_gray, right_gray = self._get_latest_stereo_tensors()

            if self.owner.show_stereo_preview:
                self._show_small_stereo_preview(left_gray, right_gray)

            self.combined_frame_count += 1
            packed_tensor = self._pack_encoded_stereo(rgb_tensor, left_gray, right_gray)
            self.owner.on_rgb_stereo_tensor(packed_tensor, self.combined_frame_count)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                combined_fps = (self.combined_frame_count - self.last_combined_count) / dt
                elapsed = max(now - self.started_at, 1e-6)
                rgb_mbps = self.rgb.byte_count * 8.0 / elapsed / 1_000_000
                left_mbps = self.left.byte_count * 8.0 / elapsed / 1_000_000
                right_mbps = self.right.byte_count * 8.0 / elapsed / 1_000_000
                self._log("info",
                    f"combined={self.combined_frame_count}, "
                    f"fps={combined_fps:.2f}, "
                    f"rgb_dec={self.rgb.decoded_frame_count}, "
                    f"left_dec={self.left.decoded_frame_count}, "
                    f"right_dec={self.right.decoded_frame_count}, "
                    f"mbps rgb/left/right={rgb_mbps:.1f}/{left_mbps:.1f}/{right_mbps:.1f}, "
                    f"pack_rows={self.owner.packed_stereo_rows}"
                )
                self.last_log_at = now
                self.last_combined_count = self.combined_frame_count

            return packed_tensor

        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event.is_set():
                raise StopIteration
            raise
        
    def release(self):
        self._log("debug", "release() called")

        if self._released:
            self._log("debug", "release() skipped: already released")
            return

        self._log("info", "Starting DepthAI resource release")

        self._released = True
        self._log("debug", "Marked object as released")

        self.stop_event.set()
        self._log("debug", "Stop event set")

        self._stereo_ready.set()
        self._log("debug", "Stereo ready event set to unblock waiting threads")

        streams = (self.rgb, self.left, self.right)

        for stream in streams:
            self._log("debug", f"Sending EOF to stream '{stream.name}'")

            try:
                stream.push_eof()
                self._log("debug", f"EOF pushed to stream '{stream.name}'")
            except Exception as e:
                self._log("warning", f"Failed to push EOF to stream '{stream.name}': {e}")

            try:
                if stream.depthai_q is not None and hasattr(stream.depthai_q, "close"):
                    self._log("debug", f"Closing DepthAI queue for stream '{stream.name}'")
                    stream.depthai_q.close()
                    self._log("debug", f"DepthAI queue closed for stream '{stream.name}'")
                else:
                    self._log("debug", f"No closable DepthAI queue for stream '{stream.name}'")
            except Exception as e:
                self._log("warning", f"Error closing DepthAI queue for stream '{stream.name}': {e}")

        for stream in streams:
            try:
                if stream.producer_thread is not None:
                    self._log("debug", f"Joining producer thread for stream '{stream.name}'")

                    stream.producer_thread.join(timeout=2.0)

                    if stream.producer_thread.is_alive():
                        self._log(
                            "warning",
                            f"Producer thread for stream '{stream.name}' did not exit within timeout"
                        )
                    else:
                        self._log("debug", f"Producer thread joined for stream '{stream.name}'")
                else:
                    self._log("debug", f"No producer thread for stream '{stream.name}'")
            except Exception as e:
                self._log("error", f"Error joining producer thread for stream '{stream.name}': {e}")

        for stream in (self.left, self.right):
            try:
                if stream.decode_thread is not None:
                    self._log("debug", f"Joining decode thread for stream '{stream.name}'")

                    stream.decode_thread.join(timeout=2.0)

                    if stream.decode_thread.is_alive():
                        self._log(
                            "warning",
                            f"Decode thread for stream '{stream.name}' did not exit within timeout"
                        )
                    else:
                        self._log("debug", f"Decode thread joined for stream '{stream.name}'")
                else:
                    self._log("debug", f"No decode thread for stream '{stream.name}'")
            except Exception as e:
                self._log("error", f"Error joining decode thread for stream '{stream.name}': {e}")

        for stream in streams:
            try:
                self._log("debug", f"Closing decoder backend for stream '{stream.name}'")
                stream.close_decoder()
                self._log("debug", f"Decoder backend closed for stream '{stream.name}'")
            except Exception as e:
                self._log("error", f"Error closing decoder backend for stream '{stream.name}': {e}")

        try:
            if self.pipeline is not None:
                self._log("debug", "Checking DepthAI pipeline state before stop")

                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self._log("info", "Stopping DepthAI pipeline")
                    self.pipeline.stop()
                    self._log("info", "DepthAI pipeline stopped")
                else:
                    self._log("debug", "DepthAI pipeline was not running")
            else:
                self._log("debug", "No DepthAI pipeline to stop")
        except Exception as e:
            self._log("warning", f"DepthAI pipeline stop failed during release: {e}")

        try:
            self._log("debug", "Closing exit stack")
            self._exit_stack.close()
            self._log("debug", "Exit stack closed")
        except Exception as e:
            self._log("warning", f"Error closing exit stack during release: {e}")

        for stream in streams:
            self._log("debug", f"Clearing cached tensors and queue references for stream '{stream.name}'")

            stream.latest_tensor = None
            stream.latest_tensor_normalized = False
            stream.depthai_q = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self._log("info", "Closing DepthAI device")
                self.device.close()
                self._log("info", "DepthAI device closed")
            else:
                self._log("debug", "No closable DepthAI device")
        except Exception as e:
            self._log("warning", f"Error closing DepthAI device during release: {e}")

        self.pipeline = None
        self.device = None
        self._log("debug", "Cleared pipeline and device references")

        for window_name in (self.owner.rgb_window_name, self.owner.stereo_window_name):
            try:
                self._log("debug", f"Destroying OpenCV window '{window_name}'")
                cv2.destroyWindow(window_name)
                self._log("debug", f"OpenCV window destroyed: '{window_name}'")
            except Exception as e:
                self._log("warning", f"Error destroying OpenCV window '{window_name}': {e}")

        self._log("info", "DepthAI resource release completed")


class DepthAIPoeRGBStereoTorchGenerator(ImageMatGenerator):
    """
    ImageMatGenerator-style DepthAI PoE RGB + stereo generator.

    This v7 version encodes RGB, left mono, and right mono on the OAK device using
    H264/H265 before sending them over PoE. It is intended for the case where
    RGB-only H26x reaches about 15 FPS, but adding raw stereo throttles the
    pipeline.

    Output per source:
        one torch.Tensor with shape [1, 3, rgb_height, rgb_width]

    Layout:
        tensor[:, :, :stereo_payload_start_row, :] is normal RGB.
        tensor[:, :, stereo_payload_start_row:, :] contains flattened stereo:
            left.flatten(), then right.flatten().

    Important:
        The bottom packed_stereo_rows RGB rows are overwritten by stereo payload.
    """

    color_types: List['ColorType'] = []

    capture_fps: float = 15.0

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800

    rgb_camera_socket: Literal["CAM_A", "RGB"] = "CAM_A"
    left_camera_socket: Literal["CAM_B", "LEFT"] = "CAM_B"
    right_camera_socket: Literal["CAM_C", "RIGHT"] = "CAM_C"

    rgb_codec: Literal["h265", "h264"] = "h265"
    stereo_codec: Literal["h265", "h264"] = "h265"

    # Kept as compatibility aliases for earlier generated files.
    codec: Literal["h265", "h264"] = "h265"
    bitrate_kbps: int = 60000

    rgb_bitrate_kbps: int = 60000
    stereo_bitrate_kbps: int = 6000

    decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"
    stereo_decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"
    # VideoEncoder accepts NV12 or 8-bit gray. On some DepthAI v3 builds,
    # Camera.requestOutput(GRAY8) can arrive as a different internal type and
    # trigger: "Arrived frame type (...) is not either NV12 or YUV400p".
    # NV12 is the safest default; with stereo_decoder_output_color="rgbp"
    # v3 defaults to stereo_decoder_output_color="rgbp" to avoid native NV12 pitch/plane interpretation issues; payload is still grayscale because channel 0 is used.
    stereo_encoder_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"

    rgb_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"
    stereo_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"

    gpu_id: int = 0
    torch_device: Optional[str] = None
    non_blocking_gpu_copy: bool = True

    # Backend selection:
    #   auto            -> gst-nvivafilter on Jetson/aarch64, PyNvVideoCodec otherwise
    #   pynvvideocodec  -> original dGPU NVDEC + DLPack path
    #   gst-nvivafilter -> Jetson nvv4l2decoder + nvivafilter CUDA tensor path
    decoder_backend: Literal["auto", "pynvvideocodec", "gst-nvivafilter"] = "auto"

    # Jetson optimized nvivafilter -> torch CUDA backend.
    # For three streams, copy_so_per_stream=True is the safe default because most
    # simple nvivafilter libraries keep the output buffer in global C state.
    gst_queue_size: int = 8
    gst_nvivafilter_so: str = "./libdepthai_cuda_preprocess.so"
    rgb_gst_nvivafilter_so: Optional[str] = None
    stereo_gst_nvivafilter_so: Optional[str] = None
    left_gst_nvivafilter_so: Optional[str] = None
    right_gst_nvivafilter_so: Optional[str] = None
    gst_nvivafilter_copy_so_per_stream: bool = True
    gst_nvivafilter_work_dir: Optional[str] = None
    gst_nvivafilter_dtype: Literal["fp16", "fp32"] = "fp16"
    gst_nvivafilter_channel_order: Literal["auto", "rgba", "bgra", "argb", "abgr"] = "rgba"
    gst_nvivafilter_clone_output: bool = True
    gst_nvivafilter_wait_timeout_sec: float = 2.0
    gst_nvivafilter_require_frame_count: bool = True
    gst_nvivafilter_disable_dpb: bool = True
    gst_nvivafilter_enable_full_frame: bool = True
    gst_nvivafilter_silent: bool = False

    # Match your original RGB-only float32 output by default. Set both False if
    # you want raw uint8 output instead.
    normalize_rgb: bool = True
    normalize_stereo: bool = True
    strict_stereo_shape: bool = True
    debug_stereo_decoded_shape: bool = False

    packed_stereo_pad_value: float = 0.0
    clear_unused_payload_tail: bool = False

    rgb_depthai_queue_size: int = 8
    stereo_depthai_queue_size: int = 8
    rgb_bitstream_queue_size: int = 64
    stereo_bitstream_queue_size: int = 64

    stereo_startup_timeout_sec: float = 2.0
    allow_missing_stereo: bool = False

    low_latency: bool = False
    log_fps: bool = True
    retain_decoded_frame_refs: int = 16

    show_rgb_preview: bool = False
    show_stereo_preview: bool = False
    preview_stride: int = 10
    rgb_window_name: str = "DepthAI RGB small preview"
    stereo_window_name: str = "DepthAI encoded stereo small preview - left | right"

    # Let the camera/encoder control FPS.
    fps: int = 0

    def __init__(self, *args, **kwargs):
        # Compatibility: previous files used codec/bitrate_kbps. If the caller
        # supplies those but not the new explicit RGB fields, mirror them.
        if "codec" in kwargs and "rgb_codec" not in kwargs:
            kwargs["rgb_codec"] = kwargs["codec"]
        if "codec" in kwargs and "stereo_codec" not in kwargs:
            kwargs["stereo_codec"] = kwargs["codec"]
        if "bitrate_kbps" in kwargs and "rgb_bitrate_kbps" not in kwargs:
            kwargs["rgb_bitrate_kbps"] = kwargs["bitrate_kbps"]
        super().__init__(*args, **kwargs)

    @property
    def packed_stereo_rows(self) -> int:
        stereo_values = 2 * int(self.stereo_height) * int(self.stereo_width)
        values_per_row = 3 * int(self.rgb_width)
        return (stereo_values + values_per_row - 1) // values_per_row

    @property
    def stereo_payload_start_row(self) -> int:
        return int(self.rgb_height) - int(self.packed_stereo_rows)

    @property
    def rgb_valid_height(self) -> int:
        return int(self.stereo_payload_start_row)

    @property
    def packed_height(self) -> int:
        return int(self.rgb_height)

    def unpack_packed_tensor(self, packed: torch.Tensor):
        """
        Returns (rgb_with_payload, stereo, left, right).

        Top rows contain the left stereo payload.
        Bottom rows contain the right stereo payload.

        Payload order inside each region is:
            channel 0 rows, channel 1 rows, channel 2 rows
        """

        if packed.ndim != 4 or packed.shape[1] != 3:
            raise ValueError(f"Expected tensor [B, 3, H, W], got {tuple(packed.shape)}.")

        b, _, rgb_h, rgb_w = packed.shape

        stereo_h = int(self.stereo_height)
        stereo_w = int(self.stereo_width)

        stereo_one_values = stereo_h * stereo_w
        values_per_payload_row = 3 * int(rgb_w)

        payload_rows_per_stereo = (
            stereo_one_values + values_per_payload_row - 1
        ) // values_per_payload_row

        total_payload_rows = 2 * payload_rows_per_stereo

        if total_payload_rows > int(rgb_h):
            raise ValueError(
                "Packed stereo payload cannot fit in top + bottom rows: "
                f"payload_rows_per_stereo={payload_rows_per_stereo}, "
                f"total_payload_rows={total_payload_rows}, rgb_height={rgb_h}."
            )

        top_payload = packed[:, :, :payload_rows_per_stereo, :]
        bottom_payload = packed[:, :, rgb_h - payload_rows_per_stereo :, :]

        def payload_region_to_flat(region: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [region[:, c, :, :].reshape(b, -1) for c in range(3)],
                dim=1,
            )

        left_flat = payload_region_to_flat(top_payload)[:, :stereo_one_values]
        right_flat = payload_region_to_flat(bottom_payload)[:, :stereo_one_values]

        left = left_flat.reshape(b, stereo_h, stereo_w)
        right = right_flat.reshape(b, stereo_h, stereo_w)

        stereo = torch.stack([left, right], dim=1)

        return packed, stereo, left, right

    def _tensor_color_type(self):
        for name in ("RGBP", "RGB_CHW", "RGB", "BGR"):
            if hasattr(ColorType, name):
                return getattr(ColorType, name)
        return ColorType.BGR

    def on_rgb_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def on_rgb_stereo_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def create_frame_generator(self, idx, source):
        tensor_color_type = self._tensor_color_type()
        if idx >= len(self.color_types):
            self.color_types.append(tensor_color_type)
        else:
            self.color_types[idx] = tensor_color_type

        capture = self.register_resource(
            _DepthAIPoeRGBStereoH26xBottomTorchTensorCapture(
                owner=self,
                source=source,
                idx=idx,
            )
        )

        def gen(capture=capture):
            while True:
                try:
                    yield capture.next_frame()
                except StopIteration:
                    return
                except Exception:
                    if capture.stop_event.is_set() or capture._released:
                        return
                    logger(f"[{self.owner.uuid}:warning] DepthAI RGB + H26x stereo bottom generator failed:")
                    traceback.print_exc()
                    raise

        return gen()


def to_small_cv(mat, s=10, rgb_to_bgr=True):
    """
    Accepts torch tensor in:
      CHW RGB: [3, H, W]
      HWC RGB: [H, W, 3]
      Gray:    [H, W]

    Returns uint8 CPU numpy image for cv2.imshow().
    """
    x = mat.detach()

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    x = x[::s, ::s]

    if x.dtype.is_floating_point:
        # Works for normalized 0-1 tensors.
        # Also safe if values are already 0-255-ish.
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = x.clamp(0, 255).to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    arr = x.cpu().numpy()

    # RGB -> BGR only for 3-channel images.
    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()

    # If shape is [H, W, 1], make it [H, W]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr

def _get_gen(decoder_backend="gst-nvivafilter"):
    return DepthAIPoeRGBStereoTorchGenerator(
        uuid="OkadCam:CamA",
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_codec="h265",
        stereo_codec="h265",
        rgb_bitrate_kbps=60000,
        stereo_bitrate_kbps=6000,
        decoder_backend=decoder_backend,
        gst_nvivafilter_so="./libdepthai_cuda_preprocess.so",
        gst_nvivafilter_dtype="fp16",
        gst_nvivafilter_channel_order="rgba",
        decoder_output_color="rgbp",
        stereo_decoder_output_color="rgbp",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )

def test_rgb_stereo(decoder_backend="pynvvideocodec"):
    """Small shape/unpack smoke test. This prints every frame and is not a benchmark."""

    gen = _get_gen(decoder_backend=decoder_backend)

    try:
        for i,mats in enumerate(gen):
            packed = mats[0].data
            rgb, stereo, left, right = gen.unpack_packed_tensor(packed)

            # Important for bottom-inplace packing:
            # bottom rows of rgb are overwritten by stereo payload.
            rgb_valid = rgb[:, :, :gen.stereo_payload_start_row, :]

            small_rgb = to_small_cv(rgb_valid[0], s=10, rgb_to_bgr=True)
            small_left = to_small_cv(left[0], s=4, rgb_to_bgr=False)
            small_right = to_small_cv(right[0], s=4, rgb_to_bgr=False)

            cv2.imshow("small_rgb", small_rgb)
            cv2.imshow("small_left", small_left)
            cv2.imshow("small_right", small_right)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # For FPS testing, do not print every frame.
            # Print every N frames instead.
            if i%10==0:
                print(
                    "packed", tuple(packed.shape), packed.device, packed.dtype,
                    "rgb", tuple(rgb.shape),
                    "stereo", tuple(stereo.shape),
                    "left", tuple(left.shape),
                    "right", tuple(right.shape),
                )

    finally:
        gen.release()
        cv2.destroyAllWindows()

def test_rgb_stereo_pynvvideocodec():
    return test_rgb_stereo(decoder_backend="pynvvideocodec")

def test_rgb_stereo_gst_nvivafilter():
    return test_rgb_stereo(decoder_backend="gst-nvivafilter")

def benchmark_rgb_stereo(duration_sec: float = 10.0, warmup_sec: float = 2.0):
    """No per-frame printing/unpacking benchmark."""

    gen = _get_gen()
    total = 0
    measured = 0
    first_shape = None
    first_dtype = None
    first_device = None
    t0 = time.monotonic()
    measure_t0 = None

    try:
        for mats in gen:
            packed = mats[0].data
            total += 1
            if first_shape is None:
                first_shape = tuple(packed.shape)
                first_dtype = packed.dtype
                first_device = packed.device
            now = time.monotonic()
            if measure_t0 is None and now - t0 >= warmup_sec:
                measure_t0 = now
                measured = 0
            if measure_t0 is not None:
                measured += 1
                if now - measure_t0 >= duration_sec:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = max(time.monotonic() - (measure_t0 or t0), 1e-6)
        fps = measured / elapsed if measure_t0 is not None else total / max(time.monotonic() - t0, 1e-6)
        print(
            f"benchmark frames={measured if measure_t0 is not None else total}, "
            f"fps={fps:.2f}, shape={first_shape}, dtype={first_dtype}, device={first_device}"
        )
    finally:
        gen.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    benchmark_rgb_stereo()
