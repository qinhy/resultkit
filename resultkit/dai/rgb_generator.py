import contextlib
import ctypes
import os
import platform
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional

import cv2
import depthai as dai
import numpy as np
import torch

from .generator import ImageMatGenerator
from ..MatModel import CodecFormat, ColorFormat
ColorType = ColorFormat
from ..logger import logger


class _DecoderBackend(ABC):
    """Compressed H264/H265 bytes -> decoded torch.Tensor frames."""

    # False: backend returns a raw single-frame tensor and capture finalizes it.
    # True: backend already returns the public generator tensor, usually
    #       [1, 3, H, W] floating point normalized on CUDA.
    returns_public_tensor = False

    def __init__(self, owner, stop_event: threading.Event):
        self.owner = owner
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
        """
        Return a single decoded uint8 tensor, without batch dimension and without
        normalization. Shape follows owner.decoder_output_color:
            rgbp -> [3, H, W]
            rgb  -> [H, W, 3]
        """

    @abstractmethod
    def close(self):
        pass

    def stats(self) -> Dict[str, float]:
        return {}


class _LiveBitstreamFeeder:
    """
    Feeds compressed OAK H264/H265 bytes into PyNvVideoCodec CreateDemuxer(callback).

    This is safer than manually creating nvc.PacketData.
    """

    def __init__(self, bitstream_queue, stop_event: threading.Event):
        self.q = bitstream_queue
        self.stop_event = stop_event
        self.pending = bytearray()
        self.eof = False
        self.total_bytes_fed = 0

    def feed_chunk(self, demuxer_buffer):
        # During shutdown, do not feed buffered partial GOP data to the demuxer.
        # Return EOF immediately so PyNvVideoCodec can unwind cleanly.
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
            except Exception:
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


class _PyNvVideoCodecBackend(_DecoderBackend):
    """dGPU backend. Keeps the original PyNvVideoCodec + DLPack behavior."""

    def __init__(self, owner, stop_event: threading.Event):
        super().__init__(owner, stop_event)
        import queue

        self.nvc = None
        self.bitstream_q = queue.Queue(maxsize=owner.bitstream_queue_size)
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
        raise ValueError(f"Unsupported decoder_output_color: {name}")

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
        # Import here so Jetson can import this module without PyNvVideoCodec installed.
        import PyNvVideoCodec as nvc

        self.nvc = nvc
        self.feeder = _LiveBitstreamFeeder(self.bitstream_q, self.stop_event)

    def _ensure_decoder(self):
        if self.decoder is not None:
            return

        nvc = self.nvc
        owner = self.owner

        logger("Creating PyNvVideoCodec demuxer...")
        self.demuxer = nvc.CreateDemuxer(self.feeder.feed_chunk)

        kwargs = {
            "gpuid": owner.gpu_id,
            "codec": self.demuxer.GetNvCodecId(),
            "usedevicememory": True,
            "maxwidth": owner.width,
            "maxheight": owner.height,
            "outputColorType": self._output_color_type(nvc, owner.decoder_output_color),
        }

        if owner.low_latency:
            latency = self._get_low_latency_enum(nvc)
            if latency is not None:
                kwargs["latency"] = latency
            else:
                logger("Warning: PyNvVideoCodec low-latency enum not found.")

        logger("Creating PyNvVideoCodec decoder...")
        self.decoder = nvc.CreateDecoder(**kwargs)
        self.packet_iter = iter(self.demuxer)

    def push_bitstream(self, data: bytes):
        if self.closed or self.stop_event.is_set():
            return

        while not self.closed and not self.stop_event.is_set():
            try:
                self.bitstream_q.put(data, timeout=0.1)
                return
            except Exception:
                continue

    def end_of_stream(self):
        try:
            self.bitstream_q.put_nowait(None)
            return
        except Exception:
            # Make one slot for EOF. Dropping a compressed packet is OK here because
            # we are shutting down and the feeder will report EOF.
            try:
                self.bitstream_q.get_nowait()
            except Exception:
                pass
            try:
                self.bitstream_q.put_nowait(None)
            except Exception:
                pass

    def _retain_decoded_frame_ref(self, frame):
        """
        Keep a few PyNvVideoCodec decoded frame objects alive.

        This helps avoid lifetime issues with DLPack/CUDA memory.
        """

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
                self.close()
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

        # Do not Flush() on forced release: the H264/H265 stream is intentionally
        # truncated, and draining can produce noisy decoder messages. Let objects
        # destruct after references are dropped.
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


class _GstNvv4l2DecoderBackend(_DecoderBackend):
    """
    Jetson fallback backend using GStreamer appsrc -> nvv4l2decoder -> appsink.

    This fallback intentionally pulls RGB frames from appsink and then moves them
    to CUDA with torch. Prefer _GstNvVivaFilterTorchBackend when the custom
    nvivafilter CUDA preprocess library is available.
    """

    def __init__(self, owner, stop_event: threading.Event):
        super().__init__(owner, stop_event)
        self.Gst = None
        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.bus = None
        self.frame_index = 0
        self.bytes_pushed = 0
        self.samples_pulled = 0
        self._lock = threading.RLock()

    def start(self):
        if self.owner.decoder_output_color == "native":
            raise ValueError(
                "gst-nvv4l2 backend currently supports decoder_output_color='rgbp' "
                "or 'rgb'. Use PyNvVideoCodec for native output, or add an NV12 path."
            )

        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as e:
            raise RuntimeError(
                "GStreamer Python bindings are required for decoder_backend='gst-nvv4l2'. "
                "On Jetson install python3-gi and the GStreamer plugins."
            ) from e

        Gst.init(None)
        self.Gst = Gst

        if self.owner.codec == "h265":
            caps = "video/x-h265,stream-format=(string)byte-stream"
            parser = "h265parse config-interval=-1"
        elif self.owner.codec == "h264":
            caps = "video/x-h264,stream-format=(string)byte-stream"
            parser = "h264parse config-interval=-1"
        else:
            raise ValueError(f"Unsupported codec: {self.owner.codec}")

        # RGBA after nvvidconv is broadly supported on Jetson. videoconvert then
        # makes a predictable tightly-packed RGB buffer for numpy/torch.
        pipeline_desc = f"""
            appsrc name=src
                is-live=true
                block=true
                format=time
                do-timestamp=true
                caps=\"{caps}\"
            ! queue max-size-buffers={int(self.owner.gst_queue_size)} max-size-time=0 max-size-bytes=0
            ! {parser}
            ! nvv4l2decoder
            ! nvvidconv
            ! video/x-raw,format=(string)RGBA,width=(int){int(self.owner.width)},height=(int){int(self.owner.height)}
            ! videoconvert
            ! video/x-raw,format=(string)RGB,width=(int){int(self.owner.width)},height=(int){int(self.owner.height)}
            ! appsink name=sink
                sync=false
                emit-signals=false
                max-buffers={int(self.owner.gst_appsink_max_buffers)}
                drop={str(bool(self.owner.gst_appsink_drop)).lower()}
        """
        pipeline_desc = " ".join(pipeline_desc.split())

        logger("Creating GStreamer nvv4l2decoder pipeline:")
        logger(f"  {pipeline_desc}")

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("sink")
        self.bus = self.pipeline.get_bus()

        if self.appsrc is None or self.appsink is None:
            raise RuntimeError("Could not create appsrc/appsink for GStreamer pipeline")

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING state")

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
                raise RuntimeError(f"GStreamer error: {err}; debug={debug}")

            if msg.type == Gst.MessageType.EOS:
                raise StopIteration

    def push_bitstream(self, data: bytes):
        if self.closed or self.stop_event.is_set():
            return

        Gst = self.Gst
        if Gst is None:
            raise RuntimeError("GStreamer backend was not started")

        with self._lock:
            if self.closed or self.stop_event.is_set() or self.appsrc is None:
                return

            self._raise_if_bus_error()

            appsrc = self.appsrc
            duration = int(Gst.SECOND / max(float(self.owner.capture_fps), 1e-6))
            pts = self.frame_index * duration
            self.frame_index += 1

        # Do not hold the backend lock while pushing. appsrc can block under
        # backpressure, and release() needs the lock to set the pipeline to NULL.
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = duration
        buf.pts = pts
        buf.dts = pts

        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            if self.closed or self.stop_event.is_set():
                return
            raise RuntimeError(f"GStreamer push-buffer failed: {ret}")

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
        Gst = self.Gst
        if Gst is None or self.appsink is None:
            raise StopIteration

        while not self.stop_event.is_set():
            self._raise_if_bus_error()

            sample = self.appsink.emit("pull-sample")
            if sample is None:
                if self.closed or self.stop_event.is_set():
                    raise StopIteration
                self._raise_if_bus_error()
                continue

            buffer = sample.get_buffer()
            ok, map_info = buffer.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("Could not map GstBuffer from appsink")

            try:
                arr = np.frombuffer(map_info.data, dtype=np.uint8)
                expected = int(self.owner.height) * int(self.owner.width) * 3
                if arr.size < expected:
                    raise RuntimeError(
                        f"GStreamer RGB buffer too small: got {arr.size}, expected {expected}"
                    )
                arr = arr[:expected].reshape(int(self.owner.height), int(self.owner.width), 3).copy()
            finally:
                buffer.unmap(map_info)

            tensor_cpu = torch.from_numpy(arr)
            if torch.cuda.is_available():
                try:
                    tensor_cpu = tensor_cpu.pin_memory()
                except Exception:
                    pass
                device = torch.device(f"cuda:{int(self.owner.gpu_id)}")
            else:
                device = torch.device("cpu")

            tensor = tensor_cpu.to(device=device, non_blocking=True)

            if self.owner.decoder_output_color == "rgbp":
                tensor = tensor.permute(2, 0, 1).contiguous()
            elif self.owner.decoder_output_color == "rgb":
                tensor = tensor.contiguous()
            else:
                raise ValueError(f"Unsupported decoder_output_color: {self.owner.decoder_output_color}")

            self.samples_pulled += 1
            return tensor

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
            self.appsink = None
            self.appsrc = None
            self.pipeline = None

    def stats(self) -> Dict[str, float]:
        return {
            "gst_samples": float(self.samples_pulled),
            "fed_mb": float(self.bytes_pushed / 1_000_000),
        }



class _GstNvVivaFilterTorchBackend(_DecoderBackend):
    """
    Jetson optimized backend:

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

    This backend returns the public generator tensor directly:
        [1, 3, H, W], CUDA, fp16/fp32, normalized RGB
    """

    returns_public_tensor = True

    _CHANNEL_ORDER_MAP = {
        "auto": 0,
        "rgba": 1,
        "bgra": 2,
        "argb": 3,
        "abgr": 4,
    }

    def __init__(self, owner, stop_event: threading.Event):
        super().__init__(owner, stop_event)
        self.Gst = None
        self.pipeline = None
        self.appsrc = None
        self.bus = None
        self.lib = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.device = torch.device(f"cuda:{int(owner.gpu_id)}")
        self.frame_index = 0
        self.bytes_pushed = 0
        self.last_returned_frame_count = 0
        self._lock = threading.RLock()

    def _load_library(self):
        owner = self.owner
        so_path = os.path.abspath(str(owner.gst_nvivafilter_so))
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"gst_nvivafilter_so does not exist: {so_path}"
            )

        lib = ctypes.CDLL(so_path)

        if not hasattr(lib, "set_torch_output_buffer"):
            raise RuntimeError(
                f"{so_path} does not export set_torch_output_buffer(...)"
            )

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
        elif bool(owner.gst_nvivafilter_require_frame_count):
            raise RuntimeError(
                f"{so_path} does not export get_torch_output_frame_count(). "
                "For a live generator this symbol is the cleanest way to know "
                "when nvivafilter has finished writing a new frame."
            )

        self.lib = lib
        return so_path

    def _allocate_output_tensor(self):
        owner = self.owner
        dtype_name = str(owner.gst_nvivafilter_dtype).lower()
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
                (1, 3, int(owner.height), int(owner.width)),
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
        if self.owner.decoder_output_color != "rgbp":
            raise ValueError(
                "gst-nvivafilter backend returns normalized RGB NCHW. "
                "Use decoder_output_color='rgbp'."
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
                "Warning: nvivafilter library does not expose set_channel_order(); "
                f"requested channel_order={channel_order!r} will be ignored."
            )

        tensor = self._allocate_output_tensor()

        if self.owner.codec == "h265":
            caps = "video/x-h265,stream-format=(string)byte-stream,alignment=(string)au"
            parser = "h265parse config-interval=-1"
        elif self.owner.codec == "h264":
            caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au"
            parser = "h264parse config-interval=-1"
        else:
            raise ValueError(f"Unsupported codec: {self.owner.codec}")

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
            ! video/x-raw(memory:NVMM),format=NV12,width=(int){int(self.owner.width)},height=(int){int(self.owner.height)}
            ! nvivafilter cuda-process=true customer-lib-name={so_path} silent={silent}
            ! video/x-raw(memory:NVMM),format=RGBA
            ! fakesink sync=false
        """
        pipeline_desc = " ".join(pipeline_desc.split())

        logger("Creating GStreamer nvivafilter torch pipeline:")
        logger(f"  {pipeline_desc}")
        logger(
            f"  output tensor: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}, clone_output={self.owner.gst_nvivafilter_clone_output}"
        )

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name("src")
        self.bus = self.pipeline.get_bus()

        if self.appsrc is None:
            raise RuntimeError("Could not create appsrc for GStreamer nvivafilter pipeline")

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer nvivafilter pipeline failed to enter PLAYING state")

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
                raise RuntimeError(f"GStreamer error: {err}; debug={debug}")

            if msg.type == Gst.MessageType.EOS:
                raise StopIteration

    def push_bitstream(self, data: bytes):
        if self.closed or self.stop_event.is_set():
            return

        Gst = self.Gst
        if Gst is None:
            raise RuntimeError("GStreamer nvivafilter backend was not started")

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
            raise RuntimeError(f"GStreamer push-buffer failed: {ret}")

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

                # nvivafilter writes on CUDA. Synchronize before snapshotting, then
                # synchronize again so the clone is complete before the source tensor
                # can be overwritten by a later frame.
                torch.cuda.synchronize(self.device)

                if bool(self.owner.gst_nvivafilter_clone_output):
                    out = self.output_tensor.detach().clone()
                    torch.cuda.synchronize(self.device)
                    return out

                return self.output_tensor

            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for nvivafilter to produce a decoded frame. "
                    f"last_frame_count={self.last_returned_frame_count}, "
                    f"bytes_pushed={self.bytes_pushed}"
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

    def stats(self) -> Dict[str, float]:
        frame_count = self._get_frame_count()
        return {
            "gst_nvivafilter_frames": float(frame_count or 0),
            "fed_mb": float(self.bytes_pushed / 1_000_000),
        }

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

    # Most Jetsons are aarch64. This alone is not definitive, but it is a good
    # auto-mode default because PyNvVideoCodec is typically unavailable there.
    return platform.machine().lower() in ("aarch64", "arm64")


def _create_decoder_backend(owner, stop_event: threading.Event) -> _DecoderBackend:
    backend = getattr(owner, "decoder_backend", "auto")

    if backend == "auto":
        backend = "gst-nvv4l2" if _looks_like_jetson() else "pynvvideocodec"

    if backend == "pynvvideocodec":
        return _PyNvVideoCodecBackend(owner, stop_event)

    if backend == "gst-nvv4l2":
        return _GstNvv4l2DecoderBackend(owner, stop_event)

    if backend == "gst-nvivafilter":
        return _GstNvVivaFilterTorchBackend(owner, stop_event)

    raise ValueError(f"Unsupported decoder_backend: {backend}")


class _DepthAIPoeTorchTensorCapture:
    """
    Internal capture/decoder resource.

    Pipeline:
        DepthAI RGB camera
        -> OAK on-device H265/H264 encoder
        -> compressed packets over PoE
        -> decoder backend
             * PyNvVideoCodec on dGPU
             * GStreamer nvv4l2decoder on Jetson
        -> torch.Tensor
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.depthai_q = None

        self.stop_event = threading.Event()
        self.producer_thread = None
        self.decoder_backend: Optional[_DecoderBackend] = None

        self.oak_packet_count = 0
        self.oak_byte_count = 0
        self.decoded_frame_count = 0

        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_decoded_count = 0

        self._released = False

        # Lets us use DepthAI v3's preferred Pipeline context-manager cleanup
        # while still keeping this object-oriented lifecycle.
        self._exit_stack = contextlib.ExitStack()

        self._start()

    def _open_device(self):
        src = str(self.source).strip()

        if src.startswith("depthai://"):
            src = src.replace("depthai://", "", 1).strip()

        if src in ("", "auto", "default", "none", "None"):
            return dai.Device()

        return dai.Device(dai.DeviceInfo(src))

    @staticmethod
    def _depthai_profile(codec: str):
        if codec == "h265":
            return dai.VideoEncoderProperties.Profile.H265_MAIN
        if codec == "h264":
            return dai.VideoEncoderProperties.Profile.H264_MAIN
        raise ValueError(f"Unsupported codec: {codec}")

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

        socket = getattr(dai.CameraBoardSocket, owner.camera_socket)
        cam = pipeline.create(dai.node.Camera).build(socket)

        rgb_nv12 = cam.requestOutput(
            (owner.width, owner.height),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            owner.capture_fps,
        )

        encoder = pipeline.create(dai.node.VideoEncoder).build(
            rgb_nv12,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.codec),
        )

        try:
            encoder.setBitrateKbps(owner.bitrate_kbps)
        except Exception as e:
            logger(f"Warning: could not set OAK encoder bitrate: {e}")

        try:
            encoder.setKeyframeFrequency(int(owner.capture_fps))
        except Exception:
            pass

        q = encoder.out.createOutputQueue(
            maxSize=owner.depthai_queue_size,
            blocking=True,
        )

        pipeline.start()
        return pipeline, q

    def _producer_loop(self):
        """
        Reads compressed packets from DepthAI and pushes them to the decoder backend.

        Important:
            Do not drop H265/H264 packets if possible. Dropping compressed packets
            corrupts the stream until the next keyframe.
        """

        try:
            while not self.stop_event.is_set():
                try:
                    if hasattr(self.depthai_q, "tryGet"):
                        pkt = self.depthai_q.tryGet()
                        if pkt is None:
                            time.sleep(0.001)
                            continue
                    else:
                        pkt = self.depthai_q.get()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    raise

                data = self._packet_to_bytes(pkt)

                self.oak_packet_count += 1
                self.oak_byte_count += len(data)

                if self.decoder_backend is not None:
                    self.decoder_backend.push_bitstream(data)

        except Exception:
            # Queue/pipeline methods may throw while we are intentionally closing.
            # Treat those as normal shutdown, not as producer failures.
            if not self.stop_event.is_set() and not self._released:
                logger("DepthAI producer thread failed:")
                traceback.print_exc()

        finally:
            self.stop_event.set()
            try:
                if self.decoder_backend is not None:
                    self.decoder_backend.end_of_stream()
            except Exception:
                pass

    def _start(self):
        owner = self.owner

        if owner.codec in ("h264", "h265") and owner.width % 32 != 0:
            raise ValueError(
                "DepthAI H264/H265 encoder requires width multiple of 32. "
                "Use 4032 instead of 4056."
            )

        logger("Opening DepthAI device...")
        self.device = self._open_device()

        self.decoder_backend = _create_decoder_backend(owner, self.stop_event)
        decoder_backend_name = self.decoder_backend.__class__.__name__.replace("_", "")

        logger("Connected DepthAI device:")
        logger(f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        logger(f"  Cameras: {self.device.getConnectedCameras()}")
        logger("")
        logger("Starting DepthAI PoE torch tensor pipeline:")
        logger(f"  Source: {self.source}")
        logger(f"  Camera socket: {owner.camera_socket}")
        logger(f"  Size: {owner.width}x{owner.height}")
        logger(f"  OAK encoder: {owner.codec.upper()} @ {owner.capture_fps} FPS")
        logger(f"  Bitrate: {owner.bitrate_kbps} kbps")
        logger(f"  Decoder output: {owner.decoder_output_color}")
        logger(f"  Decoder backend: {getattr(owner, 'decoder_backend', 'auto')} -> {decoder_backend_name}")
        logger(f"  GPU ID: {owner.gpu_id}")
        logger("")

        self.decoder_backend.start()
        self.pipeline, self.depthai_q = self._create_depthai_pipeline()

        self.producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
        )
        self.producer_thread.start()

        logger("DepthAI tensor pipeline ready.")

    def _show_small_preview_from_tensor(self, tensor: torch.Tensor):
        owner = self.owner

        stride = max(1, int(owner.preview_stride))

        if tensor.ndim != 3:
            logger(f"Cannot preview tensor shape: {tuple(tensor.shape)}")
            return

        # RGBP / CHW
        if tensor.shape[0] == 3:
            small = tensor[:, ::stride, ::stride]
            small_hwc = small.permute(1, 2, 0).contiguous()

        # RGB / HWC
        elif tensor.shape[-1] == 3:
            small_hwc = tensor[::stride, ::stride, :].contiguous()

        else:
            logger(f"Cannot preview tensor shape: {tuple(tensor.shape)}")
            return

        small_rgb_tensor = small_hwc.detach()
        if torch.is_floating_point(small_rgb_tensor):
            small_rgb_tensor = small_rgb_tensor.clamp(0, 1).mul(255).to(torch.uint8)
        else:
            small_rgb_tensor = small_rgb_tensor.to(torch.uint8)

        small_rgb = small_rgb_tensor.cpu().numpy()
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(owner.window_name, small_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _finalize_decoded_tensor(self, tensor: torch.Tensor):
        """
        Convert decoded uint8 tensor to the public generator output.

        Backend returns:
            rgbp -> [3, H, W] uint8
            rgb  -> [H, W, 3] uint8

        Public output remains:
            rgbp -> [1, 3, H, W] float, normalized
            rgb  -> [1, H, W, 3] float, normalized
        """

        self.decoded_frame_count += 1

        self.owner.on_decoded_tensor(tensor, self.decoded_frame_count)

        if self.owner.show_small_preview:
            self._show_small_preview_from_tensor(tensor)

        return tensor.unsqueeze(0).float().div_(255.0)

    def _accept_public_tensor(self, tensor: torch.Tensor):
        """
        Accept a backend tensor that is already in the public generator format.

        Expected:
            [1, 3, H, W], floating point, normalized, preferably CUDA
        """

        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise RuntimeError(
                f"Public decoder backend returned invalid tensor shape: {tuple(tensor.shape)}"
            )

        self.decoded_frame_count += 1

        single = tensor[0]
        self.owner.on_decoded_tensor(single, self.decoded_frame_count)

        if self.owner.show_small_preview:
            self._show_small_preview_from_tensor(single)

        return tensor

    def next_frame(self):
        """Return one full decoded torch.Tensor."""

        while not self.stop_event.is_set():
            if self.decoder_backend is None:
                raise StopIteration

            try:
                tensor = self.decoder_backend.next_tensor()
            except StopIteration:
                self.release()
                raise
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise

            if getattr(self.decoder_backend, "returns_public_tensor", False):
                out = self._accept_public_tensor(tensor)
            else:
                out = self._finalize_decoded_tensor(tensor)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                dec_fps = (self.decoded_frame_count - self.last_decoded_count) / dt

                avg_mbps = (
                    self.oak_byte_count
                    * 8.0
                    / max(now - self.started_at, 1e-6)
                    / 1_000_000
                )

                backend_stats = self.decoder_backend.stats() if self.decoder_backend else {}
                fed_mb = backend_stats.get("fed_mb", 0.0)

                logger(
                    f"decoded={self.decoded_frame_count}, "
                    f"dec_fps={dec_fps:.2f}, "
                    f"oak_packets={self.oak_packet_count}, "
                    f"oak_mbps={avg_mbps:.1f}, "
                    f"fed_mb={fed_mb:.1f}"
                )

                self.last_log_at = now
                self.last_decoded_count = self.decoded_frame_count

            return out

        raise StopIteration

    def release(self):
        if self._released:
            return

        self._released = True
        self.stop_event.set()

        # Close decoder first to unblock appsrc/appsink or PyNvVideoCodec feeder waits.
        try:
            if self.decoder_backend is not None:
                self.decoder_backend.close()
        except Exception as e:
            logger(f"Warning: decoder backend close during release: {e}")

        # Stop host-side queue use before stopping the DepthAI pipeline/device.
        # This avoids producer-thread messages from tryGet()/queue internals during exit.
        try:
            if self.depthai_q is not None and hasattr(self.depthai_q, "close"):
                self.depthai_q.close()
        except Exception:
            pass

        try:
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=2.0)
        except Exception as e:
            logger(f"Error joining producer thread: {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self.pipeline.stop()
        except Exception as e:
            # By this point shutdown has already been requested, so do not make
            # benign DepthAI teardown races look like runtime failures.
            logger(f"Warning: DepthAI pipeline stop during release: {e}")

        try:
            self._exit_stack.close()
        except Exception:
            pass

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self.device.close()
        except Exception:
            pass

        self.decoder_backend = None
        self.depthai_q = None
        self.pipeline = None
        self.device = None

        if self.owner.show_small_preview:
            try:
                cv2.destroyWindow(self.owner.window_name)
            except Exception:
                pass


class DepthAIPoeRGBTorchGenerator(ImageMatGenerator):
    """
    ImageMatGenerator-style DepthAI PoE realtime tensor generator.

    Output:
        List[ImageMat]

    Each ImageMat receives:
        full torch.Tensor, preferably on CUDA

    Pipeline:
        OAK RGB CAM_A
        -> OAK H265/H264 encoder
        -> Decoder backend:
             * decoder_backend='pynvvideocodec' for dGPU NVDEC + DLPack
             * decoder_backend='gst-nvv4l2' for Jetson nvv4l2decoder
             * decoder_backend='auto' chooses Jetson GStreamer on Jetson/aarch64,
               PyNvVideoCodec elsewhere
        -> ImageMat.unsafe_update_mat(torch.Tensor)

    Recommended dGPU:
        decoder_backend='pynvvideocodec'
        width=4032
        height=3040
        capture_fps=15
        codec='h265'
        decoder_output_color='rgbp'
        fps=0

    Recommended Jetson optimized path:
        decoder_backend='gst-nvivafilter'
        gst_nvivafilter_so='./libdepthai_cuda_preprocess.so'
        gst_nvivafilter_dtype='fp16'
        gst_nvivafilter_channel_order='rgba'
        decoder_output_color='rgbp'
        fps=0

    Jetson fallback path:
        decoder_backend='gst-nvv4l2'
        decoder_output_color='rgbp'
        fps=0
    """

    color_types: List['ColorType'] = []

    width: int = 4032
    height: int = 3040
    capture_fps: float = 15.0

    camera_socket: Literal["CAM_A", "CAM_B", "CAM_C"] = "CAM_A"

    codec: Literal["h265", "h264"] = "h265"
    bitrate_kbps: int = 60000

    gpu_id: int = 0

    # Backend selection:
    #   auto           -> gst-nvv4l2 on Jetson/aarch64, PyNvVideoCodec otherwise
    #   pynvvideocodec -> original dGPU path
    #   gst-nvv4l2      -> Jetson fallback path through RGB appsink/CPU
    #   gst-nvivafilter -> Jetson optimized path through nvivafilter into torch CUDA
    decoder_backend: Literal["auto", "pynvvideocodec", "gst-nvv4l2", "gst-nvivafilter"] = "auto"

    # Use rgbp for torch processing.
    # Usually returns tensor shape [1, 3, H, W] after normalization.
    decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"

    depthai_queue_size: int = 8
    bitstream_queue_size: int = 64

    # GStreamer backend tuning. We drop only decoded appsink frames in the fallback
    # backend, not compressed H264/H265 packets, because dropping compressed packets
    # corrupts the GOP.
    gst_queue_size: int = 8
    gst_appsink_max_buffers: int = 1
    gst_appsink_drop: bool = True

    # Jetson optimized nvivafilter -> torch CUDA backend.
    # The custom library should expose set_torch_output_buffer(...), and for live
    # use should also expose get_torch_output_frame_count().
    gst_nvivafilter_so: str = "./libdepthai_cuda_preprocess.so"
    gst_nvivafilter_dtype: Literal["fp16", "fp32"] = "fp16"
    gst_nvivafilter_channel_order: Literal["auto", "rgba", "bgra", "argb", "abgr"] = "rgba"
    gst_nvivafilter_clone_output: bool = True
    gst_nvivafilter_wait_timeout_sec: float = 2.0
    gst_nvivafilter_require_frame_count: bool = True
    gst_nvivafilter_disable_dpb: bool = True
    gst_nvivafilter_enable_full_frame: bool = True
    gst_nvivafilter_silent: bool = False

    low_latency: bool = False
    log_fps: bool = True

    retain_decoded_frame_refs: int = 16

    # Optional small cv2 preview from the big tensor.
    show_small_preview: bool = False
    preview_stride: int = 10
    window_name: str = "DepthAI full GPU tensor small preview"

    # Important:
    # Let the camera/encoder control FPS.
    # Do not add ImageMatGenerator sleep on top.
    fps: int = 0

    def _tensor_color_type(self):
        """
        Choose the best ColorType available in your enum.

        For rgbp, the tensor is usually BCHW after generator output:
            [1, 3, H, W]
        """

        if self.decoder_output_color == "rgbp":
            for name in ("RGBP", "RGB_CHW", "RGB"):
                if hasattr(ColorType, name):
                    return getattr(ColorType, name)

        if self.decoder_output_color == "rgb":
            if hasattr(ColorType, "RGB"):
                return ColorType.RGB

        if self.decoder_output_color == "native":
            for name in ("NV12", "YUV", "RGB"):
                if hasattr(ColorType, name):
                    return getattr(ColorType, name)

        return ColorType.BGR

    def on_decoded_tensor(self, tensor: torch.Tensor, frame_index: int):
        """
        Override this in a subclass for your real GPU processing.

        Called before batch dimension and normalization are added.

        For decoder_output_color='rgbp':
            tensor shape is usually [3, 3040, 4032]

        For decoder_output_color='rgb':
            tensor shape is usually [3040, 4032, 3]
        """

        pass

    def create_frame_generator(self, idx, source):
        tensor_color_type = self._tensor_color_type()

        if idx >= len(self.color_types):
            self.color_types.append(tensor_color_type)
        else:
            self.color_types[idx] = tensor_color_type

        capture = self.register_resource(
            _DepthAIPoeTorchTensorCapture(
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
                    raise

        return gen()


def test_dgpu_pynvvideocodec():
    gen = DepthAIPoeRGBTorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        width=4032,
        height=3040,
        capture_fps=15,
        codec="h265",
        bitrate_kbps=60000,
        decoder_backend="pynvvideocodec",
        decoder_output_color="rgbp",
        show_small_preview=False,
        preview_stride=10,
        fps=0,
    )

    try:
        for mats in gen:
            mat = mats[0]
            tensor = mat.data
            small_gpu = (
                (tensor[0].permute(1, 2, 0)[::10, ::10] * 255.0)
                .clone()
                .detach()
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()[:, :, ::-1]
            )

            cv2.imshow("test", small_gpu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        gen.release()
        cv2.destroyAllWindows()


def test_jetson_gst_nvv4l2():
    gen = DepthAIPoeRGBTorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        width=4032,
        height=3040,
        capture_fps=15,
        codec="h265",
        bitrate_kbps=60000,
        decoder_backend="gst-nvv4l2",
        decoder_output_color="rgbp",
        show_small_preview=False,
        preview_stride=10,
        fps=0,
    )

    try:
        for mats in gen:
            mat = mats[0]
            tensor = mat.data
            small_gpu = (
                (tensor[0].permute(1, 2, 0)[::10, ::10] * 255.0)
                .clone()
                .detach()
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()[:, :, ::-1]
            )

            # cv2.imwrite("jetson-gst-test.png", small_gpu)
            cv2.imshow("jetson-gst-test", small_gpu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        gen.release()
        cv2.destroyAllWindows()



def test_jetson_gst_nvivafilter():
    gen = DepthAIPoeRGBTorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        width=4032,
        height=3040,
        capture_fps=15,
        codec="h265",
        bitrate_kbps=60000,
        decoder_backend="gst-nvivafilter",
        gst_nvivafilter_so="./libdepthai_cuda_preprocess.so",
        gst_nvivafilter_dtype="fp16",
        gst_nvivafilter_channel_order="rgba",
        gst_nvivafilter_clone_output=True,
        decoder_output_color="rgbp",
        show_small_preview=False,
        preview_stride=10,
        fps=0,
    )

    try:
        for mats in gen:
            mat = mats[0]
            tensor = mat.data  # [1, 3, H, W], fp16/fp32, CUDA, normalized RGB
            small_gpu = (
                tensor[0]
                .permute(1, 2, 0)[::10, ::10]
                .clamp(0, 1)
                .mul(255)
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()[:, :, ::-1]
            )

            cv2.imshow("jetson-nvivafilter-test", small_gpu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        gen.release()
        cv2.destroyAllWindows()
