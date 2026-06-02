"""
Clean H264 FFmpeg pub/sub demo for resultkit.Model4Mat.EncodedImageMatPubSub.

Streaming model:
    one decoded frame
    -> one H264 Annex B access-unit payload
    -> one EncodedImageMatPubSub packet
    -> pub/sub transport
    -> one decoded ImageMat

Run publisher:
    python gen_codec_vis_cuda.py pub

Run subscriber:
    python gen_codec_vis_cuda.py sub

Legacy compatible subscriber flag:
    python gen_codec_vis_cuda.py --sub

Custom FFmpeg binary:
    python gen_codec_vis_cuda.py pub --ffmpeg-bin /path/to/ffmpeg
    FFMPEG_BIN=/path/to/ffmpeg python gen_codec_vis_cuda.py sub
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
import torch

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
except ImportError:
    print("warning: pycuda is not available")
    cuda = None
    gpuarray = None

# Adjust this exactly like your current test file if needed.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


DEFAULT_TOPIC = "EncodedImageMatPubSub:test"
DEFAULT_FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
NS_PER_SECOND = 1_000_000_000


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is missing or returns a non-zero exit code."""


@dataclass(frozen=True)
class StreamConfig:
    topic: str = DEFAULT_TOPIC
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    width: int = 96
    height: int = 64
    fps: int = 30
    crf: int = 18
    color_format: ColorFormat = ColorFormat.BGR
    stats_every: int = 100
    max_frames: Optional[int] = None
    capacity_scale: float = 2.0
    min_capacity_bytes: int = 64 * 1024
    display: bool = True
    window_name: str = "resultkit h264 pub/sub"
    wait_ms: Optional[int] = None
    poll_sleep_s: float = 0.001
    viewer: str = "gl"
    device: int = 0
    image_topic: str = "ImageMatCUDAPubSub:decoded"
    num_slots: int = 3
    flip_y: bool = True

    @property
    def frame_period_ns(self) -> int:
        return int(NS_PER_SECOND / max(self.fps, 1))

    @property
    def cv_wait_ms(self) -> int:
        if self.wait_ms is not None:
            return max(1, int(self.wait_ms))
        return max(1, int(1000 / max(self.fps, 1)))


class FFmpegRunner:
    def __init__(self, ffmpeg_bin: str = DEFAULT_FFMPEG_BIN):
        if not ffmpeg_bin:
            raise ValueError("ffmpeg_bin must be a non-empty string")
        self.ffmpeg_bin = ffmpeg_bin

    def resolve(self) -> str:
        resolved = shutil.which(self.ffmpeg_bin)
        if resolved is None:
            raise FFmpegError(
                f"ffmpeg executable was not found: {self.ffmpeg_bin!r}. "
                "Pass --ffmpeg-bin /path/to/ffmpeg or set FFMPEG_BIN."
            )
        return resolved

    def run(self, args: list[str], input_bytes: bytes) -> bytes:
        cmd = [self.resolve(), *args]
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")
            raise FFmpegError(
                "ffmpeg failed\n\n"
                f"CMD:\n{' '.join(cmd)}\n\n"
                f"STDERR:\n{stderr}"
            )

        return proc.stdout


class H264AnnexBCodec:
    """Stateless FFmpeg helper for one-keyframe-per-packet H264 testing."""

    def __init__(self, runner: FFmpegRunner, fps: int = 30, crf: int = 18):
        self.runner = runner
        self.fps = int(fps)
        self.crf = int(crf)

    @staticmethod
    def as_bgr24_frame(
        frame: np.ndarray,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> np.ndarray:
        color_format = ColorFormat(color_format)

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected HWC-3 frame, got shape {arr.shape}")

        if color_format == ColorFormat.BGR:
            return np.ascontiguousarray(arr)

        if color_format == ColorFormat.RGB:
            return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        raise ValueError(f"H264 pub/sub demo expects RGB or BGR input, got {color_format}")

    def encode_access_unit(
        self,
        frame: np.ndarray,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> bytes:
        """
        Encode one decoded frame into one H264 Annex B access unit.

        The returned payload may contain multiple NAL units. For this demo,
        every payload is independently decodable because libx264 is configured
        to emit an IDR/keyframe and repeat SPS/PPS headers for every frame.
        """
        bgr = self.as_bgr24_frame(frame, color_format=color_format)
        h, w = bgr.shape[:2]

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            str(self.crf),
            "-x264-params",
            "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
            "pipe:1",
        ]

        payload = self.runner.run(args, bgr.tobytes())
        if not payload:
            raise FFmpegError("FFmpeg returned an empty H264 payload")
        return payload

    def decode_access_unit(
        self,
        packet_bytes: bytes,
        width: int,
        height: int,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> np.ndarray:
        """Decode one H264 Annex B access unit into one HWC uint8 frame."""
        color_format = ColorFormat(color_format)

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        raw = self.runner.run(args, packet_bytes)
        expected = int(width) * int(height) * 3
        if len(raw) != expected:
            raise FFmpegError(f"Expected {expected} decoded bytes, got {len(raw)}")

        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(int(height), int(width), 3).copy()

        if color_format == ColorFormat.BGR:
            return bgr
        if color_format == ColorFormat.RGB:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise ValueError(f"Unsupported output color_format: {color_format}")


class FpsMeter:
    def __init__(self, label: str, stats_every: int = 100):
        self.label = label
        self.stats_every = max(1, int(stats_every))
        self.start_t = time.perf_counter()
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += int(n)
        if self.count % self.stats_every != 0:
            return

        elapsed = time.perf_counter() - self.start_t
        if elapsed <= 0:
            return

        print(f"{self.label} FPS: {self.count / elapsed:.2f}")


class PyNvH264Decoder:
    """Small adapter around PyNvVideoCodec's low-level Decoder API.

    The important part is constructing PacketData using the fields that the
    native decoder actually consumes: bsl_data, bsl, pts, and decode_flag.
    Some PyNvVideoCodec docs also mention bitstream/size/flags, so this class
    sets both names when available for version compatibility.
    """

    @staticmethod
    def require_pynvvideocodec():
        try:
            import PyNvVideoCodec as nvc
        except ImportError:
            try:
                import pynvvideocodec as nvc
            except ImportError as exc:
                raise RuntimeError(
                    "PyNvVideoCodec is required for cuda-sub. Install NVIDIA PyNvVideoCodec."
                ) from exc
        return nvc

    def __init__(
        self,
        *,
        device: int,
        width: int,
        height: int,
        use_device_memory: bool = False,
    ):
        self.nvc = self.require_pynvvideocodec()
        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        # OpenCV display needs a CPU array.  Keep the NVDEC hardware decoder,
        # but request host output by default so the demo does not require a
        # CUDA-enabled PyTorch/CuPy install just to call cv2.imshow().  Set
        # use_device_memory=True if your downstream consumer is GPU-only.
        self.use_device_memory = bool(use_device_memory)
        self.packet_index = 0
        # Keep recent encoded packet buffers alive while the native decoder uses
        # their pointers. This matters when PacketData.bsl_data is an address.
        self._packet_buffers: list[np.ndarray] = []
        self.decoder = self._create_decoder()

    def _enum_value(self, enum_name: str, member: str, default=None):
        enum_obj = getattr(self.nvc, enum_name, None)
        if enum_obj is None:
            return default
        return getattr(enum_obj, member, default)

    def _create_decoder(self):
        codec = getattr(getattr(self.nvc, "cudaVideoCodec", None), "H264", None)
        if codec is None:
            raise RuntimeError("PyNvVideoCodec cudaVideoCodec.H264 was not found")

        kwargs = {
            "gpuid": self.device,
            "codec": codec,
            "cudacontext": 0,
            "cudastream": 0,
            "usedevicememory": self.use_device_memory,
            "maxwidth": self.width,
            "maxheight": self.height,
        }

        # Ask the decoder to output RGB if the installed version supports it.
        output_color_type = getattr(self.nvc, "OutputColorType", None)
        if output_color_type is not None:
            rgb = getattr(output_color_type, "RGB", None)
            if rgb is not None:
                kwargs["outputColorType"] = rgb

        # Your stream is one complete access unit per packet with no B frames.
        # LOW/ZERO plus ENDOFPICTURE tells the parser to return immediately.
        latency_enum = getattr(self.nvc, "DisplayDecodeLatencyType", None)
        if latency_enum is not None:
            latency = getattr(latency_enum, "ZERO", getattr(latency_enum, "LOW", None))
            if latency is not None:
                kwargs["latency"] = latency

        try:
            return self.nvc.CreateDecoder(**kwargs)
        except TypeError:
            # Older builds may expose fewer keyword parameters even when the
            # enums exist. Fall back to the minimal core decoder parameters.
            kwargs.pop("latency", None)
            kwargs.pop("outputColorType", None)
            return self.nvc.CreateDecoder(**kwargs)

    @staticmethod
    def _try_set(obj, name: str, value) -> bool:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

    def _end_of_picture_flag(self):
        flag_enum = getattr(self.nvc, "VideoPacketFlag", None)
        if flag_enum is None:
            return 0
        for name in ("ENDOFPICTURE", "END_OF_PICTURE", "ENDOFPICTURE_FLAG"):
            value = getattr(flag_enum, name, None)
            if value is not None:
                return value
        return 0

    def _make_packet(self, payload: bytes):
        payload = bytes(payload)
        if not payload:
            raise ValueError("Empty H264 payload")

        packet_cls = getattr(self.nvc, "PacketData", None)
        if packet_cls is None:
            # Very old/alternate builds may accept bytes directly.
            return payload

        try:
            packet = packet_cls()
        except Exception:
            # Some documented builds support keyword construction instead.
            packet = packet_cls(
                bitstream=payload,
                size=len(payload),
                pts=self.packet_index,
                flags=0,
            )

        # Use a numpy-owned byte buffer and pass its address.  The native
        # PyNvVideoCodec 2.x decoder path reads packetData.bsl_data and
        # packetData.bsl, not only bitstream/size.
        packet_buf = np.frombuffer(payload, dtype=np.uint8).copy()
        self._packet_buffers.append(packet_buf)
        if len(self._packet_buffers) > 16:
            del self._packet_buffers[:-16]

        # Native/internal field names used by PyNvDecoder::Decode.
        data_ok = self._try_set(packet, "bsl_data", int(packet_buf.ctypes.data))
        if not data_ok:
            data_ok = self._try_set(packet, "bsl_data", payload)
        size_ok = self._try_set(packet, "bsl", int(packet_buf.nbytes))

        # Documented/newer field names. Set these too when available.
        data_ok = self._try_set(packet, "bitstream", payload) or data_ok
        size_ok = self._try_set(packet, "size", int(packet_buf.nbytes)) or size_ok

        self._try_set(packet, "pts", int(self.packet_index))
        self._try_set(packet, "dts", int(self.packet_index))
        self._try_set(packet, "duration", 1)
        self._try_set(packet, "key", True)

        decode_flag = self._end_of_picture_flag()
        self._try_set(packet, "decode_flag", decode_flag)
        self._try_set(packet, "flags", decode_flag)

        if not data_ok:
            raise ValueError("Could not set PacketData payload field: bsl_data or bitstream")
        if not size_ok:
            raise ValueError("Could not set PacketData size field: bsl or size")

        return packet

    @staticmethod
    def _looks_like_decoded_frame(obj) -> bool:
        return (
            obj is not None
            and (
                obj.__class__.__name__ == "DecodedFrame"
                or hasattr(obj, "__dlpack__")
                or hasattr(obj, "cuda")
                or hasattr(obj, "framesize")
            )
        )

    @staticmethod
    def _numeric_numpy_array(value) -> Optional[np.ndarray]:
        """Return value as a real numeric ndarray, never as an object scalar."""
        if value is None:
            return None

        if PyNvH264Decoder._looks_like_decoded_frame(value):
            return None

        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "__array_interface__"):
            arr = np.asarray(value)
        else:
            return None

        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            return None
        return arr

    def _frames_as_list(self, frames) -> list:
        if frames is None:
            return []
        if self._looks_like_decoded_frame(frames):
            return [frames]
        if isinstance(frames, (list, tuple)):
            return list(frames)
        try:
            return list(frames)
        except TypeError:
            return [frames]

    def decode_one(self, payload: bytes) -> list:
        packet = self._make_packet(payload)
        frames = self.decoder.Decode(packet)
        self.packet_index += 1
        if hasattr(self.decoder, "SyncOnCUStream"):
            self.decoder.SyncOnCUStream()
        return self._frames_as_list(frames)

    def _decoded_frame_to_numpy(self, decoded_frame) -> np.ndarray:
        """Copy a DecodedFrame to a CPU ndarray without object-array fallbacks."""
        errors: list[str] = []

        arr = self._numeric_numpy_array(decoded_frame)
        if arr is not None:
            return arr

        # Host-memory DecodedFrame path.  NumPy can import CPU DLPack, but it
        # cannot import CUDA DLPack directly.  Device type 1 is kDLCPU.
        try:
            if hasattr(np, "from_dlpack") and hasattr(decoded_frame, "__dlpack_device__"):
                device_type, _device_id = decoded_frame.__dlpack_device__()
                if int(device_type) == 1:
                    arr = np.from_dlpack(decoded_frame)
                    arr = self._numeric_numpy_array(arr)
                    if arr is not None:
                        return arr
        except Exception as exc:
            errors.append(f"numpy/DLPack: {exc}")

        # GPU-memory DecodedFrame path.  This requires a CUDA-enabled PyTorch
        # install.  NVIDIA's docs show torch.from_dlpack(frame) for this case.
        try:
            import torch

            tensor = torch.from_dlpack(decoded_frame)
            tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            arr = tensor.numpy()
            arr = self._numeric_numpy_array(arr)
            if arr is not None:
                return arr
            errors.append("[torch/DLPack] returned a non-numeric array, your torch has no CUDA support?")
        except Exception as exc:
            errors.append(f"[torch/DLPack]: {exc}")

        # GPU-memory fallback through CUDA Array Interface.  This requires CuPy.
        try:
            import cupy as cp

            views = decoded_frame.cuda()
            if isinstance(views, (list, tuple)):
                if not views:
                    raise ValueError("decoded_frame.cuda() returned no planes")
                view = views[0]
            else:
                view = views
            arr = cp.asarray(view).get()
            arr = self._numeric_numpy_array(arr)
            if arr is not None:
                return arr
            errors.append("[cupy/cuda] returned a non-numeric array")
        except Exception as exc:
            errors.append(f"[cupy/cuda]: {exc}")

        # Some wrapper builds expose get(); accept it only when it really returns
        # numeric image memory.  Do not feed DecodedFrame itself to np.asarray(),
        # because that creates an object array and later np.clip fails with:
        # "'>=' not supported between instances of '_PyNvVideoCodec.DecodedFrame' and 'int'".
        try:
            get = getattr(decoded_frame, "get", None)
            if callable(get):
                value = get()
                arr = self._numeric_numpy_array(value)
                if arr is not None:
                    return arr
                errors.append(f"get() returned {type(value).__name__}, not a numeric array")
        except Exception as exc:
            errors.append(f"get(): {exc}")

        raise RuntimeError("Could not copy DecodedFrame to CPU: " + "; ".join(errors))

    def frame_to_bgr(self, decoded_frame) -> np.ndarray:
        """Copy one PyNvVideoCodec DecodedFrame to a CPU BGR image for cv2.imshow."""
        arr = self._decoded_frame_to_numpy(decoded_frame)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # RGBP/CHW -> HWC.
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr[:, : self.height, : self.width]
            arr = np.transpose(arr, (1, 2, 0))

        # Native NV12 fallback: shape is usually (height * 3 // 2, width).
        if arr.ndim == 2:
            if arr.shape[0] >= self.height * 3 // 2 and arr.shape[1] >= self.width:
                nv12 = np.ascontiguousarray(arr[: self.height * 3 // 2, : self.width])
                return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            raise ValueError(f"Unsupported decoded 2D frame shape: {arr.shape}")

        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"Unsupported decoded frame shape: {arr.shape}")

        # Crop possible pitch padding and extra channels.  The decoder output is
        # requested as RGB; OpenCV display expects BGR.
        rgb = np.ascontiguousarray(arr[: self.height, : self.width, :3])
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_frame(i: int, h: int = 64, w: int = 96) -> np.ndarray:
    """Make one deterministic BGR test frame with shape HWC."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frame[:, :, 1] = int(i * 20) % 255
    frame[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    return frame


def payload_as_array(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.uint8).copy()

def packet_payload_bytes(pkt: "Model4Mat.EncodedImageMat") -> bytes:
    valid_nbytes = int(getattr(pkt, "valid_nbytes", 0))

    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        if isinstance(payload, np.ndarray):
            b = payload.tobytes()
        else:
            b = bytes(payload)
        return b[:valid_nbytes] if valid_nbytes > 0 else b

    valid_nbytes = valid_nbytes or len(pkt.data)
    return np.asarray(pkt.data).reshape(-1)[:valid_nbytes].tobytes()

def packet_nbytes(pkt: "Model4Mat.EncodedImageMat") -> int:
    if hasattr(pkt, "nbytes") and callable(pkt.nbytes):
        return int(pkt.nbytes())
    return int(getattr(pkt, "valid_nbytes", len(pkt.data)))


def initial_pubsub_buffer(payload: bytes, config: StreamConfig) -> np.ndarray:
    """
    Create the initial buffer used to size the pub/sub service.

    Your working version used payload * 2. This keeps that idea, but makes the
    capacity explicit and easier to tune with --capacity-scale and
    --min-capacity-bytes.
    """
    payload_arr = payload_as_array(payload)
    capacity = max(
        int(np.ceil(payload_arr.size * max(config.capacity_scale, 1.0))),
        int(config.min_capacity_bytes),
    )
    data = np.zeros((capacity,), dtype=np.uint8)
    data[: payload_arr.size] = payload_arr
    return data


def update_packet_metadata(
    pkt: "Model4Mat.EncodedImageMatPubSub",
    *,
    frame_index: int,
    pts_ns: int,
    width: int,
    height: int,
    valid_nbytes: int,
) -> None:
    pkt.codec = CodecFormat.H264
    pkt.frame_index = int(frame_index)
    pkt.pts_ns = int(pts_ns)
    pkt.dts_ns = int(pts_ns)
    pkt.is_keyframe = True
    pkt.width = int(width)
    pkt.height = int(height)
    pkt.valid_nbytes = int(valid_nbytes)


def make_seed_packet(
    codec: H264AnnexBCodec,
    config: StreamConfig,
) -> "Model4Mat.EncodedImageMatPubSub":
    """Create an initialized packet model with enough buffer capacity."""
    frame = make_frame(0, h=config.height, w=config.width)
    payload = codec.encode_access_unit(frame, color_format=config.color_format)
    data = initial_pubsub_buffer(payload, config)

    return Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.H264,
        color_format=config.color_format,
        frame_index=0,
        pts_ns=0,
        dts_ns=0,
        is_keyframe=True,
        width=int(config.width),
        height=int(config.height),
        valid_nbytes=len(payload),
        data=data,
    )


def make_endpoint(
    codec: H264AnnexBCodec,
    config: StreamConfig,
    *,
    is_pub: bool,
) -> "Model4Mat.EncodedImageMatPubSub":
    endpoint = make_seed_packet(codec, config)
    endpoint.set_id(config.topic).init()
    endpoint.is_pub = bool(is_pub)
    return endpoint


def publish_frame(
    publisher: "Model4Mat.EncodedImageMatPubSub",
    codec: H264AnnexBCodec,
    config: StreamConfig,
    frame_index: int,
) -> int:
    frame = make_frame(frame_index, h=config.height, w=config.width)
    payload = codec.encode_access_unit(frame, color_format=config.color_format)
    pts_ns = frame_index * config.frame_period_ns

    update_packet_metadata(
        publisher,
        frame_index=frame_index,
        pts_ns=pts_ns,
        width=config.width,
        height=config.height,
        valid_nbytes=len(payload),
    )

    # Keep the actual published unit as one encoded access-unit payload.
    # This matches your working code and your streaming model.
    publisher.pub(data=payload_as_array(payload))
    return len(payload)


def decode_packet_to_image_mat(
    pkt: "Model4Mat.EncodedImageMat",
    codec: H264AnnexBCodec,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> "Model4Mat.ImageMat":
    if CodecFormat(pkt.codec) != CodecFormat.H264:
        raise ValueError(f"Expected H264 packet, got {pkt.codec}")

    frame = codec.decode_access_unit(
        packet_payload_bytes(pkt),
        width=int(pkt.width),
        height=int(pkt.height),
        color_format=color_format,
    )

    return Model4Mat.ImageMat(
        color_format=ColorFormat(color_format),
        data=frame,
    )


def publisher_loop(config: StreamConfig) -> None:
    runner = FFmpegRunner(config.ffmpeg_bin)
    codec = H264AnnexBCodec(runner, fps=config.fps, crf=config.crf)
    publisher = make_endpoint(codec, config, is_pub=True)
    meter = FpsMeter("Pub", stats_every=config.stats_every)

    print(
        f"Publishing H264 packets to topic={config.topic!r}, "
        f"size={config.width}x{config.height}, fps={config.fps}, crf={config.crf}"
    )

    frame_index = 0
    while config.max_frames is None or frame_index < config.max_frames:
        frame_index += 1
        publish_frame(publisher, codec, config, frame_index)
        meter.tick()




def make_cuda_image_endpoint(
    *,
    width: int,
    height: int,
    color_format: Union[str, ColorFormat],
    topic_id: str,
    num_slots: int,
):
    """Create an ImageMatCUDAPubSub endpoint for the OpenGL/PBO viewer."""
    try:
        import pycuda.gpuarray as gpuarray
    except ImportError as exc:
        raise RuntimeError("pycuda is required for the GL viewer path") from exc

    try:
        from resultkit.MatModel import ImageShapeType
        from resultkit.mat import DataType, MatDevice
    except ImportError as exc:
        raise RuntimeError(
            "ImageMatCUDAPubSub requires resultkit.MatModel.ImageShapeType "
            "and resultkit.mat.DataType/MatDevice"
        ) from exc

    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError(
            "Model4Mat.ImageMatCUDAPubSub is not available. "
            "Use the PyCUDA patched MatModel.py that defines ImageMatCUDAPubSub."
        )

    data = gpuarray.empty((int(height), int(width), 3), dtype=np.uint8)
    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat(color_format),
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=MatDevice.CUDA0,
        data=data,
        num_slots=int(num_slots),
    )
    img.set_id(topic_id).init()
    return img


def publish_bgr_to_cuda_image(image_pub, bgr: np.ndarray) -> None:
    """Publish one HWC/BGR uint8 CPU frame into an ImageMatCUDAPubSub CUDA slot."""
    frame = np.asarray(bgr)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HWC/BGR uint8 frame, got shape={frame.shape}, dtype={frame.dtype}")
    frame = np.ascontiguousarray(frame)

    def upload(slot):
        if tuple(slot.shape) != tuple(frame.shape):
            raise ValueError(f"CUDA slot shape {slot.shape} != decoded frame shape {frame.shape}")
        # PyCUDA GPUArray.set() performs the host-to-device upload in the active context.
        slot.set(frame)
        return slot

    image_pub.pub(edit_func=upload)


class ForeignGpuPointer(cuda.PointerHolderBase):
    """PyCUDA pointer wrapper for memory owned by another library."""
    def __init__(self, ptr: int, owner=None):
        super().__init__()
        self.ptr = int(ptr)
        self.owner = owner   # keep DecodedFrame / CAI view alive

    def get_pointer(self):
        return self.ptr


def gpuarray_from_cuda_array_interface(cai_obj, owner=None):
    """
    Zero-copy PyCUDA GPUArray view from an object exposing
    __cuda_array_interface__.
    """
    iface = cai_obj.__cuda_array_interface__

    ptr = int(iface["data"][0])
    shape = tuple(int(x) for x in iface["shape"])
    dtype = np.dtype(iface["typestr"])

    strides = iface.get("strides", None)
    if strides is not None:
        strides = tuple(int(x) for x in strides)

    holder = ForeignGpuPointer(ptr, owner=owner)

    return gpuarray.GPUArray(
        shape=shape,
        dtype=dtype,
        gpudata=holder,
        strides=strides,
    )


def decoded_frame_to_gpuarray(frame, plane: int = 0):
    """
    Convert PyNvVideoCodec DecodedFrame -> PyCUDA GPUArray view.

    For RGB output: usually one HWC plane.
    For RGBP output: may be CHW or separate planes depending on version.
    For NV12/native output: plane 0 is Y, plane 1 is UV.
    """
    views = frame.cuda()
    view = views[plane]
    return gpuarray_from_cuda_array_interface(view, owner=(frame, view))
class H264ToCudaImageBridge(threading.Thread):
    """Background bridge: EncodedImageMatPubSub -> decoded BGR -> ImageMatCUDAPubSub."""

    def __init__(self, config: StreamConfig, stop_event: threading.Event):
        super().__init__(name="h264-to-cuda-image-bridge", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self._run_impl()
        except BaseException as exc:  # keep the viewer process from silently hanging
            self.error = exc
            self.stop_event.set()
            print(f"GL decode bridge stopped with error: {exc}", file=sys.stderr, flush=True)

    def _run_impl(self) -> None:
        try:
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("pycuda is required for the GL viewer path") from exc

        config = self.config
        cuda.init()
        ctx = cuda.Device(config.device).make_context()
        image_pub = None
        try:
            runner = FFmpegRunner(config.ffmpeg_bin)
            codec = H264AnnexBCodec(runner, fps=config.fps, crf=config.crf)
            encoded_sub = make_endpoint(codec, config, is_pub=False)
            image_pub = make_cuda_image_endpoint(
                width=config.width,
                height=config.height,
                color_format=ColorFormat.RGB,
                topic_id=config.image_topic,
                num_slots=config.num_slots,
            )
            meter = FpsMeter("Sub/GL", stats_every=config.stats_every)

            # Keep host output here on purpose: it still uses NVDEC, but it gives
            # a CPU ndarray that can be uploaded into the ImageMatCUDAPubSub slot.
            # This avoids fragile direct pointer conversion between PyNvVideoCodec
            # DecodedFrame and PyCUDA GPUArray across PyNvVideoCodec versions.
            cuda_dec = PyNvH264Decoder(
                device=config.device,
                width=config.width,
                height=config.height,
                use_device_memory=False,
            )

            print(
                f"Subscribing H264 packets from topic={config.topic!r}; "
                f"publishing decoded CUDA frames to topic={config.image_topic!r}",
                flush=True,
            )

            frame_count = 0
            while not self.stop_event.is_set() and (
                config.max_frames is None or frame_count < config.max_frames
            ):
                pkt = encoded_sub.sub()

                if packet_nbytes(pkt) <= 0:
                    time.sleep(config.poll_sleep_s)
                    continue

                payload = packet_payload_bytes(pkt)
                imgs = cuda_dec.decode_one(payload)
                
                image_pub.pub(data=decoded_frame_to_gpuarray(imgs[-1]))

                # torch also works well
                # imgs = [torch.from_dlpack(decoded_frame) for decoded_frame in imgs]
                # print(imgs[-1].sum())

                frame_count += 1
                meter.tick()

            self.stop_event.set()
        finally:
            if image_pub is not None:
                try:
                    image_pub.close()
                except Exception:
                    pass
            ctx.pop()
            ctx.detach()


def subscriber_gl_loop(config: StreamConfig) -> None:
    """Run the OpenGL/PBO viewer from resultkit.cudavis."""
    try:
        from resultkit.cudavis import ImageMatCudaGlViewer
    except ImportError as exc:
        raise RuntimeError("resultkit.cudavis.ImageMatCudaGlViewer is required") from exc

    stop_event = threading.Event()
    bridge = H264ToCudaImageBridge(config, stop_event)
    bridge.start()

    viewer_img = None
    try:
        vis = ImageMatCudaGlViewer(
            width=int(config.width),
            height=int(config.height),
            fps=float(config.fps),
            device=int(config.device),
            flip_y=bool(config.flip_y),
            max_frames=config.max_frames,
        )
        vis.init()
        viewer_img = make_cuda_image_endpoint(
            width=config.width,
            height=config.height,
            color_format=ColorFormat.BGR,
            topic_id=config.image_topic,
            num_slots=config.num_slots,
        )
        vis.run(img=viewer_img)
    finally:
        stop_event.set()
        if viewer_img is not None:
            try:
                viewer_img.close()
            except Exception:
                pass
        bridge.join(timeout=2.0)
        if bridge.error is not None:
            raise RuntimeError("GL decode bridge failed") from bridge.error

def subscriber_loop(config: StreamConfig) -> None:
    if config.display and config.viewer == "gl":
        subscriber_gl_loop(config)
        return

    runner = FFmpegRunner(config.ffmpeg_bin)
    codec = H264AnnexBCodec(runner, fps=config.fps, crf=config.crf)
    subscriber = make_endpoint(codec, config, is_pub=False)
    meter = FpsMeter("Sub", stats_every=config.stats_every)
    cuda_dec = PyNvH264Decoder(device=config.device, width=config.width, height=config.height, use_device_memory=False)

    print(f"Subscribing H264 packets from topic={config.topic!r}")

    frame_count = 0
    try:
        while config.max_frames is None or frame_count < config.max_frames:
            pkt = subscriber.sub()

            if packet_nbytes(pkt) <= 0:
                time.sleep(config.poll_sleep_s)
                continue
                
            payload = packet_payload_bytes(pkt)
            imgs = cuda_dec.decode_one(payload)
            print(imgs)
            if not imgs:
                continue

            frame_count += 1
            meter.tick()

            if config.display:
                try:
                    bgr = cuda_dec.frame_to_bgr(imgs[-1])
                except Exception as exc:
                    # Keep the demo visible even if the installed PyNvVideoCodec
                    # frame object cannot be copied through DLPack on this host.
                    print(f"CUDA frame copy failed, falling back to FFmpeg decode: {exc}")
                    bgr = decode_packet_to_image_mat(
                        pkt, codec, color_format=ColorFormat.BGR
                    ).data
                cv2.imshow(config.window_name, bgr)
                if cv2.waitKey(config.cv_wait_ms) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(config.poll_sleep_s)
    finally:
        if config.display:
            cv2.destroyAllWindows()


def parse_color_format(value: str) -> ColorFormat:
    try:
        return ColorFormat(value)
    except ValueError:
        allowed = ", ".join(v.value for v in ColorFormat)
        raise argparse.ArgumentTypeError(f"invalid color format {value!r}; expected one of: {allowed}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or subscribe one H264 Annex B access-unit per EncodedImageMatPubSub."
    )

    parser.add_argument(
        "role",
        nargs="?",
        choices=("pub", "sub"),
        help="Run as publisher or subscriber. Default is pub unless --sub is used.",
    )
    parser.add_argument("--sub", action="store_true", help="Legacy alias for role=sub.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ffmpeg-bin", default=DEFAULT_FFMPEG_BIN)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--color-format", type=parse_color_format, default=ColorFormat.BGR)
    parser.add_argument("--stats-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--capacity-scale", type=float, default=2.0)
    parser.add_argument("--min-capacity-bytes", type=int, default=64 * 1024)
    parser.add_argument("--wait-ms", type=int, default=None)
    parser.add_argument("--poll-sleep-s", type=float, default=0.001)
    parser.add_argument("--viewer", choices=("gl", "cv2"), default="gl", help="Display backend for subscriber mode.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id for NVDEC / GL viewer.")
    parser.add_argument("--image-topic", default="ImageMatCUDAPubSub:decoded", help="Intermediate decoded ImageMatCUDAPubSub topic used by the GL viewer.")
    parser.add_argument("--num-slots", type=int, default=3, help="Number of CUDA image pub/sub slots for the GL viewer.")
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true", default=True)
    display_group.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--window-name", default="resultkit h264 pub/sub")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> tuple[str, StreamConfig]:
    role = "sub" if args.sub else (args.role or "pub")
    config = StreamConfig(
        topic=args.topic,
        ffmpeg_bin=args.ffmpeg_bin,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        color_format=args.color_format,
        stats_every=args.stats_every,
        max_frames=args.max_frames,
        capacity_scale=args.capacity_scale,
        min_capacity_bytes=args.min_capacity_bytes,
        display=args.display,
        window_name=args.window_name,
        wait_ms=args.wait_ms,
        poll_sleep_s=args.poll_sleep_s,
        viewer=args.viewer,
        device=args.device,
        image_topic=args.image_topic,
        num_slots=args.num_slots,
        flip_y=args.flip_y,
    )
    return role, config


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    role, config = config_from_args(args)

    try:
        if role == "sub":
            subscriber_loop(config)
        else:
            publisher_loop(config)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
