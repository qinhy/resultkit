import contextlib
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import depthai as dai

try:
    from ..logger import logger
except Exception:  # Allows this file to run standalone during local tests.
    def logger(msg: str, level: str = "info"):
        print(f"[{level}] {msg}")


@dataclass(frozen=True)
class MjpegFrame:
    """One encoded JPEG/MJPEG frame from one DepthAI RGB VideoEncoder queue."""

    name: str
    data: bytes
    sequence: int
    byte_count: int
    host_time: float
    camera_index: int
    source: str
    device_timestamp: Optional[Any] = None


@dataclass(frozen=True)
class DualRGBMjpegFrame:
    """One host-side sample containing encoded RGB JPEG frames from two cameras."""

    camera0: MjpegFrame
    camera1: MjpegFrame
    frame_index: int
    host_time: float

    @property
    def rgb0(self) -> MjpegFrame:
        """Alias for the first camera RGB frame."""
        return self.camera0

    @property
    def rgb1(self) -> MjpegFrame:
        """Alias for the second camera RGB frame."""
        return self.camera1

    def as_dict(self) -> Dict[str, bytes]:
        """Return just the encoded JPEG bytes for simple network/HTTP use."""
        return {
            "camera0": self.camera0.data,
            "camera1": self.camera1.data,
            "rgb0": self.camera0.data,
            "rgb1": self.camera1.data,
        }


class _MjpegStreamRuntime:
    """Host-side state for one encoded MJPEG DepthAI stream."""

    def __init__(self, name: str):
        self.name = name
        self.depthai_q = None
        self.packet_count = 0
        self.byte_count = 0
        self.last_log_byte_count = 0
        self.dropped_count = 0
        self.latest_at = 0.0


class _DepthAIPoeRGBMjpegCapture:
    """
    RGB-only MJPEG capture for one DepthAI/OAK device.

    Device side:
        CAM_A RGB -> requested frame type -> VideoEncoder MJPEG

    Host side:
        No JPEG decode, no cv2, no numpy, no torch, no image tensor packing.
        The output is encoded JPEG bytes that can be forwarded directly as MJPEG.
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.stop_event = False

        self.rgb = _MjpegStreamRuntime("rgb")

        self.frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_frame_count = 0

        self._released = False
        self._exit_stack = contextlib.ExitStack()

        self._start()
        self._reset_stats_clock()

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
        aliases = {"RGB": "CAM_A"}
        alias = aliases.get(socket_name)
        if alias and hasattr(dai.CameraBoardSocket, alias):
            return getattr(dai.CameraBoardSocket, alias)
        raise ValueError(f"Unsupported RGB camera socket: {socket_name}")

    @staticmethod
    def _depthai_mjpeg_profile():
        profile = dai.VideoEncoderProperties.Profile
        for name in ("MJPEG", "Mjpeg", "JPEG", "Jpeg"):
            if hasattr(profile, name):
                return getattr(profile, name)
        raise RuntimeError("This DepthAI build does not expose VideoEncoderProperties.Profile.MJPEG")

    @staticmethod
    def _img_frame_type(type_name: str):
        if hasattr(dai.ImgFrame.Type, type_name):
            return getattr(dai.ImgFrame.Type, type_name)
        raise ValueError(f"Unsupported DepthAI ImgFrame.Type: {type_name}")

    @staticmethod
    def _resize_mode(mode_name: str):
        if hasattr(dai.ImgResizeMode, mode_name):
            return getattr(dai.ImgResizeMode, mode_name)
        raise ValueError(f"Unsupported DepthAI ImgResizeMode: {mode_name}")

    @staticmethod
    def _packet_to_bytes(packet) -> bytes:
        data = packet.getData()
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, memoryview):
            return data.tobytes()
        return bytes(data)

    @staticmethod
    def _packet_timestamp(packet):
        if hasattr(packet, "getTimestamp"):
            try:
                return packet.getTimestamp()
            except Exception:
                return None
        return None

    def _create_mjpeg_encoder(self, pipeline, source_output, *, frame_rate: float, quality: int):
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            source_output,
            frameRate=frame_rate,
            profile=self._depthai_mjpeg_profile(),
        )

        quality = int(max(1, min(100, quality)))
        for setter_name in ("setQuality", "setLossless"):
            if hasattr(encoder, setter_name):
                try:
                    if setter_name == "setQuality":
                        getattr(encoder, setter_name)(quality)
                    else:
                        getattr(encoder, setter_name)(quality >= 100)
                    break
                except Exception as e:
                    self._log("warning", f"could not set OAK RGB MJPEG quality with {setter_name}: {e}")

        return encoder

    def _create_depthai_pipeline(self):
        owner = self.owner

        raw_pipeline = dai.Pipeline(self.device)
        if hasattr(raw_pipeline, "__enter__"):
            pipeline = self._exit_stack.enter_context(raw_pipeline)
        else:
            pipeline = raw_pipeline

        rgb_socket = self._camera_socket(owner.rgb_camera_socket)
        rgb_cam = pipeline.create(dai.node.Camera).build(rgb_socket)

        rgb_frame = rgb_cam.requestOutput(
            (owner.rgb_width, owner.rgb_height),
            self._img_frame_type(owner.rgb_mjpeg_input_type),
            self._resize_mode(owner.rgb_resize_mode),
            owner.capture_fps,
        )

        rgb_encoder = self._create_mjpeg_encoder(
            pipeline,
            rgb_frame,
            frame_rate=owner.capture_fps,
            quality=owner.rgb_mjpeg_quality,
        )

        self.rgb.depthai_q = rgb_encoder.out.createOutputQueue(
            maxSize=owner.rgb_depthai_queue_size,
            blocking=owner.depthai_queue_blocking,
        )

        pipeline.start()
        return pipeline

    def _log(self, path: str, msg: str):
        level = "info"
        if path in ["error", "warning", "debug", "info"]:
            level = path
        logger(f"[{self.owner.uuid}:cam{self.idx}:{path}] {msg}", level=level)

    def _start(self):
        owner = self.owner

        self._log("info", "Opening DepthAI RGB device...")
        self.device = self._open_device()

        self._log("info", "Connected DepthAI RGB device:")
        self._log("status", f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        self._log("status", f"  Cameras: {self.device.getConnectedCameras()}")

        self._log("info", "Starting DepthAI RGB MJPEG streaming pipeline:")
        self._log("status", f"  Source: {self.source}")
        self._log("status", f"  RGB socket: {owner.rgb_camera_socket}")
        self._log("status", f"  RGB size: {owner.rgb_width}x{owner.rgb_height}")
        self._log("status", f"  Capture FPS: {owner.capture_fps}")
        self._log("status", "  OAK encoder: MJPEG")
        self._log("status", f"  RGB MJPEG quality: {owner.rgb_mjpeg_quality}")
        self._log("status", f"  RGB MJPEG input type: {owner.rgb_mjpeg_input_type}")
        self._log("status", "  Host output: encoded JPEG bytes only")

        self.pipeline = self._create_depthai_pipeline()

        self._log("info", "DepthAI RGB MJPEG streaming pipeline ready.")

    def _reset_stats_clock(self):
        """Start FPS/bitrate timing after the pipeline is ready, not during device open."""
        now = time.monotonic()
        self.started_at = now
        self.last_log_at = now
        self.last_frame_count = self.frame_count
        self.rgb.last_log_byte_count = self.rgb.byte_count

    def _read_packet(self, stream: _MjpegStreamRuntime):
        if self._released or self.stop_event:
            raise StopIteration

        try:
            if hasattr(stream.depthai_q, "get"):
                return stream.depthai_q.get()
            raise RuntimeError(f"DepthAI queue for stream {stream.name!r} has no get() method")
        except Exception:
            if self._released or self.stop_event:
                raise StopIteration
            raise

    def _drain_to_latest_packet(self, stream: _MjpegStreamRuntime, packet, *, max_packets: int = 128):
        """
        Drop already-queued packets and return the newest available packet.

        This is important for live viewing. When camera0 is opened before camera1,
        camera0 can already have queued old frames before the paired loop starts.
        Reading only one packet with get() returns the oldest queued packet and
        creates visible lag. Draining makes the next delivered frame close to live.
        """
        q = stream.depthai_q
        dropped = 0
        if q is None:
            return packet, dropped

        try:
            if hasattr(q, "tryGetAll"):
                packets = q.tryGetAll()
                if packets:
                    dropped = len(packets)
                    packet = packets[-1]
                return packet, dropped
        except Exception:
            pass

        if not hasattr(q, "tryGet"):
            return packet, dropped

        for _ in range(max(0, int(max_packets))):
            try:
                next_packet = q.tryGet()
            except Exception:
                break
            if next_packet is None:
                break
            packet = next_packet
            dropped += 1
        return packet, dropped

    def read_frame(self, *, drain: Optional[bool] = None) -> MjpegFrame:
        """Read one encoded RGB JPEG frame from this camera."""
        if self._released:
            raise StopIteration

        if drain is None:
            drain = bool(getattr(self.owner, "drain_depthai_queue", True))

        stream = self.rgb
        packet = self._read_packet(stream)
        if drain:
            packet, dropped = self._drain_to_latest_packet(
                stream,
                packet,
                max_packets=getattr(self.owner, "max_depthai_drain_packets", 128),
            )
        else:
            dropped = 0

        encoded = self._packet_to_bytes(packet)

        stream.packet_count += 1 + int(dropped)
        stream.dropped_count += int(dropped)
        stream.byte_count += len(encoded)
        stream.latest_at = time.monotonic()
        self.frame_count += 1

        frame = MjpegFrame(
            name=f"camera{self.idx}",
            data=encoded,
            sequence=stream.packet_count,
            byte_count=len(encoded),
            host_time=stream.latest_at,
            camera_index=self.idx,
            source=str(self.source),
            device_timestamp=self._packet_timestamp(packet),
        )

        now = time.monotonic()
        if self.owner.log_camera_fps and now - self.last_log_at >= 1.0:
            dt = max(now - self.last_log_at, 1e-6)
            fps = (self.frame_count - self.last_frame_count) / dt
            delta_bytes = self.rgb.byte_count - self.rgb.last_log_byte_count
            rgb_mbps = delta_bytes * 8.0 / dt / 1_000_000
            self._log(
                "info",
                f"frames={self.frame_count}, fps={fps:.2f}, "
                f"packets={self.rgb.packet_count}, dropped={self.rgb.dropped_count}, mbps={rgb_mbps:.1f}",
            )
            self.last_log_at = now
            self.last_frame_count = self.frame_count
            self.rgb.last_log_byte_count = self.rgb.byte_count

        return frame

    def next_frame(self) -> MjpegFrame:
        """Compatibility helper for single-camera use."""
        try:
            frame = self.read_frame()
            self.owner.on_mjpeg_frame(frame, self.idx, frame.sequence)
            return frame
        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event:
                raise StopIteration
            raise

    def iter_mjpeg_parts(self, *, boundary: str = "frame") -> Iterator[bytes]:
        """
        Yield multipart/x-mixed-replace MJPEG chunks for this RGB camera.

        Example HTTP header:
            Content-Type: multipart/x-mixed-replace; boundary=frame
        """
        boundary = boundary[2:] if boundary.startswith("--") else boundary
        while True:
            frame = self.read_frame()
            yield self.to_mjpeg_part(frame.data, boundary=boundary)

    @staticmethod
    def to_mjpeg_part(encoded_jpeg: bytes, *, boundary: str = "frame") -> bytes:
        """Wrap encoded JPEG bytes as one multipart MJPEG HTTP chunk."""
        boundary = boundary[2:] if boundary.startswith("--") else boundary
        header = (
            f"--{boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(encoded_jpeg)}\r\n\r\n"
        ).encode("ascii")
        return header + encoded_jpeg + b"\r\n"

    def release(self):
        self._log("debug", "release() called")

        if self._released:
            self._log("debug", "release() skipped: already released")
            return

        self._log("info", "Starting DepthAI RGB MJPEG resource release")
        self._released = True
        self.stop_event = True

        try:
            if self.rgb.depthai_q is not None and hasattr(self.rgb.depthai_q, "close"):
                self._log("debug", "Closing DepthAI RGB queue")
                self.rgb.depthai_q.close()
        except Exception as e:
            self._log("warning", f"Error closing DepthAI RGB queue: {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self._log("info", "Stopping DepthAI RGB pipeline")
                    self.pipeline.stop()
                    self._log("info", "DepthAI RGB pipeline stopped")
        except Exception as e:
            self._log("warning", f"DepthAI RGB pipeline stop failed during release: {e}")

        try:
            self._exit_stack.close()
        except Exception as e:
            self._log("warning", f"Error closing exit stack during release: {e}")

        self.rgb.depthai_q = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self._log("info", "Closing DepthAI RGB device")
                self.device.close()
                self._log("info", "DepthAI RGB device closed")
        except Exception as e:
            self._log("warning", f"Error closing DepthAI RGB device during release: {e}")

        self.pipeline = None
        self.device = None
        self._log("info", "DepthAI RGB MJPEG resource release completed")


class _DepthAIPoeDualRGBMjpegCapture:
    """
    Host-side wrapper that reads live RGB frames from two devices.

    By default this uses one reader thread per camera. Each reader continuously
    drains its DepthAI output queue and stores only the latest frame. The paired
    next_frame() call then publishes the newest camera0/camera1 pair instead of
    walking through old queued frames.
    """

    def __init__(self, owner, sources: Sequence[str]):
        self.owner = owner
        self.sources = list(sources[:2])
        self.captures: List[_DepthAIPoeRGBMjpegCapture] = []
        self.frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_frame_count = 0
        self.last_cam0_byte_count = 0
        self.last_cam1_byte_count = 0
        self.stop_event = False
        self._released = False

        self._latest: List[Optional[MjpegFrame]] = [None, None]
        self._last_emitted_sequences: List[int] = [0, 0]
        self._reader_errors: List[Optional[BaseException]] = [None, None]
        self._reader_threads: List[threading.Thread] = []
        self._cond = threading.Condition()

        if len(self.sources) != 2:
            raise ValueError("Dual RGB MJPEG capture requires exactly two sources")

        try:
            for idx, source in enumerate(self.sources):
                self.captures.append(_DepthAIPoeRGBMjpegCapture(owner=owner, source=source, idx=idx))
            self._reset_stats_clock()
            if bool(getattr(self.owner, "use_reader_threads", True)):
                self._start_reader_threads()
        except Exception:
            self.release()
            raise

    def _reset_stats_clock(self):
        """Start paired FPS/bitrate timing after both devices are ready."""
        now = time.monotonic()
        self.started_at = now
        self.last_log_at = now
        self.last_frame_count = self.frame_count
        if len(self.captures) >= 2:
            self.last_cam0_byte_count = self.captures[0].rgb.byte_count
            self.last_cam1_byte_count = self.captures[1].rgb.byte_count

    def _log(self, path: str, msg: str):
        level = "info"
        if path in ["error", "warning", "debug", "info"]:
            level = path
        logger(f"[{self.owner.uuid}:dual:{path}] {msg}", level=level)

    def _start_reader_threads(self) -> None:
        for idx in range(2):
            thread = threading.Thread(
                target=self._reader_loop,
                args=(idx,),
                name=f"{self.owner.uuid}-dual-rgb-reader-{idx}",
                daemon=True,
            )
            self._reader_threads.append(thread)
            thread.start()

    def _reader_loop(self, idx: int) -> None:
        capture = self.captures[idx]
        try:
            while not self._released and not self.stop_event and not capture.stop_event:
                frame = capture.read_frame(drain=bool(getattr(self.owner, "drain_depthai_queue", True)))
                self.owner.on_mjpeg_frame(frame, idx, frame.sequence)
                with self._cond:
                    self._latest[idx] = frame
                    self._reader_errors[idx] = None
                    self._cond.notify_all()
        except StopIteration:
            pass
        except BaseException as exc:
            with self._cond:
                self._reader_errors[idx] = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _raise_reader_error_if_any(self) -> None:
        for idx, exc in enumerate(self._reader_errors):
            if exc is not None:
                raise RuntimeError(f"Dual RGB camera reader {idx} failed") from exc

    def _next_frame_threaded(self) -> DualRGBMjpegFrame:
        timeout_s = float(getattr(self.owner, "pair_wait_timeout_s", 2.0))
        allow_reuse = bool(getattr(self.owner, "allow_reuse_on_timeout", False))
        deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s

        with self._cond:
            while True:
                if self._released or self.stop_event:
                    raise StopIteration
                self._raise_reader_error_if_any()

                cam0 = self._latest[0]
                cam1 = self._latest[1]
                if cam0 is not None and cam1 is not None:
                    both_new = (
                        cam0.sequence != self._last_emitted_sequences[0]
                        and cam1.sequence != self._last_emitted_sequences[1]
                    )
                    if both_new or (allow_reuse and self.frame_count > 0):
                        self._last_emitted_sequences = [cam0.sequence, cam1.sequence]
                        break

                if deadline is None:
                    self._cond.wait(timeout=0.1)
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if allow_reuse and cam0 is not None and cam1 is not None:
                        self._last_emitted_sequences = [cam0.sequence, cam1.sequence]
                        break
                    raise TimeoutError(
                        "Timed out waiting for fresh dual RGB frames. "
                        "Check the slower camera/network, or set allow_reuse_on_timeout=True."
                    )
                self._cond.wait(timeout=min(0.1, remaining))

        return self._make_pair(cam0, cam1)

    def _next_frame_sequential(self) -> DualRGBMjpegFrame:
        cam0 = self.captures[0].read_frame(drain=bool(getattr(self.owner, "drain_depthai_queue", True)))
        cam1 = self.captures[1].read_frame(drain=bool(getattr(self.owner, "drain_depthai_queue", True)))
        return self._make_pair(cam0, cam1)

    def _make_pair(self, cam0: MjpegFrame, cam1: MjpegFrame) -> DualRGBMjpegFrame:
        self.frame_count += 1
        frame = DualRGBMjpegFrame(
            camera0=cam0,
            camera1=cam1,
            frame_index=self.frame_count,
            host_time=time.monotonic(),
        )

        self.owner.on_dual_mjpeg_frame(frame, self.frame_count)
        self._log_stats_if_needed()
        return frame

    def _log_stats_if_needed(self) -> None:
        now = time.monotonic()
        if self.owner.log_fps and now - self.last_log_at >= 1.0:
            dt = max(now - self.last_log_at, 1e-6)
            fps = (self.frame_count - self.last_frame_count) / dt
            cam0_bytes = self.captures[0].rgb.byte_count
            cam1_bytes = self.captures[1].rgb.byte_count
            cam0_mbps = (cam0_bytes - self.last_cam0_byte_count) * 8.0 / dt / 1_000_000
            cam1_mbps = (cam1_bytes - self.last_cam1_byte_count) * 8.0 / dt / 1_000_000
            cam0_dropped = self.captures[0].rgb.dropped_count
            cam1_dropped = self.captures[1].rgb.dropped_count
            self._log(
                "info",
                f"frames={self.frame_count}, fps={fps:.2f}, "
                f"packets cam0/cam1={self.captures[0].rgb.packet_count}/{self.captures[1].rgb.packet_count}, "
                f"dropped cam0/cam1={cam0_dropped}/{cam1_dropped}, "
                f"mbps cam0/cam1={cam0_mbps:.1f}/{cam1_mbps:.1f}",
            )
            self.last_log_at = now
            self.last_frame_count = self.frame_count
            self.last_cam0_byte_count = cam0_bytes
            self.last_cam1_byte_count = cam1_bytes

    def next_frame(self) -> DualRGBMjpegFrame:
        """Read one live encoded RGB sample from both cameras without decoding image data."""
        if self._released:
            raise StopIteration

        try:
            if bool(getattr(self.owner, "use_reader_threads", True)):
                return self._next_frame_threaded()
            return self._next_frame_sequential()
        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event:
                raise StopIteration
            raise

    def release(self):
        if self._released:
            return
        self._released = True
        self.stop_event = True
        with self._cond:
            self._cond.notify_all()
        for capture in list(self.captures):
            try:
                capture.release()
            except Exception as e:
                self._log("warning", f"Error releasing RGB capture: {e}")
        for thread in list(self._reader_threads):
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        self.captures.clear()
        self._reader_threads.clear()


class DepthAIPoeDualRGBMjpegGenerator:
    """
    Lightweight DepthAI PoE dual-camera RGB MJPEG generator.

    This class opens two DepthAI/OAK devices and streams only RGB MJPEG from each
    device. It does not import or use cv2, numpy, torch, TorchVision, tensor
    packing, stereo streams, or preview UI.

    Iteration yields DualRGBMjpegFrame objects:
        frame.camera0.data -> encoded JPEG bytes from first source CAM_A
        frame.camera1.data -> encoded JPEG bytes from second source CAM_A

    For PoE cameras, pass both IP addresses explicitly:
        gen = DepthAIPoeDualRGBMjpegGenerator(
            sources=["169.254.1.222", "169.254.1.223"],
            rgb_width=4032,
            rgb_height=3040,
            capture_fps=15,
        )
    """

    capture_fps: float = 15.0

    rgb_width: int = 4032
    rgb_height: int = 3040

    rgb_camera_socket: Literal["CAM_A", "RGB"] = "CAM_A"

    rgb_mjpeg_quality: int = 90
    mjpeg_quality: int = 90

    rgb_mjpeg_input_type: Literal["NV12", "BGR888p", "RGB888p", "YUV420p"] = "NV12"
    rgb_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"

    # Live-view defaults: keep host queues shallow and drop old packets.
    rgb_depthai_queue_size: int = 1
    depthai_queue_blocking: bool = False

    # Low-latency pairing options.
    use_reader_threads: bool = True
    drain_depthai_queue: bool = True
    max_depthai_drain_packets: int = 128
    pair_wait_timeout_s: float = 2.0
    allow_reuse_on_timeout: bool = False

    log_fps: bool = True
    log_camera_fps: bool = False
    fps: int = 0

    _config_fields = (
        "capture_fps",
        "rgb_width",
        "rgb_height",
        "rgb_camera_socket",
        "rgb_mjpeg_quality",
        "mjpeg_quality",
        "rgb_mjpeg_input_type",
        "rgb_resize_mode",
        "rgb_depthai_queue_size",
        "depthai_queue_blocking",
        "use_reader_threads",
        "drain_depthai_queue",
        "max_depthai_drain_packets",
        "pair_wait_timeout_s",
        "allow_reuse_on_timeout",
        "log_fps",
        "log_camera_fps",
        "fps",
    )

    def __init__(self, *args, uuid: str = "DepthAI-DualRGB-MJPEG", sources=None, **kwargs):
        if args:
            raise TypeError(
                "DepthAIPoeDualRGBMjpegGenerator accepts keyword arguments only. "
                "Pass uuid=..., sources=[cam0, cam1], rgb_width=..., etc."
            )

        for key in ("codec", "rgb_codec"):
            if key in kwargs and str(kwargs.pop(key)).lower() not in ("mjpeg", "jpeg", "jpg"):
                logger(
                    f"[DepthAIPoeDualRGBMjpegGenerator:warning] Ignoring {key}; MJPEG is always used.",
                    level="warning",
                )

        if "mjpeg_quality" in kwargs:
            kwargs.setdefault("rgb_mjpeg_quality", kwargs["mjpeg_quality"])

        self.uuid = uuid
        self.sources = list(sources) if sources is not None else ["auto", "auto"]
        self._resources = []

        for field in self._config_fields:
            setattr(self, field, kwargs.pop(field, getattr(type(self), field)))

        # Silently accept removed stereo/image/tensor options so old config files do not fail.
        removed_options = {
            "gpu_id",
            "torch_device",
            "non_blocking_gpu_copy",
            "mjpeg_decode_backend",
            "mjpeg_decode_fallback_to_opencv",
            "normalize_rgb",
            "normalize_stereo",
            "strict_rgb_shape",
            "strict_stereo_shape",
            "packed_stereo_pad_value",
            "clear_unused_payload_tail",
            "stereo_startup_timeout_sec",
            "allow_missing_stereo",
            "show_rgb_preview",
            "show_stereo_preview",
            "preview_stride",
            "rgb_window_name",
            "stereo_window_name",
            "color_types",
            "bitrate_kbps",
            "rgb_bitrate_kbps",
            "stereo_bitrate_kbps",
            "stereo_width",
            "stereo_height",
            "left_camera_socket",
            "right_camera_socket",
            "stereo_mjpeg_quality",
            "stereo_mjpeg_input_type",
            "stereo_encoder_input_type",
            "stereo_resize_mode",
            "stereo_depthai_queue_size",
        }
        for key in list(kwargs):
            if key in removed_options:
                kwargs.pop(key)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected dual RGB MJPEG generator option(s): {unknown}")

    def _resolve_source(self, source: Optional[str], idx: int) -> str:
        if source is not None:
            return source
        return self.sources[idx] if idx < len(self.sources) else "auto"

    def open(self, source: Optional[str] = None, idx: int = 0) -> _DepthAIPoeRGBMjpegCapture:
        """Open one RGB-only camera capture. Useful for single-camera MJPEG streaming."""
        capture = _DepthAIPoeRGBMjpegCapture(owner=self, source=self._resolve_source(source, idx), idx=idx)
        self._resources.append(capture)
        return capture

    def open_pair(self, sources: Optional[Sequence[str]] = None) -> _DepthAIPoeDualRGBMjpegCapture:
        """Open both RGB cameras and return a paired capture object."""
        pair_sources = list(sources) if sources is not None else list(self.sources)
        if len(pair_sources) < 2:
            raise ValueError(
                "DepthAIPoeDualRGBMjpegGenerator requires two sources, for example: "
                "sources=['169.254.1.222', '169.254.1.223']"
            )
        capture = _DepthAIPoeDualRGBMjpegCapture(owner=self, sources=pair_sources[:2])
        self._resources.append(capture)
        return capture

    def create_frame_generator(self, sources: Optional[Sequence[str]] = None):
        """Compatibility helper: yield DualRGBMjpegFrame objects forever."""
        capture = self.open_pair(sources=sources)

        def gen():
            while True:
                try:
                    yield capture.next_frame()
                except StopIteration:
                    return
                except Exception:
                    if capture.stop_event or capture._released:
                        return
                    logger(f"[{self.uuid}:warning] DepthAI dual RGB MJPEG generator failed:", level="warning")
                    traceback.print_exc()
                    raise

        return gen()

    def stream_mjpeg(
        self,
        *,
        source: Optional[str] = None,
        idx: int = 0,
        boundary: str = "frame",
    ) -> Iterator[bytes]:
        """Yield multipart MJPEG chunks for one selected RGB camera."""
        capture = self.open(source=source, idx=idx)
        try:
            yield from capture.iter_mjpeg_parts(boundary=boundary)
        finally:
            capture.release()

    def __iter__(self):
        return self.create_frame_generator()

    def release(self):
        for capture in list(self._resources):
            try:
                capture.release()
            except Exception as e:
                logger(f"[{self.uuid}:warning] Error releasing capture: {e}", level="warning")
        self._resources.clear()

    def on_mjpeg_frame(self, frame: MjpegFrame, camera_index: int, frame_index: int):
        """Hook for subclasses. Called after each single-camera RGB frame."""
        pass

    def on_dual_mjpeg_frame(self, frame: DualRGBMjpegFrame, frame_index: int):
        """Hook for subclasses. Called after each paired RGB/RGB sample."""
        pass


# Backward-friendly aliases for projects that use the old naming style.
# The output is encoded MJPEG bytes, not torch tensors.
DepthAIPoeDualRGBMjpegTorchGenerator = DepthAIPoeDualRGBMjpegGenerator
DepthAIPoeDualRGBTorchGenerator = DepthAIPoeDualRGBMjpegGenerator


def _get_gen():
    return DepthAIPoeDualRGBMjpegGenerator(
        uuid="OkadCam:DualRGB",
        sources=["169.254.1.222", "169.254.1.223"],
        rgb_width=4032,
        rgb_height=3040,
        capture_fps=15,
        rgb_mjpeg_quality=90,
        rgb_mjpeg_input_type="NV12",
        rgb_camera_socket="CAM_A",
        log_fps=True,
        log_camera_fps=False,
        fps=0,
    )


def benchmark_dual_rgb(duration_sec: float = 10.0, warmup_sec: float = 2.0):
    """Benchmark encoded MJPEG packet throughput from two RGB cameras without decoding."""
    gen = _get_gen()
    total = 0
    measured = 0
    first_sizes = None
    t0 = time.monotonic()
    measure_t0 = None

    try:
        for frame in gen:
            total += 1
            if first_sizes is None:
                first_sizes = {
                    "camera0": len(frame.camera0.data),
                    "camera1": len(frame.camera1.data),
                }
            now = time.monotonic()
            if measure_t0 is None and now - t0 >= warmup_sec:
                measure_t0 = now
                measured = 0
            if measure_t0 is not None:
                measured += 1
                if now - measure_t0 >= duration_sec:
                    break

        elapsed = max(time.monotonic() - (measure_t0 or t0), 1e-6)
        fps = measured / elapsed if measure_t0 is not None else total / max(time.monotonic() - t0, 1e-6)
        print(
            f"benchmark frames={measured if measure_t0 is not None else total}, "
            f"fps={fps:.2f}, first_jpeg_sizes={first_sizes}"
        )
    finally:
        gen.release()


if __name__ == "__main__":
    benchmark_dual_rgb()
