import contextlib
import json
from pathlib import Path
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Literal, Optional

import depthai as dai

try:
    from ..logger import logger
except Exception:  # Allows this file to run standalone during local tests.
    def logger(msg: str, level: str = "info"):
        print(f"[{level}] {msg}")


DEPTHAI_DISTORTION_COEFF_NAMES = [
    "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    "s1", "s2", "s3", "s4", "tau_x", "tau_y",
]


def _json_float_matrix(value) -> list[list[float]]:
    return [[float(x) for x in row] for row in value]


def _json_float_vector(value) -> list[float]:
    return [float(x) for x in value]


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class MjpegFrame:
    """One encoded JPEG/MJPEG frame from a DepthAI VideoEncoder queue."""

    name: str
    data: bytes
    sequence: int
    byte_count: int
    host_time: float
    device_timestamp: Optional[Any] = None


@dataclass(frozen=True)
class RGBStereoMjpegFrame:
    """One host-side sample containing encoded RGB, left, and right JPEG frames."""

    rgb: MjpegFrame
    left: MjpegFrame
    right: MjpegFrame
    frame_index: int
    host_time: float

    def as_dict(self) -> Dict[str, bytes]:
        """Return just the encoded JPEG bytes for simple network/HTTP use."""
        return {
            "rgb": self.rgb.data,
            "left": self.left.data,
            "right": self.right.data,
        }


class _MjpegStreamRuntime:
    """Host-side state for one encoded MJPEG DepthAI stream."""

    def __init__(self, name: str):
        self.name = name
        self.depthai_q = None
        self.packet_count = 0
        self.byte_count = 0
        self.latest_at = 0.0


class _DepthAIPoeRGBStereoMjpegCapture:
    """
    RGB + stereo capture using MJPEG on all three DepthAI camera streams.

    Device side only:
        CAM_A RGB   -> requested frame type -> VideoEncoder MJPEG
        CAM_B left  -> requested frame type -> VideoEncoder MJPEG
        CAM_C right -> requested frame type -> VideoEncoder MJPEG

    Host side:
        No JPEG decode, no cv2, no numpy, no torch, no image tensor packing.
        The output is encoded JPEG bytes that can be forwarded directly as MJPEG.
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner:DepthAIPoeRGBStereoMjpegGenerator = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.stop_event = False

        self.rgb = _MjpegStreamRuntime("rgb")
        self.left = _MjpegStreamRuntime("left")
        self.right = _MjpegStreamRuntime("right")

        self.frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_frame_count = 0

        self._released = False
        self._exit_stack = contextlib.ExitStack()

        self.calibration: dict[str, Any] | None = None

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

    def _create_mjpeg_encoder(self, pipeline, source_output, *, frame_rate: float, quality: int, name: str):
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
                    self._log("warning", f"could not set OAK {name} MJPEG quality with {setter_name}: {e}")

        return encoder

    def _create_depthai_pipeline(self):
        owner:DepthAIPoeRGBStereoMjpegGenerator = self.owner

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

        rgb_frame = rgb_cam.requestOutput(
            (owner.rgb_width, owner.rgb_height),
            self._img_frame_type(owner.rgb_mjpeg_input_type),
            self._resize_mode(owner.rgb_resize_mode),
            owner.capture_fps,
        )

        stereo_input_type = self._img_frame_type(owner.stereo_mjpeg_input_type)
        stereo_resize_mode = self._resize_mode(owner.stereo_resize_mode)
        left_frame = left_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            stereo_input_type,
            stereo_resize_mode,
            owner.capture_fps,
        )
        right_frame = right_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            stereo_input_type,
            stereo_resize_mode,
            owner.capture_fps,
        )

        rgb_encoder = self._create_mjpeg_encoder(
            pipeline,
            rgb_frame,
            frame_rate=owner.capture_fps,
            quality=owner.rgb_mjpeg_quality,
            name="RGB",
        )
        left_encoder = self._create_mjpeg_encoder(
            pipeline,
            left_frame,
            frame_rate=owner.capture_fps,
            quality=owner.stereo_mjpeg_quality,
            name="left",
        )
        right_encoder = self._create_mjpeg_encoder(
            pipeline,
            right_frame,
            frame_rate=owner.capture_fps,
            quality=owner.stereo_mjpeg_quality,
            name="right",
        )

        self.rgb.depthai_q = rgb_encoder.out.createOutputQueue(
            maxSize=owner.rgb_depthai_queue_size,
            blocking=owner.depthai_queue_blocking,
        )
        self.left.depthai_q = left_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=owner.depthai_queue_blocking,
        )
        self.right.depthai_q = right_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=owner.depthai_queue_blocking,
        )

        pipeline.start()
        return pipeline

    def _log(self, path: str, msg: str):
        level = "info"
        if path in ["error", "warning", "debug", "info"]:
            level = path
        logger(f"[{self.owner.uuid}:{path}] {msg}", level=level)

    def _start(self):
        owner:DepthAIPoeRGBStereoMjpegGenerator = self.owner

        self._log("info", "Opening DepthAI device...")
        self.device = self._open_device()

        self._log("info", "Connected DepthAI device:")
        self._log("status", f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        self._log("status", f"  Cameras: {self.device.getConnectedCameras()}")

        self.calibration = self.owner.calibration = self.read_calibration_dict()
        self._log("info", "DepthAI calibration loaded.")
        self._log("status", f"DepthAI calibration {self.calibration}")

        self._log("info", "Starting DepthAI RGB + stereo MJPEG streaming pipeline:")
        self._log("status", f"  Source: {self.source}")
        self._log("status", f"  RGB socket: {owner.rgb_camera_socket}")
        self._log("status", f"  Left socket: {owner.left_camera_socket}")
        self._log("status", f"  Right socket: {owner.right_camera_socket}")
        self._log("status", f"  RGB size: {owner.rgb_width}x{owner.rgb_height}")
        self._log("status", f"  Stereo size: {owner.stereo_width}x{owner.stereo_height}")
        self._log("status", f"  Capture FPS: {owner.capture_fps}")
        self._log("status", "  OAK encoder: MJPEG")
        self._log("status", f"  RGB MJPEG quality: {owner.rgb_mjpeg_quality}")
        self._log("status", f"  Stereo MJPEG quality: {owner.stereo_mjpeg_quality}")
        self._log("status", f"  RGB MJPEG input type: {owner.rgb_mjpeg_input_type}")
        self._log("status", f"  Stereo MJPEG input type: {owner.stereo_mjpeg_input_type}")
        self._log("status", "  Host output: encoded JPEG bytes only")

        self.pipeline = self._create_depthai_pipeline()

        self._log("info", "DepthAI RGB + stereo MJPEG streaming pipeline ready.")

    def _read_packet(self, stream: _MjpegStreamRuntime):
        """
        Read one packet from a DepthAI output queue.

        Low-latency behavior:
            - get() waits until at least one packet is available.
            - when drain_depthai_queue=True, tryGet() is then used to discard
              older queued packets and keep only the newest packet available now.

        This avoids returning stale frames after the host loop, publisher, or
        downstream viewer falls behind.
        """
        if self._released or self.stop_event:
            raise StopIteration

        try:
            if not hasattr(stream.depthai_q, "get"):
                raise RuntimeError(f"DepthAI queue for stream {stream.name!r} has no get() method")

            packet = stream.depthai_q.get()

            if getattr(self.owner, "drain_depthai_queue", False) and hasattr(stream.depthai_q, "tryGet"):
                while True:
                    newer = stream.depthai_q.tryGet()
                    if newer is None:
                        break
                    packet = newer

            return packet

        except Exception:
            if self._released or self.stop_event:
                raise StopIteration
            raise

    def read_stream_frame(self, stream_name: Literal["rgb", "left", "right"] = "rgb") -> MjpegFrame:
        """Read one encoded JPEG frame from a single DepthAI stream."""
        stream = getattr(self, stream_name)
        packet = self._read_packet(stream)
        encoded = self._packet_to_bytes(packet)

        stream.packet_count += 1
        stream.byte_count += len(encoded)
        stream.latest_at = time.monotonic()

        return MjpegFrame(
            name=stream.name,
            data=encoded,
            sequence=stream.packet_count,
            byte_count=len(encoded),
            host_time=stream.latest_at,
            device_timestamp=self._packet_timestamp(packet),
        )

    def next_frame(self) -> RGBStereoMjpegFrame:
        """Read one encoded RGB/left/right sample without decoding any image data."""
        if self._released:
            raise StopIteration

        try:
            rgb = self.read_stream_frame("rgb")
            left = self.read_stream_frame("left")
            right = self.read_stream_frame("right")

            self.frame_count += 1
            frame = RGBStereoMjpegFrame(
                rgb=rgb,
                left=left,
                right=right,
                frame_index=self.frame_count,
                host_time=time.monotonic(),
            )

            self.owner.on_mjpeg_frame(frame, self.frame_count)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                fps = (self.frame_count - self.last_frame_count) / dt
                elapsed = max(now - self.started_at, 1e-6)
                rgb_mbps = self.rgb.byte_count * 8.0 / elapsed / 1_000_000
                left_mbps = self.left.byte_count * 8.0 / elapsed / 1_000_000
                right_mbps = self.right.byte_count * 8.0 / elapsed / 1_000_000
                self._log(
                    "info",
                    f"frames={self.frame_count}, fps={fps:.2f}, "
                    f"packets rgb/left/right={self.rgb.packet_count}/{self.left.packet_count}/{self.right.packet_count}, "
                    f"mbps rgb/left/right={rgb_mbps:.1f}/{left_mbps:.1f}/{right_mbps:.1f}",
                )
                self.last_log_at = now
                self.last_frame_count = self.frame_count

            return frame

        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event:
                raise StopIteration
            raise

    def iter_mjpeg_parts(
        self,
        stream_name: Literal["rgb", "left", "right"] = "rgb",
        *,
        boundary: str = "frame",
    ) -> Iterator[bytes]:
        """
        Yield multipart/x-mixed-replace MJPEG chunks for one stream.

        Example HTTP header:
            Content-Type: multipart/x-mixed-replace; boundary=frame
        """
        boundary = boundary[2:] if boundary.startswith("--") else boundary
        while True:
            frame = self.read_stream_frame(stream_name)
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

        self._log("info", "Starting DepthAI MJPEG resource release")
        self._released = True
        self.stop_event = True

        for stream in (self.rgb, self.left, self.right):
            try:
                if stream.depthai_q is not None and hasattr(stream.depthai_q, "close"):
                    self._log("debug", f"Closing DepthAI queue for stream '{stream.name}'")
                    stream.depthai_q.close()
            except Exception as e:
                self._log("warning", f"Error closing DepthAI queue for stream '{stream.name}': {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self._log("info", "Stopping DepthAI pipeline")
                    self.pipeline.stop()
                    self._log("info", "DepthAI pipeline stopped")
        except Exception as e:
            self._log("warning", f"DepthAI pipeline stop failed during release: {e}")

        try:
            self._exit_stack.close()
        except Exception as e:
            self._log("warning", f"Error closing exit stack during release: {e}")

        for stream in (self.rgb, self.left, self.right):
            stream.depthai_q = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self._log("info", "Closing DepthAI device")
                self.device.close()
                self._log("info", "DepthAI device closed")
        except Exception as e:
            self._log("warning", f"Error closing DepthAI device during release: {e}")

        self.pipeline = None
        self.device = None
        self._log("info", "DepthAI MJPEG resource release completed")


    def _read_default_resolution(self, calib, socket, fallback: tuple[int, int]) -> list[int]:
        try:
            _matrix, width, height = calib.getDefaultIntrinsics(socket)
            return [int(width), int(height)]
        except Exception as e:
            self._log("warning", f"could not read default resolution for {socket}: {e}")
            return [int(fallback[0]), int(fallback[1])]

    def _read_intrinsics(self, calib, socket, width: int, height: int) -> list[list[float]]:
        try:
            return _json_float_matrix(calib.getCameraIntrinsics(socket, int(width), int(height)))
        except TypeError:
            # Some DepthAI versions also allow getCameraIntrinsics(socket)
            return _json_float_matrix(calib.getCameraIntrinsics(socket))
        except Exception as e:
            self._log("warning", f"could not read intrinsics for {socket}: {e}")
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    def _read_distortion(self, calib, socket) -> list[float]:
        try:
            return _json_float_vector(calib.getDistortionCoefficients(socket))
        except Exception as e:
            self._log("warning", f"could not read distortion coefficients for {socket}: {e}")
            return []

    def _read_extrinsics(self, calib, from_socket, to_socket) -> list[list[float]]:
        try:
            return _json_float_matrix(calib.getCameraExtrinsics(from_socket, to_socket))
        except Exception as e:
            self._log("warning", f"could not read extrinsics {from_socket} -> {to_socket}: {e}")
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

    def read_calibration_dict(self) -> dict[str, Any]:
        """
        Read DepthAI calibration from the connected device and normalize it into
        the same shape as DEFAULT_CALIBRATION.

        This does not decode frames and does not require cv2/numpy/torch.
        """
        if self.device is None:
            raise RuntimeError("DepthAI device is not open")

        owner:DepthAIPoeRGBStereoMjpegGenerator = self.owner
        calib = self.device.readCalibration()

        rgb_socket = self._camera_socket(owner.rgb_camera_socket)
        left_socket = self._camera_socket(owner.left_camera_socket)
        right_socket = self._camera_socket(owner.right_camera_socket)

        rgb_resolution = self._read_default_resolution(
            calib,
            rgb_socket,
            fallback=(owner.rgb_width, owner.rgb_height),
        )
        left_resolution = self._read_default_resolution(
            calib,
            left_socket,
            fallback=(owner.stereo_width, owner.stereo_height),
        )
        right_resolution = self._read_default_resolution(
            calib,
            right_socket,
            fallback=(owner.stereo_width, owner.stereo_height),
        )

        result: dict[str, Any] = {
            "rgb_resolution": rgb_resolution,
            "left_resolution": left_resolution,
            "right_resolution": right_resolution,

            # In practice, DepthAI stereo baseline APIs report centimeters.
            # Your sample translation values around -7.5 match an OAK stereo baseline in cm.
            "stereo_translation_units_hint": (
                "DepthAI calibration extrinsic translation values are device calibration "
                "units; Luxonis stereo baseline APIs report centimeters."
            ),

            "rgb_intrinsics": self._read_intrinsics(
                calib, rgb_socket, owner.rgb_width, owner.rgb_height
            ),
            "left_intrinsics": self._read_intrinsics(
                calib, left_socket, owner.stereo_width, owner.stereo_height
            ),
            "right_intrinsics": self._read_intrinsics(
                calib, right_socket, owner.stereo_width, owner.stereo_height
            ),

            "left_to_right_extrinsics": self._read_extrinsics(
                calib, left_socket, right_socket
            ),
            "left_to_rgb_extrinsics": self._read_extrinsics(
                calib, left_socket, rgb_socket
            ),

            "rgb_distortion": self._read_distortion(calib, rgb_socket),
            "left_distortion": self._read_distortion(calib, left_socket),
            "right_distortion": self._read_distortion(calib, right_socket),

            "distortion_coeff_order": DEPTHAI_DISTORTION_COEFF_NAMES,
        }

        # Optional helpful metadata.
        try:
            eeprom = calib.getEepromData()
            result["board_name"] = getattr(eeprom, "boardName", None)
            result["product_name"] = getattr(eeprom, "productName", None)
        except Exception:
            pass

        try:
            result["device_id"] = self.device.getDeviceInfo().getDeviceId()
        except Exception:
            pass

        try:
            if hasattr(calib, "getBaselineDistance"):
                result["stereo_baseline_cm"] = _safe_float(calib.getBaselineDistance())
        except Exception:
            pass

        try:
            if hasattr(calib, "getFov"):
                result["rgb_fov_deg"] = _safe_float(calib.getFov(rgb_socket))
                result["left_fov_deg"] = _safe_float(calib.getFov(left_socket))
                result["right_fov_deg"] = _safe_float(calib.getFov(right_socket))
        except Exception:
            pass

        return result

    def save_calibration_json(self, path: str | Path) -> None:
        calibration = self.read_calibration_dict()
        path = Path(path)
        path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    def save_raw_depthai_eeprom_json(self, path: str | Path) -> None:
        """
        Save Luxonis' raw EEPROM calibration JSON, if you also want the vendor dump.
        """
        if self.device is None:
            raise RuntimeError("DepthAI device is not open")

        calib = self.device.readCalibration()
        calib.eepromToJsonFile(str(path))

class DepthAIPoeRGBStereoMjpegGenerator:
    """
    Low-latency DepthAI PoE RGB + stereo MJPEG generator.

    This class keeps only camera setup and encoded MJPEG streaming. It does not
    import or use cv2, numpy, torch, TorchVision, tensor packing, or preview UI.

    Iteration yields RGBStereoMjpegFrame objects:
        frame.rgb.data   -> encoded JPEG bytes from CAM_A
        frame.left.data  -> encoded JPEG bytes from CAM_B
        frame.right.data -> encoded JPEG bytes from CAM_C
    """

    capture_fps: float = 15.0

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800

    rgb_camera_socket: Literal["CAM_A", "RGB"] = "CAM_A"
    left_camera_socket: Literal["CAM_B", "LEFT"] = "CAM_B"
    right_camera_socket: Literal["CAM_C", "RIGHT"] = "CAM_C"

    rgb_mjpeg_quality: int = 90
    stereo_mjpeg_quality: int = 85
    mjpeg_quality: int = 90

    rgb_mjpeg_input_type: Literal["NV12", "BGR888p", "RGB888p", "YUV420p"] = "NV12"
    stereo_mjpeg_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"
    stereo_encoder_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"

    rgb_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"
    stereo_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"

    # Low-latency defaults: keep only the newest host-side packet.
    # The old defaults were queue_size=8 and blocking=True, which can show stale
    # frames if the host, publisher, or viewer falls behind.
    rgb_depthai_queue_size: int = 1
    stereo_depthai_queue_size: int = 1
    depthai_queue_blocking: bool = False
    drain_depthai_queue: bool = True

    log_fps: bool = True
    fps: int = 0

    _config_fields = (
        "capture_fps",
        "rgb_width",
        "rgb_height",
        "stereo_width",
        "stereo_height",
        "rgb_camera_socket",
        "left_camera_socket",
        "right_camera_socket",
        "rgb_mjpeg_quality",
        "stereo_mjpeg_quality",
        "mjpeg_quality",
        "rgb_mjpeg_input_type",
        "stereo_mjpeg_input_type",
        "stereo_encoder_input_type",
        "rgb_resize_mode",
        "stereo_resize_mode",
        "rgb_depthai_queue_size",
        "stereo_depthai_queue_size",
        "depthai_queue_blocking",
        "drain_depthai_queue",
        "log_fps",
        "fps",
    )

    def __init__(self, *args, uuid: str = "DepthAI-MJPEG", sources=None, **kwargs):
        self.calibration: dict[str, Any] | None = None

        if args:
            raise TypeError(
                "DepthAIPoeRGBStereoMjpegGenerator accepts keyword arguments only. "
                "Pass uuid=..., sources=[...], rgb_width=..., etc."
            )

        for key in ("codec", "rgb_codec", "stereo_codec"):
            if key in kwargs and str(kwargs.pop(key)).lower() not in ("mjpeg", "jpeg", "jpg"):
                logger(
                    f"[DepthAIPoeRGBStereoMjpegGenerator:warning] Ignoring {key}; MJPEG is always used.",
                    level="warning",
                )

        if "mjpeg_quality" in kwargs:
            kwargs.setdefault("rgb_mjpeg_quality", kwargs["mjpeg_quality"])
            kwargs.setdefault("stereo_mjpeg_quality", kwargs["mjpeg_quality"])
        if "stereo_encoder_input_type" in kwargs and "stereo_mjpeg_input_type" not in kwargs:
            kwargs["stereo_mjpeg_input_type"] = kwargs["stereo_encoder_input_type"]

        self.uuid = uuid
        self.sources = list(sources) if sources is not None else ["auto"]
        self._resources = []

        for field in self._config_fields:
            setattr(self, field, kwargs.pop(field, getattr(type(self), field)))

        # Silently accept removed image/tensor options so old config files do not fail.
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
        }
        for key in list(kwargs):
            if key in removed_options:
                kwargs.pop(key)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MJPEG generator option(s): {unknown}")

    def open(self, source: Optional[str] = None, idx: int = 0) -> _DepthAIPoeRGBStereoMjpegCapture:
        if source is None:
            source = self.sources[idx] if idx < len(self.sources) else "auto"
        capture = _DepthAIPoeRGBStereoMjpegCapture(owner=self, source=source, idx=idx)
        self._resources.append(capture)
        return capture

    def create_frame_generator(self, idx: int = 0, source: Optional[str] = None):
        """Compatibility helper: yield RGBStereoMjpegFrame objects forever."""
        capture = self.open(source=source, idx=idx)

        def gen():
            while True:
                try:
                    yield capture.next_frame()
                except StopIteration:
                    return
                except Exception:
                    if capture.stop_event or capture._released:
                        return
                    logger(f"[{self.uuid}:warning] DepthAI MJPEG generator failed:", level="warning")
                    traceback.print_exc()
                    raise

        return gen()

    def stream_mjpeg(
        self,
        *,
        source: Optional[str] = None,
        idx: int = 0,
        stream_name: Literal["rgb", "left", "right"] = "rgb",
        boundary: str = "frame",
    ) -> Iterator[bytes]:
        """Yield multipart MJPEG chunks for one camera stream."""
        capture = self.open(source=source, idx=idx)
        try:
            yield from capture.iter_mjpeg_parts(stream_name=stream_name, boundary=boundary)
        finally:
            capture.release()

    def __iter__(self):
        return self.create_frame_generator(idx=0)

    def release(self):
        for capture in list(self._resources):
            try:
                capture.release()
            except Exception as e:
                logger(f"[{self.uuid}:warning] Error releasing capture: {e}", level="warning")
        self._resources.clear()

    def on_mjpeg_frame(self, frame: RGBStereoMjpegFrame, frame_index: int):
        """Hook for subclasses. Called after each RGB/left/right encoded sample."""
        pass


# Backward-friendly aliases for imports that used the old tensor class names.
# The output is now encoded MJPEG bytes, not torch tensors.
DepthAIPoeRGBStereoMjpegTorchGenerator = DepthAIPoeRGBStereoMjpegGenerator
DepthAIPoeRGBStereoTorchGenerator = DepthAIPoeRGBStereoMjpegGenerator


def _get_gen():
    return DepthAIPoeRGBStereoMjpegGenerator(
        uuid="OkadCam:CamA",
        sources=["169.254.1.222"],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_mjpeg_quality=90,
        stereo_mjpeg_quality=85,
        rgb_mjpeg_input_type="NV12",
        stereo_mjpeg_input_type="NV12",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        rgb_depthai_queue_size=1,
        stereo_depthai_queue_size=1,
        depthai_queue_blocking=False,
        drain_depthai_queue=True,
        fps=0,
    )


def benchmark_rgb_stereo(duration_sec: float = 10.0, warmup_sec: float = 2.0):
    """Benchmark encoded MJPEG packet throughput without decoding."""
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
                    "rgb": len(frame.rgb.data),
                    "left": len(frame.left.data),
                    "right": len(frame.right.data),
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
    benchmark_rgb_stereo()
