from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as py_queue
import struct
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common import EmptyParams, RpcModel, openapi_doc
from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat
from resultkit.dai.rgb_stereo_mjpeg_generator import DepthAIPoeRGBStereoMjpegGenerator


parser = argparse.ArgumentParser(description="Run the DepthAI MJPEG camera RPC server.")
parser.add_argument("--service-name", default="jrpc", help="iceoryx2 service name")
parser.add_argument("--controller-name", default="cameraRgbd", help="camera controller name")
parser.add_argument("--device", default="169.254.1.222", help="DepthAI device IP or identifier")
args = parser.parse_args()


NS_PER_SECOND = 1_000_000_000
PUBSUB_HEADER_BYTES = 8

BUNDLE_MAGIC = b"RSMJ"
BUNDLE_VERSION = 1
BUNDLE_FORMAT = "rgb_stereo_mjpeg_bundle_v1"

# magic[4], version:u16, header_nbytes:u16,
# frame_index:u64, pts_ns:u64,
# rgb_width:u32, rgb_height:u32, stereo_width:u32, stereo_height:u32,
# rgb_nbytes:u32, left_nbytes:u32, right_nbytes:u32
BUNDLE_HEADER = struct.Struct("<4sHHQQIIIIIII")


class CameraBaseModel(RpcModel):
    service: str = args.service_name


class CameraConfig(CameraBaseModel):
    """DepthAI camera settings plus one EncodedImageMatPubSub output topic."""

    uuid: str = f"{args.service_name}:{args.controller_name}"
    sources: list[str] = [args.device]

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800
    capture_fps: int = 15

    rgb_camera_socket: str = "CAM_A"
    left_camera_socket: str = "CAM_B"
    right_camera_socket: str = "CAM_C"

    rgb_mjpeg_quality: int = 90
    stereo_mjpeg_quality: int = 85
    rgb_mjpeg_input_type: str = "NV12"
    stereo_mjpeg_input_type: str = "NV12"
    rgb_resize_mode: str = "CROP"
    stereo_resize_mode: str = "CROP"

    rgb_depthai_queue_size: int = 1
    stereo_depthai_queue_size: int = 1
    depthai_queue_blocking: bool = False
    log_fps: bool = True

    encoded_topic: str = f"{args.service_name}:{args.controller_name}:rgb_stereo_mjpeg"
    encoded_buffer_capacity_bytes: int = 64 * 1024 * 1024

    retry_forever: bool = True
    retry_delay_s: float = 1.0
    worker_watchdog_interval_s: float = 0.5
    close_join_timeout_s: float = 5.0
    close_terminate_timeout_s: float = 2.0


class CameraStatusResult(CameraBaseModel):
    opened: bool
    worker_alive: bool
    worker_state: str | None = None
    worker_exitcode: int | None = None
    last_error: str | None = None

    device_id: str
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    fps: int

    encoded_topic: str
    encoded_payload_format: str = BUNDLE_FORMAT
    encoded_buffer_capacity_bytes: int

    last_frame_id: int | None = None
    last_frame_age_s: float | None = None
    last_encoded_nbytes: int | None = None
    last_rgb_nbytes: int | None = None
    last_left_nbytes: int | None = None
    last_right_nbytes: int | None = None

    camera_restart_count: int = 0
    process_restart_count: int = 0


@dataclass(frozen=True)
class RGBStereoMjpegBundle:
    frame_index: int
    pts_ns: int
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    rgb: bytes
    left: bytes
    right: bytes

    @property
    def total_nbytes(self) -> int:
        return BUNDLE_HEADER.size + len(self.rgb) + len(self.left) + len(self.right)


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------


def model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def put_drop_oldest(q: Any, item: Any) -> None:
    try:
        q.put_nowait(item)
        return
    except py_queue.Full:
        pass

    try:
        q.get_nowait()
    except Exception:
        pass

    try:
        q.put_nowait(item)
    except py_queue.Full:
        pass


def drain_queue(q: Any) -> None:
    while q is not None:
        try:
            q.get_nowait()
        except Exception:
            return


def int_or_keep(current: int | None, value: Any) -> int | None:
    try:
        return int(value) if value is not None else current
    except Exception:
        return current


def float_or_keep(current: float | None, value: Any) -> float | None:
    try:
        return float(value) if value is not None else current
    except Exception:
        return current


def sleep_until_stopped(stop_event: Any, seconds: float) -> None:
    end_s = time.monotonic() + max(0.0, seconds)
    while not stop_event.is_set() and time.monotonic() < end_s:
        time.sleep(min(0.1, end_s - time.monotonic()))


def as_uint8_array(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        arr = payload.reshape(-1)
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        return np.frombuffer(payload, dtype=np.uint8).copy()
    else:
        arr = np.asarray(payload).reshape(-1)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Bundle packing/unpacking
# ---------------------------------------------------------------------------


def pack_rgb_stereo_mjpeg_bundle(
    frame: Any,
    *,
    frame_index: int,
    pts_ns: int,
    rgb_width: int,
    rgb_height: int,
    stereo_width: int,
    stereo_height: int,
) -> np.ndarray:
    """Return one uint8 payload containing RGB, left, and right JPEG bytes."""

    try:
        rgb = bytes(frame.rgb.data)
        left = bytes(frame.left.data)
        right = bytes(frame.right.data)
    except AttributeError as exc:
        raise TypeError(
            "Expected RGBStereoMjpegFrame with .rgb.data, .left.data, and .right.data bytes."
        ) from exc

    if not rgb or not left or not right:
        raise ValueError("RGB, left, and right MJPEG payloads must all be non-empty.")

    header = BUNDLE_HEADER.pack(
        BUNDLE_MAGIC,
        BUNDLE_VERSION,
        BUNDLE_HEADER.size,
        int(frame_index),
        int(pts_ns),
        int(rgb_width),
        int(rgb_height),
        int(stereo_width),
        int(stereo_height),
        len(rgb),
        len(left),
        len(right),
    )

    out = np.empty(BUNDLE_HEADER.size + len(rgb) + len(left) + len(right), dtype=np.uint8)
    pos = 0
    for part in (header, rgb, left, right):
        part_arr = np.frombuffer(part, dtype=np.uint8)
        out[pos : pos + part_arr.size] = part_arr
        pos += part_arr.size
    return out


def packet_payload_bytes(pkt: Any) -> bytes:
    """Read the valid payload bytes from an EncodedImageMat/EncodedImageMatPubSub sample."""

    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        if isinstance(payload, np.ndarray):
            return payload.tobytes()
        return bytes(payload)

    data = getattr(pkt, "data", pkt)
    arr = as_uint8_array(data)
    valid_nbytes = int(getattr(pkt, "valid_nbytes", arr.size))
    valid_nbytes = max(0, min(valid_nbytes, int(arr.size)))
    return arr[:valid_nbytes].tobytes()


def unpack_rgb_stereo_mjpeg_bundle(payload_or_packet: Any) -> RGBStereoMjpegBundle:
    """Subscriber helper. The returned rgb/left/right fields are standalone JPEG bytes."""

    payload = packet_payload_bytes(payload_or_packet)
    if len(payload) < BUNDLE_HEADER.size:
        raise ValueError(f"Payload too small for {BUNDLE_FORMAT}: {len(payload)} bytes")

    (
        magic,
        version,
        header_nbytes,
        frame_index,
        pts_ns,
        rgb_width,
        rgb_height,
        stereo_width,
        stereo_height,
        rgb_nbytes,
        left_nbytes,
        right_nbytes,
    ) = BUNDLE_HEADER.unpack(payload[: BUNDLE_HEADER.size])

    if magic != BUNDLE_MAGIC:
        raise ValueError(f"Invalid RGB+stereo MJPEG bundle magic: {magic!r}")
    if version != BUNDLE_VERSION:
        raise ValueError(f"Unsupported RGB+stereo MJPEG bundle version: {version}")
    if int(header_nbytes) < BUNDLE_HEADER.size:
        raise ValueError(f"Invalid RGB+stereo MJPEG bundle header size: {header_nbytes}")

    rgb_start = int(header_nbytes)
    left_start = rgb_start + int(rgb_nbytes)
    right_start = left_start + int(left_nbytes)
    payload_end = right_start + int(right_nbytes)
    if payload_end > len(payload):
        raise ValueError(f"Truncated RGB+stereo MJPEG bundle: need {payload_end}, got {len(payload)}")

    return RGBStereoMjpegBundle(
        frame_index=int(frame_index),
        pts_ns=int(pts_ns),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        stereo_width=int(stereo_width),
        stereo_height=int(stereo_height),
        rgb=payload[rgb_start:left_start],
        left=payload[left_start:right_start],
        right=payload[right_start:payload_end],
    )


# ---------------------------------------------------------------------------
# Publisher and DepthAI worker helpers
# ---------------------------------------------------------------------------


def frame_period_ns(config: CameraConfig) -> int:
    return int(NS_PER_SECOND / max(int(config.capture_fps), 1))


def make_encoded_publisher(config: CameraConfig) -> "Model4Mat.EncodedImageMatPubSub":
    capacity = max(int(config.encoded_buffer_capacity_bytes), PUBSUB_HEADER_BYTES + BUNDLE_HEADER.size + 1)
    publisher = Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.MJPEG,
        color_format=ColorFormat.BGR,
        frame_index=0,
        pts_ns=0,
        dts_ns=0,
        is_keyframe=True,
        width=int(config.rgb_width),
        height=int(config.rgb_height),
        valid_nbytes=0,
        data=np.zeros((capacity,), dtype=np.uint8),
    )
    publisher.set_id(config.encoded_topic).init()
    publisher.is_pub = True
    publisher.valid_nbytes = 0
    return publisher


def publish_bundle(
    publisher: "Model4Mat.EncodedImageMatPubSub",
    payload: np.ndarray,
    *,
    frame_index: int,
    pts_ns: int,
    width: int,
    height: int,
) -> int:
    payload = as_uint8_array(payload)
    payload_nbytes = int(payload.size)
    required_nbytes = payload_nbytes + PUBSUB_HEADER_BYTES
    capacity_nbytes = int(publisher.data.size)

    if payload_nbytes <= 0:
        raise ValueError("Cannot publish an empty RGB+stereo MJPEG bundle.")
    if required_nbytes > capacity_nbytes:
        raise ValueError(
            "RGB+stereo MJPEG bundle exceeds EncodedImageMatPubSub capacity: "
            f"payload={payload_nbytes}, required={required_nbytes}, capacity={capacity_nbytes}. "
            "Increase encoded_buffer_capacity_bytes."
        )

    publisher.codec = CodecFormat.MJPEG
    publisher.color_format = ColorFormat.BGR
    publisher.frame_index = int(frame_index)
    publisher.pts_ns = int(pts_ns)
    publisher.dts_ns = int(pts_ns)
    publisher.is_keyframe = True
    publisher.width = int(width)
    publisher.height = int(height)
    publisher.valid_nbytes = payload_nbytes
    publisher.pub(data=payload)
    return payload_nbytes


def build_dai_generator(config: CameraConfig) -> DepthAIPoeRGBStereoMjpegGenerator:
    return DepthAIPoeRGBStereoMjpegGenerator(
        uuid=config.uuid,
        sources=config.sources,
        rgb_width=config.rgb_width,
        rgb_height=config.rgb_height,
        stereo_width=config.stereo_width,
        stereo_height=config.stereo_height,
        capture_fps=config.capture_fps,
        rgb_camera_socket=config.rgb_camera_socket,
        left_camera_socket=config.left_camera_socket,
        right_camera_socket=config.right_camera_socket,
        rgb_mjpeg_quality=config.rgb_mjpeg_quality,
        stereo_mjpeg_quality=config.stereo_mjpeg_quality,
        rgb_mjpeg_input_type=config.rgb_mjpeg_input_type,
        stereo_mjpeg_input_type=config.stereo_mjpeg_input_type,
        rgb_resize_mode=config.rgb_resize_mode,
        stereo_resize_mode=config.stereo_resize_mode,
        rgb_depthai_queue_size=config.rgb_depthai_queue_size,
        stereo_depthai_queue_size=config.stereo_depthai_queue_size,
        depthai_queue_blocking=config.depthai_queue_blocking,
        log_fps=config.log_fps,
        fps=0,
    )


def emit_status(status_queue: Any, state: str, **fields: Any) -> None:
    fields.setdefault("timestamp_s", time.time())
    fields.setdefault("error", None)
    put_drop_oldest(status_queue, {"state": state, **fields})


def release_generator(gen: Any) -> str | None:
    try:
        if gen is not None:
            gen.release()
    except Exception:
        return traceback.format_exc()
    return None


@dataclass
class CameraWorker:
    config_dict: dict[str, Any]
    status_queue: Any
    stop_event: Any
    frame_id: int = 0
    camera_restart_count: int = 0

    def run(self) -> None:
        print("DepthAI MJPEG camera worker starting.", flush=True)
        config = CameraConfig(**self.config_dict)
        publisher = make_encoded_publisher(config)

        emit_status(
            self.status_queue,
            "publisher_ready",
            encoded_topic=config.encoded_topic,
            encoded_payload_format=BUNDLE_FORMAT,
            encoded_buffer_capacity_bytes=config.encoded_buffer_capacity_bytes,
        )

        while not self.stop_event.is_set():
            self.camera_restart_count += 1
            last_timestamp_s = 0.0

            try:
                last_timestamp_s = self.run_camera_session(config, publisher)
            except KeyboardInterrupt:
                self.stop_event.set()
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                emit_status(
                    self.status_queue,
                    "error",
                    error=err,
                    camera_restart_count=self.camera_restart_count,
                    last_frame_id=self.frame_id or None,
                )

            if self.stop_event.is_set() or not config.retry_forever:
                break

            emit_status(
                self.status_queue,
                "retry_wait",
                camera_restart_count=self.camera_restart_count,
                last_frame_id=self.frame_id or None,
                last_frame_timestamp_s=last_timestamp_s or None,
            )
            sleep_until_stopped(self.stop_event, config.retry_delay_s)

        emit_status(
            self.status_queue,
            "stopped",
            camera_restart_count=self.camera_restart_count,
            last_frame_id=self.frame_id or None,
        )
        print("DepthAI MJPEG camera worker stopped.", flush=True)

    def run_camera_session(
        self,
        config: CameraConfig,
        publisher: "Model4Mat.EncodedImageMatPubSub",
    ) -> float:
        gen = None
        last_timestamp_s = 0.0
        period_ns = frame_period_ns(config)

        try:
            emit_status(
                self.status_queue,
                "starting",
                camera_restart_count=self.camera_restart_count,
                last_frame_id=self.frame_id or None,
            )
            gen = build_dai_generator(config)
            emit_status(
                self.status_queue,
                "running",
                camera_restart_count=self.camera_restart_count,
                last_frame_id=self.frame_id or None,
                encoded_topic=config.encoded_topic,
                encoded_payload_format=BUNDLE_FORMAT,
            )

            for frame in gen:
                if self.stop_event.is_set():
                    break

                self.frame_id += 1
                last_timestamp_s = time.time()
                pts_ns = self.frame_id * period_ns

                payload = pack_rgb_stereo_mjpeg_bundle(
                    frame,
                    frame_index=self.frame_id,
                    pts_ns=pts_ns,
                    rgb_width=config.rgb_width,
                    rgb_height=config.rgb_height,
                    stereo_width=config.stereo_width,
                    stereo_height=config.stereo_height,
                )
                encoded_nbytes = publish_bundle(
                    publisher,
                    payload,
                    frame_index=self.frame_id,
                    pts_ns=pts_ns,
                    width=config.rgb_width,
                    height=config.rgb_height,
                )

                emit_status(
                    self.status_queue,
                    "running",
                    camera_restart_count=self.camera_restart_count,
                    last_frame_id=self.frame_id,
                    last_frame_timestamp_s=last_timestamp_s,
                    last_encoded_nbytes=encoded_nbytes,
                    last_rgb_nbytes=len(frame.rgb.data),
                    last_left_nbytes=len(frame.left.data),
                    last_right_nbytes=len(frame.right.data),
                    encoded_topic=config.encoded_topic,
                    encoded_payload_format=BUNDLE_FORMAT,
                )

            if not self.stop_event.is_set():
                emit_status(
                    self.status_queue,
                    "ended",
                    error="DepthAI generator ended; restarting camera session.",
                    camera_restart_count=self.camera_restart_count,
                    last_frame_id=self.frame_id or None,
                    last_frame_timestamp_s=last_timestamp_s or None,
                )
        finally:
            err = release_generator(gen)
            if err:
                print(err, flush=True)
                emit_status(
                    self.status_queue,
                    "release_error",
                    error=err,
                    camera_restart_count=self.camera_restart_count,
                    last_frame_id=self.frame_id or None,
                )

        return last_timestamp_s


def camera_worker_entry(config_dict: dict[str, Any], status_queue: Any, stop_event: Any) -> None:
    CameraWorker(config_dict=config_dict, status_queue=status_queue, stop_event=stop_event).run()


# ---------------------------------------------------------------------------
# RPC controller. The parent process keeps only status and worker lifecycle.
# ---------------------------------------------------------------------------


@dataclass
class CameraController:
    service_name: str = "serverCam"
    controller_name: str = "camera"
    config: CameraConfig = field(default_factory=CameraConfig)
    opened: bool = False

    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _ctx: Any = field(default=None, init=False, repr=False)
    _status_queue: Any = field(default=None, init=False, repr=False)
    _stop_event: Any = field(default=None, init=False, repr=False)
    _process: Any = field(default=None, init=False, repr=False)

    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_worker_state: str | None = field(default=None, init=False, repr=False)
    _last_frame_id: int | None = field(default=None, init=False, repr=False)
    _last_frame_timestamp_s: float | None = field(default=None, init=False, repr=False)
    _last_encoded_nbytes: int | None = field(default=None, init=False, repr=False)
    _last_rgb_nbytes: int | None = field(default=None, init=False, repr=False)
    _last_left_nbytes: int | None = field(default=None, init=False, repr=False)
    _last_right_nbytes: int | None = field(default=None, init=False, repr=False)
    _camera_restart_count: int = field(default=0, init=False, repr=False)
    _process_restart_count: int = field(default=0, init=False, repr=False)

    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @staticmethod
    def openapi_examples() -> dict[str, Any]:
        return {
            **openapi_doc("camera_status", id=1, params={}),
            **openapi_doc("camera_start", id=2, params=model_to_dict(CameraConfig())),
            **openapi_doc("camera_stop", id=3, params={}),
        }

    def _is_worker_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _worker_exitcode(self) -> int | None:
        try:
            return None if self._process is None else self._process.exitcode
        except Exception:
            return None

    def _reset_status_unlocked(self) -> None:
        self._last_error = None
        self._last_worker_state = None
        self._last_frame_id = None
        self._last_frame_timestamp_s = None
        self._last_encoded_nbytes = None
        self._last_rgb_nbytes = None
        self._last_left_nbytes = None
        self._last_right_nbytes = None
        self._camera_restart_count = 0
        self._process_restart_count = 0

    def _apply_worker_status_unlocked(self, msg: dict[str, Any]) -> None:
        if msg.get("state"):
            self._last_worker_state = str(msg["state"])
        if msg.get("error"):
            self._last_error = str(msg["error"])

        self._camera_restart_count = int_or_keep(
            self._camera_restart_count,
            msg.get("camera_restart_count"),
        ) or 0
        self._last_frame_id = int_or_keep(self._last_frame_id, msg.get("last_frame_id"))
        self._last_frame_timestamp_s = float_or_keep(
            self._last_frame_timestamp_s,
            msg.get("last_frame_timestamp_s"),
        )
        self._last_encoded_nbytes = int_or_keep(
            self._last_encoded_nbytes,
            msg.get("last_encoded_nbytes"),
        )
        self._last_rgb_nbytes = int_or_keep(self._last_rgb_nbytes, msg.get("last_rgb_nbytes"))
        self._last_left_nbytes = int_or_keep(self._last_left_nbytes, msg.get("last_left_nbytes"))
        self._last_right_nbytes = int_or_keep(self._last_right_nbytes, msg.get("last_right_nbytes"))

    def _drain_status_queue_unlocked(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                msg = self._status_queue.get_nowait()
            except py_queue.Empty:
                return
            except Exception as exc:
                self._last_error = f"Failed reading worker status queue: {exc!r}"
                return
            if isinstance(msg, dict):
                self._apply_worker_status_unlocked(msg)

    def _cleanup_ipc_unlocked(self) -> None:
        if self._status_queue is not None:
            try:
                drain_queue(self._status_queue)
                self._status_queue.close()
                self._status_queue.join_thread()
            except Exception:
                pass
        self._status_queue = None
        self._stop_event = None

    def _start_worker_unlocked(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._status_queue = self._ctx.Queue(maxsize=32)
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=camera_worker_entry,
            args=(model_to_dict(self.config), self._status_queue, self._stop_event),
            daemon=False,
        )
        self._process.start()
        self.opened = True
        self._last_worker_state = "process_started"
        self._ensure_watchdog_unlocked()

    def _stop_worker_unlocked(self) -> None:
        self._watchdog_stop.set()

        if self._stop_event is not None:
            try:
                self._stop_event.set()
            except Exception:
                pass

        if self._process is not None:
            try:
                self._process.join(timeout=float(self.config.close_join_timeout_s))
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=float(self.config.close_terminate_timeout_s))
            except Exception:
                pass

        self._process = None
        self.opened = False
        self._drain_status_queue_unlocked()
        self._cleanup_ipc_unlocked()
        self._last_worker_state = "closed"

    def _restart_dead_worker_unlocked(self) -> None:
        if not self.opened or self._process is None or self._is_worker_alive():
            return

        self._drain_status_queue_unlocked()
        exitcode = self._worker_exitcode()
        self._process_restart_count += 1
        self._last_error = f"Camera worker process exited with code {exitcode}; restarting."

        try:
            self._process.join(timeout=0.0)
        except Exception:
            pass

        self._process = None
        self._cleanup_ipc_unlocked()
        self._last_worker_state = "process_restarting"

        try:
            self._start_worker_unlocked()
        except Exception:
            self.opened = False
            self._last_worker_state = "process_restart_failed"
            self._last_error = traceback.format_exc()

    def _ensure_watchdog_unlocked(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="camera-rpc-worker-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            interval_s = max(0.1, float(self.config.worker_watchdog_interval_s))
            if self._watchdog_stop.wait(interval_s):
                return
            try:
                with self._state_lock:
                    if self.opened and self._process is not None and not self._is_worker_alive():
                        self._restart_dead_worker_unlocked()
                    else:
                        self._drain_status_queue_unlocked()
            except Exception:
                self._last_error = traceback.format_exc()

    def _status_unlocked(self) -> CameraStatusResult:
        if self.opened and self._process is not None and not self._is_worker_alive():
            self._restart_dead_worker_unlocked()
        self._drain_status_queue_unlocked()

        age_s = None
        if self._last_frame_timestamp_s is not None:
            age_s = max(0.0, time.time() - self._last_frame_timestamp_s)

        return CameraStatusResult(
            opened=bool(self.opened),
            worker_alive=self._is_worker_alive(),
            worker_state=self._last_worker_state,
            worker_exitcode=self._worker_exitcode(),
            last_error=self._last_error,
            device_id=",".join(self.config.sources),
            rgb_width=self.config.rgb_width,
            rgb_height=self.config.rgb_height,
            stereo_width=self.config.stereo_width,
            stereo_height=self.config.stereo_height,
            fps=self.config.capture_fps,
            encoded_topic=self.config.encoded_topic,
            encoded_payload_format=BUNDLE_FORMAT,
            encoded_buffer_capacity_bytes=self.config.encoded_buffer_capacity_bytes,
            last_frame_id=self._last_frame_id,
            last_frame_age_s=age_s,
            last_encoded_nbytes=self._last_encoded_nbytes,
            last_rgb_nbytes=self._last_rgb_nbytes,
            last_left_nbytes=self._last_left_nbytes,
            last_right_nbytes=self._last_right_nbytes,
            camera_restart_count=self._camera_restart_count,
            process_restart_count=self._process_restart_count,
        )

    def open(self, params: CameraConfig) -> CameraStatusResult:
        config = params or self.config
        with self._state_lock:
            try:
                if self._is_worker_alive() and config == self.config:
                    self.opened = True
                    return self._status_unlocked()
                if self._process is not None:
                    self._stop_worker_unlocked()

                self.config = config
                self._reset_status_unlocked()
                self._start_worker_unlocked()
            except Exception:
                self.opened = False
                self._last_worker_state = "open_failed"
                self._last_error = traceback.format_exc()
            return self._status_unlocked()

    def close(self, params: EmptyParams) -> CameraStatusResult:
        with self._state_lock:
            self._stop_worker_unlocked()
            return self._status_unlocked()

    def status(self, params: EmptyParams) -> CameraStatusResult:
        with self._state_lock:
            return self._status_unlocked()


def run_server(service_name: str = "serverCam", controller_name: str = "camera") -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    mp.freeze_support()
    Iox2JsonRpcServer(
        CameraController(service_name=service_name, controller_name=controller_name)
    ).run_forever()


if __name__ == "__main__":
    run_server(service_name=args.service_name, controller_name=args.controller_name)
