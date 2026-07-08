from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as py_queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from common import (EmptyParams, RpcModel, openapi_doc,
                    AsyncPreviewProcess,
                    AsyncYoloProcess, CaptureRequestProcess,
                    WorkerFrame, WorkerContext, WorkerPipelineProcess)

try:
    # Preferred module name for the MJPEG + torchvision/nvJPEG decode generator.
    from resultkit.dai.rgb_stereo_mjpeg_cuda_generator import DepthAIPoeRGBStereoMjpegTorchGenerator
except ImportError:
    # Fallback if you installed the same class under the shorter MJPEG module name.
    from resultkit.dai.rgb_stereo_mjpeg_generator import DepthAIPoeRGBStereoMjpegTorchGenerator

StreamName = Literal["rgb", "stereo", "left", "right"]
STREAMS: tuple[StreamName, ...] = ("rgb", "stereo", "left", "right")

parser = argparse.ArgumentParser(
    description="Run the camera controller JSON-RPC server."
)
parser.add_argument(
    "--service-name",
    default="OkadCamA",
    help="Name of the iceoryx2 service.",
)
parser.add_argument(
    "--controller-name",
    default="camera",
    help="Name of the camera controller.",
)
parser.add_argument(
    "--device",
    default="169.254.1.222",
    help="IP address or device identifier of the camera controller.",
)
args = parser.parse_args()


class CameraBaseModel(RpcModel):
    service: str = args.service_name


class CameraConfig(CameraBaseModel):
    """Runtime settings for the DepthAI PoE RGB+stereo RPC camera server.

    Keep DepthAI/Torch/OpenCV frame work in the child process. The RPC parent
    only owns queues, metadata, and JPEG base64 strings, so native decoder
    failures cannot directly crash the RPC server process.
    """

    uuid: str = f"{args.service_name}:{args.controller_name}"
    sources: list[str] = [args.device]

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800
    capture_fps: int = 15

    # MJPEG camera generator settings. The old H26x fields below are kept only
    # for RPC/config compatibility with older clients; _build_dai_gen_from_config
    # does not pass them to the MJPEG generator.
    rgb_mjpeg_quality: int = 90
    stereo_mjpeg_quality: int = 85
    rgb_mjpeg_input_type: str = "NV12"
    stereo_mjpeg_input_type: str = "NV12"
    mjpeg_decode_backend: str = "torchvision-cuda"
    mjpeg_decode_fallback_to_opencv: bool = True

    # CUDA device used by the MJPEG decoder. Leave torch_device as None to use
    # cuda:{gpu_id}; set torch_device="cuda:0" if you prefer explicit config.
    gpu_id: int = 0
    torch_device: str | None = None
    non_blocking_gpu_copy: bool = True

    # Deprecated H26x compatibility fields. They are accepted in requests but
    # ignored by the MJPEG generator path.
    rgb_codec: str = "mjpeg"
    stereo_codec: str = "mjpeg"
    rgb_bitrate_kbps: int = 0
    stereo_bitrate_kbps: int = 0
    decoder_backend: str = "mjpeg"
    gst_nvivafilter_so: str = "./libdepthai_cuda_preprocess.so"
    gst_nvivafilter_dtype: str = "fp16"
    gst_nvivafilter_channel_order: str = "rgba"
    decoder_output_color: str = "rgbp"
    stereo_decoder_output_color: str = "rgbp"

    rgb_camera_socket: str = "CAM_A"
    left_camera_socket: str = "CAM_B"
    right_camera_socket: str = "CAM_C"

    normalize_rgb: bool = True
    normalize_stereo: bool = True

    preview: bool = True
    preview_rgb_downsample: int = 10
    preview_stereo_downsample: int = 4
    preview_yolo_overlay: bool = True

    # Async zero-copy CUDA Ultralytics YOLO inference. The model is created
    # inside the CameraWorker child process and inside the async YOLO thread.
    # Frames stay as CUDA tensors; only post-NMS detection summaries are copied
    # back to CPU for JSON status output.
    yolo_enabled: bool = True
    yolo_model_path: str = "yolov8n.pt"
    yolo_stream: StreamName = "rgb"
    yolo_imgsz: int = 1280
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_max_det: int = 100
    yolo_max_results: int = 20
    yolo_device: str = "0"
    yolo_half: bool = True
    yolo_verbose: bool = False  # kept for config compatibility; raw-model path does not use it
    # None means: follow normalize_rgb for rgb, normalize_stereo otherwise.
    # Set False only when your frame tensors are float 0..255 or uint8-like.
    yolo_input_normalized: bool | None = None
    yolo_letterbox: bool = True
    yolo_pad_value: float = 114.0 / 255.0
    # GPU slicing before resize/letterbox. Keeps CUDA-only path, useful for quick load tests.
    yolo_downsample: int = 10
    yolo_frame_interval: int = 1
    yolo_queue_max_size: int = 1
    yolo_join_timeout_s: float = 2.0
    yolo_empty_cache_on_stop: bool = False

    # Debug-only paired overlay: disabled by default because the normal preview
    # stage is the single owner of OpenCV HighGUI windows. Enable only when you
    # specifically need exact YOLO-frame debug visualization.
    yolo_debug_preview: bool = False
    yolo_debug_window_name: str = "yolo_debug_rgb"
    yolo_debug_downsample: int = 1
    yolo_debug_wait_key_ms: int = 1
    yolo_debug_font_scale: float = 0.6
    yolo_debug_text_thickness: int = 2

    queue_max_size: int = 8
    capture_wait_s: float = 2.0
    close_join_timeout_s: float = 5.0
    close_terminate_timeout_s: float = 2.0
    worker_watchdog_interval_s: float = 0.5

    retry_forever: bool = True
    retry_delay_s: float = 1.0


class CaptureParams(CameraBaseModel):
    """Params for camera.capture.

    exposure_ms and iso are kept for wire compatibility. This generator path does
    not apply manual exposure unless the generator implementation adds that API.
    """

    stream: StreamName = "rgb"
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    exposure_ms: int | None = Field(default=None, ge=1, le=33)
    iso: int | None = Field(default=None, ge=100, le=1600)


class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int
    device_id: str
    worker_alive: bool
    width: int
    height: int
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    fps: int
    last_frame_id: int | None = None
    last_frame_age_s: float | None = None
    last_error: str | None = None
    worker_state: str | None = None
    restart_count: int = 0
    process_restart_count: int = 0
    worker_exitcode: int | None = None

    yolo_enabled: bool = False
    yolo_model_path: str | None = None
    yolo_stream: str | None = None
    yolo_frame_id: int | None = None
    yolo_frame_age_s: float | None = None
    yolo_latency_ms: float | None = None
    yolo_num_detections: int = 0
    yolo_detections: list[dict[str, Any]] = []
    yolo_zero_copy_cuda: bool = True

    mjpeg_decode_backend: str = "torchvision-cuda"
    rgb_mjpeg_quality: int = 90
    stereo_mjpeg_quality: int = 85
    rgb_valid_height: int | None = None


class CaptureResult(CameraStatusResult):
    frame_id: int
    frame_timestamp_s: float
    stream: StreamName
    tensor_shape: tuple[int, ...]
    jpeg_base64: str


# ---------------------------------------------------------------------------
# Shared helpers. Parent helpers must stay JSON/pydantic/queue-only.
# ---------------------------------------------------------------------------


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _put_drop_oldest(q: Any, item: Any) -> None:
    """Put without blocking forever; if full, drop one stale item."""

    try:
        q.put_nowait(item)
        return
    except py_queue.Full:
        pass

    try:
        q.get_nowait()
    except (py_queue.Empty, Exception):
        pass

    try:
        q.put_nowait(item)
    except py_queue.Full:
        pass


def _drain_queue(q: Any) -> None:
    while q is not None:
        try:
            q.get_nowait()
        except (py_queue.Empty, Exception):
            return


def _shape(x: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _int_or_keep(current: int | None, value: Any) -> int | None:
    try:
        return int(value) if value is not None else current
    except Exception:
        return current


def _float_or_keep(current: float | None, value: Any) -> float | None:
    try:
        return float(value) if value is not None else current
    except Exception:
        return current


# ---------------------------------------------------------------------------
# Worker-only helpers. Keep Torch/OpenCV/DepthAI imports inside functions.
# ---------------------------------------------------------------------------

def _sleep_until_stopped(stop_event: Any, seconds: float) -> None:
    end_s = time.monotonic() + max(0.0, seconds)
    while not stop_event.is_set() and time.monotonic() < end_s:
        time.sleep(min(0.1, end_s - time.monotonic()))


def _packed_stereo_rows_per_side_from_config(config: Any) -> int:
    stereo_values = int(config.stereo_width) * int(config.stereo_height)
    values_per_row = 3 * int(config.rgb_width)
    return (stereo_values + values_per_row - 1) // values_per_row


def _rgb_valid_height_from_config(config: Any) -> int:
    return max(0, int(config.rgb_height) - 2 * _packed_stereo_rows_per_side_from_config(config))

def _build_dai_gen_from_config(config: dict[str, Any]) -> Any:

    camera_config = CameraConfig(**config)
    return DepthAIPoeRGBStereoMjpegTorchGenerator(
        uuid=camera_config.uuid,
        sources=camera_config.sources,
        rgb_width=camera_config.rgb_width,
        rgb_height=camera_config.rgb_height,
        stereo_width=camera_config.stereo_width,
        stereo_height=camera_config.stereo_height,
        capture_fps=camera_config.capture_fps,
        rgb_mjpeg_quality=camera_config.rgb_mjpeg_quality,
        stereo_mjpeg_quality=camera_config.stereo_mjpeg_quality,
        rgb_mjpeg_input_type=camera_config.rgb_mjpeg_input_type,
        stereo_mjpeg_input_type=camera_config.stereo_mjpeg_input_type,
        mjpeg_decode_backend=camera_config.mjpeg_decode_backend,
        mjpeg_decode_fallback_to_opencv=camera_config.mjpeg_decode_fallback_to_opencv,
        gpu_id=camera_config.gpu_id,
        torch_device=camera_config.torch_device,
        non_blocking_gpu_copy=camera_config.non_blocking_gpu_copy,
        rgb_camera_socket=camera_config.rgb_camera_socket,
        left_camera_socket=camera_config.left_camera_socket,
        right_camera_socket=camera_config.right_camera_socket,
        normalize_rgb=camera_config.normalize_rgb,
        normalize_stereo=camera_config.normalize_stereo,
        color_types=[],
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )


def _emit_worker_status(status_queue: Any, state: str, **fields: Any) -> None:
    fields.setdefault("timestamp_s", time.time())
    fields.setdefault("error", None)
    _put_drop_oldest(status_queue, {"state": state, **fields})


def _unpack_frame(gen: Any, mats: Any) -> dict[StreamName, Any]:
    rgb_with_payload, stereo, left, right = gen.unpack_packed_tensor(mats[0].data)

    # The MJPEG generator packs left stereo into top rows and right stereo into
    # bottom rows of the returned RGB tensor. For preview, capture, and YOLO,
    # publish a clean RGB view with those payload rows removed.
    if hasattr(gen, "rgb_without_stereo_payload"):
        rgb = gen.rgb_without_stereo_payload(rgb_with_payload)
    else:
        rgb = rgb_with_payload

    return {"rgb": rgb, "stereo": stereo, "left": left, "right": right}


def _emit_frame_status(
    status_queue: Any,
    tensors: dict[StreamName, Any],
    *,
    frame_id: int,
    frame_timestamp_s: float,
    restart_count: int,
) -> None:
    shapes = {f"{stream}_shape": _shape(tensor) for stream, tensor in tensors.items()}
    _emit_worker_status(
        status_queue,
        "running",
        restart_count=restart_count,
        last_frame_id=frame_id,
        last_frame_timestamp_s=frame_timestamp_s,
        **shapes,
    )


def _release_worker_resources(gen: Any) -> str | None:
    error = None
    try:
        if gen is not None:
            gen.release()
    except Exception:
        error = traceback.format_exc()
    return error



@dataclass
class FrameStatusProcess(WorkerPipelineProcess):
    """Periodically report the latest frame metadata to the parent process."""

    emit_interval_s: float = 1.0
    _last_emit_monotonic_s: float = field(default=0.0, init=False, repr=False)

    def on_start(self, context: WorkerContext) -> None:
        self._last_emit_monotonic_s = 0.0

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        now = time.monotonic()
        if now - self._last_emit_monotonic_s < self.emit_interval_s:
            return

        self._last_emit_monotonic_s = now
        _emit_frame_status(
            context.status_queue,
            frame.tensors,
            frame_id=frame.frame_id,
            frame_timestamp_s=frame.timestamp_s,
            restart_count=context.restart_count,
        )


def build_worker_pipeline(config: dict[str, Any]) -> list[WorkerPipelineProcess]:
    """Create the ordered worker-side processing pipeline.

    Status and capture request handling stay inline. Preview and YOLO are async
    latest-frame stages so OpenCV display or model inference cannot block the
    camera acquisition loop.
    """

    processes: list[WorkerPipelineProcess] = [
        FrameStatusProcess(),
        CaptureRequestProcess(),
    ]

    if bool(config.get("yolo_enabled", False)):
        processes.append(
            AsyncYoloProcess(
                queue_max_size=int(config.get("yolo_queue_max_size", 1)),
                join_timeout_s=float(config.get("yolo_join_timeout_s", 2.0)),
            )
        )


    if bool(config.get("preview", False)):
        processes.append(
            AsyncPreviewProcess(
                queue_max_size=1,
                join_timeout_s=2.0,
            )
        )

    return processes


@dataclass
class CameraWorker:
    """Child process owner for DepthAI and all frame pipeline processing."""

    config: dict[str, Any]
    status_queue: Any
    capture_request_queue: Any
    capture_result_queue: Any
    stop_event: Any
    frame_id: int = 0
    restart_count: int = 0

    def run(self) -> None:
        print("DepthAI RPC camera worker starting.", flush=True)
        retry_forever = bool(self.config.get("retry_forever", True))
        retry_delay_s = float(self.config.get("retry_delay_s", 1.0))        
        pipeline = build_worker_pipeline(self.config)

        while not self.stop_event.is_set():
            self.restart_count += 1
            latest_timestamp_s = 0.0
            context = WorkerContext(
                config=self.config,
                status_queue=self.status_queue,
                capture_request_queue=self.capture_request_queue,
                capture_result_queue=self.capture_result_queue,
                stop_event=self.stop_event,
                restart_count=self.restart_count,
            )

            _emit_worker_status(
                self.status_queue,
                "starting",
                restart_count=self.restart_count,
                last_frame_id=self.frame_id or None,
            )

            try:
                latest_timestamp_s = self._run_session(context, pipeline)
            except KeyboardInterrupt:
                self.stop_event.set()
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                _emit_worker_status(
                    self.status_queue,
                    "error",
                    error=err,
                    restart_count=self.restart_count,
                    last_frame_id=self.frame_id or None,
                )

            if self.stop_event.is_set() or not retry_forever:
                break

            _emit_worker_status(
                self.status_queue,
                "retry_wait",
                restart_count=self.restart_count,
                last_frame_id=self.frame_id or None,
                last_frame_timestamp_s=latest_timestamp_s or None,
            )
            _sleep_until_stopped(self.stop_event, retry_delay_s)

        _emit_worker_status(
            self.status_queue,
            "stopped",
            restart_count=self.restart_count,
            last_frame_id=self.frame_id or None,
        )
        print("DepthAI RPC camera worker stopped.", flush=True)

    def _run_session(self, context: WorkerContext, pipeline: list[WorkerPipelineProcess]) -> float:
        gen = None
        latest_timestamp_s = 0.0

        try:
            for process in pipeline:
                process.on_start(context)

            gen = _build_dai_gen_from_config(self.config)
            _emit_worker_status(
                self.status_queue,
                "running",
                restart_count=self.restart_count,
                last_frame_id=self.frame_id or None,
            )

            for mats in gen:
                if self.stop_event.is_set():
                    break

                self.frame_id += 1
                latest_timestamp_s = time.time()
                frame = WorkerFrame(
                    frame_id=self.frame_id,
                    timestamp_s=latest_timestamp_s,
                    tensors=_unpack_frame(gen, mats),
                )

                for process in pipeline:
                    process.on_frame(context, frame)

            if not self.stop_event.is_set():
                _emit_worker_status(
                    self.status_queue,
                    "ended",
                    error="DepthAI generator ended; restarting camera session.",
                    restart_count=self.restart_count,
                    last_frame_id=self.frame_id or None,
                    last_frame_timestamp_s=latest_timestamp_s or None,
                )
        finally:
            for process in reversed(pipeline):
                process.on_stop(context)
            err = _release_worker_resources(gen)
            if err:
                print(err, flush=True)
                _emit_worker_status(
                    self.status_queue,
                    "release_error",
                    error=err,
                    restart_count=self.restart_count,
                    last_frame_id=self.frame_id or None,
                )

        return latest_timestamp_s


def _camera_worker(
    config: dict[str, Any],
    status_queue: Any,
    capture_request_queue: Any,
    capture_result_queue: Any,
    stop_event: Any,
) -> None:
    """Child process entry point. Must stay top-level for multiprocessing spawn."""

    CameraWorker(
        config=config,
        status_queue=status_queue,
        capture_request_queue=capture_request_queue,
        capture_result_queue=capture_result_queue,
        stop_event=stop_event,
    ).run()


# ---------------------------------------------------------------------------
# RPC controller. Parent process owns only queues, metadata, and JPEG text.
# ---------------------------------------------------------------------------


@dataclass
class CameraController:
    """JSON-RPC controller backed by an isolated DepthAI worker process."""

    service_name: str = "serverCam"
    controller_name: str = "camera"
    config: CameraConfig = field(default_factory=CameraConfig)
    opened: bool = False
    captures: int = 0

    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _ctx: Any = field(default=None, init=False, repr=False)
    _status_queue: Any = field(default=None, init=False, repr=False)
    _capture_request_queue: Any = field(default=None, init=False, repr=False)
    _capture_result_queue: Any = field(default=None, init=False, repr=False)
    _stop_event: Any = field(default=None, init=False, repr=False)
    _process: Any = field(default=None, init=False, repr=False)

    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_worker_state: str | None = field(default=None, init=False, repr=False)
    _restart_count: int = field(default=0, init=False, repr=False)
    _process_restart_count: int = field(default=0, init=False, repr=False)
    _last_frame_id: int | None = field(default=None, init=False, repr=False)
    _last_frame_timestamp_s: float | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)

    _last_yolo_model_path: str | None = field(default=None, init=False, repr=False)
    _last_yolo_stream: str | None = field(default=None, init=False, repr=False)
    _last_yolo_frame_id: int | None = field(default=None, init=False, repr=False)
    _last_yolo_frame_timestamp_s: float | None = field(default=None, init=False, repr=False)
    _last_yolo_latency_ms: float | None = field(default=None, init=False, repr=False)
    _last_yolo_num_detections: int = field(default=0, init=False, repr=False)
    _last_yolo_detections: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @staticmethod
    def openapi_examples():
        return {
            **openapi_doc("camera_status", id=1, params={}),
            **openapi_doc("camera_start", id=2, params=CameraConfig().model_dump()),
            **openapi_doc("camera_stop", id=3, params={}),
            **openapi_doc("camera_capture", id=4, params=CaptureParams().model_dump()),
        }

    def _is_worker_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _worker_exitcode(self) -> int | None:
        try:
            return None if self._process is None else self._process.exitcode
        except Exception:
            return None

    def _reset_runtime_state_unlocked(self) -> None:
        self._last_error = None
        self._last_worker_state = None
        self._last_frame_id = None
        self._last_frame_timestamp_s = None
        self._restart_count = 0
        self._process_restart_count = 0
        self._request_id = 0
        self._last_yolo_model_path = None
        self._last_yolo_stream = None
        self._last_yolo_frame_id = None
        self._last_yolo_frame_timestamp_s = None
        self._last_yolo_latency_ms = None
        self._last_yolo_num_detections = 0
        self._last_yolo_detections = []

    def _apply_worker_status_unlocked(self, msg: dict[str, Any]) -> None:
        state = msg.get("state")
        if state:
            self._last_worker_state = str(state)
        self._restart_count = _int_or_keep(self._restart_count, msg.get("restart_count")) or 0
        self._last_frame_id = _int_or_keep(self._last_frame_id, msg.get("last_frame_id"))
        self._last_frame_timestamp_s = _float_or_keep(
            self._last_frame_timestamp_s,
            msg.get("last_frame_timestamp_s"),
        )
        if msg.get("error"):
            self._last_error = str(msg["error"])

        if any(str(key).startswith("yolo_") for key in msg):
            self._last_yolo_model_path = msg.get("yolo_model_path", self._last_yolo_model_path)
            self._last_yolo_stream = msg.get("yolo_stream", self._last_yolo_stream)
            self._last_yolo_frame_id = _int_or_keep(
                self._last_yolo_frame_id,
                msg.get("yolo_frame_id"),
            )
            self._last_yolo_frame_timestamp_s = _float_or_keep(
                self._last_yolo_frame_timestamp_s,
                msg.get("yolo_frame_timestamp_s"),
            )
            self._last_yolo_latency_ms = _float_or_keep(
                self._last_yolo_latency_ms,
                msg.get("yolo_latency_ms"),
            )
            self._last_yolo_num_detections = _int_or_keep(
                self._last_yolo_num_detections,
                msg.get("yolo_num_detections"),
            ) or 0
            detections = msg.get("yolo_detections")
            if isinstance(detections, list):
                self._last_yolo_detections = [d for d in detections if isinstance(d, dict)]

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                msg = self._status_queue.get_nowait()
            except py_queue.Empty:
                break
            except Exception as exc:
                self._last_error = f"Failed reading worker status queue: {exc!r}"
                break
            if isinstance(msg, dict):
                self._apply_worker_status_unlocked(msg)

    def _cleanup_queues_unlocked(self) -> None:
        for attr in ("_status_queue", "_capture_request_queue", "_capture_result_queue"):
            q = getattr(self, attr)
            if q is None:
                continue
            try:
                _drain_queue(q)
                q.close()
                q.join_thread()
            except Exception:
                pass
            setattr(self, attr, None)
        self._stop_event = None

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
            try:
                interval_s = max(0.1, float(self.config.worker_watchdog_interval_s))
            except Exception:
                interval_s = 0.5

            if self._watchdog_stop.wait(interval_s):
                break

            try:
                with self._state_lock:
                    if self.opened and self._process is not None and not self._is_worker_alive():
                        self._restart_dead_worker_unlocked()
                    else:
                        self._drain_status_queue()
            except Exception:
                self._last_error = traceback.format_exc()

    def _start_worker_unlocked(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._status_queue = self._ctx.Queue(maxsize=32)
        self._capture_request_queue = self._ctx.Queue(maxsize=int(self.config.queue_max_size))
        self._capture_result_queue = self._ctx.Queue(maxsize=int(self.config.queue_max_size))
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_camera_worker,
            args=(
                _model_to_dict(self.config),
                self._status_queue,
                self._capture_request_queue,
                self._capture_result_queue,
                self._stop_event,
            ),
            daemon=False,
        )
        self._process.start()
        self.opened = True
        self._last_worker_state = "process_started"
        self._ensure_watchdog_unlocked()

    def _restart_dead_worker_unlocked(self) -> None:
        """Recover if the worker process died from a native crash."""

        if not self.opened or self._process is None or self._is_worker_alive():
            return

        self._drain_status_queue()
        exitcode = self._worker_exitcode()
        self._process_restart_count += 1
        self._last_error = f"Camera worker process exited with code {exitcode}; restarting."

        try:
            self._process.join(timeout=0.0)
        except Exception:
            pass

        self._process = None
        self._cleanup_queues_unlocked()
        self._last_worker_state = "process_restarting"

        try:
            self._start_worker_unlocked()
        except Exception:
            self.opened = False
            self._last_worker_state = "process_restart_failed"
            self._last_error = traceback.format_exc()

    def _stop_worker_unlocked(self) -> None:
        process, stop_event = self._process, self._stop_event

        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass

        if process is not None:
            try:
                process.join(timeout=float(self.config.close_join_timeout_s))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=float(self.config.close_terminate_timeout_s))
            except Exception:
                pass

        self._process = None
        self.opened = False
        self._drain_status_queue()
        self._cleanup_queues_unlocked()
        self._last_worker_state = "closed"

    def _status_unlocked(self) -> CameraStatusResult:
        if self.opened and self._process is not None and not self._is_worker_alive():
            self._restart_dead_worker_unlocked()
        self._drain_status_queue()

        age_s = None
        if self._last_frame_timestamp_s is not None:
            age_s = max(0.0, time.time() - self._last_frame_timestamp_s)

        yolo_age_s = None
        if self._last_yolo_frame_timestamp_s is not None:
            yolo_age_s = max(0.0, time.time() - self._last_yolo_frame_timestamp_s)

        return CameraStatusResult(**{
            "opened": bool(self.opened),
            "captures": self.captures,
            "device_id": ",".join(self.config.sources),
            "worker_alive": self._is_worker_alive(),
            "width": self.config.rgb_width,
            "height": self.config.rgb_height,
            "rgb_width": self.config.rgb_width,
            "rgb_height": self.config.rgb_height,
            "stereo_width": self.config.stereo_width,
            "stereo_height": self.config.stereo_height,
            "fps": self.config.capture_fps,
            "last_frame_id": self._last_frame_id,
            "last_frame_age_s": age_s,
            "last_error": self._last_error,
            "worker_state": self._last_worker_state,
            "restart_count": self._restart_count,
            "process_restart_count": self._process_restart_count,
            "worker_exitcode": self._worker_exitcode(),
            "yolo_enabled": bool(self.config.yolo_enabled),
            "yolo_model_path": self._last_yolo_model_path or self.config.yolo_model_path,
            "yolo_stream": self._last_yolo_stream or self.config.yolo_stream,
            "yolo_frame_id": self._last_yolo_frame_id,
            "yolo_frame_age_s": yolo_age_s,
            "yolo_latency_ms": self._last_yolo_latency_ms,
            "yolo_num_detections": self._last_yolo_num_detections,
            "yolo_detections": self._last_yolo_detections,
            "yolo_zero_copy_cuda": True,
            "mjpeg_decode_backend": self.config.mjpeg_decode_backend,
            "rgb_mjpeg_quality": self.config.rgb_mjpeg_quality,
            "stereo_mjpeg_quality": self.config.stereo_mjpeg_quality,
            "rgb_valid_height": _rgb_valid_height_from_config(self.config),
        })

    def open(self, params: CameraConfig) -> CameraStatusResult:
        """Start the isolated camera worker process."""

        config = params or self.config
        with self._state_lock:
            try:
                if self._is_worker_alive() and config == self.config:
                    self.opened = True
                    return self._status_unlocked()

                if self._process is not None:
                    self._stop_worker_unlocked()

                self.config = config
                self._reset_runtime_state_unlocked()
                self._start_worker_unlocked()
            except Exception:
                self.opened = False
                self._last_worker_state = "open_failed"
                self._last_error = traceback.format_exc()
            return self._status_unlocked()

    def close(self, params: EmptyParams) -> CameraStatusResult:
        """Stop the worker process and release the camera."""

        with self._state_lock:
            self._stop_worker_unlocked()
            return self._status_unlocked()

    def status(self, params: EmptyParams) -> CameraStatusResult:
        with self._state_lock:
            return self._status_unlocked()

    def _put_capture_request_unlocked(self, request: dict[str, Any]) -> None:
        if self._capture_request_queue is None:
            raise RuntimeError("Camera worker request queue is not available")
        try:
            self._capture_request_queue.put_nowait(request)
        except py_queue.Full:
            _drain_queue(self._capture_request_queue)
            self._capture_request_queue.put_nowait(request)

    def _wait_capture_result_unlocked(self, request_id: int, wait_s: float) -> dict[str, Any] | None:
        if self._capture_result_queue is None:
            return None

        deadline = time.monotonic() + max(0.0, wait_s)
        while time.monotonic() < deadline:
            try:
                result = self._capture_result_queue.get(
                    timeout=min(0.1, max(0.0, deadline - time.monotonic()))
                )
            except py_queue.Empty:
                if self.opened and self._process is not None and not self._is_worker_alive():
                    return None
                continue
            except Exception:
                return None

            if isinstance(result, dict) and int(result.get("request_id", -999999)) == request_id:
                return result
        return None

    def _capture_once_unlocked(self, request: dict[str, Any]) -> dict[str, Any] | None:
        self._request_id += 1
        request = {**request, "request_id": self._request_id, "timestamp_s": time.time()}
        self._put_capture_request_unlocked(request)
        return self._wait_capture_result_unlocked(
            self._request_id,
            wait_s=float(self.config.capture_wait_s),
        )

    def _capture_size(self, stream: str) -> tuple[int, int]:
        if stream == "rgb":
            return self.config.rgb_width, _rgb_valid_height_from_config(self.config)
        if stream == "stereo":
            return 2 * self.config.stereo_width, self.config.stereo_height
        return self.config.stereo_width, self.config.stereo_height

    def _build_capture_result(
        self,
        result: dict[str, Any],
        status: CameraStatusResult,
        stream: str,
    ) -> CaptureResult:
        width, height = self._capture_size(stream)
        payload = _model_to_dict(status)
        payload.update({
            "captures": self.captures,
            "width": width,
            "height": height,
            "frame_id": int(result["frame_id"]),
            "frame_timestamp_s": float(result["frame_timestamp_s"]),
            "stream": stream,
            "tensor_shape": tuple(int(x) for x in result["tensor_shape"]),
            "jpeg_base64": str(result["jpeg_base64"]),
        })
        return CaptureResult(**payload)

    def capture(self, params: CaptureParams) -> CaptureResult:
        """Ask the worker to JPEG-encode the latest stream and return it."""

        with self._state_lock:
            if self._process is not None and not self._is_worker_alive() and self.opened:
                self._restart_dead_worker_unlocked()
            elif not self._is_worker_alive():
                self.open(self.config)

            if not self.opened or not self._is_worker_alive():
                status = self._status_unlocked()
                raise RuntimeError(
                    "Camera worker is not running. "
                    f"worker_state={status.worker_state!r}, last_error={status.last_error!r}"
                )

            _drain_queue(self._capture_result_queue)
            request = {"stream": params.stream, "jpeg_quality": int(params.jpeg_quality)}
            result = self._capture_once_unlocked(request)

            # If a native crash killed the worker while capture was waiting,
            # restart once and give the new worker one more capture interval.
            if result is None and self.opened and self._process is not None and not self._is_worker_alive():
                self._restart_dead_worker_unlocked()
                if self.opened and self._is_worker_alive():
                    _drain_queue(self._capture_result_queue)
                    result = self._capture_once_unlocked(request)

            if result is None:
                status = self._status_unlocked()
                raise RuntimeError(
                    f"No JPEG capture received within {self.config.capture_wait_s}s. "
                    f"worker_alive={status.worker_alive}, "
                    f"worker_state={status.worker_state!r}, "
                    f"last_error={status.last_error!r}"
                )

            if not result.get("ok", False):
                self._last_error = str(result.get("error", "capture failed"))
                raise RuntimeError(self._last_error)

            self.captures += 1
            self._last_frame_id = int(result["frame_id"])
            self._last_frame_timestamp_s = float(result["frame_timestamp_s"])
            status = self._status_unlocked()
            stream = str(result.get("stream", params.stream))
            return self._build_capture_result(result, status, stream)


def run_server(service_name: str = "serverCam", controller_name: str = "camera") -> None:
    """Run the real camera controller as an iceoryx2 JSON-RPC service."""

    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    mp.freeze_support()
    Iox2JsonRpcServer(
        CameraController(service_name=service_name, controller_name=controller_name)
    ).run_forever()


if __name__ == "__main__":
    run_server(
        service_name=args.service_name,
        controller_name=args.controller_name,
    )
