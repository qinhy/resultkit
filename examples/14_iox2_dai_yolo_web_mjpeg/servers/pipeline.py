from __future__ import annotations
import argparse
import base64
import multiprocessing as mp
import queue as py_queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import cv2


StreamName = Literal["rgb", "stereo", "left", "right"]
STREAMS: tuple[StreamName, ...] = ("rgb", "stereo", "left", "right")

# ---------------------------------------------------------------------------
# Shared helpers. Parent helpers must stay JSON/pydantic/queue-only.
# ---------------------------------------------------------------------------

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


def _shape(x: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


# ---------------------------------------------------------------------------
# Worker-only helpers. Keep Torch/OpenCV/DepthAI imports inside functions.
# ---------------------------------------------------------------------------


def _to_cv_array(mat: Any, *, downsample: int = 1, rgb_to_bgr: bool = True) -> Any:
    """Convert a Torch tensor image to a CPU OpenCV array inside the worker."""
    
    x = mat.detach()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = x.permute(1, 2, 0)

    step = max(1, int(downsample))
    if step > 1:
        x = x[::step, ::step]

    if x.dtype.is_floating_point:
        max_value = float(x.max()) if x.numel() else 1.0
        x = x * 255.0 if max_value <= 1.5 else x
        x = x.clamp(0, 255)
    x = x.to(dtype=torch.uint8)

    arr = x.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr


def _first_image(tensor: Any) -> Any:
    """Return the first image when a tensor contains a leading batch dimension."""

    try:
        return tensor[0] if int(tensor.ndim) == 4 else tensor
    except Exception:
        return tensor


def _tensor_to_jpeg_base64_in_worker(tensor: Any, jpeg_quality: int, *, rgb_to_bgr: bool) -> str:

    ok, encoded = cv2.imencode(
        ".jpg",
        _to_cv_array(tensor, downsample=1, rgb_to_bgr=rgb_to_bgr),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed while encoding camera frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _drain_capture_requests(capture_request_queue: Any) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    while True:
        try:
            req = capture_request_queue.get_nowait()
        except (py_queue.Empty, Exception):
            return requests
        if isinstance(req, dict):
            requests.append(req)


@dataclass
class WorkerFrame:
    """One frame passed through the worker-side processing pipeline."""

    frame_id: int
    timestamp_s: float
    tensors: dict[StreamName, Any]


@dataclass
class WorkerContext:
    """Mutable session context shared by worker-side processing steps."""

    config: dict[str, Any]
    status_queue: Any
    capture_request_queue: Any
    capture_result_queue: Any
    stop_event: Any
    restart_count: int


class WorkerPipelineProcess:
    """Base class for one worker-side processing step.

    Add a new pipeline behavior by subclassing this class and registering it in
    ``build_worker_pipeline``. Keep heavy imports inside the methods because the
    parent RPC process should not import DepthAI/Torch/OpenCV modules.
    """

    def on_start(self, context: WorkerContext) -> None:
        pass

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        pass

    def on_stop(self, context: WorkerContext) -> None:
        pass


class AsyncWorkerPipelineProcess(WorkerPipelineProcess):
    """Run another WorkerPipelineProcess in a thread inside the worker process.

    This keeps CUDA/Torch tensors in the same process, avoiding CUDA
    multiprocessing/IPC issues on Jetson, while preventing slow stages such as
    preview or inference from blocking the camera acquisition loop.

    The default queue size of 1 implements latest-frame behavior: if the inner
    process is still busy, the stale queued frame is dropped and replaced by the
    newest frame.
    """

    def __init__(
        self,
        inner: WorkerPipelineProcess,
        *,
        name: str | None = None,
        queue_max_size: int = 1,
        join_timeout_s: float = 2.0,
    ) -> None:
        self.inner = inner
        self.name = name or inner.__class__.__name__
        self.queue_max_size = max(1, int(queue_max_size))
        self.join_timeout_s = float(join_timeout_s)
        self._queue: py_queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self._context: WorkerContext | None = None
        self._sentinel = object()

    def on_start(self, context: WorkerContext) -> None:
        self._context = context
        self._queue = py_queue.Queue(maxsize=self.queue_max_size)

        # Initialize heavy resources in the async thread, not in the camera loop.
        self._thread = threading.Thread(
            target=self._run,
            name=f"worker-pipeline-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        q = self._queue
        if q is None:
            return

        try:
            q.put_nowait(frame)
            return
        except py_queue.Full:
            pass

        # Latest-frame policy: discard one stale frame and enqueue the newest.
        try:
            q.get_nowait()
        except (py_queue.Empty, Exception):
            pass

        try:
            q.put_nowait(frame)
        except py_queue.Full:
            pass

    def on_stop(self, context: WorkerContext) -> None:
        q = self._queue
        if q is not None:
            try:
                q.put_nowait(self._sentinel)
            except py_queue.Full:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(self._sentinel)
                except Exception:
                    pass

        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=self.join_timeout_s)

    def _run(self) -> None:
        context = self._context
        q = self._queue
        if context is None or q is None:
            return

        try:
            self.inner.on_start(context)

            while not context.stop_event.is_set():
                try:
                    item = q.get(timeout=0.1)
                except py_queue.Empty:
                    continue

                if item is self._sentinel:
                    break

                try:
                    self.inner.on_frame(context, item)
                except Exception:
                    _put_drop_oldest(
                        context.status_queue,
                        {
                            "state": "pipeline_error",
                            "pipeline": self.name,
                            "error": traceback.format_exc(),
                            "timestamp_s": time.time(),
                        },
                    )
        except Exception:
            _put_drop_oldest(
                context.status_queue,
                {
                    "state": "pipeline_error",
                    "pipeline": self.name,
                    "error": traceback.format_exc(),
                    "timestamp_s": time.time(),
                },
            )
        finally:
            try:
                self.inner.on_stop(context)
            except Exception:
                _put_drop_oldest(
                    context.status_queue,
                    {
                        "state": "pipeline_stop_error",
                        "pipeline": self.name,
                        "error": traceback.format_exc(),
                        "timestamp_s": time.time(),
                    },
                )


class CaptureRequestProcess(WorkerPipelineProcess):
    """Encode requested streams from the latest frame and return JPEG text."""

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        for req in _drain_capture_requests(context.capture_request_queue):
            request_id = int(req.get("request_id", -1))
            stream = str(req.get("stream", "rgb"))
            quality = int(req.get("jpeg_quality", 85))

            try:
                if stream not in frame.tensors:
                    raise RuntimeError(f"Unsupported stream: {stream!r}")

                tensor = frame.tensors[stream]  # type: ignore[index]
                _put_drop_oldest(
                    context.capture_result_queue,
                    {
                        "ok": True,
                        "request_id": request_id,
                        "frame_id": frame.frame_id,
                        "frame_timestamp_s": frame.timestamp_s,
                        "stream": stream,
                        "tensor_shape": _shape(tensor),
                        "jpeg_base64": _tensor_to_jpeg_base64_in_worker(
                            tensor,
                            quality,
                            rgb_to_bgr=(stream == "rgb"),
                        ),
                    },
                )
            except Exception:
                _put_drop_oldest(
                    context.capture_result_queue,
                    {"ok": False, "request_id": request_id, "error": traceback.format_exc()},
                )


class PreviewProcess(WorkerPipelineProcess):
    """Optional local OpenCV preview windows for RGB/left/right streams."""

    window_names: tuple[str, str, str] = (
        "camera_rgb",
        "camera_left",
        "camera_right",
    )

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        if not bool(context.config.get("preview", False)):
            return
        
        rgb_step = int(context.config["preview_rgb_downsample"])
        stereo_step = int(context.config["preview_stereo_downsample"])
        cv2.imshow(
            self.window_names[0],
            _to_cv_array(_first_image(frame.tensors["rgb"]), downsample=rgb_step, rgb_to_bgr=True),
        )
        cv2.imshow(
            self.window_names[1],
            _to_cv_array(_first_image(frame.tensors["left"]), downsample=stereo_step, rgb_to_bgr=False),
        )
        cv2.imshow(
            self.window_names[2],
            _to_cv_array(_first_image(frame.tensors["right"]), downsample=stereo_step, rgb_to_bgr=False),
        )

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            context.stop_event.set()

    def on_stop(self, context: WorkerContext) -> None:
        if not bool(context.config.get("preview", False)):
            return

        try:
            for window_name in self.window_names:
                cv2.destroyWindow(window_name)
        except Exception:
            pass
 
