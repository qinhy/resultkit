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
    """Convert a Torch tensor image to a CPU OpenCV array inside the worker.

    This generic converter keeps the old normalization auto-detection behavior
    for non-preview callers such as JPEG capture. Do not use it in the live
    preview hot path because ``float(x.max())`` synchronizes CUDA tensors.
    """

    x = mat.detach()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 2 and x.shape[-1] not in (1, 3, 4):
        # Stereo tensor [2, H, W] -> side-by-side grayscale preview/capture [H, 2W].
        x = torch.cat((x[0], x[1]), dim=1)
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


def _to_cv_array_preview(
    mat: Any,
    *,
    downsample: int = 1,
    rgb_to_bgr: bool = True,
    input_normalized: bool = True,
) -> Any:
    """Convert a Torch tensor image for live preview without CUDA max() sync.

    Preview knows from config whether decoder tensors are normalized. Avoiding
    data-dependent auto-detection removes an extra CUDA synchronization point
    when YOLO is busy. A CUDA -> CPU copy is still required for OpenCV display.
    """

    x = mat.detach()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 2 and x.shape[-1] not in (1, 3, 4):
        # Stereo tensor [2, H, W] -> side-by-side grayscale preview/capture [H, 2W].
        x = torch.cat((x[0], x[1]), dim=1)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = x.permute(1, 2, 0)

    step = max(1, int(downsample))
    if step > 1:
        x = x[::step, ::step]

    if x.dtype.is_floating_point:
        if input_normalized:
            x = x * 255.0
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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_list(values: Any) -> list[float]:
    return [float(v) for v in values]


def _class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
    except Exception:
        pass
    return str(class_id)


def _cuda_device_from_config(value: Any) -> torch.device:
    """Resolve a CUDA device from config without falling back to CPU silently."""

    text = _optional_str(value) or "0"
    if text.isdigit():
        text = f"cuda:{text}"
    elif text == "cuda":
        text = "cuda:0"

    device = torch.device(text)
    if device.type != "cuda":
        raise RuntimeError(
            f"Zero-copy YOLO requires a CUDA device, got {text!r}. "
            "Use yolo_device='0' or yolo_device='cuda:0'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Zero-copy YOLO requires torch.cuda.is_available() == True")
    return device


def _yolo_input_is_normalized(config: dict[str, Any], stream: str) -> bool:
    """Decide whether frame tensors are already in the 0..1 range.

    No max()/min() auto-detection is used because that would synchronize CUDA
    and defeat the purpose of keeping the hot path GPU-only. If not specified,
    reuse the camera generator's normalize_rgb/normalize_stereo settings.
    """

    explicit = config.get("yolo_input_normalized", None)
    if explicit is not None:
        return bool(explicit)
    if stream == "rgb":
        return bool(config.get("normalize_rgb", True))
    return bool(config.get("normalize_stereo", True))


def _as_bchw_cuda_tensor(
    tensor: Any,
    *,
    device: torch.device,
    half: bool,
    input_normalized: bool,
    downsample: int = 1,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert an existing CUDA image tensor to BCHW RGB float tensor on CUDA.

    This function intentionally never calls .cpu(), .numpy(), float(tensor.max()),
    or OpenCV. It accepts common HWC/CHW/NHWC/NCHW layouts, strips alpha, repeats
    grayscale to 3 channels, optionally downsamples by CUDA slicing, and returns
    the original H/W after slicing.
    """

    if not torch.is_tensor(tensor):
        raise RuntimeError(f"YOLO expected a torch.Tensor, got {type(tensor)!r}")

    x = _first_image(tensor).detach()

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    elif x.ndim == 3:
        # CHW when the first dim is channels, otherwise HWC.
        if int(x.shape[0]) in (1, 3, 4) and int(x.shape[-1]) not in (1, 3, 4):
            x = x.unsqueeze(0)
        elif int(x.shape[-1]) in (1, 3, 4):
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            raise RuntimeError(f"Unsupported YOLO tensor shape: {tuple(x.shape)}")
    elif x.ndim == 4:
        # NCHW or NHWC. Keep only the first image for live latest-frame inference.
        if int(x.shape[1]) in (1, 3, 4):
            x = x[:1]
        elif int(x.shape[-1]) in (1, 3, 4):
            x = x[:1].permute(0, 3, 1, 2)
        else:
            raise RuntimeError(f"Unsupported YOLO tensor shape: {tuple(x.shape)}")
    else:
        raise RuntimeError(f"Unsupported YOLO tensor ndim: {int(x.ndim)}")

    if int(x.shape[1]) == 4:
        x = x[:, :3]
    elif int(x.shape[1]) == 1:
        x = x.repeat(1, 3, 1, 1)
    elif int(x.shape[1]) != 3:
        raise RuntimeError(f"YOLO expects 1/3/4 channels, got shape {tuple(x.shape)}")

    step = max(1, int(downsample))
    if step > 1:
        x = x[:, :, ::step, ::step]

    orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
    dtype = torch.float16 if half else torch.float32
    needs_scale = (not x.dtype.is_floating_point) or (not input_normalized)
    x = x.to(device=device, dtype=dtype, non_blocking=True)
    if needs_scale:
        x = x / 255.0
    return x.contiguous(), (orig_h, orig_w)


def _letterbox_bchw_cuda(
    x: torch.Tensor,
    *,
    imgsz: int,
    pad_value: float = 114.0 / 255.0,
) -> tuple[torch.Tensor, float, int, int, tuple[int, int]]:
    """Letterbox BCHW tensor to a square CUDA tensor using torch ops only."""

    import torch.nn.functional as F

    target = max(1, int(imgsz))
    _, _, h, w = x.shape
    scale = min(target / float(h), target / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if new_h != h or new_w != w:
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    if pad_x == 0 and pad_y == 0 and new_h == target and new_w == target:
        return x.contiguous(), scale, pad_x, pad_y, (h, w)

    out = torch.full(
        (x.shape[0], x.shape[1], target, target),
        float(pad_value),
        dtype=x.dtype,
        device=x.device,
    )
    out[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = x
    return out.contiguous(), scale, pad_x, pad_y, (h, w)


def _resize_bchw_cuda(
    x: torch.Tensor,
    *,
    imgsz: int,
) -> tuple[torch.Tensor, float, int, int, tuple[int, int]]:
    """Direct square resize on CUDA. Faster, but boxes need non-uniform scaling."""

    import torch.nn.functional as F

    target = max(1, int(imgsz))
    _, _, h, w = x.shape
    if int(x.shape[-2]) != target or int(x.shape[-1]) != target:
        x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    # A negative scale marks direct resize; postprocess uses x/y scale separately.
    return x.contiguous(), -1.0, 0, 0, (h, w)


def _scale_yolo_boxes_to_original(
    det: torch.Tensor,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_hw: tuple[int, int],
    imgsz: int,
) -> torch.Tensor:
    """Map YOLO xyxy boxes from model input coordinates back to frame coords."""

    orig_h, orig_w = orig_hw
    boxes = det[:, :4].clone()
    if scale >= 0:
        boxes[:, [0, 2]] -= float(pad_x)
        boxes[:, [1, 3]] -= float(pad_y)
        boxes[:, :4] /= max(scale, 1e-9)
    else:
        # Direct square resize fallback: width and height have separate scales.
        boxes[:, [0, 2]] *= float(orig_w) / float(imgsz)
        boxes[:, [1, 3]] *= float(orig_h) / float(imgsz)

    boxes[:, [0, 2]].clamp_(0, float(orig_w))
    boxes[:, [1, 3]].clamp_(0, float(orig_h))
    return boxes


def _detections_from_nms_tensor(
    det: torch.Tensor,
    *,
    names: Any,
    max_results: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_hw: tuple[int, int],
    imgsz: int,
) -> list[dict[str, Any]]:
    """Convert a small post-NMS CUDA tensor to JSON-safe detection dicts."""

    limit = max(0, int(max_results))
    if det is None or limit == 0 or int(det.numel()) == 0:
        return []

    det = det[:limit]
    boxes = _scale_yolo_boxes_to_original(
        det,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_hw=orig_hw,
        imgsz=imgsz,
    )
    scores = det[:, 4]
    classes = det[:, 5].to(torch.int64)

    # This is the only CPU transfer in the YOLO path, and it copies only the
    # small post-NMS summary, not the camera frame.
    rows = torch.cat(
        [boxes, scores[:, None], classes.to(dtype=boxes.dtype)[:, None]],
        dim=1,
    ).detach().cpu().tolist()

    detections: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row[5])
        detections.append({
            "class_id": class_id,
            "name": _class_name(names, class_id),
            "confidence": round(float(row[4]), 5),
            "xyxy": [round(float(v), 2) for v in row[:4]],
        })
    return detections


def _set_latest_yolo_result(context: Any, result: dict[str, Any]) -> None:
    """Publish tiny YOLO metadata to other async stages in this worker process."""

    try:
        with context.shared_lock:
            context.shared_state["yolo_latest"] = result
    except Exception:
        pass


def _draw_yolo_detections_on_bgr(image: Any, result: dict[str, Any], *, config: dict[str, Any]) -> None:
    """Draw YOLO boxes on a CPU BGR OpenCV preview image.

    YOLO inference remains zero-copy CUDA. This function runs only in the preview
    path after the preview frame has already been converted to a CPU OpenCV image.
    """

    detections = result.get("detections")
    if not isinstance(detections, list) or not detections:
        return

    orig_hw = result.get("orig_hw")
    if not isinstance(orig_hw, (list, tuple)) or len(orig_hw) != 2:
        return

    try:
        orig_h, orig_w = float(orig_hw[0]), float(orig_hw[1])
        view_h, view_w = image.shape[:2]
        if orig_h <= 0 or orig_w <= 0 or view_h <= 0 or view_w <= 0:
            return
        sx = float(view_w) / orig_w
        sy = float(view_h) / orig_h
    except Exception:
        return

    thickness = max(1, int(config.get("preview_yolo_thickness", 2)))
    font_scale = float(config.get("preview_yolo_font_scale", 0.5))
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        if not isinstance(det, dict):
            continue
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
            p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
        except Exception:
            continue

        p1 = (max(0, min(view_w - 1, p1[0])), max(0, min(view_h - 1, p1[1])))
        p2 = (max(0, min(view_w - 1, p2[0])), max(0, min(view_h - 1, p2[1])))
        if p2[0] <= p1[0] or p2[1] <= p1[1]:
            continue

        name = str(det.get("name", det.get("class_id", "object")))
        conf = det.get("confidence")
        label = f"{name} {float(conf):.2f}" if isinstance(conf, (int, float)) else name

        cv2.rectangle(image, p1, p2, color, thickness)
        text_org = (p1[0], max(0, p1[1] - 6))
        cv2.putText(image, label, text_org, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_yolo_debug_preview_from_pair(
    context: Any,
    frame: WorkerFrame,
    result: dict[str, Any],
    *,
    stream: str,
    yolo_downsample: int,
) -> None:
    """Debug-only paired YOLO preview rendered inside UltralyticsYoloProcess.

    This draws detections from the exact same frame/result pair produced by the
    YOLO thread. It intentionally converts that paired frame to a CPU OpenCV
    image only for debugging. Normal YOLO inference remains zero-copy CUDA.
    """

    cfg = context.config
    if not bool(cfg.get("yolo_debug_preview", False)):
        return
    if stream not in frame.tensors:
        return

    display_downsample = max(1, int(cfg.get("yolo_debug_downsample", 1)))
    total_downsample = max(1, int(yolo_downsample)) * display_downsample
    window_name = str(cfg.get("yolo_debug_window_name", f"yolo_debug_{stream}"))

    image = _to_cv_array(
        _first_image(frame.tensors[stream]),
        downsample=total_downsample,
        rgb_to_bgr=(stream == "rgb"),
    )

    # cv2 drawing with tuple colors expects BGR images. Convert grayscale debug
    # views to BGR so boxes/labels are visible on left/right/stereo streams too.
    try:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass

    _draw_yolo_detections_on_bgr(image, result, config=cfg)

    # Add small debug metadata so it is obvious this is the paired YOLO frame,
    # not the async preview thread's latest-frame overlay.
    try:
        label = (
            f"YOLO pair frame={result.get('frame_id')} "
            f"det={len(result.get('detections', []) or [])} "
            f"{float(result.get('latency_ms', 0.0)):.1f}ms"
        )
        cv2.putText(
            image,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(cfg.get("yolo_debug_font_scale", 0.6)),
            (0, 255, 255),
            max(1, int(cfg.get("yolo_debug_text_thickness", 2))),
            cv2.LINE_AA,
        )
    except Exception:
        pass

    cv2.imshow(window_name, image)
    if cv2.waitKey(max(1, int(cfg.get("yolo_debug_wait_key_ms", 1)))) & 0xFF in (ord("q"), 27):
        context.stop_event.set()


@dataclass
class WorkerFrame:
    """One frame passed through the worker-side processing pipeline."""

    frame_id: int
    timestamp_s: float
    tensors: dict[StreamName, Any]


@dataclass
class WorkerContext:
    """Mutable session context shared by worker-side processing steps.

    shared_state/shared_lock are worker-process-local and are safe for the
    async pipeline threads. Use them only for small metadata such as detection
    boxes; do not put CPU images or large CUDA tensors here.
    """

    config: dict[str, Any]
    status_queue: Any
    capture_request_queue: Any
    capture_result_queue: Any
    stop_event: Any
    restart_count: int
    shared_state: dict[str, Any] = field(default_factory=dict)
    shared_lock: threading.RLock = field(default_factory=threading.RLock)


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

    Use this for expensive CUDA/Torch/OpenCV steps that must stay in the same
    process as the CUDA tensors on Jetson, but should not block the camera frame
    acquisition loop.

    The default queue size of 1 implements latest-frame behavior: if the inner
    process is still busy, one stale queued frame is dropped and replaced by the
    newest frame. This is usually the right policy for live preview/inference.
    """

    def __init__(
        self,
        inner: WorkerPipelineProcess,
        *,
        name: str | None = None,
        queue_max_size: int = 1,
        join_timeout_s: float = 2.0,
        poll_timeout_s: float = 0.1,
    ) -> None:
        self.inner = inner
        self.name = name or inner.__class__.__name__
        self.queue_max_size = max(1, int(queue_max_size))
        self.join_timeout_s = max(0.0, float(join_timeout_s))
        self.poll_timeout_s = max(0.01, float(poll_timeout_s))
        self._queue: py_queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self._context: WorkerContext | None = None
        self._sentinel = object()
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def on_start(self, context: WorkerContext) -> None:
        self._context = context
        self._queue = py_queue.Queue(maxsize=self.queue_max_size)
        self._dropped_frames = 0
        self._thread = threading.Thread(
            target=self._run_loop,
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
            self._dropped_frames += 1
        except (py_queue.Empty, Exception):
            pass

        try:
            q.put_nowait(frame)
        except py_queue.Full:
            self._dropped_frames += 1

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

    def _emit_error(self, context: WorkerContext, state: str) -> None:
        _put_drop_oldest(
            context.status_queue,
            {
                "state": state,
                "pipeline": self.name,
                "error": traceback.format_exc(),
                "dropped_frames": self._dropped_frames,
                "timestamp_s": time.time(),
            },
        )

    def _run_loop(self) -> None:
        context = self._context
        q = self._queue
        if context is None or q is None:
            return

        try:
            # Run inner startup in the async thread. This keeps resources such as
            # OpenCV windows or Torch/CUDA stream ownership consistent with the
            # thread that will process frames.
            self.inner.on_start(context)

            while not context.stop_event.is_set():
                try:
                    item = q.get(timeout=self.poll_timeout_s)
                except py_queue.Empty:
                    continue

                if item is self._sentinel:
                    break

                try:
                    self.inner.on_frame(context, item)
                except Exception:
                    self._emit_error(context, "pipeline_error")
        except Exception:
            self._emit_error(context, "pipeline_thread_error")
        finally:
            try:
                self.inner.on_stop(context)
            except Exception:
                self._emit_error(context, "pipeline_stop_error")


class UltralyticsYoloProcess(WorkerPipelineProcess):
    """Run Ultralytics YOLO from existing CUDA frame tensors.

    This intentionally avoids ``model.predict(source=np_array)`` and avoids
    OpenCV/numpy frame conversion. The frame tensor remains in the CUDA-owning
    CameraWorker process, preprocessing is done with torch CUDA ops, the raw
    YOLO model is called directly, and only the small post-NMS detection summary
    is copied to CPU for JSON status output.

    Designed to be wrapped by AsyncWorkerPipelineProcess so model inference does
    not block the camera acquisition loop.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.yolo: Any = None
        self.model: Any = None
        self.non_max_suppression: Any = None
        self.names: Any = {}
        self.device: torch.device | None = None
        self.cuda_stream: torch.cuda.Stream | None = None
        self.model_path = ""
        self.stream = "rgb"
        self.frame_interval = 1

    def on_start(self, context: WorkerContext) -> None:
        cfg = context.config
        self.enabled = bool(cfg.get("yolo_enabled", False))
        if not self.enabled:
            return

        self.model_path = str(cfg.get("yolo_model_path", "yolov8n.pt"))
        self.stream = str(cfg.get("yolo_stream", "rgb"))
        self.frame_interval = max(1, int(cfg.get("yolo_frame_interval", 1)))
        half = bool(cfg.get("yolo_half", True))
        self.device = _cuda_device_from_config(cfg.get("yolo_device", "0"))

        _put_drop_oldest(
            context.status_queue,
            {
                "state": "yolo_starting",
                "yolo_model_path": self.model_path,
                "yolo_stream": self.stream,
                "timestamp_s": time.time(),
                "error": None,
            },
        )

        from ultralytics import YOLO

        try:
            from ultralytics.utils.nms import non_max_suppression
        except Exception:  # older Ultralytics releases
            from ultralytics.utils.ops import non_max_suppression

        self.non_max_suppression = non_max_suppression
        self.yolo = YOLO(self.model_path)
        self.model = self.yolo.model
        self.model.to(self.device)
        self.model.eval()
        if half:
            self.model.half()
        else:
            self.model.float()

        try:
            self.model.fuse()
        except Exception:
            pass

        self.names = getattr(self.yolo, "names", None) or getattr(self.model, "names", {})
        self.cuda_stream = torch.cuda.Stream(device=self.device)

        _put_drop_oldest(
            context.status_queue,
            {
                "state": "yolo_ready",
                "yolo_enabled": True,
                "yolo_model_path": self.model_path,
                "yolo_stream": self.stream,
                "yolo_device": str(self.device),
                "yolo_zero_copy_cuda": True,
                "timestamp_s": time.time(),
                "error": None,
            },
        )

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        if not self.enabled or self.model is None or self.device is None:
            return
        if self.frame_interval > 1 and frame.frame_id % self.frame_interval != 0:
            return

        cfg = context.config
        stream = str(cfg.get("yolo_stream", self.stream))
        if stream not in frame.tensors:
            raise RuntimeError(f"YOLO unsupported stream: {stream!r}")

        imgsz = int(cfg.get("yolo_imgsz", 640))
        conf = float(cfg.get("yolo_conf", 0.25))
        iou = float(cfg.get("yolo_iou", 0.45))
        max_det = int(cfg.get("yolo_max_det", 100))
        max_results = int(cfg.get("yolo_max_results", 20))
        half = bool(cfg.get("yolo_half", True))
        downsample = max(1, int(cfg.get("yolo_downsample", 1)))
        letterbox = bool(cfg.get("yolo_letterbox", True))
        pad_value = float(cfg.get("yolo_pad_value", 114.0 / 255.0))
        input_normalized = _yolo_input_is_normalized(cfg, stream)

        started = time.monotonic()
        cuda_stream = self.cuda_stream

        if cuda_stream is None:
            cuda_stream = torch.cuda.current_stream(device=self.device)

        with torch.inference_mode():
            with torch.cuda.stream(cuda_stream):
                x, orig_hw = _as_bchw_cuda_tensor(
                    frame.tensors[stream].clone(),
                    device=self.device,
                    half=half,
                    input_normalized=input_normalized,
                    downsample=downsample,
                )

                if letterbox:
                    x, scale, pad_x, pad_y, orig_hw = _letterbox_bchw_cuda(
                        x,
                        imgsz=imgsz,
                        pad_value=pad_value,
                    )
                else:
                    x, scale, pad_x, pad_y, orig_hw = _resize_bchw_cuda(x, imgsz=imgsz)

                pred = self.model(x)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]

                nms_result = self.non_max_suppression(
                    pred,
                    conf_thres=conf,
                    iou_thres=iou,
                    max_det=max_det,
                )
                det = nms_result[0] if nms_result else torch.empty((0, 6), device=self.device)

            # We use a dedicated CUDA stream; synchronize it before copying the
            # tiny post-NMS summary to CPU and before reporting latency.
            cuda_stream.synchronize()

            detections = _detections_from_nms_tensor(
                det,
                names=self.names,
                max_results=max_results,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                orig_hw=orig_hw,
                imgsz=imgsz,
            )

        latency_ms = (time.monotonic() - started) * 1000.0
        now_s = time.time()
        yolo_result = {
            "frame_id": frame.frame_id,
            "frame_timestamp_s": frame.timestamp_s,
            "timestamp_s": now_s,
            "latency_ms": round(latency_ms, 2),
            "stream": stream,
            "model_path": self.model_path,
            "device": str(self.device),
            "orig_hw": orig_hw,
            "detections": detections,
        }

        # Debug-only paired preview: draw boxes in this YOLO thread from the
        # same frame/result pair that produced the detections. This may block
        # the YOLO async worker, but it does not block the camera acquisition
        # loop because YOLO itself is already wrapped by AsyncYoloProcess.
        _draw_yolo_debug_preview_from_pair(
            context,
            frame,
            yolo_result,
            stream=stream,
            yolo_downsample=downsample,
        )

        _set_latest_yolo_result(context, yolo_result)

        _put_drop_oldest(
            context.status_queue,
            {
                "state": "yolo",
                "yolo_enabled": True,
                "yolo_model_path": self.model_path,
                "yolo_stream": stream,
                "yolo_device": str(self.device),
                "yolo_zero_copy_cuda": True,
                "yolo_frame_id": frame.frame_id,
                "yolo_frame_timestamp_s": frame.timestamp_s,
                "yolo_timestamp_s": now_s,
                "yolo_latency_ms": round(latency_ms, 2),
                "yolo_num_detections": len(detections),
                "yolo_detections": detections,
                "timestamp_s": now_s,
                "error": None,
            },
        )

    def on_stop(self, context: WorkerContext) -> None:
        if bool(context.config.get("yolo_debug_preview", False)):
            try:
                stream = str(context.config.get("yolo_stream", self.stream))
                window_name = str(context.config.get("yolo_debug_window_name", f"yolo_debug_{stream}"))
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        self.model = None
        self.yolo = None
        self.non_max_suppression = None
        self.cuda_stream = None
        if bool(context.config.get("yolo_empty_cache_on_stop", False)):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class AsyncYoloProcess(AsyncWorkerPipelineProcess):
    """Async latest-frame wrapper for zero-copy CUDA Ultralytics YOLO inference."""

    def __init__(
        self,
        *,
        queue_max_size: int = 1,
        join_timeout_s: float = 2.0,
        poll_timeout_s: float = 0.1,
    ) -> None:
        super().__init__(
            UltralyticsYoloProcess(),
            name="yolo",
            queue_max_size=queue_max_size,
            join_timeout_s=join_timeout_s,
            poll_timeout_s=poll_timeout_s,
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
    """Optional local OpenCV preview windows for RGB/left/right streams.

    This is the only pipeline stage that should own OpenCV HighGUI windows.
    YOLO publishes small detection metadata to ``context.shared_state``; this
    preview stage can optionally overlay that metadata without making the YOLO
    thread call ``cv2.imshow`` / ``cv2.waitKey``.
    """

    window_names: tuple[str, str, str] = (
        "camera_rgb",
        "camera_left",
        "camera_right",
    )

    def _latest_yolo_result(self, context: WorkerContext) -> dict[str, Any] | None:
        try:
            with context.shared_lock:
                result = context.shared_state.get("yolo_latest")
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _make_rgb_preview(self, context: WorkerContext, frame: WorkerFrame) -> Any:
        rgb_step = int(context.config["preview_rgb_downsample"])
        image = _to_cv_array_preview(
            _first_image(frame.tensors["rgb"]),
            downsample=rgb_step,
            rgb_to_bgr=True,
            input_normalized=bool(context.config.get("normalize_rgb", True)),
        )

        if bool(context.config.get("preview_yolo_overlay", True)):
            result = self._latest_yolo_result(context)
            if result is not None and result.get("stream") == "rgb":
                _draw_yolo_detections_on_bgr(image, result, config=context.config)

        return image

    def on_frame(self, context: WorkerContext, frame: WorkerFrame) -> None:
        if not bool(context.config.get("preview", False)):
            return

        stereo_step = int(context.config["preview_stereo_downsample"])
        stereo_normalized = bool(context.config.get("normalize_stereo", True))

        cv2.imshow(self.window_names[0], self._make_rgb_preview(context, frame))
        cv2.imshow(
            self.window_names[1],
            _to_cv_array_preview(
                _first_image(frame.tensors["left"]),
                downsample=stereo_step,
                rgb_to_bgr=False,
                input_normalized=stereo_normalized,
            ),
        )
        cv2.imshow(
            self.window_names[2],
            _to_cv_array_preview(
                _first_image(frame.tensors["right"]),
                downsample=stereo_step,
                rgb_to_bgr=False,
                input_normalized=stereo_normalized,
            ),
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


class AsyncPreviewProcess(AsyncWorkerPipelineProcess):
    """Threaded latest-frame OpenCV preview process for quick testing.

    This is only a thin convenience wrapper around PreviewProcess so the server
    can use AsyncPreviewProcess() directly when config["preview"] is true.
    """

    def __init__(
        self,
        *,
        queue_max_size: int = 1,
        join_timeout_s: float = 2.0,
    ) -> None:
        super().__init__(
            PreviewProcess(),
            name="preview",
            queue_max_size=queue_max_size,
            join_timeout_s=join_timeout_s,
        )


