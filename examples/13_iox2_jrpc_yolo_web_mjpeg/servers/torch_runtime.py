#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

# Keep this if the demo lives in an examples/tests folder next to resultkit.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuda_ipc_runtime import (  # noqa: E402
    Config,
    FpsMeter,
    FramePacer,
    StoppableLoop,
    make_cuda_image_endpoint,
    mat_device,
)
from utils import (  # noqa: E402
    DEFAULT_COLOR_PALETTE_RGB,
    draw_boxes_gpu_with_bitmap_labels,
)


YOLO_TOPIC = "ImageMatCUDAPubSub:yolo"

# Environment overrides:
#   set YOLO_MODEL=yolov8n.pt
#   set YOLO_CONF=0.25
#   set YOLO_IOU=0.45
#   set YOLO_MAX_DET=100
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "100"))


@contextmanager
def pushed_cuda_context(ctx: Any):
    """
    Temporarily make the CUDA primary context current for PyCUDA/resultkit calls.

    Important:
        Do not keep this pushed while running YOLO / torchvision / PyTorch.
        Push only around resultkit CUDA IPC operations.
    """
    ctx.push()
    try:
        yield
    finally:
        ctx.pop()


@dataclass(frozen=True)
class YoloSettings:
    """Runtime settings for model inference and detection serialization."""

    model_name: str = YOLO_MODEL
    confidence: float = YOLO_CONF
    iou: float = YOLO_IOU
    max_detections: int = YOLO_MAX_DET
    stride: int = 32

    @classmethod
    def from_env(cls) -> "YoloSettings":
        return cls(
            model_name=os.environ.get("YOLO_MODEL", YOLO_MODEL),
            confidence=float(os.environ.get("YOLO_CONF", str(YOLO_CONF))),
            iou=float(os.environ.get("YOLO_IOU", str(YOLO_IOU))),
            max_detections=int(os.environ.get("YOLO_MAX_DET", str(YOLO_MAX_DET))),
        )


@dataclass(frozen=True)
class CropRegion:
    """A centered crop whose height and width are YOLO-stride aligned."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    @classmethod
    def centered_stride_crop(cls, height: int, width: int, stride: int) -> "CropRegion":
        crop_h = height - (height % stride)
        crop_w = width - (width % stride)

        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError(
                "image too small after stride crop: "
                f"original shape=({height}, {width}, 3), stride={stride}"
            )

        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return cls(top=top, left=left, bottom=top + crop_h, right=left + crop_w)


@dataclass
class DetectionResult:
    """YOLO boxes in full-frame coordinates."""

    boxes_xyxy: torch.Tensor | None = None
    conf: torch.Tensor | None = None
    cls: torch.Tensor | None = None

    @property
    def is_empty(self) -> bool:
        return self.boxes_xyxy is None or self.conf is None or self.cls is None


class DetectionPixelEncoder:
    """
    Encodes YOLO detections into the first bytes of an RGB output image.

    Layout:
        bytes 0..7    magic: b"YOLORES1"
        bytes 8..9    detection count uint16 little-endian
        bytes 10..15   reserved

    Per detection, 12 bytes:
        x1 uint16
        y1 uint16
        x2 uint16
        y2 uint16
        conf uint16, confidence * 10000
        cls uint16
    """

    MAGIC = b"YOLORES1"
    HEADER_RESERVED_BYTES = 6

    def __init__(self, max_detections: int) -> None:
        self.max_detections = int(max_detections)

    def encode(self, out_img: torch.Tensor, detections: DetectionResult) -> torch.Tensor:
        self._validate_image(out_img)

        if not out_img.is_contiguous():
            out_img = out_img.contiguous()

        payload = self._build_payload(detections)
        flat = out_img.reshape(-1)

        if len(payload) > flat.numel():
            raise RuntimeError(
                f"YOLO encoded payload needs {len(payload)} bytes, "
                f"but output image only has {flat.numel()} bytes"
            )

        payload_np = np.frombuffer(payload, dtype=np.uint8).copy()
        encoded = torch.from_numpy(payload_np).to(device=out_img.device, dtype=torch.uint8)
        flat[: encoded.numel()] = encoded
        return out_img

    @staticmethod
    def uint16_le(value: int) -> bytes:
        return struct.pack("<H", max(0, min(65535, int(value))))

    @staticmethod
    def _validate_image(out_img: torch.Tensor) -> None:
        if out_img.dtype != torch.uint8:
            raise ValueError("out_img must be torch.uint8")

        if out_img.ndim != 3 or int(out_img.shape[-1]) != 3:
            raise ValueError(f"out_img must be HWC RGB, got {tuple(out_img.shape)}")

    def _build_payload(self, detections: DetectionResult) -> bytes:
        if detections.is_empty or detections.boxes_xyxy.numel() == 0:
            return self.MAGIC + self.uint16_le(0) + bytes(self.HEADER_RESERVED_BYTES)

        boxes_cpu = (
            detections.boxes_xyxy[: self.max_detections]
            .detach()
            .round()
            .to(torch.int32)
            .cpu()
        )
        conf_cpu = detections.conf[: self.max_detections].detach().cpu()
        cls_cpu = detections.cls[: self.max_detections].detach().round().to(torch.int32).cpu()
        count = min(int(boxes_cpu.shape[0]), self.max_detections)

        payload = bytearray()
        payload += self.MAGIC
        payload += self.uint16_le(count)
        payload += bytes(self.HEADER_RESERVED_BYTES)

        for i in range(count):
            x1, y1, x2, y2 = [int(v) for v in boxes_cpu[i].tolist()]
            score = int(float(conf_cpu[i]) * 10000.0)
            class_id = int(cls_cpu[i])

            payload += self.uint16_le(x1)
            payload += self.uint16_le(y1)
            payload += self.uint16_le(x2)
            payload += self.uint16_le(y2)
            payload += self.uint16_le(score)
            payload += self.uint16_le(class_id)

        return bytes(payload)


class YoloDetector:
    """Runs YOLO on CUDA HWC RGB images and draws encoded results."""

    LABEL_NAMES = {
        0: "person",
        1: "自行车",      # bicycle - CN
        2: "車",          # car - JP
        3: "오토바이",    # motorcycle - KR
        4: "飞机",        # airplane - CN
        5: "バス",        # bus - JP
        6: "기차",        # train - KR
        7: "卡车",        # truck - CN
        8: "船",          # boat - CN/JP
        9: "신호등",      # traffic light - KR
    }

    def __init__(self, settings: YoloSettings | None = None) -> None:
        self.settings = settings or YoloSettings.from_env()
        self.encoder = DetectionPixelEncoder(self.settings.max_detections)

    def process(self, img_uint8: torch.Tensor) -> torch.Tensor:
        """
        Input:
            CUDA HWC RGB uint8 tensor, shape [H, W, 3].

        Output:
            CUDA HWC RGB uint8 tensor, same original shape.

        Behavior:
            - Crops edges so H and W are divisible by the configured stride.
            - For 720x1280 input, YOLO sees 704x1280 when stride is 32.
            - Boxes are shifted back to full-frame coordinates.
            - Output image keeps the original shape.
        """
        self._validate_input(img_uint8)
        img_uint8 = img_uint8.contiguous()

        device_index = int(img_uint8.device.index or 0)
        model = self._get_model(self.settings.model_name, device_index)

        crop = CropRegion.centered_stride_crop(
            height=int(img_uint8.shape[0]),
            width=int(img_uint8.shape[1]),
            stride=self.settings.stride,
        )
        img_crop = img_uint8[crop.top : crop.bottom, crop.left : crop.right, :].contiguous()
        model_input = self._to_model_input(img_crop)

        result = self._predict(model, model_input, device_index)
        detections = self._extract_detections(result, crop)
        output = self._draw_detections(img_uint8, detections)
        output = self.encoder.encode(output, detections)
        return output.contiguous()

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_model(model_name: str, device_index: int):
        """
        Load the YOLO model once per model/device pair.

        Requires:
            pip install ultralytics opencv-python
        """
        from ultralytics import YOLO

        model = YOLO(model_name)
        model.to(f"cuda:{int(device_index)}")
        return model

    @staticmethod
    def _validate_input(img_uint8: torch.Tensor) -> None:
        if not img_uint8.is_cuda:
            raise RuntimeError("yolo_step expects a CUDA tensor")

        if img_uint8.dtype != torch.uint8:
            raise RuntimeError(f"yolo_step expects torch.uint8, got {img_uint8.dtype}")

        if img_uint8.ndim != 3 or int(img_uint8.shape[-1]) != 3:
            raise RuntimeError(
                f"yolo_step expects HWC RGB image, got shape {tuple(img_uint8.shape)}"
            )

    @staticmethod
    def _to_model_input(img_crop: torch.Tensor) -> torch.Tensor:
        # Ultralytics tensor input must be BCHW and H/W divisible by stride 32.
        # Input is RGB float in range 0..1.
        return (
            img_crop.permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )

    def _predict(self, model: Any, model_input: torch.Tensor, device_index: int) -> Any:
        with torch.inference_mode():
            results = model.predict(
                source=model_input,
                device=f"cuda:{device_index}",
                conf=self.settings.confidence,
                iou=self.settings.iou,
                max_det=self.settings.max_detections,
                verbose=False,
            )
        return results[0]

    @staticmethod
    def _extract_detections(result: Any, crop: CropRegion) -> DetectionResult:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionResult()

        boxes_xyxy = boxes.xyxy.clone()
        boxes_xyxy[:, [0, 2]] += crop.left
        boxes_xyxy[:, [1, 3]] += crop.top
        return DetectionResult(
            boxes_xyxy=boxes_xyxy,
            conf=boxes.conf,
            cls=boxes.cls,
        )

    def _draw_detections(
        self,
        img_uint8: torch.Tensor,
        detections: DetectionResult,
    ) -> torch.Tensor:
        return draw_boxes_gpu_with_bitmap_labels(
            img_uint8,
            boxes_xyxy=detections.boxes_xyxy,
            conf=detections.conf,
            cls=detections.cls,
            names=self.LABEL_NAMES,
            font_scale=2,
            color_rgb=DEFAULT_COLOR_PALETTE_RGB,
        )


class CudaYoloEndpointFactory:
    """Creates resultkit CUDA image endpoints."""

    @staticmethod
    def make_yolo_endpoint(cfg: Config, *, is_pub: bool, output_topic: str = YOLO_TOPIC):
        import pycuda.gpuarray as gpuarray
        from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
        from resultkit.mat import DataType

        if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
            raise RuntimeError(
                "Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build"
            )

        data = gpuarray.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
        img = Model4Mat.ImageMatCUDAPubSub(
            color_format=ColorFormat.RGB,
            shape_type=ImageShapeType.HWC,
            dtype=DataType.UINT8,
            device=mat_device(cfg.device),
            data=data,
            num_slots=int(cfg.num_slots),
        )

        img.set_id(output_topic).init()
        img.init()

        try:
            img.is_pub = bool(is_pub)
        except Exception:
            pass

        return img


class CudaPrimaryContext:
    """Owns the PyCUDA primary context used for resultkit CUDA IPC calls."""

    def __init__(self, device: int) -> None:
        import pycuda.driver as cuda

        cuda.init()
        self.device = int(device)
        self._cuda = cuda
        self._ctx = None

    @property
    def ctx(self) -> Any:
        if self._ctx is None:
            raise RuntimeError("CUDA primary context has not been initialized")
        return self._ctx

    def initialize_for_torch(self) -> None:
        # Initialize PyTorch CUDA first.
        torch.cuda.set_device(self.device)
        torch.empty(1, device=f"cuda:{self.device}")

        # Use CUDA primary context, same context family PyTorch uses.
        self._ctx = self._cuda.Device(self.device).retain_primary_context()

    def detach(self) -> None:
        if self._ctx is not None:
            self._ctx.detach()
            self._ctx = None

    @contextmanager
    def pushed(self):
        with pushed_cuda_context(self.ctx):
            yield



class YoloLoop(StoppableLoop):
    """CUDA IPC image subscriber -> YOLO CUDA image publisher loop."""

    def __init__(
        self,
        cfg: Config,
        *,
        output_topic: str = YOLO_TOPIC,
        detector: YoloDetector | None = None,
        pause_sleep_seconds: float = 0.01,
    ) -> None:
        super().__init__(cfg)
        self.output_topic = output_topic
        self.detector = detector or YoloDetector()
        self._detector_lock = threading.RLock()
        self.pause_sleep_seconds = float(pause_sleep_seconds)
        self._pause_event = threading.Event()
        self.cuda_context = CudaPrimaryContext(cfg.device)
        self.endpoint_factory = CudaYoloEndpointFactory()
        self.pacer = FramePacer(cfg.fps)
        self.meter = FpsMeter("yolo-pub", cfg.stats_every)
        self.image_sub = None
        self.yolo_pub = None
        self.last_sequence = -1
        self.published = 0

    def pause(self) -> None:
        """Pause after the current frame finishes processing."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume frame processing."""
        self._pause_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def change_detector(
        self,
        detector: YoloDetector | None = None,
        *,
        settings: YoloSettings | None = None,
    ) -> YoloDetector:
        """Safely replace the detector used by the running loop.

        The swap is protected by the same lock used during frame processing, so
        this method waits until the current inference finishes before replacing
        ``self.detector``. The next frame will use the new detector.

        Args:
            detector: A fully constructed detector-like object. It must expose a
                callable ``process(frame)`` method.
            settings: Optional settings used to create a new ``YoloDetector``
                when ``detector`` is not provided.

        Returns:
            The detector now installed on this loop.
        """
        if detector is not None and settings is not None:
            raise ValueError("pass either detector or settings, not both")

        new_detector = detector or YoloDetector(settings)
        new_detector._get_model(new_detector.settings.model_name, 0)
        process = getattr(new_detector, "process", None)
        if not callable(process):
            raise TypeError("detector must provide a callable process(frame) method")

        with self._detector_lock:
            self.detector = new_detector
            return self.detector

    def close(self) -> None:
        try:
            with self.cuda_context.pushed():
                self._safe_close(self.yolo_pub)
                self._safe_close(self.image_sub)
        finally:
            self.cuda_context.detach()
            self.yolo_pub = None
            self.image_sub = None

    def _run(self) -> None:
        """Run until stopped. When paused, no new frames are received or published."""
        self.cuda_context.initialize_for_torch()

        try:
            self._open_endpoints()
            self._log_startup()

            while not self._should_stop():
                if self._wait_if_paused():
                    return

                frame = self._receive_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self._process_and_publish(frame)

                if self._reached_max_frames():
                    return
        finally:
            self.close()

    def _open_endpoints(self) -> None:
        # Create resultkit endpoints while PyCUDA primary context is current.
        with self.cuda_context.pushed():
            self.image_sub = make_cuda_image_endpoint(self.cfg, is_pub=False)
            self.yolo_pub = self.endpoint_factory.make_yolo_endpoint(
                self.cfg,
                is_pub=True,
                output_topic=self.output_topic,
            )

    def _log_startup(self) -> None:
        print(
            f"torch-yolo: subscribing {self.cfg.image_topic!r}, "
            f"publishing {self.output_topic!r}",
            flush=True,
        )

    def _receive_frame(self) -> torch.Tensor | None:
        # Resultkit/PyCUDA IPC receive must happen with the PyCUDA context current.
        # CRITICAL: do not let YOLO run directly on the remote IPC tensor. Copy it
        # into local PyTorch CUDA memory immediately, then release the remote tensor
        # before leaving the PyCUDA/resultkit section.
        with self.cuda_context.pushed():
            self.image_sub.sub(copy=False, sync=True)

            if getattr(self.image_sub, "_remote_mem", None) is None:
                return None

            sequence = int(getattr(self.image_sub, "sequence", -1))
            if sequence == self.last_sequence:
                return None

            self.last_sequence = sequence
            remote_t = self.image_sub.get_data_torch(copy=False, sync=False)

            # This clone breaks the lifetime dependency on the IPC memory handle.
            # YOLO/torchvision will only see normal PyTorch CUDA memory, not
            # resultkit's remote IPC slot.
            frame = remote_t.clone(memory_format=torch.contiguous_format)

            # Make the clone complete before resultkit is allowed to open, close,
            # or switch IPC handles on the next iteration.
            torch.cuda.synchronize(self.cfg.device)
            del remote_t
            return frame

    def _process_and_publish(self, frame: torch.Tensor) -> None:
        try:
            # Run YOLO outside the pushed PyCUDA context, using only the local clone.
            # Hold the detector lock for the whole inference so change_detector()
            # cannot replace self.detector while the current frame is using it.
            with self._detector_lock:
                yolo_res = self.detector.process(frame)

            # Make sure PyTorch/YOLO kernels and copies are complete before publishing.
            torch.cuda.synchronize(self.cfg.device)

            # Publish result with PyCUDA/resultkit context current.
            with self.cuda_context.pushed():
                self.yolo_pub.pub(data=yolo_res)

            self.published += 1
            self.meter.tick()
            self.pacer.sleep()
        finally:
            # Drop references aggressively so Python does not hold old IPC-backed
            # tensors across resultkit slot switches.
            del frame
            if "yolo_res" in locals():
                del yolo_res

    def _reached_max_frames(self) -> bool:
        return self.cfg.max_frames is not None and self.published >= self.cfg.max_frames

    def _should_stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _wait_if_paused(self) -> bool:
        """Return True when the loop was stopped while waiting in pause state."""
        while self._pause_event.is_set():
            if self._should_stop():
                return True
            time.sleep(self.pause_sleep_seconds)
        return False

    @staticmethod
    def _safe_close(endpoint: Any) -> None:
        if endpoint is None:
            return
        try:
            endpoint.close()
        except Exception:
            pass
