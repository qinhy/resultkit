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

    model_name: str = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    confidence: float = float(os.environ.get("YOLO_CONF", "0.25"))
    iou: float = float(os.environ.get("YOLO_IOU", "0.45"))
    max_detections: int = int(os.environ.get("YOLO_MAX_DET", "100"))
    stride: int = 32


@dataclass
class DetectionResult:
    """YOLO boxes in full-frame coordinates."""

    boxes_xyxy: torch.Tensor | None = None
    conf: torch.Tensor | None = None
    cls: torch.Tensor | None = None
    count: int = 0

    @property
    def is_empty(self) -> bool:
        return self.count == 0 or self.boxes_xyxy is None or self.conf is None or self.cls is None


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
    HEADER_BYTES = 16
    BYTES_PER_DETECTION = 12

    def __init__(self, max_detections: int) -> None:
        self.max_detections = int(max_detections)

        if self.max_detections < 0:
            raise ValueError("max_detections must be >= 0")

        if self.max_detections > 65535:
            raise ValueError("max_detections must be <= 65535 because count is uint16")

        self._magic_cache = {}

    def encode(self, out_img: torch.Tensor, detections: DetectionResult) -> torch.Tensor:
        self._validate_image(out_img)

        # Fastest option: require contiguous output.
        # Copying a whole image just to encode a few bytes can be expensive.
        if not out_img.is_contiguous():
            raise ValueError("out_img must be contiguous for fast encoding")

        flat = out_img.view(-1)

        count = self._detection_count(detections)
        needed = self.HEADER_BYTES + count * self.BYTES_PER_DETECTION

        if needed > flat.numel():
            raise RuntimeError(
                f"YOLO encoded payload needs {needed} bytes, "
                f"but output image only has {flat.numel()} bytes"
            )

        payload = torch.empty(
            needed,
            device=out_img.device,
            dtype=torch.uint8,
        )

        # Header
        payload[:8] = self._magic_tensor(out_img.device)
        payload[8] = count & 0xFF
        payload[9] = (count >> 8) & 0xFF
        payload[10:16].zero_()

        if count > 0:
            values = self._detections_as_uint16_values(detections, count, out_img.device)

            # Convert uint16-ish int32 values to little-endian bytes:
            # [x1_lo, x1_hi, y1_lo, y1_hi, ...]
            bytes_le = torch.empty(
                (count, 6, 2),
                device=out_img.device,
                dtype=torch.uint8,
            )

            bytes_le[:, :, 0] = torch.bitwise_and(values, 0xFF).to(torch.uint8)
            bytes_le[:, :, 1] = torch.bitwise_right_shift(values, 8).to(torch.uint8)

            payload[16:] = bytes_le.reshape(-1)

        flat[:needed] = payload
        return out_img

    def _magic_tensor(self, device: torch.device) -> torch.Tensor:
        key = str(device)

        cached = self._magic_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                list(self.MAGIC),
                device=device,
                dtype=torch.uint8,
            )
            self._magic_cache[key] = cached

        return cached

    def _detection_count(self, detections: DetectionResult) -> int:
        if detections.is_empty or detections.boxes_xyxy.numel() == 0:
            return 0

        return min(
            int(detections.boxes_xyxy.shape[0]),
            int(detections.conf.shape[0]),
            int(detections.cls.shape[0]),
            self.max_detections,
            65535,
        )

    @staticmethod
    def _detections_as_uint16_values(
        detections: DetectionResult,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        boxes = (
            detections.boxes_xyxy[:count]
            .detach()
            .round()
            .to(device=device, dtype=torch.int32)
            .clamp_(0, 65535)
        )

        conf = detections.conf[:count].detach().to(device=device)
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
        conf = conf.clamp_(0.0, 1.0)
        conf_u16 = (conf * 10000.0).round().to(torch.int32).unsqueeze(1)

        cls = (
            detections.cls[:count]
            .detach()
            .round()
            .to(device=device, dtype=torch.int32)
            .clamp_(0, 65535)
            .unsqueeze(1)
        )

        # Shape: [N, 6]
        # Columns: x1, y1, x2, y2, conf, cls
        return torch.cat([boxes, conf_u16, cls], dim=1)

    @staticmethod
    def _validate_image(out_img: torch.Tensor) -> None:
        if out_img.dtype != torch.uint8:
            raise ValueError("out_img must be torch.uint8")

        if out_img.ndim != 3 or int(out_img.shape[-1]) != 3:
            raise ValueError(f"out_img must be HWC RGB, got {tuple(out_img.shape)}")
        

class DetectionPixelDecoder:
    """
    Decodes YOLO detections from the first bytes of an RGB image.

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
    HEADER_BYTES = 16
    BYTES_PER_DETECTION = 12

    def __init__(self) -> None:
        self._magic_cache = {}

    def decode(self, encoded_img: torch.Tensor) -> DetectionResult:
        self._validate_image(encoded_img)

        # For maximum speed, require contiguous memory.
        # This matches the fast encoder behavior.
        if not encoded_img.is_contiguous():
            raise ValueError("encoded_img must be contiguous for fast decoding")

        flat = encoded_img.view(-1)

        if flat.numel() < self.HEADER_BYTES:
            raise ValueError(
                f"encoded_img is too small: needs at least {self.HEADER_BYTES} bytes"
            )

        self._validate_magic(flat)

        count = self._read_uint16_le(flat, offset=8)

        payload_bytes = self.HEADER_BYTES + count * self.BYTES_PER_DETECTION

        if payload_bytes > flat.numel():
            raise ValueError(
                f"Encoded payload says it needs {payload_bytes} bytes, "
                f"but image only has {flat.numel()} bytes"
            )

        if count == 0:
            return self._empty_result(encoded_img.device)

        body = flat[
            self.HEADER_BYTES : self.HEADER_BYTES + count * self.BYTES_PER_DETECTION
        ]

        # Shape: [N, 6, 2]
        # 6 uint16 fields per detection, 2 little-endian bytes per field.
        byte_pairs = body.view(count, 6, 2).to(torch.int32)

        # uint16 little-endian decode:
        # value = low_byte + high_byte * 256
        values = byte_pairs[:, :, 0] + byte_pairs[:, :, 1] * 256

        boxes_xyxy = values[:, 0:4].to(torch.float32)
        conf = values[:, 4].to(torch.float32) / 10000.0
        cls = values[:, 5].to(torch.int64)

        return DetectionResult(
            boxes_xyxy=boxes_xyxy,
            conf=conf,
            cls=cls,
            count=count,
        )

    def _validate_magic(self, flat: torch.Tensor) -> None:
        expected = self._magic_tensor(flat.device)

        if not torch.equal(flat[:8], expected):
            found = bytes(flat[:8].detach().cpu().tolist())
            raise ValueError(
                f"Invalid YOLO result magic header: expected {self.MAGIC!r}, got {found!r}"
            )

    def _magic_tensor(self, device: torch.device) -> torch.Tensor:
        key = str(device)

        cached = self._magic_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                list(self.MAGIC),
                device=device,
                dtype=torch.uint8,
            )
            self._magic_cache[key] = cached

        return cached

    @staticmethod
    def _read_uint16_le(flat: torch.Tensor, offset: int) -> int:
        low = int(flat[offset].item())
        high = int(flat[offset + 1].item())
        return low | (high << 8)

    @staticmethod
    def _empty_result(device: torch.device) -> DetectionResult:
        return DetectionResult(
            boxes_xyxy=torch.empty((0, 4), device=device, dtype=torch.float32),
            conf=torch.empty((0,), device=device, dtype=torch.float32),
            cls=torch.empty((0,), device=device, dtype=torch.int64),
            count=0,
        )

    @staticmethod
    def _validate_image(encoded_img: torch.Tensor) -> None:
        if encoded_img.dtype != torch.uint8:
            raise ValueError("encoded_img must be torch.uint8")

        if encoded_img.ndim != 3 or int(encoded_img.shape[-1]) != 3:
            raise ValueError(f"encoded_img must be HWC RGB, got {tuple(encoded_img.shape)}")
        

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
        self.settings = settings or YoloSettings()
        self.encoder = DetectionPixelEncoder(self.settings.max_detections)
        self._validated_inputs = {}

    def process(self, img_uint8: torch.Tensor) -> torch.Tensor:
        """
        Input:
            CUDA HWC RGB uint8 tensor, shape [H, W, 3].

        Output:
            CUDA HWC RGB uint8 tensor, same original shape.

        Behavior:
            - Crops only the bottom/right edges so H and W are divisible by stride.
            - For 3040x4032 input, YOLO sees when stride is 32.
            - No box offset is needed because crop starts at top-left corner.
            - Output image keeps the original shape.
        """
        crop_h, crop_w = self._validate_input(img_uint8)
        img_uint8 = img_uint8.contiguous()

        device_index = int(img_uint8.device.index or 0)
        model = self._get_model(self.settings.model_name, device_index)

        img_crop = img_uint8[:crop_h, :crop_w, :].contiguous()
        model_input = self._to_model_input(img_crop)

        result = self._predict(model, model_input, device_index)

        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            detections = DetectionResult()
        else:
            detections = DetectionResult(
                boxes_xyxy=boxes.xyxy,
                conf=boxes.conf,
                cls=boxes.cls,
                count=len(boxes.conf),
            )

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

    def _validate_input(self, img_uint8: torch.Tensor) -> tuple[int, int]:
        key = (tuple(img_uint8.shape), img_uint8.dtype, img_uint8.is_cuda, int(self.settings.stride))

        if key in self._validated_inputs:
            crop_h, crop_w = self._validated_inputs[key]
            return crop_h, crop_w

        if not img_uint8.is_cuda:
            raise RuntimeError("yolo_step expects a CUDA tensor")

        if img_uint8.dtype != torch.uint8:
            raise RuntimeError(f"yolo_step expects torch.uint8, got {img_uint8.dtype}")

        if img_uint8.ndim != 3 or int(img_uint8.shape[-1]) != 3:
            raise RuntimeError(
                f"yolo_step expects HWC RGB image, got shape {tuple(img_uint8.shape)}"
            )

        h = int(img_uint8.shape[0])
        w = int(img_uint8.shape[1])
        stride = int(self.settings.stride)

        crop_h = h - (h % stride)
        crop_w = w - (w % stride)

        self._validated_inputs[key] = (crop_h, crop_w)
        return crop_h, crop_w


    @staticmethod
    def _to_model_input(img_crop: torch.Tensor) -> torch.Tensor:
        # Ultralytics tensor input must be BCHW and H/W divisible by stride 32.
        # Input is RGB float in range 0..1.
        # For YOLO object detection, you normally should not apply ImageNet mean/std
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
    def make_yolo_endpoint(cfg: Config, *, is_pub: bool, output_topic: str = "ImageMatCUDAPubSub:yolo"):
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
        output_topic: str,
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

