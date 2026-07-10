from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file
import torch.nn.functional as F
from ultralytics import YOLO

from common import EmptyParams, HookDispatcher, RpcModel, openapi_doc
from store.custom_record_store import CustomRecord
from resultkit.logger import logger


DEFAULT_SERVICE_NAME = "jrpc"
DEFAULT_CONTROLLER_NAME = "yolo"

logger(
    f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:init] module loaded",
    extra={"module": __name__},
)


SizeMode = Literal["resize", "tiling"]
YoloTask = Literal["detect", "segment"]
MaskEncoding = Literal["rle"]
MaskOrder = Literal["row_major"]
TiledMaskOutput = Literal["tile", "full_image"]


class YoloBaseModel(RpcModel):
    service:str = DEFAULT_SERVICE_NAME


class StartYoloParams(YoloBaseModel):
    """Parameters for one YOLO inference request.
    """

    size_mode: SizeMode = "tiling"
    cuda_device: int = 0
    db_record: CustomRecord = CustomRecord.empty()
    input_jpg_paths: List[str] = [] # field(default_factory=list)
    output_json_paths: List[str] = [] # field(default_factory=list)

    hook_urls:list[list[str]] = [[]]

    
    def model_post_init(self, context):
        if not self.db_record.is_empty():
            input_jpg_paths = [p for p in self.db_record.listup_rgb_image_paths]
            # jpg_parent_names = [p.parent.name for p in input_jpg_paths]
            output_json_paths = [p.parents[2] / "yolo" / p.parent.name /f"{p.stem}.json" for p in input_jpg_paths]

            self.input_jpg_paths = [str(p) for p in input_jpg_paths]
            self.output_json_paths = [str(p) for p in output_json_paths]
            logger(
                f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:params] image paths derived from record",
                extra={
                    "record_id": getattr(self.db_record, "record_id", None),
                    "input_count": len(self.input_jpg_paths),
                    "output_count": len(self.output_json_paths),
                },
            )

        return super().model_post_init(context)


class YoloSettings(YoloBaseModel):
    """Runtime settings for model inference and detection serialization."""

    model_name: str = "yolov8n-seg.pt"

    # Kept as ``imgz`` for compatibility with the original JSON-RPC contract.
    # Ultralytics calls this parameter ``imgsz``.
    imgz: int = Field(default=1280, gt=0)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, gt=0)
    stride: int = Field(default=32, gt=0)
    tile_overlap: int = Field(default=128, ge=0)
    half: bool = True
    include_masks: bool = True
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # In tiling mode, "full_image" writes every final mask on the original
    # image canvas: size=[image_height, image_width], origin_xy=[0, 0].
    tiled_mask_output: TiledMaskOutput = "full_image"
    # Union duplicate instances detected in overlapping tiles before RLE.
    merge_tiled_masks: bool = True
    # Intersection-over-smaller-box threshold used in addition to IoU.
    # This helps join two clipped halves of an object at a tile boundary.
    tile_merge_iom: float = Field(default=0.15, ge=0.0, le=1.0)

    @property
    def imgsz(self) -> int:
        return self.imgz


class YoloResult(StartYoloParams):
    model_config = {"extra": "ignore"}

    running: bool = False
    queued: bool = False
    queue_size: int = 0
    model: YoloSettings | None = None
    error: str | None = None
    last_output_json_path: str | None = None


class YoloTile(BaseModel):
    """Tile bounds for detections produced in tiling mode."""

    model_config = ConfigDict(extra="ignore")

    left: int
    top: int
    right: int
    bottom: int


class YoloInstanceMask(BaseModel):
    """Compact instance mask encoded with run-length encoding.

    ``origin_xy`` tells where this mask canvas starts in the original image.
    In resize mode it is usually ``[0, 0]`` and ``size`` is the full image.
    In tiling mode it is the tile's top-left coordinate and ``size`` is the
    tile canvas size, which avoids writing a full-image mask for every tile.
    Counts use row-major order and always start with the background run.
    """

    model_config = ConfigDict(extra="ignore")

    encoding: MaskEncoding = "rle"
    order: MaskOrder = "row_major"
    size: list[int] = Field(min_length=2, max_length=2)
    origin_xy: list[int] = Field(default_factory=lambda: [0, 0], min_length=2, max_length=2)
    counts: list[int] = Field(default_factory=list)
    area: int = 0
    threshold: float = 0.5


class YoloDetection(BaseModel):
    """One YOLO detection or segmentation instance in original-image coordinates."""

    model_config = ConfigDict(extra="ignore")

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    tile: YoloTile | None = None
    mask: YoloInstanceMask | None = None


class YoloDetectResult(BaseModel):
    """Complete JSON-serializable result returned by ``YoloDetector.detect``."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    input_jpg_path: str
    output_json_path: str
    size_mode: SizeMode
    cuda_device: int
    task: YoloTask = "detect"
    image_width: int
    image_height: int
    detections: list[YoloDetection] = Field(default_factory=list)
    has_masks: bool | None = None
    num_detections: int | None = None
    yolo_config: YoloSettings
    tile_size: int | None = None
    tile_overlap: int | None = None
    tile_count: int | None = None

    def model_post_init(self, context: Any) -> None:
        if self.num_detections is None:
            self.num_detections = len(self.detections)
        if self.has_masks is None:
            self.has_masks = any(det.mask is not None for det in self.detections)

    def to_json_text(self, *, indent: int = 2) -> str:
        """Serialize with ``None`` fields omitted, matching the old payload style."""
        return json.dumps(self.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=indent)


class YoloDetector:
    """High-throughput Torch-first wrapper around Ultralytics YOLO weights.

    Performance-oriented changes compared with the original implementation:

    * JPEG files are decoded with ``torchvision.io.decode_jpeg``. On CUDA this
      uses nvJPEG and returns a CHW uint8 tensor directly on the GPU.
    * The model is moved/cast only when the requested device or precision
      changes, instead of once per image or tile.
    * Letterbox resize, normalization, padding, and tile crops stay in Torch.
    * Tiles are submitted to the model in batches. Configure the batch size by
      adding ``tile_batch_size`` to ``YoloSettings``; otherwise it defaults to 8.
    * Detection tensors are copied to CPU in bulk instead of synchronizing once
      for every scalar.
    * Mask RLE creation uses vectorized run-boundary detection rather than a
      Python loop over every pixel.

    This class assumes the same project-level dependencies and data models as
    the original class: ``YoloSettings``, ``StartYoloParams``,
    ``YoloDetectResult``, ``YoloDetection``, ``YoloInstanceMask``, ``YoloTile``,
    ``YOLO``, ``logger``, ``DEFAULT_SERVICE_NAME``, and
    ``DEFAULT_CONTROLLER_NAME``.
    """

    def __init__(
        self,
        settings: YoloSettings | None = None,
        service_name: str = DEFAULT_SERVICE_NAME,
        controller_name: str = DEFAULT_CONTROLLER_NAME,
    ) -> None:
        self.service_name = service_name
        self.controller_name = controller_name
        self.settings = settings or YoloSettings()

        self._model: Any | None = None
        self._model_name_loaded: str | None = None
        self._model_device: Any | None = None
        self._model_dtype: Any | None = None

        self._names: dict[int, str] = {}
        self._nc: int = 0
        self._mask_coeff_count: int = 0
        self._is_segment_model: bool = False

        self._lock = Lock()
        self._nvjpeg_fallback_warning_emitted = False

        logger(
            f"[{self.service_name}:{self.controller_name}:detector:init] detector initialized",
            extra={
                "model_name": self.settings.model_name,
                "imgsz": self.settings.imgsz,
                "confidence": self.settings.confidence,
                "iou": self.settings.iou,
                "include_masks": self.settings.include_masks,
                "tile_batch_size": self._tile_batch_size(),
            },
        )

    def change_settings(self, settings: YoloSettings) -> None:
        with self._lock:
            previous_model_name = self.settings.model_name
            reload_model = settings.model_name != previous_model_name

            logger(
                f"[{self.service_name}:{self.controller_name}:detector:settings] applying detector settings",
                extra={
                    "previous_model_name": previous_model_name,
                    "model_name": settings.model_name,
                    "reload_model": reload_model,
                    "imgsz": settings.imgsz,
                    "confidence": settings.confidence,
                    "iou": settings.iou,
                    "max_detections": settings.max_detections,
                    "include_masks": settings.include_masks,
                    "tile_batch_size": max(
                        1,
                        int(getattr(settings, "tile_batch_size", 8) or 8),
                    ),
                },
            )

            self.settings = settings

            if reload_model:
                self._model = None
                self._model_name_loaded = None
                self._model_device = None
                self._model_dtype = None
                self._names = {}
                self._nc = 0
                self._mask_coeff_count = 0
                self._is_segment_model = False

            logger(
                f"[{self.service_name}:{self.controller_name}:detector:settings] detector settings applied",
                extra={
                    "model_name": self.settings.model_name,
                    "reload_model": reload_model,
                },
            )

    def _tile_batch_size(self) -> int:
        """Return a safe tile batch size without requiring a schema change."""
        return max(1, int(getattr(self.settings, "tile_batch_size", 8) or 8))

    def _load_model(self) -> Any:
        """Load the underlying PyTorch module from Ultralytics weights."""
        if (
            self._model is not None
            and self._model_name_loaded == self.settings.model_name
        ):
            return self._model

        with self._lock:
            if (
                self._model is not None
                and self._model_name_loaded == self.settings.model_name
            ):
                return self._model

            logger(
                f"[{self.service_name}:{self.controller_name}:model:load] loading YOLO model",
                extra={"model_name": self.settings.model_name},
            )

            try:
                yolo = YOLO(self.settings.model_name)
                model = yolo.model
                model.eval()

                try:
                    model.fuse()
                except Exception as exc:
                    logger(
                        f"[{self.service_name}:{self.controller_name}:model:fuse] model fuse skipped",
                        level="warning",
                        extra={
                            "model_name": self.settings.model_name,
                            "error": str(exc),
                        },
                    )

                names = (
                    getattr(yolo, "names", None)
                    or getattr(model, "names", {})
                    or {}
                )
                if isinstance(names, dict):
                    self._names = {int(k): str(v) for k, v in names.items()}
                else:
                    self._names = {
                        idx: str(name) for idx, name in enumerate(names)
                    }

                head = None
                try:
                    head = model.model[-1]
                except Exception:
                    head = None

                self._nc = int(
                    getattr(head, "nc", len(self._names) or 0) or 0
                )
                if self._nc and not self._names:
                    self._names = {
                        idx: str(idx) for idx in range(self._nc)
                    }

                self._mask_coeff_count = int(
                    getattr(head, "nm", 0) or 0
                )
                self._is_segment_model = bool(
                    self._mask_coeff_count
                    or getattr(yolo, "task", None) == "segment"
                    or getattr(model, "task", None) == "segment"
                    or (
                        head is not None
                        and "Segment" in head.__class__.__name__
                    )
                )
            except Exception:
                logger(
                    f"[{self.service_name}:{self.controller_name}:model:load:error] failed to load YOLO model",
                    level="error",
                    extra={"model_name": self.settings.model_name},
                )
                raise

            self._model = model
            self._model_name_loaded = self.settings.model_name
            self._model_device = None
            self._model_dtype = None

            logger(
                f"[{self.service_name}:{self.controller_name}:model:load] YOLO model loaded",
                extra={
                    "model_name": self.settings.model_name,
                    "class_count": self._nc,
                    "mask_coeff_count": self._mask_coeff_count,
                    "task": (
                        "segment" if self._is_segment_model else "detect"
                    ),
                },
            )

            return self._model

    def _prepare_model(self, device: Any) -> tuple[Any, Any]:
        """Move/cast the model only when device or precision changes."""

        model = self._load_model()
        use_half = bool(self.settings.half and device.type == "cuda")
        dtype = torch.float16 if use_half else torch.float32

        if self._model_device != device or self._model_dtype != dtype:
            with self._lock:
                if self._model_device != device or self._model_dtype != dtype:
                    model.to(device=device, dtype=dtype)
                    model.eval()
                    self._model_device = device
                    self._model_dtype = dtype

                    if device.type == "cuda":
                        # Letterboxed model inputs have a stable spatial size.
                        torch.backends.cudnn.benchmark = True

                    logger(
                        f"[{self.service_name}:{self.controller_name}:model:prepare] model prepared",
                        extra={
                            "device": str(device),
                            "dtype": str(dtype),
                        },
                    )

        return model, dtype

    def detect(self, params: StartYoloParams) -> YoloDetectResult:
        input_count = len(params.input_jpg_paths)
        output_count = len(params.output_json_paths)

        if input_count == 0:
            raise ValueError("At least one input JPEG path is required")
        if input_count != output_count:
            raise ValueError(
                "input_jpg_paths and output_json_paths must have equal lengths: "
                f"{input_count} != {output_count}"
            )

        logger(
            f"[{self.service_name}:{self.controller_name}:detect] inference request started",
            extra={
                "input_count": input_count,
                "output_count": output_count,
                "size_mode": params.size_mode,
                "cuda_device": params.cuda_device,
                "model_name": self.settings.model_name,
                "tile_batch_size": self._tile_batch_size(),
            },
        )

        start_time = time.perf_counter()
        processed_count = 0
        result: YoloDetectResult | None = None

        # Prepare once before entering image/tile loops.
        device = self._device(params.cuda_device)
        self._prepare_model(device)

        for image_path_value, output_json_path_value in zip(
            params.input_jpg_paths,
            params.output_json_paths,
        ):
            image_path = Path(image_path_value)
            output_json_path = Path(output_json_path_value)

            logger(
                f"[{self.service_name}:{self.controller_name}:detect:image] processing image",
                extra={
                    "input_jpg_path": image_path.as_posix(),
                    "output_json_path": output_json_path.as_posix(),
                    "size_mode": params.size_mode,
                    "cuda_device": params.cuda_device,
                },
            )

            if not image_path.is_file():
                logger(
                    f"[{self.service_name}:{self.controller_name}:detect:image] input image is missing",
                    level="warning",
                    extra={"input_jpg_path": image_path.as_posix()},
                )
                raise FileNotFoundError(
                    f"Input image does not exist: {image_path}"
                )

            if params.size_mode == "tiling":
                image_payload = self._detect_tiled(
                    image_path,
                    params.cuda_device,
                )
            else:
                image_payload = self._detect_resized(
                    image_path,
                    params.cuda_device,
                )

            result = YoloDetectResult(
                input_jpg_path=str(image_path),
                output_json_path=str(output_json_path),
                size_mode=params.size_mode,
                cuda_device=params.cuda_device,
                yolo_config=self.settings,
                **image_payload,
            )

            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(
                result.to_json_text(),
                encoding="utf-8",
            )
            processed_count += 1

            logger(
                f"[{self.service_name}:{self.controller_name}:detect:image] inference output written",
                extra={
                    "input_jpg_path": result.input_jpg_path,
                    "output_json_path": result.output_json_path,
                    "task": result.task,
                    "num_detections": result.num_detections,
                    "has_masks": result.has_masks,
                    "tile_count": result.tile_count,
                },
            )

        elapsed = time.perf_counter() - start_time
        # assert result is not None
        seconds_per_imag = elapsed/processed_count if processed_count else 0
        logger(
            f"[{self.service_name}:{self.controller_name}:detect] inference request completed ({seconds_per_imag:.2f} sec/image)",
            extra={
                "processed_count": processed_count,
                "last_output_json_path": result.output_json_path,
                "seconds_per_image": seconds_per_imag,
            },
        )
        return result

    def _device(self, cuda_device: int) -> Any:
        if cuda_device >= 0 and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if cuda_device >= device_count:
                raise ValueError(
                    f"CUDA device {cuda_device} is unavailable; "
                    f"device_count={device_count}"
                )
            return torch.device(f"cuda:{cuda_device}")
        return torch.device("cpu")

    def _decode_jpeg(self, image_path: Path, device: Any) -> Any:
        """Decode a JPEG to a CHW uint8 tensor, using nvJPEG on CUDA.

        ``read_file`` returns the compressed bytes as a CPU tensor, as required
        by ``decode_jpeg``. When ``device`` is CUDA, torchvision asks nvJPEG to
        place the decoded pixels directly on that GPU.
        """
        encoded = read_file(str(image_path))

        try:
            image = decode_jpeg(
                encoded,
                mode=ImageReadMode.RGB,
                device=device,
            )
        except Exception as exc:
            if device.type != "cuda":
                raise

            # Keep the service usable if torchvision was installed without the
            # CUDA image extension or nvJPEG rejects a particular JPEG.
            if not self._nvjpeg_fallback_warning_emitted:
                logger(
                    f"[{self.service_name}:{self.controller_name}:image:decode] nvJPEG decode failed; using torchvision CPU decode fallback",
                    level="warning",
                    extra={
                        "input_jpg_path": image_path.as_posix(),
                        "device": str(device),
                        "error": str(exc),
                    },
                )
                self._nvjpeg_fallback_warning_emitted = True

            image = decode_jpeg(
                encoded,
                mode=ImageReadMode.RGB,
                device="cpu",
            ).to(device=device, non_blocking=True)

        if image.ndim != 3 or int(image.shape[0]) != 3:
            raise RuntimeError(
                "Expected decoded RGB CHW tensor, got shape "
                f"{tuple(image.shape)} for {image_path}"
            )

        return image.contiguous()

    def _predict_batch(
        self,
        images: list[Any],
        cuda_device: int,
    ) -> list[list[YoloDetection]]:
        """Run one batched forward pass for CHW uint8 image tensors."""

        if not images:
            return []

        device = self._device(cuda_device)
        model, dtype = self._prepare_model(device)

        network_inputs: list[Any] = []
        metadata: list[tuple[int, int, float, tuple[int, int]]] = []

        for image in images:
            if image.ndim != 3:
                raise ValueError(
                    f"Expected CHW image tensor, got {tuple(image.shape)}"
                )

            if image.device != device:
                image = image.to(device=device, non_blocking=True)

            original_height = int(image.shape[-2])
            original_width = int(image.shape[-1])
            network_image, gain, pad = self._letterbox_tensor(image, dtype)

            network_inputs.append(network_image)
            metadata.append(
                (original_width, original_height, gain, pad)
            )

        tensor = torch.stack(network_inputs, dim=0).contiguous()
        input_height = int(tensor.shape[-2])
        input_width = int(tensor.shape[-1])

        with torch.inference_mode():
            output = model(tensor)

        predictions, proto = self._unwrap_model_output(output)
        prediction_batch = self._split_prediction_batch(
            predictions,
            expected_batch_size=len(images),
        )
        proto_batch = self._split_proto_batch(
            proto,
            expected_batch_size=len(images),
        )

        batch_results: list[list[YoloDetection]] = []

        for batch_index, prediction in enumerate(prediction_batch):
            original_width, original_height, gain, pad = metadata[batch_index]
            image_proto = proto_batch[batch_index]

            boxes, scores, class_ids, mask_coefficients = (
                self._decode_predictions(
                    predictions=prediction,
                    original_width=original_width,
                    original_height=original_height,
                    gain=gain,
                    pad=pad,
                    proto=image_proto,
                )
            )

            keep = self._torch_class_aware_nms(
                boxes,
                scores,
                class_ids,
                self.settings.iou,
                self.settings.max_detections,
            )

            masks: list[YoloInstanceMask | None] = [
                None
            ] * int(keep.numel())

            if (
                self.settings.include_masks
                and image_proto is not None
                and mask_coefficients is not None
                and keep.numel() > 0
            ):
                masks = self._process_instance_masks(
                    proto=image_proto,
                    mask_coefficients=mask_coefficients[keep],
                    boxes=boxes[keep],
                    original_width=original_width,
                    original_height=original_height,
                    input_width=input_width,
                    input_height=input_height,
                    gain=gain,
                    pad=pad,
                )

            if keep.numel() == 0:
                batch_results.append([])
                continue

            # Three bulk copies instead of repeated .cpu().item() calls.
            kept_boxes = boxes[keep].detach().to("cpu")
            kept_scores = scores[keep].detach().to("cpu")
            kept_class_ids = class_ids[keep].detach().to("cpu")

            detections: list[YoloDetection] = []
            detection_count = int(kept_boxes.shape[0])

            for output_index in range(detection_count):
                class_id = int(kept_class_ids[output_index].item())
                bbox = kept_boxes[output_index].tolist()

                detections.append(
                    YoloDetection(
                        class_id=class_id,
                        class_name=self._names.get(
                            class_id,
                            str(class_id),
                        ),
                        confidence=float(
                            kept_scores[output_index].item()
                        ),
                        bbox_xyxy=[float(value) for value in bbox],
                        mask=(
                            masks[output_index]
                            if output_index < len(masks)
                            else None
                        ),
                    )
                )

            batch_results.append(detections)

        return batch_results

    def _detect_resized(
        self,
        image_path: Path,
        cuda_device: int,
    ) -> dict[str, Any]:
        device = self._device(cuda_device)
        image = self._decode_jpeg(image_path, device)

        height = int(image.shape[-2])
        width = int(image.shape[-1])
        detections = self._predict_batch([image], cuda_device)[0]

        return {
            "task": "segment" if self._is_segment_model else "detect",
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "has_masks": any(det.mask is not None for det in detections),
            "num_detections": len(detections),
        }

    def _detect_tiled(
        self,
        image_path: Path,
        cuda_device: int,
    ) -> dict[str, Any]:
        device = self._device(cuda_device)
        image = self._decode_jpeg(image_path, device)

        height = int(image.shape[-2])
        width = int(image.shape[-1])

        tile_size = self._make_divisible(
            self.settings.imgsz,
            self.settings.stride,
        )
        overlap = min(
            self.settings.tile_overlap,
            max(tile_size - 1, 0),
        )
        step = max(tile_size - overlap, 1)
        tile_batch_size = self._tile_batch_size()

        all_detections: list[YoloDetection] = []
        tile_count = 0

        logger(
            f"[{self.service_name}:{self.controller_name}:detect:tiling] tiled inference started",
            extra={
                "input_jpg_path": image_path.as_posix(),
                "image_width": width,
                "image_height": height,
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "step": step,
                "tile_batch_size": tile_batch_size,
            },
        )

        pending_images: list[Any] = []
        pending_tiles: list[tuple[int, int, int, int]] = []

        def flush_tile_batch() -> None:
            nonlocal tile_count

            if not pending_images:
                return

            batch_detections = self._predict_batch(
                pending_images,
                cuda_device,
            )

            for tile_bounds, tile_detections in zip(
                pending_tiles,
                batch_detections,
            ):
                left, top, right, bottom = tile_bounds
                tile_count += 1

                for det in tile_detections:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    mask = (
                        self._shift_mask_origin(det.mask, left, top)
                        if det.mask is not None
                        else None
                    )

                    all_detections.append(
                        det.model_copy(
                            update={
                                "bbox_xyxy": [
                                    x1 + left,
                                    y1 + top,
                                    x2 + left,
                                    y2 + top,
                                ],
                                "tile": YoloTile(
                                    left=left,
                                    top=top,
                                    right=right,
                                    bottom=bottom,
                                ),
                                "mask": mask,
                            }
                        )
                    )

            pending_images.clear()
            pending_tiles.clear()

        for top in self._tile_positions(height, tile_size, step):
            for left in self._tile_positions(width, tile_size, step):
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)

                # This is a GPU tensor view; the JPEG is not decoded again and
                # no pixel data is copied until preprocessing creates the batch.
                tile = image[:, top:bottom, left:right]
                pending_images.append(tile)
                pending_tiles.append((left, top, right, bottom))

                if len(pending_images) >= tile_batch_size:
                    flush_tile_batch()

        flush_tile_batch()

        if (
            self.settings.include_masks
            and self._is_segment_model
            and self.settings.tiled_mask_output == "full_image"
        ):
            if self.settings.merge_tiled_masks:
                merged = self._merge_tiled_instances(
                    all_detections,
                    image_width=width,
                    image_height=height,
                    iou_threshold=self.settings.iou,
                    iom_threshold=self.settings.tile_merge_iom,
                )
            else:
                merged = self._nms(all_detections, self.settings.iou)
                merged = [
                    det.model_copy(
                        update={
                            "tile": None,
                            "mask": self._mask_to_full_image_rle(
                                det.mask,
                                image_width=width,
                                image_height=height,
                            )
                            if det.mask is not None
                            else None,
                        }
                    )
                    for det in merged
                ]
        else:
            # Detection models, or callers that explicitly request compact
            # tile-local masks, keep the original fast NMS behavior.
            merged = self._nms(all_detections, self.settings.iou)

        logger(
            f"[{self.service_name}:{self.controller_name}:detect:tiling] tiled inference completed",
            extra={
                "input_jpg_path": image_path.as_posix(),
                "tile_count": tile_count,
                "raw_detection_count": len(all_detections),
                "merged_detection_count": len(merged),
            },
        )

        return {
            "task": "segment" if self._is_segment_model else "detect",
            "image_width": width,
            "image_height": height,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "tile_count": tile_count,
            "detections": merged,
            "has_masks": any(det.mask is not None for det in merged),
            "num_detections": len(merged),
        }

    def _letterbox_tensor(
        self,
        image: Any,
        dtype: Any,
    ) -> tuple[Any, float, tuple[int, int]]:
        """Letterbox a CHW uint8 tensor entirely with Torch operations.

        Returns a normalized CHW floating-point tensor with a fixed square
        spatial size, together with the scale and left/top padding needed for
        restoring detection coordinates.
        """

        if image.ndim != 3:
            raise ValueError(
                f"Expected CHW image tensor, got shape {tuple(image.shape)}"
            )

        _, height, width = image.shape
        height = int(height)
        width = int(width)

        img_size = self._make_divisible(
            self.settings.imgsz,
            self.settings.stride,
        )
        gain = min(img_size / width, img_size / height)

        resized_width = int(round(width * gain))
        resized_height = int(round(height * gain))

        pad_w = img_size - resized_width
        pad_h = img_size - resized_height
        pad_left = int(round(pad_w / 2 - 0.1))
        pad_top = int(round(pad_h / 2 - 0.1))
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        tensor = image.to(dtype=dtype)
        tensor.mul_(1.0 / 255.0)

        if (resized_height, resized_width) != (height, width):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if pad_w or pad_h:
            tensor = F.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=114.0 / 255.0,
            )

        return tensor.contiguous(), gain, (pad_left, pad_top)

    @staticmethod
    def _make_divisible(value: int, divisor: int) -> int:
        divisor = max(int(divisor), 1)
        return int(math.ceil(int(value) / divisor) * divisor)

    @staticmethod
    def _tile_positions(
        length: int,
        tile_size: int,
        step: int,
    ) -> list[int]:
        if length <= tile_size:
            return [0]

        positions = list(
            range(0, max(length - tile_size + 1, 1), step)
        )
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _unwrap_model_output(self, output: Any) -> tuple[Any, Any | None]:
        """Return the inference prediction tensor and optional mask prototypes.

        Ultralytics has used more than one segmentation return layout:

        * older YOLOv8: ``(pred, (features, mask_coefficients, proto))``
        * newer releases: ``((pred, proto), raw_predictions)``
        * exported models: commonly ``(pred, proto)``

        Detection models usually return ``(pred, raw_predictions)``.  The old
        implementation assumed that ``output[0]`` was always a tensor, which is
        not true for the newer segmentation layout.
        """
        if not isinstance(output, (tuple, list)):
            return output, None

        target_mask_dim = int(self._mask_coeff_count or 0)

        def find_proto(value: Any) -> Any | None:
            candidates: list[Any] = []
            stack = [value]

            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                    continue
                if isinstance(item, (tuple, list)):
                    stack.extend(item)
                    continue

                shape = getattr(item, "shape", None)
                ndim = getattr(item, "ndim", None)
                if shape is None or ndim != 4 or len(shape) != 4:
                    continue

                channels = int(shape[1])
                if target_mask_dim and channels == target_mask_dim:
                    candidates.append(item)
                elif not target_mask_dim and channels <= 128:
                    candidates.append(item)

            if not candidates:
                return None

            # The prototype tensor is normally the largest HxW 4-D tensor with
            # exactly ``nm`` channels.
            return max(
                candidates,
                key=lambda item: int(item.shape[-2]) * int(item.shape[-1]),
            )

        first = output[0]
        proto = None

        # Newer segmentation layout: ((predictions, proto), raw_predictions)
        if isinstance(first, (tuple, list)):
            if not first:
                raise RuntimeError("YOLO returned an empty primary output")
            predictions = first[0]
            if len(first) > 1:
                proto = find_proto(first[1:])
        else:
            # Detection and older YOLOv8 segmentation layouts.
            predictions = first

        # Older segmentation layout and exported-model fallbacks.
        if proto is None and len(output) > 1:
            proto = find_proto(output[1:])
        if proto is None:
            proto = find_proto(output)

        if getattr(predictions, "ndim", None) not in (2, 3):
            raise RuntimeError(
                "Unable to locate YOLO prediction tensor in model output; "
                f"primary_type={type(first).__name__}, "
                f"prediction_type={type(predictions).__name__}"
            )

        return predictions, proto

    @staticmethod
    def _split_prediction_batch(
        predictions: Any,
        expected_batch_size: int,
    ) -> list[Any]:
        """Split raw predictions without guessing channel orientation."""
        if getattr(predictions, "ndim", None) == 2:
            if expected_batch_size != 1:
                raise RuntimeError(
                    "Model returned unbatched predictions for batch size "
                    f"{expected_batch_size}: {tuple(predictions.shape)}"
                )
            return [predictions]

        if getattr(predictions, "ndim", None) != 3:
            raise RuntimeError(
                "Unexpected YOLO prediction shape: "
                f"{tuple(getattr(predictions, 'shape', ())) }"
            )

        if int(predictions.shape[0]) != expected_batch_size:
            raise RuntimeError(
                "YOLO output batch mismatch: "
                f"expected={expected_batch_size}, "
                f"actual={int(predictions.shape[0])}, "
                f"shape={tuple(predictions.shape)}"
            )

        return [predictions[index] for index in range(expected_batch_size)]

    @staticmethod
    def _split_proto_batch(
        proto: Any | None,
        expected_batch_size: int,
    ) -> list[Any | None]:
        if proto is None:
            return [None] * expected_batch_size

        if getattr(proto, "ndim", None) == 3:
            if expected_batch_size != 1:
                raise RuntimeError(
                    "Model returned unbatched mask prototypes for batch size "
                    f"{expected_batch_size}: {tuple(proto.shape)}"
                )
            return [proto]

        if getattr(proto, "ndim", None) != 4:
            return [None] * expected_batch_size

        if int(proto.shape[0]) != expected_batch_size:
            raise RuntimeError(
                "YOLO prototype batch mismatch: "
                f"expected={expected_batch_size}, "
                f"actual={int(proto.shape[0])}, "
                f"shape={tuple(proto.shape)}"
            )

        return [proto[index] for index in range(expected_batch_size)]

    def _decode_predictions(
        self,
        predictions: Any,
        original_width: int,
        original_height: int,
        gain: float,
        pad: tuple[int, int],
        proto: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:

        pred = predictions[0] if predictions.ndim == 3 else predictions

        if pred.ndim != 2:
            raise RuntimeError(
                f"Unexpected YOLO output shape: {tuple(predictions.shape)}"
            )

        nc = int(self._nc or len(self._names) or 0)
        proto_mask_dim = (
            int(proto.shape[0])
            if proto is not None and getattr(proto, "ndim", 0) == 3
            else int(proto.shape[1])
            if proto is not None and getattr(proto, "ndim", 0) == 4
            else 0
        )
        mask_dim = int(self._mask_coeff_count or proto_mask_dim or 0)

        possible_channels = {6}
        if nc:
            possible_channels.update({4 + nc, 5 + nc})
            if mask_dim:
                possible_channels.update(
                    {
                        4 + nc + mask_dim,
                        5 + nc + mask_dim,
                    }
                )

        channels_first_raw = (
            pred.shape[0] in possible_channels
            and pred.shape[1] not in possible_channels
        )
        unknown_names_channels_first = (
            nc == 0
            and pred.shape[0] < pred.shape[1]
            and pred.shape[0] <= 512
        )
        if channels_first_raw or unknown_names_channels_first:
            pred = pred.transpose(0, 1).contiguous()

        channels = int(pred.shape[-1])
        if channels < 6:
            empty = torch.empty((0,), device=pred.device)
            return (
                torch.empty((0, 4), device=pred.device),
                empty,
                empty.long(),
                None,
            )

        boxes: Any
        scores: Any
        class_ids: Any
        mask_coefficients = None

        if nc and mask_dim and channels == 4 + nc + mask_dim:
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4 : 4 + nc].max(dim=1)
            mask_coefficients = pred[
                :, 4 + nc : 4 + nc + mask_dim
            ]
        elif nc and mask_dim and channels == 5 + nc + mask_dim:
            boxes = self._xywh_to_xyxy(pred[:, :4])
            class_scores, class_ids = pred[:, 5 : 5 + nc].max(dim=1)
            scores = pred[:, 4] * class_scores
            mask_coefficients = pred[
                :, 5 + nc : 5 + nc + mask_dim
            ]
        elif nc and channels == 4 + nc:
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4:].max(dim=1)
        elif nc and channels == 5 + nc:
            boxes = self._xywh_to_xyxy(pred[:, :4])
            class_scores, class_ids = pred[:, 5:].max(dim=1)
            scores = pred[:, 4] * class_scores
        elif channels == 6:
            boxes = pred[:, :4].clone()
            scores = pred[:, 4]
            class_ids = pred[:, 5].long()
        elif mask_dim and channels > 4 + mask_dim:
            inferred_nc = channels - 4 - mask_dim
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[
                :, 4 : 4 + inferred_nc
            ].max(dim=1)
            mask_coefficients = pred[
                :, 4 + inferred_nc : 4 + inferred_nc + mask_dim
            ]
        else:
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4:].max(dim=1)

        keep = scores >= self.settings.confidence
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep].long()
        if mask_coefficients is not None:
            mask_coefficients = mask_coefficients[keep]

        if boxes.numel() == 0:
            return (
                boxes.reshape(0, 4),
                scores,
                class_ids,
                mask_coefficients,
            )

        boxes = self._scale_boxes_from_letterbox(
            boxes,
            original_width,
            original_height,
            gain,
            pad,
        )
        return boxes, scores, class_ids, mask_coefficients

    @staticmethod
    def _xywh_to_xyxy(boxes: Any) -> Any:
        converted = boxes.clone()
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    @staticmethod
    def _scale_boxes_from_letterbox(
        boxes: Any,
        original_width: int,
        original_height: int,
        gain: float,
        pad: tuple[int, int],
    ) -> Any:
        pad_left, pad_top = pad
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes[:, :4] /= gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0,
            original_width,
        )
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0,
            original_height,
        )
        return boxes

    def _process_instance_masks(
        self,
        proto: Any,
        mask_coefficients: Any,
        boxes: Any,
        original_width: int,
        original_height: int,
        input_width: int,
        input_height: int,
        gain: float,
        pad: tuple[int, int],
    ) -> list[YoloInstanceMask | None]:
        """Convert YOLO segmentation prototypes and coefficients to RLE."""

        if mask_coefficients.numel() == 0:
            return []

        if proto.ndim == 4:
            proto = proto[0]
        if proto.ndim != 3:
            logger(
                f"[{self.service_name}:{self.controller_name}:mask:decode] unexpected prototype shape",
                level="warning",
                extra={"proto_shape": tuple(proto.shape)},
            )
            return [None] * int(mask_coefficients.shape[0])

        proto = proto.float()
        mask_coefficients = mask_coefficients.float()
        mask_dim, proto_height, proto_width = proto.shape

        if int(mask_coefficients.shape[1]) != int(mask_dim):
            logger(
                f"[{self.service_name}:{self.controller_name}:mask:decode] mask coefficient mismatch",
                level="warning",
                extra={
                    "coefficient_shape": tuple(mask_coefficients.shape),
                    "proto_shape": tuple(proto.shape),
                },
            )
            return [None] * int(mask_coefficients.shape[0])

        masks = mask_coefficients @ proto.reshape(mask_dim, -1)
        masks = masks.reshape(
            -1,
            proto_height,
            proto_width,
        ).sigmoid()

        masks = F.interpolate(
            masks[:, None],
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        pad_left, pad_top = pad
        resized_width = int(round(original_width * gain))
        resized_height = int(round(original_height * gain))
        crop_left = max(pad_left, 0)
        crop_top = max(pad_top, 0)
        crop_right = min(crop_left + resized_width, input_width)
        crop_bottom = min(crop_top + resized_height, input_height)

        if crop_right <= crop_left or crop_bottom <= crop_top:
            return [None] * int(mask_coefficients.shape[0])

        masks = masks[
            :,
            crop_top:crop_bottom,
            crop_left:crop_right,
        ]

        if masks.shape[-2:] != (original_height, original_width):
            masks = F.interpolate(
                masks[:, None],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        binary_masks = masks >= self.settings.mask_threshold
        binary_masks = self._crop_masks_to_boxes(binary_masks, boxes)

        # Transfer the complete mask batch once.
        cpu_masks = binary_masks.detach().to("cpu")
        return [self._binary_mask_to_rle(mask) for mask in cpu_masks]

    @staticmethod
    def _crop_masks_to_boxes(masks: Any, boxes: Any) -> Any:
        """Zero out mask pixels outside each detection box."""

        if masks.numel() == 0:
            return masks

        n, height, width = masks.shape
        x = torch.arange(
            width,
            device=masks.device,
        ).view(1, 1, width)
        y = torch.arange(
            height,
            device=masks.device,
        ).view(1, height, 1)

        x1 = boxes[:, 0].floor().view(n, 1, 1)
        y1 = boxes[:, 1].floor().view(n, 1, 1)
        x2 = boxes[:, 2].ceil().view(n, 1, 1)
        y2 = boxes[:, 3].ceil().view(n, 1, 1)

        inside = (
            (x >= x1)
            & (x < x2)
            & (y >= y1)
            & (y < y2)
        )
        return masks & inside

    def _binary_mask_to_rle(self, mask: Any) -> YoloInstanceMask:
        """Encode a boolean CPU mask as row-major uncompressed RLE."""

        flat = mask.reshape(-1).to(dtype=torch.uint8)
        pixel_count = int(flat.numel())

        if pixel_count == 0:
            counts = [0]
            area = 0
        else:
            changes = torch.nonzero(
                flat[1:] != flat[:-1],
                as_tuple=False,
            ).flatten() + 1

            boundaries = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long),
                    changes.to(dtype=torch.long),
                    torch.tensor([pixel_count], dtype=torch.long),
                )
            )
            counts = (boundaries[1:] - boundaries[:-1]).tolist()

            # Uncompressed COCO-style RLE starts with a zero run. If the first
            # pixel is foreground, prepend an empty background run.
            if int(flat[0].item()) == 1:
                counts.insert(0, 0)

            area = int(flat.sum().item())

        height = int(mask.shape[0])
        width = int(mask.shape[1])

        return YoloInstanceMask(
            size=[height, width],
            origin_xy=[0, 0],
            counts=[int(value) for value in counts],
            area=area,
            threshold=float(self.settings.mask_threshold),
        )

    @staticmethod
    def _shift_mask_origin(
        mask: YoloInstanceMask,
        dx: int,
        dy: int,
    ) -> YoloInstanceMask:
        """Shift a tile-local mask origin into original-image coordinates."""
        ox, oy = mask.origin_xy
        return mask.model_copy(
            update={"origin_xy": [int(ox + dx), int(oy + dy)]}
        )

    @staticmethod
    def _rle_to_binary_mask(mask: YoloInstanceMask) -> Any:
        """Decode this service's row-major uncompressed RLE on CPU."""
        height, width = (int(mask.size[0]), int(mask.size[1]))
        pixel_count = height * width
        counts = torch.as_tensor(mask.counts, dtype=torch.long)

        if counts.numel() == 0:
            return torch.zeros((height, width), dtype=torch.bool)
        if bool((counts < 0).any()):
            raise ValueError("RLE counts must be non-negative")
        if int(counts.sum().item()) != pixel_count:
            raise ValueError(
                "RLE count total does not match mask size: "
                f"sum={int(counts.sum().item())}, expected={pixel_count}"
            )

        # Runs alternate background/foreground and always begin with
        # background. repeat_interleave avoids looping over every pixel.
        values = torch.arange(counts.numel(), dtype=torch.long).remainder(2)
        flat = torch.repeat_interleave(values, counts).to(torch.bool)
        return flat.reshape(height, width)

    @classmethod
    def _paste_instance_mask(
        cls,
        canvas: Any,
        mask: YoloInstanceMask,
    ) -> None:
        """OR a compact mask into its original-image position."""
        local = cls._rle_to_binary_mask(mask)
        canvas_height, canvas_width = canvas.shape
        origin_x, origin_y = (int(mask.origin_xy[0]), int(mask.origin_xy[1]))
        mask_height, mask_width = local.shape

        dst_left = max(origin_x, 0)
        dst_top = max(origin_y, 0)
        dst_right = min(origin_x + mask_width, canvas_width)
        dst_bottom = min(origin_y + mask_height, canvas_height)
        if dst_right <= dst_left or dst_bottom <= dst_top:
            return

        src_left = dst_left - origin_x
        src_top = dst_top - origin_y
        src_right = src_left + (dst_right - dst_left)
        src_bottom = src_top + (dst_bottom - dst_top)
        canvas[dst_top:dst_bottom, dst_left:dst_right] |= local[
            src_top:src_bottom, src_left:src_right
        ]

    def _mask_to_full_image_rle(
        self,
        mask: YoloInstanceMask,
        image_width: int,
        image_height: int,
    ) -> YoloInstanceMask:
        """Place one tile-local mask on a full original-image canvas."""
        canvas = torch.zeros(
            (int(image_height), int(image_width)),
            dtype=torch.bool,
        )
        self._paste_instance_mask(canvas, mask)
        return self._binary_mask_to_rle(canvas)

    @staticmethod
    def _box_iou_and_iom(
        box: Any,
        boxes: Any,
    ) -> tuple[Any, Any]:
        """Return IoU and intersection-over-smaller-area for one vs many."""
        xx1 = torch.maximum(box[0], boxes[:, 0])
        yy1 = torch.maximum(box[1], boxes[:, 1])
        xx2 = torch.minimum(box[2], boxes[:, 2])
        yy2 = torch.minimum(box[3], boxes[:, 3])
        intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

        box_area = (box[2] - box[0]).clamp(min=0) * (
            box[3] - box[1]
        ).clamp(min=0)
        boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp(min=0)
        union = box_area + boxes_area - intersection
        smaller = torch.minimum(box_area.expand_as(boxes_area), boxes_area)
        iou = intersection / union.clamp(min=1e-7)
        iom = intersection / smaller.clamp(min=1e-7)
        return iou, iom

    def _merge_tiled_instances(
        self,
        detections: list[YoloDetection],
        image_width: int,
        image_height: int,
        iou_threshold: float,
        iom_threshold: float,
    ) -> list[YoloDetection]:
        """Merge duplicate tiled instances and emit full-image-coordinate masks.

        Detections are clustered class-aware, beginning with the highest
        confidence item. Normal IoU catches standard duplicates; IoM also
        catches clipped halves whose overlap is only the tile-overlap strip.
        Every cluster's masks are OR'ed on one original-image canvas.
        """
        if not detections:
            return []

        boxes = torch.tensor(
            [det.bbox_xyxy for det in detections], dtype=torch.float32
        )
        scores = torch.tensor(
            [det.confidence for det in detections], dtype=torch.float32
        )
        class_ids = torch.tensor(
            [det.class_id for det in detections], dtype=torch.long
        )
        remaining = scores.argsort(descending=True)
        merged: list[YoloDetection] = []

        while remaining.numel() > 0:
            seed_index = int(remaining[0].item())
            candidate_indices = remaining
            candidate_boxes = boxes[candidate_indices]
            iou, iom = self._box_iou_and_iom(
                boxes[seed_index], candidate_boxes
            )
            same_class = (
                class_ids[candidate_indices] == class_ids[seed_index]
            )
            belongs = same_class & (
                (iou >= float(iou_threshold))
                | (iom >= float(iom_threshold))
            )
            # The seed always belongs, including degenerate-box edge cases.
            belongs[0] = True
            cluster_indices = candidate_indices[belongs]
            remaining = candidate_indices[~belongs]

            representative = detections[seed_index]
            canvas = torch.zeros(
                (int(image_height), int(image_width)),
                dtype=torch.bool,
            )
            cluster_boxes = []
            mask_found = False

            for index_tensor in cluster_indices:
                index = int(index_tensor.item())
                det = detections[index]
                cluster_boxes.append(det.bbox_xyxy)
                if det.mask is not None:
                    self._paste_instance_mask(canvas, det.mask)
                    mask_found = True

            cluster_box_tensor = torch.tensor(
                cluster_boxes, dtype=torch.float32
            )
            union_box = [
                float(cluster_box_tensor[:, 0].min().clamp(0, image_width).item()),
                float(cluster_box_tensor[:, 1].min().clamp(0, image_height).item()),
                float(cluster_box_tensor[:, 2].max().clamp(0, image_width).item()),
                float(cluster_box_tensor[:, 3].max().clamp(0, image_height).item()),
            ]

            full_mask = (
                self._binary_mask_to_rle(canvas) if mask_found else None
            )
            merged.append(
                representative.model_copy(
                    update={
                        "bbox_xyxy": union_box,
                        "tile": None,
                        "mask": full_mask,
                    }
                )
            )

        return merged

    @staticmethod
    def _torch_class_aware_nms(
        boxes: Any,
        scores: Any,
        class_ids: Any,
        iou_threshold: float,
        max_detections: int,
    ) -> Any:

        if boxes.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=boxes.device,
            )

        try:
            from torchvision.ops import batched_nms

            keep = batched_nms(
                boxes,
                scores,
                class_ids,
                iou_threshold,
            )
        except Exception:
            keep_parts = []
            for cls in class_ids.unique():
                cls_indices = torch.where(class_ids == cls)[0]
                cls_keep_local = YoloDetector._torch_nms_single_class(
                    boxes[cls_indices],
                    scores[cls_indices],
                    iou_threshold,
                )
                keep_parts.append(cls_indices[cls_keep_local])

            if not keep_parts:
                return torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=boxes.device,
                )

            keep = torch.cat(keep_parts)
            keep = keep[scores[keep].argsort(descending=True)]

        return keep[:max_detections]

    @staticmethod
    def _torch_nms_single_class(
        boxes: Any,
        scores: Any,
        iou_threshold: float,
    ) -> Any:

        if boxes.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=boxes.device,
            )

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (
            (x2 - x1).clamp(min=0)
            * (y2 - y1).clamp(min=0)
        )
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[current], x1[rest])
            yy1 = torch.maximum(y1[current], y1[rest])
            xx2 = torch.minimum(x2[current], x2[rest])
            yy2 = torch.minimum(y2[current], y2[rest])

            inter_w = (xx2 - xx1).clamp(min=0)
            inter_h = (yy2 - yy1).clamp(min=0)
            intersection = inter_w * inter_h
            union = areas[current] + areas[rest] - intersection
            iou = intersection / union.clamp(min=1e-7)
            order = rest[iou <= iou_threshold]

        if keep:
            return torch.stack(keep)
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device,
        )

    @staticmethod
    def _nms(
        detections: list[YoloDetection],
        iou_threshold: float,
    ) -> list[YoloDetection]:
        """Class-aware NMS for detections merged across overlapping tiles."""
        if not detections:
            return []


        boxes = torch.tensor(
            [det.bbox_xyxy for det in detections],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [det.confidence for det in detections],
            dtype=torch.float32,
        )
        class_ids = torch.tensor(
            [det.class_id for det in detections],
            dtype=torch.long,
        )

        keep = YoloDetector._torch_class_aware_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold,
            max_detections=len(detections),
        )
        return [detections[index] for index in keep.tolist()]


class YoloRunner:
    """Background worker that consumes StartYoloParams from a queue."""

    def __init__(
        self,
        input_queue: Queue[StartYoloParams] | None = None,
        detector: YoloDetector | None = None,
        service_name: str = DEFAULT_SERVICE_NAME,
        controller_name: str = DEFAULT_CONTROLLER_NAME,
    ) -> None:
        self.service_name = service_name
        self.controller_name = controller_name
        self.input_queue: Queue[StartYoloParams] = input_queue or Queue()
        self.detector = detector or YoloDetector(
            service_name=self.service_name,
            controller_name=self.controller_name,
        )
        self.exception: Exception | None = None
        self.last_output_json_path: str | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._hooks: HookDispatcher = HookDispatcher()
        logger(
            f"[{self.service_name}:{self.controller_name}:runner:init] runner initialized",
            extra={
                "queue_size": self.input_queue.qsize(),
                "model_name": self.detector.settings.model_name,
            },
        )

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                logger(
                    f"[{self.service_name}:{self.controller_name}:runner:start] runner already active",
                    extra={"queue_size": self.input_queue.qsize()},
                )
                return

            logger(
                f"[{self.service_name}:{self.controller_name}:runner:start] starting runner",
                extra={"queue_size": self.input_queue.qsize()},
            )
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="YoloRunner", daemon=True)
            self._thread.start()
            logger(
                f"[{self.service_name}:{self.controller_name}:runner:start] runner thread started",
                extra={"thread_name": self._thread.name},
            )

    def stop(self, timeout: float = 5.0) -> None:
        logger(
            f"[{self.service_name}:{self.controller_name}:runner:stop] stopping runner",
            extra={"timeout_s": timeout, "queue_size": self.input_queue.qsize()},
        )
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

        if thread and thread.is_alive():
            logger(
                f"[{self.service_name}:{self.controller_name}:runner:stop] runner did not stop before timeout",
                level="warning",
                extra={"timeout_s": timeout, "queue_size": self.input_queue.qsize()},
            )
        else:
            logger(
                f"[{self.service_name}:{self.controller_name}:runner:stop] runner stopped",
                extra={"queue_size": self.input_queue.qsize()},
            )

    def change_detector(self, settings: YoloSettings) -> None:
        self.detector.change_settings(settings)

    def _run(self) -> None:
        logger(
            f"[{self.service_name}:{self.controller_name}:runner:loop] worker loop started",
            extra={"thread_name": self._thread.name if self._thread is not None else "YoloRunner"},
        )
        while not self._stop_event.is_set():
            try:
                params = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            logger(
                f"[{self.service_name}:{self.controller_name}:runner:request] inference request dequeued",
                extra={
                    "queue_size": self.input_queue.qsize(),
                    "input_count": len(params.input_jpg_paths),
                    "output_count": len(params.output_json_paths),
                    "size_mode": params.size_mode,
                    "cuda_device": params.cuda_device,
                },
            )

            try:
                result = self.detector.detect(params)
                if params.hook_urls and params.hook_urls[0]:
                    logger(
                        f"[{self.service_name}:{self.controller_name}:runner:hooks] dispatching hooks",
                        extra={
                            "output_json_path": result.output_json_path,
                            "hook_chains": params.hook_urls,
                        },
                    )
                    self._hooks.dispatch(
                        db_record=params.db_record,
                        hook_chains=params.hook_urls,
                    )
                    logger(
                        f"[{self.service_name}:{self.controller_name}:runner:hooks] hooks dispatched",
                        extra={"output_json_path": result.output_json_path},
                    )

                self.last_output_json_path = result.output_json_path
                self.exception = None
                logger(
                    f"[{self.service_name}:{self.controller_name}:runner:request] inference request completed",
                    extra={
                        "output_json_path": result.output_json_path,
                        "num_detections": result.num_detections,
                        "task": result.task,
                        "queue_size": self.input_queue.qsize(),
                    },
                )
            except Exception as exc:  # Keep the worker alive after bad requests.
                self.exception = exc
                logger(f"[{self.service_name}:{self.controller_name}:runner:request:error] inference request failed",level="error",
                    extra={
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "input_jpg_paths": params.input_jpg_paths,
                        "output_json_paths": params.output_json_paths,
                        "size_mode": params.size_mode,
                        "cuda_device": params.cuda_device,
                    },
                )
            finally:
                self.input_queue.task_done()

        logger(
            f"[{self.service_name}:{self.controller_name}:runner:loop] worker loop stopped",
            extra={"queue_size": self.input_queue.qsize()},
        )


@dataclass
class YoloController:
    running: bool = False
    yolo_runner: YoloRunner | None = None
    yolo_input_queue: Queue[StartYoloParams] = field(default_factory=Queue)
    service_name: str = DEFAULT_SERVICE_NAME
    controller_name: str = DEFAULT_CONTROLLER_NAME

    def __post_init__(self) -> None:
        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(
                input_queue=self.yolo_input_queue,
                service_name=self.service_name,
                controller_name=self.controller_name,
            )
        else:
            self.yolo_input_queue = self.yolo_runner.input_queue

        logger(
            f"[{self.service_name}:{self.controller_name}:init] controller initialized",
            extra={
                "queue_size": self.yolo_input_queue.qsize(),
                "runner_active": self.yolo_runner.is_running,
                "model_name": self.yolo_runner.detector.settings.model_name,
            },
        )

    @staticmethod
    def openapi_examples() -> dict[str, Any]:
        return {
            **openapi_doc("yolo_status", id=1, params={}),
            **openapi_doc("yolo_start", id=2, params=StartYoloParams().model_dump()),
            **openapi_doc("yolo_stop", id=3, params={}),
            **openapi_doc("yolo_set_model", id=4, params=YoloSettings().model_dump()),
        }

    def _result(self, params: StartYoloParams | None = None, err: str | None = None) -> YoloResult:
        runner = self.yolo_runner
        model = runner.detector.settings if runner is not None else None
        running = runner.is_running if runner is not None else False
        last_output = runner.last_output_json_path if runner is not None else None
        runner_error = None

        if runner is not None and runner.exception is not None:
            runner_error = f"{runner.exception.__class__.__name__}: {runner.exception}"

        return YoloResult(
            running=running,
            queued=False,
            queue_size=self.yolo_input_queue.qsize(),
            model=model,
            error=err or runner_error,
            input_jpg_paths=params.input_jpg_paths if params else [""],
            size_mode=params.size_mode if params else "tiling",
            cuda_device=params.cuda_device if params else 0,
            last_output_json_path=last_output,
        )

    def start(self, params: StartYoloParams) -> YoloResult:
        logger(
            f"[{self.service_name}:{self.controller_name}:start] inference requested",
            extra={
                "input_jpg_paths": params.input_jpg_paths,
                "output_json_paths": params.output_json_paths,
                "size_mode": params.size_mode,
                "cuda_device": params.cuda_device,
                "queue_size": self.yolo_input_queue.qsize(),
            },
        )

        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(
                input_queue=self.yolo_input_queue,
                service_name=self.service_name,
                controller_name=self.controller_name,
            )

        if not self.yolo_runner.is_running:
            self.yolo_runner.start()

        self.running = self.yolo_runner.is_running
        self.yolo_input_queue.put(params)

        result = self._result(params=params)
        result.queued = True
        logger(
            f"[{self.service_name}:{self.controller_name}:start] inference queued",
            extra={
                "running": result.running,
                "queue_size": result.queue_size,
                "input_count": len(params.input_jpg_paths),
            },
        )
        return result

    def stop(self, params: EmptyParams) -> YoloResult:
        logger(
            f"[{self.service_name}:{self.controller_name}:stop] stop requested",
            extra={"queue_size": self.yolo_input_queue.qsize()},
        )
        if self.yolo_runner is not None:
            self.yolo_runner.stop()

        self.running = False
        result = self._result()
        logger(
            f"[{self.service_name}:{self.controller_name}:stop] controller stopped",
            extra={
                "running": result.running,
                "queue_size": result.queue_size,
            },
        )
        return result

    def status(self, params: EmptyParams) -> YoloResult:
        result = self._result()
        logger(
            f"[{self.service_name}:{self.controller_name}:status] status requested",
            extra={
                "running": result.running,
                "queue_size": result.queue_size,
                "model_name": result.model.model_name if result.model is not None else None,
                "has_error": result.error is not None,
                "last_output_json_path": result.last_output_json_path,
            },
        )
        return result

    def set_model(self, params: YoloSettings) -> YoloResult:
        logger(
            f"[{self.service_name}:{self.controller_name}:set_model] model settings requested",
            extra={
                "model_name": params.model_name,
                "imgsz": params.imgsz,
                "confidence": params.confidence,
                "iou": params.iou,
                "max_detections": params.max_detections,
                "include_masks": params.include_masks,
            },
        )

        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(
                input_queue=self.yolo_input_queue,
                service_name=self.service_name,
                controller_name=self.controller_name,
            )

        try:
            self.yolo_runner.change_detector(settings=params)
        except Exception as exc:
            logger(f"[{self.service_name}:{self.controller_name}:set_model:error] failed to apply model settings",level="error",
                extra={
                    "model_name": params.model_name,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            return self._result(err=str(exc))

        result = self._result()
        logger(
            f"[{self.service_name}:{self.controller_name}:set_model] model settings applied",
            extra={"model_name": params.model_name},
        )
        return result


def main() -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    logger(
        f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:run_server] starting RPC server",
        extra={
            "service_name": DEFAULT_SERVICE_NAME,
            "controller_name": DEFAULT_CONTROLLER_NAME,
        },
    )
    try:
        server = Iox2JsonRpcServer(
            YoloController(
                service_name=DEFAULT_SERVICE_NAME,
                controller_name=DEFAULT_CONTROLLER_NAME,
            )
        )
        server.run_forever()
    except KeyboardInterrupt:
        logger(
            f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:run_server] server interrupted",
            level="warning",
        )
        raise
    except Exception:
        logger(f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:run_server:error] server stopped with an error",level="error",
            extra={
                "service_name": DEFAULT_SERVICE_NAME,
                "controller_name": DEFAULT_CONTROLLER_NAME,
            },
        )
        raise
    finally:
        logger(
            f"[{DEFAULT_SERVICE_NAME}:{DEFAULT_CONTROLLER_NAME}:run_server] server stopped"
        )


if __name__ == "__main__":
    main()
