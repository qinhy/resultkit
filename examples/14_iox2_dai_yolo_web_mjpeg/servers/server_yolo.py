from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file
import torch.nn.functional as F
from ultralytics import YOLO

from common import EmptyParams, HookDispatcher, RpcModel, openapi_doc
from store.custom_record_store import CustomRecord
from resultkit.logger import logger
from yolo_utils import (
    binary_mask_to_polygon_contours,
    binary_mask_to_uncompressed_rle,
    decode_instance_masks,
    decode_yolo_predictions,
    greedy_class_aware_clusters,
    letterbox_tensor,
    make_divisible,
    paste_binary_mask,
    split_prediction_batch,
    split_proto_batch,
    tile_positions,
    torch_class_aware_nms,
    uncompressed_rle_to_binary_mask,
    unwrap_yolo_model_output,
)


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


class YoloPolygon(BaseModel):
    """One polygon ring in absolute source-image XY coordinates."""

    points_xy: list[list[float]]
    is_hole: bool = False
    parent_index: int | None = None


class YoloInstancePolygon(BaseModel):
    """JSON-friendly polygon representation of one instance mask."""

    format: Literal["polygon"] = "polygon"
    size: list[int]
    polygons: list[YoloPolygon]
    area: int
    threshold: float


class YoloBaseModel(RpcModel):
    service:str = DEFAULT_SERVICE_NAME


class StartYoloParams(YoloBaseModel):
    """Parameters for one YOLO inference request.
    """

    size_mode: SizeMode = "tiling"
    cuda_device: int = 0
    db_record: Optional[CustomRecord] = field(default_factory=CustomRecord.empty)
    input_jpg_paths: List[str] = [] # field(default_factory=list)
    output_json_paths: List[str] = [] # field(default_factory=list)

    # Optional inference ROI in absolute original-image XYXY coordinates.
    # None means detect on the full image. Values are clipped to image bounds.
    detection_bbox_xyxy: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )

    hook_urls:list[list[str]] = [[]]

    
    def model_post_init(self, context):
        if self.db_record is None:
            self.db_record = CustomRecord.empty()

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

    model_name: str = "yolo11l-seg.pt"

    # Number of already-loaded model weights to keep in the in-memory LRU cache.
    # A value of 1 behaves similarly to the old single-model implementation.
    model_cache_size: int = Field(default=4, ge=1)
    # False: inactive cached models are moved to CPU/float32 to conserve VRAM.
    # True: keep models on their last device/dtype for the fastest switching,
    #       at the cost of additional GPU memory usage.
    model_cache_keep_on_device: bool = True

    imgsz: int = Field(default=1280, gt=0)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, gt=0)
    
    stride: int = Field(default=32, gt=0)
    tile_overlap: int = Field(default=416, ge=0) # good for image of 3872 × 3008
    tile_batch_size: int = Field(default=4, ge=0)

    half: bool = True
    include_masks: bool = True
    mask_format: Literal["polygon", "rle"] = "polygon"
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # In tiling mode, "full_image" writes every final mask on the original
    # image canvas: size=[image_height, image_width], origin_xy=[0, 0].
    tiled_mask_output: TiledMaskOutput = "full_image"
    # Union duplicate instances detected in overlapping tiles before RLE.
    merge_tiled_masks: bool = True
    # Intersection-over-smaller-box threshold used in addition to IoU.
    # This helps join two clipped halves of an object at a tile boundary.
    tile_merge_iom: float = Field(default=0.15, ge=0.0, le=1.0)
    
    polygon_epsilon: float = 1.0
    polygon_min_area: float = 1.0


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
    mask: YoloInstanceMask | YoloInstancePolygon | None = None


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
    # Effective ROI after clipping to the source image. Omitted for full-image inference.
    detection_bbox_xyxy: list[int] | None = None


    def model_post_init(self, context: Any) -> None:
        if self.num_detections is None:
            self.num_detections = len(self.detections)
        if self.has_masks is None:
            self.has_masks = any(det.mask is not None for det in self.detections)

    def to_json_text(self, *, indent: int = 2) -> str:
        """Serialize with ``None`` fields omitted, matching the old payload style."""
        return json.dumps(self.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=indent)


@dataclass
class _YoloModelCacheEntry:
    """One loaded YOLO model plus metadata needed by the raw forward path."""

    model: Any
    names: dict[int, str]
    nc: int
    mask_coeff_count: int
    is_segment_model: bool
    device: Any
    dtype: Any


class YoloDetector:
    """High-throughput Torch-first wrapper around Ultralytics YOLO weights.

    Model weights are cached in an in-memory LRU cache. Switching back to a
    previously-used ``model_name`` therefore avoids re-reading and rebuilding
    the model from disk.

    By default inactive cached models are moved to CPU/float32 to reduce VRAM
    pressure. Set ``model_cache_keep_on_device=True`` for the fastest possible
    model switching when enough GPU memory is available.
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

        # Active model view. These fields describe whichever cache entry is
        # currently selected for inference.
        self._model: Any | None = None
        self._model_name_loaded: str | None = None
        self._model_device: Any | None = None
        self._model_dtype: Any | None = None

        self._names: dict[int, str] = {}
        self._nc: int = 0
        self._mask_coeff_count: int = 0
        self._is_segment_model: bool = False

        # OrderedDict order is LRU -> MRU.
        self._model_cache: OrderedDict[str, _YoloModelCacheEntry] = OrderedDict()

        self._lock = Lock()
        self._nvjpeg_fallback_warning_emitted = False

        logger(
            f"[{self.service_name}:{self.controller_name}:detector:init] detector initialized",
            extra={
                "model_name": self.settings.model_name,
                "model_cache_size": self._model_cache_limit(),
                "model_cache_keep_on_device": bool(
                    self.settings.model_cache_keep_on_device
                ),
                "imgsz": self.settings.imgsz,
                "confidence": self.settings.confidence,
                "iou": self.settings.iou,
                "include_masks": self.settings.include_masks,
                "tile_batch_size": self._tile_batch_size(),
                "mask_format": self._mask_format(),
            },
        )

    def change_settings(self, settings: YoloSettings) -> None:
        """Apply runtime settings without discarding already-loaded models."""
        with self._lock:
            previous_model_name = self.settings.model_name
            switch_model = settings.model_name != previous_model_name

            logger(
                f"[{self.service_name}:{self.controller_name}:detector:settings] applying detector settings",
                extra={
                    "previous_model_name": previous_model_name,
                    "model_name": settings.model_name,
                    "switch_model": switch_model,
                    "model_cached": settings.model_name in self._model_cache,
                    "cached_model_names": list(self._model_cache.keys()),
                    "model_cache_size": max(1, int(settings.model_cache_size)),
                    "model_cache_keep_on_device": bool(
                        settings.model_cache_keep_on_device
                    ),
                    "imgsz": settings.imgsz,
                    "confidence": settings.confidence,
                    "iou": settings.iou,
                    "max_detections": settings.max_detections,
                    "include_masks": settings.include_masks,
                    "mask_format": str(
                        getattr(settings, "mask_format", "polygon") or "polygon"
                    ).lower(),
                    "polygon_epsilon": float(
                        getattr(settings, "polygon_epsilon", 1.0) or 0.0
                    ),
                    "polygon_min_area": float(
                        getattr(settings, "polygon_min_area", 1.0) or 0.0
                    ),
                    "tile_batch_size": max(
                        1,
                        int(getattr(settings, "tile_batch_size", 8) or 8),
                    ),
                },
            )

            self.settings = settings

            # Do NOT destroy the active model when model_name changes. _load_model()
            # will either activate a cached entry or load a new one on the next
            # inference. Keeping the current active entry here also avoids moving a
            # model while another thread may still be finishing a forward pass.
            if (
                not switch_model
                and self._model_name_loaded == self.settings.model_name
            ):
                self._trim_model_cache_locked()

            logger(
                f"[{self.service_name}:{self.controller_name}:detector:settings] detector settings applied",
                extra={
                    "model_name": self.settings.model_name,
                    "switch_model": switch_model,
                    "cached_model_names": list(self._model_cache.keys()),
                },
            )

    def _tile_batch_size(self) -> int:
        """Return a safe tile batch size without requiring a schema change."""
        return max(1, int(getattr(self.settings, "tile_batch_size", 8) or 8))

    def _model_cache_limit(self) -> int:
        return max(1, int(getattr(self.settings, "model_cache_size", 2) or 2))

    def cached_model_names(self) -> list[str]:
        """Return cached model names in LRU -> MRU order."""
        with self._lock:
            return list(self._model_cache.keys())

    def clear_model_cache(self, *, keep_active: bool = True) -> None:
        """Clear cached weights, optionally retaining the active model.

        This is useful if a model file on disk has been replaced and must be
        force-reloaded even though its ``model_name`` string did not change.
        """
        with self._lock:
            active_name = self._model_name_loaded if keep_active else None
            victims = [
                name for name in self._model_cache.keys()
                if name != active_name
            ]
            for name in victims:
                self._evict_cache_entry_locked(name)

            if not keep_active:
                self._model = None
                self._model_name_loaded = None
                self._model_device = None
                self._model_dtype = None
                self._names = {}
                self._nc = 0
                self._mask_coeff_count = 0
                self._is_segment_model = False

    @staticmethod
    def _model_placement(model: Any) -> tuple[Any, Any]:
        """Inspect a module's current device and floating-point dtype."""
        try:
            parameter = next(model.parameters())
            return parameter.device, parameter.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32

    def _build_cache_entry(self, yolo: Any, model: Any) -> _YoloModelCacheEntry:
        names = (
            getattr(yolo, "names", None)
            or getattr(model, "names", {})
            or {}
        )
        if isinstance(names, dict):
            parsed_names = {int(k): str(v) for k, v in names.items()}
        else:
            parsed_names = {
                idx: str(name) for idx, name in enumerate(names)
            }

        head = None
        try:
            head = model.model[-1]
        except Exception:
            head = None

        nc = int(getattr(head, "nc", len(parsed_names) or 0) or 0)
        if nc and not parsed_names:
            parsed_names = {idx: str(idx) for idx in range(nc)}

        mask_coeff_count = int(getattr(head, "nm", 0) or 0)
        is_segment_model = bool(
            mask_coeff_count
            or getattr(yolo, "task", None) == "segment"
            or getattr(model, "task", None) == "segment"
            or (
                head is not None
                and "Segment" in head.__class__.__name__
            )
        )
        device, dtype = self._model_placement(model)

        return _YoloModelCacheEntry(
            model=model,
            names=parsed_names,
            nc=nc,
            mask_coeff_count=mask_coeff_count,
            is_segment_model=is_segment_model,
            device=device,
            dtype=dtype,
        )

    def _offload_cache_entry_locked(
        self,
        model_name: str,
        entry: _YoloModelCacheEntry,
    ) -> None:
        """Move an inactive cached model to CPU/float32 to release VRAM."""
        if getattr(entry.device, "type", str(entry.device)) == "cpu" and entry.dtype == torch.float32:
            return

        entry.model.to(device=torch.device("cpu"), dtype=torch.float32)
        entry.model.eval()
        entry.device = torch.device("cpu")
        entry.dtype = torch.float32

        logger(
            f"[{self.service_name}:{self.controller_name}:model:cache] inactive model moved to CPU",
            extra={"model_name": model_name},
        )

    def _evict_cache_entry_locked(self, model_name: str) -> None:
        entry = self._model_cache.pop(model_name, None)
        if entry is None:
            return

        # Explicitly move an evicted GPU model to CPU before dropping our
        # reference, so its CUDA allocation can be released promptly.
        if getattr(entry.device, "type", str(entry.device)) == "cuda":
            entry.model.to(device=torch.device("cpu"), dtype=torch.float32)

        logger(
            f"[{self.service_name}:{self.controller_name}:model:cache] model evicted",
            extra={
                "model_name": model_name,
                "cached_model_names": list(self._model_cache.keys()),
            },
        )

    def _trim_model_cache_locked(self) -> None:
        limit = self._model_cache_limit()
        while len(self._model_cache) > limit:
            victim_name = next(
                (
                    name for name in self._model_cache.keys()
                    if name != self._model_name_loaded
                ),
                None,
            )
            if victim_name is None:
                break
            self._evict_cache_entry_locked(victim_name)

    def _activate_cache_entry_locked(
        self,
        model_name: str,
        entry: _YoloModelCacheEntry,
    ) -> None:
        previous_name = self._model_name_loaded

        if (
            previous_name is not None
            and previous_name != model_name
            and not bool(self.settings.model_cache_keep_on_device)
        ):
            previous_entry = self._model_cache.get(previous_name)
            if previous_entry is not None:
                self._offload_cache_entry_locked(previous_name, previous_entry)

        self._model = entry.model
        self._model_name_loaded = model_name
        self._model_device = entry.device
        self._model_dtype = entry.dtype
        self._names = dict(entry.names)
        self._nc = int(entry.nc)
        self._mask_coeff_count = int(entry.mask_coeff_count)
        self._is_segment_model = bool(entry.is_segment_model)

        self._model_cache.move_to_end(model_name)

    def _load_model(self) -> Any:
        """Activate a cached model or load/fuse it once and cache it."""
        target_name = self.settings.model_name

        if (
            self._model is not None
            and self._model_name_loaded == target_name
        ):
            return self._model

        with self._lock:
            if (
                self._model is not None
                and self._model_name_loaded == target_name
            ):
                self._model_cache.move_to_end(target_name)
                return self._model

            cached = self._model_cache.get(target_name)
            if cached is not None:
                self._activate_cache_entry_locked(target_name, cached)
                self._trim_model_cache_locked()

                logger(
                    f"[{self.service_name}:{self.controller_name}:model:cache] cache hit",
                    extra={
                        "model_name": target_name,
                        "device": str(cached.device),
                        "dtype": str(cached.dtype),
                        "cached_model_names": list(self._model_cache.keys()),
                    },
                )
                return cached.model

            logger(
                f"[{self.service_name}:{self.controller_name}:model:load] loading YOLO model",
                extra={
                    "model_name": target_name,
                    "cached_model_names": list(self._model_cache.keys()),
                },
            )

            try:
                yolo = YOLO(target_name)
                model = yolo.model
                model.eval()

                try:
                    model.fuse()
                except Exception as exc:
                    logger(
                        f"[{self.service_name}:{self.controller_name}:model:fuse] model fuse skipped",
                        level="warning",
                        extra={
                            "model_name": target_name,
                            "error": str(exc),
                        },
                    )

                entry = self._build_cache_entry(yolo, model)
            except Exception:
                logger(
                    f"[{self.service_name}:{self.controller_name}:model:load:error] failed to load YOLO model",
                    level="error",
                    extra={"model_name": target_name},
                )
                raise

            self._model_cache[target_name] = entry
            self._activate_cache_entry_locked(target_name, entry)
            self._trim_model_cache_locked()

            logger(
                f"[{self.service_name}:{self.controller_name}:model:load] YOLO model loaded and cached ({target_name})",
                extra={
                    "model_name": target_name,
                    "class_count": self._nc,
                    "mask_coeff_count": self._mask_coeff_count,
                    "task": "segment" if self._is_segment_model else "detect",
                    "cached_model_names": list(self._model_cache.keys()),
                },
            )

            return self._model

    def _prepare_model(self, device: Any) -> tuple[Any, Any]:
        """Move/cast only the active cached model when placement changes."""
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

                    active_name = self._model_name_loaded
                    if active_name is not None:
                        entry = self._model_cache.get(active_name)
                        if entry is not None and entry.model is model:
                            entry.device = device
                            entry.dtype = dtype
                            self._model_cache.move_to_end(active_name)

                    if device.type == "cuda":
                        # Letterboxed model inputs have a stable spatial size.
                        torch.backends.cudnn.benchmark = True

                    logger(
                        f"[{self.service_name}:{self.controller_name}:model:prepare] model prepared",
                        extra={
                            "model_name": self._model_name_loaded,
                            "device": str(device),
                            "dtype": str(dtype),
                            "cached_model_names": list(self._model_cache.keys()),
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
                "mask_format": self._mask_format(),
                "detection_bbox_xyxy": params.detection_bbox_xyxy,
            },
        )

        start_time = time.perf_counter()
        processed_count = 0
        result: YoloDetectResult | None = None

        # Prepare once before entering image/tile loops.
        device = self._device(params.cuda_device)
        self._prepare_model(device)

        group = zip(
            params.input_jpg_paths,
            params.output_json_paths,
        )

        for image_path_value, output_json_path_value in group:
            image_path = Path(image_path_value)
            output_json_path = Path(output_json_path_value)

            logger(
                f"[{self.service_name}:{self.controller_name}:detect:image] processing image",
                extra={
                    "input_jpg_path": image_path.as_posix(),
                    "output_json_path": output_json_path.as_posix(),
                    "size_mode": params.size_mode,
                    "cuda_device": params.cuda_device,
                    "detection_bbox_xyxy": params.detection_bbox_xyxy,
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
                    detection_bbox_xyxy=params.detection_bbox_xyxy,
                )
            else:
                image_payload = self._detect_resized(
                    image_path,
                    params.cuda_device,
                    detection_bbox_xyxy=params.detection_bbox_xyxy,
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
                    "detection_bbox_xyxy": result.detection_bbox_xyxy,
                },
            )

        elapsed = time.perf_counter() - start_time
        seconds_per_imag = elapsed / processed_count if processed_count else 0
        logger(
            f"[{self.service_name}:{self.controller_name}:detect] "
            f"inference {input_count} item completed ({seconds_per_imag:.2f} sec/item)",
            extra={
                "processed_count": processed_count,
                "last_output_json_path": result.output_json_path if result is not None else None,
                "seconds_per_image": seconds_per_imag,
            },
        )

        if result is None:
            raise RuntimeError("YOLO inference completed without producing a result")
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

    def _mask_format(self) -> str:
        """Return the configured external mask representation."""
        mask_format = str(
            getattr(self.settings, "mask_format", "polygon") or "polygon"
        ).strip().lower()
        if mask_format not in {"polygon", "rle"}:
            raise ValueError(
                "mask_format must be either 'polygon' or 'rle', got "
                f"{mask_format!r}"
            )
        return mask_format

    def _rle_to_polygon(
        self,
        mask: YoloInstanceMask,
        *,
        image_width: int,
        image_height: int,
    ) -> YoloInstancePolygon:
        """Convert one internal RLE mask to absolute source-image polygons."""
        binary_mask = self._rle_to_binary_mask(mask)
        contours = binary_mask_to_polygon_contours(
            binary_mask,
            simplify_epsilon=float(
                getattr(self.settings, "polygon_epsilon", 1.0) or 0.0
            ),
            minimum_area=float(
                getattr(self.settings, "polygon_min_area", 1.0) or 0.0
            ),
        )
        origin_x, origin_y = int(mask.origin_xy[0]), int(mask.origin_xy[1])

        polygons = [
            YoloPolygon(
                points_xy=[
                    [float(x + origin_x), float(y + origin_y)]
                    for x, y in contour.points_xy
                ],
                is_hole=contour.is_hole,
                parent_index=contour.parent_index,
            )
            for contour in contours
        ]

        return YoloInstancePolygon(
            size=[int(image_height), int(image_width)],
            polygons=polygons,
            area=int(mask.area),
            threshold=float(mask.threshold),
        )

    def _convert_detection_masks_for_output(
        self,
        detections: list[YoloDetection],
        *,
        image_width: int,
        image_height: int,
    ) -> list[YoloDetection]:
        """Convert final masks to the configured external representation."""
        if self._mask_format() == "rle":
            return detections

        converted: list[YoloDetection] = []
        for detection in detections:
            polygon_mask = (
                self._rle_to_polygon(
                    detection.mask,
                    image_width=image_width,
                    image_height=image_height,
                )
                if detection.mask is not None
                else None
            )
            converted.append(
                detection.model_copy(update={"mask": polygon_mask})
            )
        return converted

    @staticmethod
    def _resolve_detection_bbox(
        bbox_xyxy: list[float] | None,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        """Clip an optional original-image ROI and return integer XYXY bounds.

        ``None`` selects the full image. The left/top edges are floored and
        right/bottom edges are ceiled so a fractional caller ROI does not lose
        border pixels.
        """
        if bbox_xyxy is None:
            return 0, 0, int(image_width), int(image_height)

        if len(bbox_xyxy) != 4:
            raise ValueError(
                "detection_bbox_xyxy must contain exactly four values: "
                "[x1, y1, x2, y2]"
            )

        values = [float(value) for value in bbox_xyxy]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detection_bbox_xyxy values must be finite numbers")

        x1 = math.floor(values[0])
        y1 = math.floor(values[1])
        x2 = math.ceil(values[2])
        y2 = math.ceil(values[3])

        x1 = max(0, min(x1, int(image_width)))
        y1 = max(0, min(y1, int(image_height)))
        x2 = max(0, min(x2, int(image_width)))
        y2 = max(0, min(y2, int(image_height)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "detection_bbox_xyxy has no positive area after clipping to "
                f"image bounds: {[x1, y1, x2, y2]} for "
                f"image {image_width}x{image_height}"
            )

        return x1, y1, x2, y2

    def _detect_resized(
        self,
        image_path: Path,
        cuda_device: int,
        detection_bbox_xyxy: list[float] | None = None,
    ) -> dict[str, Any]:
        device = self._device(cuda_device)
        image = self._decode_jpeg(image_path, device)

        full_height = int(image.shape[-2])
        full_width = int(image.shape[-1])
        roi_left, roi_top, roi_right, roi_bottom = self._resolve_detection_bbox(
            detection_bbox_xyxy,
            image_width=full_width,
            image_height=full_height,
        )

        # Tensor slicing is a view, so the decoded full image is not copied here.
        roi_image = image[:, roi_top:roi_bottom, roi_left:roi_right]
        detections = self._predict_batch([roi_image], cuda_device)[0]

        shifted_detections: list[YoloDetection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            mask = (
                self._shift_mask_origin(det.mask, roi_left, roi_top)
                if det.mask is not None
                else None
            )
            shifted_detections.append(
                det.model_copy(
                    update={
                        "bbox_xyxy": [
                            x1 + roi_left,
                            y1 + roi_top,
                            x2 + roi_left,
                            y2 + roi_top,
                        ],
                        "mask": mask,
                    }
                )
            )

        detections = self._convert_detection_masks_for_output(
            shifted_detections,
            image_width=full_width,
            image_height=full_height,
        )

        return {
            "task": "segment" if self._is_segment_model else "detect",
            "image_width": full_width,
            "image_height": full_height,
            "detection_bbox_xyxy": (
                [roi_left, roi_top, roi_right, roi_bottom]
                if detection_bbox_xyxy is not None
                else None
            ),
            "detections": detections,
            "has_masks": any(det.mask is not None for det in detections),
            "num_detections": len(detections),
        }

    def _detect_tiled(
        self,
        image_path: Path,
        cuda_device: int,
        detection_bbox_xyxy: list[float] | None = None,
    ) -> dict[str, Any]:
        device = self._device(cuda_device)
        image = self._decode_jpeg(image_path, device)

        full_height = int(image.shape[-2])
        full_width = int(image.shape[-1])
        roi_left, roi_top, roi_right, roi_bottom = self._resolve_detection_bbox(
            detection_bbox_xyxy,
            image_width=full_width,
            image_height=full_height,
        )

        # Work only inside the requested ROI. All final coordinates are shifted
        # back to the original full-image coordinate system below.
        image = image[:, roi_top:roi_bottom, roi_left:roi_right]
        roi_height = int(image.shape[-2])
        roi_width = int(image.shape[-1])

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
                "image_width": full_width,
                "image_height": full_height,
                "roi_left": roi_left,
                "roi_top": roi_top,
                "roi_right": roi_right,
                "roi_bottom": roi_bottom,
                "roi_width": roi_width,
                "roi_height": roi_height,
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "step": step,
                "tile_batch_size": tile_batch_size,
            },
        )

        pending_images: list[Any] = []
        # Tile bounds here are ROI-local until flush_tile_batch shifts them.
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
                global_left = roi_left + left
                global_top = roi_top + top
                global_right = roi_left + right
                global_bottom = roi_top + bottom
                tile_count += 1

                for det in tile_detections:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    mask = (
                        self._shift_mask_origin(
                            det.mask,
                            global_left,
                            global_top,
                        )
                        if det.mask is not None
                        else None
                    )

                    all_detections.append(
                        det.model_copy(
                            update={
                                "bbox_xyxy": [
                                    x1 + global_left,
                                    y1 + global_top,
                                    x2 + global_left,
                                    y2 + global_top,
                                ],
                                "tile": YoloTile(
                                    left=global_left,
                                    top=global_top,
                                    right=global_right,
                                    bottom=global_bottom,
                                ),
                                "mask": mask,
                            }
                        )
                    )

            pending_images.clear()
            pending_tiles.clear()

        for top in self._tile_positions(roi_height, tile_size, step):
            for left in self._tile_positions(roi_width, tile_size, step):
                right = min(left + tile_size, roi_width)
                bottom = min(top + tile_size, roi_height)

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
                    image_width=full_width,
                    image_height=full_height,
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
                                image_width=full_width,
                                image_height=full_height,
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

        merged = self._convert_detection_masks_for_output(
            merged,
            image_width=full_width,
            image_height=full_height,
        )

        logger(
            f"[{self.service_name}:{self.controller_name}:detect:tiling] tiled inference completed",
            extra={
                "input_jpg_path": image_path.as_posix(),
                "tile_count": tile_count,
                "raw_detection_count": len(all_detections),
                "merged_detection_count": len(merged),
                "detection_bbox_xyxy": (
                    [roi_left, roi_top, roi_right, roi_bottom]
                    if detection_bbox_xyxy is not None
                    else None
                ),
            },
        )

        return {
            "task": "segment" if self._is_segment_model else "detect",
            "image_width": full_width,
            "image_height": full_height,
            "detection_bbox_xyxy": (
                [roi_left, roi_top, roi_right, roi_bottom]
                if detection_bbox_xyxy is not None
                else None
            ),
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
        """Letterbox and normalize a CHW tensor using reusable utilities."""
        return letterbox_tensor(
            image,
            image_size=self._make_divisible(
                self.settings.imgsz,
                self.settings.stride,
            ),
            dtype=dtype,
        )


    @staticmethod
    def _make_divisible(value: int, divisor: int) -> int:
        return make_divisible(value, divisor)


    @staticmethod
    def _tile_positions(
        length: int,
        tile_size: int,
        step: int,
    ) -> list[int]:
        return tile_positions(length, tile_size, step)


    def _unwrap_model_output(self, output: Any) -> tuple[Any, Any | None]:
        """Locate prediction and prototype tensors across Ultralytics layouts."""
        return unwrap_yolo_model_output(
            output,
            target_mask_dim=int(self._mask_coeff_count or 0),
        )


    @staticmethod
    def _split_prediction_batch(
        predictions: Any,
        expected_batch_size: int,
    ) -> list[Any]:
        return split_prediction_batch(predictions, expected_batch_size)


    @staticmethod
    def _split_proto_batch(
        proto: Any | None,
        expected_batch_size: int,
    ) -> list[Any | None]:
        return split_proto_batch(proto, expected_batch_size)


    def _decode_predictions(
        self,
        predictions: Any,
        original_width: int,
        original_height: int,
        gain: float,
        pad: tuple[int, int],
        proto: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """Adapt detector metadata to the reusable YOLO tensor decoder."""
        return decode_yolo_predictions(
            predictions=predictions,
            class_count=self._nc,
            class_name_count=len(self._names),
            mask_coeff_count=self._mask_coeff_count,
            proto=proto,
            confidence_threshold=float(self.settings.confidence),
            original_width=original_width,
            original_height=original_height,
            gain=gain,
            pad=pad,
        )






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
        """Decode masks in ``utils.py`` and adapt them to project RLE models."""
        if mask_coefficients.numel() == 0:
            return []

        try:
            cpu_masks = decode_instance_masks(
                proto=proto,
                mask_coefficients=mask_coefficients,
                boxes=boxes,
                original_width=original_width,
                original_height=original_height,
                input_width=input_width,
                input_height=input_height,
                gain=gain,
                pad=pad,
                threshold=float(self.settings.mask_threshold),
            )
        except ValueError as exc:
            logger(
                f"[{self.service_name}:{self.controller_name}:mask:decode] mask decoding skipped",
                level="warning",
                extra={
                    "coefficient_shape": tuple(mask_coefficients.shape),
                    "proto_shape": tuple(getattr(proto, "shape", ())),
                    "error": str(exc),
                },
            )
            return [None] * int(mask_coefficients.shape[0])

        return [self._binary_mask_to_rle(mask) for mask in cpu_masks]




    def _binary_mask_to_rle(self, mask: Any) -> YoloInstanceMask:
        """Adapt reusable row-major RLE primitives to ``YoloInstanceMask``."""
        counts, area = binary_mask_to_uncompressed_rle(mask)
        height = int(mask.shape[0])
        width = int(mask.shape[1])

        return YoloInstanceMask(
            size=[height, width],
            origin_xy=[0, 0],
            counts=counts,
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
        return uncompressed_rle_to_binary_mask(mask.counts, mask.size)


    @classmethod
    def _paste_instance_mask(
        cls,
        canvas: Any,
        mask: YoloInstanceMask,
    ) -> None:
        local = cls._rle_to_binary_mask(mask)
        paste_binary_mask(canvas, local, origin_xy=mask.origin_xy)


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



    def _merge_tiled_instances(
        self,
        detections: list[YoloDetection],
        image_width: int,
        image_height: int,
        iou_threshold: float,
        iom_threshold: float,
    ) -> list[YoloDetection]:
        """Merge tiled duplicates and OR their masks on full-image canvases."""
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
        clusters = greedy_class_aware_clusters(
            boxes,
            scores,
            class_ids,
            iou_threshold=iou_threshold,
            iom_threshold=iom_threshold,
        )

        merged: list[YoloDetection] = []
        for cluster_indices in clusters:
            representative = detections[cluster_indices[0]]
            cluster_boxes = boxes[cluster_indices]
            canvas = torch.zeros(
                (int(image_height), int(image_width)),
                dtype=torch.bool,
            )
            mask_found = False

            for index in cluster_indices:
                mask = detections[index].mask
                if mask is not None:
                    self._paste_instance_mask(canvas, mask)
                    mask_found = True

            union_box = [
                float(cluster_boxes[:, 0].min().clamp(0, image_width).item()),
                float(cluster_boxes[:, 1].min().clamp(0, image_height).item()),
                float(cluster_boxes[:, 2].max().clamp(0, image_width).item()),
                float(cluster_boxes[:, 3].max().clamp(0, image_height).item()),
            ]
            merged.append(
                representative.model_copy(
                    update={
                        "bbox_xyxy": union_box,
                        "tile": None,
                        "mask": (
                            self._binary_mask_to_rle(canvas)
                            if mask_found
                            else None
                        ),
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
        return torch_class_aware_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold,
            max_detections,
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
        keep = torch_class_aware_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold,
            max_detections=len(detections),
        )
        return [detections[index] for index in keep.tolist()]



def drain_queue(queue: Queue[Any], timeout: float | None = None) -> list[Any]:
    items = [queue.get(timeout=timeout)]
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            return items
            
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
                batch_params:List[StartYoloParams] = drain_queue(self.input_queue, timeout=0.1)
            except Empty:
                continue

            logger(
                f"[{self.service_name}:{self.controller_name}:runner:request] inference request dequeued"
                f" (input_count: {sum([len(params.input_jpg_paths) for params in batch_params])})"
                f" (queue_size: {self.input_queue.qsize()})"
                ,
                extra={
                    "queue_size": self.input_queue.qsize(),
                    "input_count":sum([len(params.input_jpg_paths) for params in batch_params]),
                    "output_count": sum([len(params.output_json_paths) for params in batch_params]),
                    "size_mode": batch_params[0].size_mode,
                    "cuda_device": batch_params[0].cuda_device,
                },
            )

            try:
                results:List[YoloDetectResult] = []
                for params in batch_params:
                    result = self.detector.detect(params)
                    results.append(result)
                    
                for params,result in zip(batch_params,results):
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
