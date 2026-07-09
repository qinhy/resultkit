from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field
import requests

from common import EmptyParams, RpcModel, openapi_doc
from store.custom_record_store import CustomRecord

logger = logging.getLogger(__name__)


SizeMode = Literal["resize", "tiling"]
YoloTask = Literal["detect", "segment"]
MaskEncoding = Literal["rle"]
MaskOrder = Literal["row_major"]


class YoloBaseModel(RpcModel):
    service: Literal["yolo"] = "yolo"

    def model_post_init(self, context: Any) -> None:
        logger.debug("[Yolo %s]", self.__class__.__name__)
        return super().model_post_init(context)


class StartYoloParams(YoloBaseModel):
    """Parameters for one YOLO inference request.
    """

    size_mode: SizeMode = "tiling"
    cuda_device: int = 0
    db_record: CustomRecord = CustomRecord.empty()
    input_jpg_paths: List[str] = [] # field(default_factory=list)
    output_json_paths: List[str] = [] # field(default_factory=list)
    
    hook_urls:List[str] = []

    
    def model_post_init(self, context):
        if not self.db_record.is_empty():
            self.input_jpg_paths = [str(p) for p in self.db_record.listup_rgb_image_paths]
            self.output_json_paths = [self.to_output_json_paths(p) for p in self.input_jpg_paths]
            if self.db_record.mode == "dual_rgb":
                self.hook_urls = []
            if self.db_record.mode == "rgb_stereo":
                self.hook_urls = ["http://localhost:8000/controllers/pcd/to_pcd"]
        return super().model_post_init(context)

    @staticmethod
    def to_output_json_paths(p) -> str:
        jpg = Path(p)
        # Original behavior was: jpg.parent.parent.parent / "yolo" / file.json.
        # Keep it when possible, but avoid raising on short/relative paths.
        try:
            base_dir = jpg.parents[2]
        except IndexError:
            base_dir = jpg.parent
        return str(base_dir / "yolo" / f"{jpg.stem}.json")


class YoloSettings(YoloBaseModel):
    """Runtime settings for model inference and detection serialization."""

    model_name: str = "yolov8n.pt"

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
    """Torch-first wrapper around Ultralytics YOLO weights.

    This avoids ``YOLO.predict()`` and runs a direct PyTorch forward pass,
    followed by torch/torchvision NMS. It keeps Ultralytics only for loading
    ``.pt`` weights and model metadata.
    """

    def __init__(self, settings: YoloSettings | None = None) -> None:
        self.settings = settings or YoloSettings()
        self._model: Any | None = None
        self._model_name_loaded: str | None = None
        self._names: dict[int, str] = {}
        self._nc: int = 0
        self._mask_coeff_count: int = 0
        self._is_segment_model: bool = False
        self._lock = Lock()

    def change_settings(self, settings: YoloSettings) -> None:
        with self._lock:
            reload_model = settings.model_name != self.settings.model_name
            self.settings = settings
            if reload_model:
                self._model = None
                self._model_name_loaded = None
                self._names = {}
                self._nc = 0
                self._mask_coeff_count = 0
                self._is_segment_model = False

    def _load_model(self) -> Any:
        """Load the underlying PyTorch module from Ultralytics weights."""
        if self._model is not None and self._model_name_loaded == self.settings.model_name:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'ultralytics' package is required to load YOLO weights. "
                "Install it with: pip install ultralytics"
            ) from exc

        yolo = YOLO(self.settings.model_name)
        model = yolo.model
        model.eval()

        # fuse() is available on Ultralytics PyTorch models and usually makes
        # Conv+BatchNorm inference a little faster. If it is unavailable or not
        # supported by a custom model, continue without failing startup.
        try:
            model.fuse()
        except Exception:
            logger.debug("YOLO model fuse() skipped", exc_info=True)

        names = getattr(yolo, "names", None) or getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        else:
            self._names = {idx: str(name) for idx, name in enumerate(names)}

        head = None
        try:
            head = model.model[-1]
        except Exception:
            head = None

        self._nc = int(getattr(head, "nc", len(self._names) or 0) or 0)
        if self._nc and not self._names:
            self._names = {idx: str(idx) for idx in range(self._nc)}

        self._mask_coeff_count = int(getattr(head, "nm", 0) or 0)
        self._is_segment_model = bool(
            self._mask_coeff_count
            or getattr(yolo, "task", None) == "segment"
            or getattr(model, "task", None) == "segment"
            or (head is not None and "Segment" in head.__class__.__name__)
        )

        self._model = model
        self._model_name_loaded = self.settings.model_name
        return self._model

    def detect(self, params: StartYoloParams) -> YoloDetectResult:
        for image_path, output_json_path in zip(params.input_jpg_paths,params.output_json_paths):
            image_path = Path(image_path)
            output_json_path = Path(output_json_path)

            if not image_path.is_file():
                raise FileNotFoundError(f"Input image does not exist: {image_path}")

            if params.size_mode == "tiling":
                image_payload = self._detect_tiled(image_path, params.cuda_device)
            else:
                image_payload = self._detect_resized(image_path, params.cuda_device)

            result = YoloDetectResult(
                input_jpg_path=str(image_path),
                output_json_path=str(output_json_path),
                size_mode=params.size_mode,
                cuda_device=params.cuda_device,
                yolo_config=self.settings,
                **image_payload,
            )

            output_path = Path(result.output_json_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.to_json_text(), encoding="utf-8")

            if not params.db_record.is_empty():
                for hook_url in params.hook_urls:
                    payload = {"db_record": json.loads(params.db_record.model_dump_json())}
                    try:
                        def push_request(hook_url=hook_url, payload=payload):
                            try:
                                requests.post(hook_url, json=payload, timeout=0.5)
                            except requests.RequestException:
                                pass  # ignore timeout / connection errors
                        executor = ThreadPoolExecutor(max_workers=8)
                        executor.submit(push_request)
                    except Exception:
                        pass

        return result

    def _device(self, cuda_device: int) -> Any:
        import torch

        if cuda_device >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_device}")
        return torch.device("cpu")

    def _predict(self, source: Any, cuda_device: int) -> list[YoloDetection]:
        """Run one torch forward pass, torch NMS, and optional mask decoding."""
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Torch YOLO inference requires torch, numpy, and Pillow. "
                "Install them with: pip install torch torchvision numpy pillow"
            ) from exc

        image = Image.open(source).convert("RGB") if isinstance(source, (str, Path)) else source.convert("RGB")
        original_width, original_height = image.size

        model = self._load_model()
        device = self._device(cuda_device)
        model.to(device)
        model.eval()

        use_half = bool(self.settings.half and device.type == "cuda")
        if use_half:
            model.half()
        else:
            model.float()

        input_image, gain, pad = self._letterbox(image)
        input_width, input_height = input_image.size
        array = np.asarray(input_image)
        tensor = torch.from_numpy(array).to(device)
        tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        tensor = tensor.half() if use_half else tensor.float()
        tensor = tensor / 255.0

        with torch.inference_mode():
            output = model(tensor)

        predictions, proto = self._unwrap_model_output(output)
        boxes, scores, class_ids, mask_coefficients = self._decode_predictions(
            predictions=predictions,
            original_width=original_width,
            original_height=original_height,
            gain=gain,
            pad=pad,
            proto=proto,
        )
        keep = self._torch_class_aware_nms(boxes, scores, class_ids, self.settings.iou, self.settings.max_detections)

        masks: list[YoloInstanceMask | None] = [None] * int(keep.numel())
        if (
            self.settings.include_masks
            and proto is not None
            and mask_coefficients is not None
            and keep.numel() > 0
        ):
            kept_coefficients = mask_coefficients[keep]
            kept_boxes = boxes[keep]
            masks = self._process_instance_masks(
                proto=proto,
                mask_coefficients=kept_coefficients,
                boxes=kept_boxes,
                original_width=original_width,
                original_height=original_height,
                input_width=input_width,
                input_height=input_height,
                gain=gain,
                pad=pad,
            )

        detections: list[YoloDetection] = []
        keep_indices = keep.detach().cpu().tolist()
        for out_idx, idx in enumerate(keep_indices):
            class_id = int(class_ids[idx].detach().cpu().item())
            bbox = boxes[idx].detach().cpu().tolist()
            detections.append(
                YoloDetection(
                    class_id=class_id,
                    class_name=self._names.get(class_id, str(class_id)),
                    confidence=float(scores[idx].detach().cpu().item()),
                    bbox_xyxy=[float(v) for v in bbox],
                    mask=masks[out_idx] if out_idx < len(masks) else None,
                )
            )
        return detections

    def _detect_resized(self, image_path: Path, cuda_device: int) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "The 'Pillow' package is required for YOLO inference. "
                "Install it with: pip install pillow"
            ) from exc

        with Image.open(image_path) as image:
            width, height = image.size
        detections = self._predict(str(image_path), cuda_device)
        return {
            "task": "segment" if self._is_segment_model else "detect",
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "has_masks": any(det.mask is not None for det in detections),
            "num_detections": len(detections),
        }

    def _detect_tiled(self, image_path: Path, cuda_device: int) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "The 'Pillow' package is required for tiled YOLO inference. "
                "Install it with: pip install pillow"
            ) from exc

        tile_size = self._make_divisible(self.settings.imgsz, self.settings.stride)
        overlap = min(self.settings.tile_overlap, max(tile_size - 1, 0))
        step = max(tile_size - overlap, 1)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        all_detections: list[YoloDetection] = []
        tile_count = 0

        for top in self._tile_positions(height, tile_size, step):
            for left in self._tile_positions(width, tile_size, step):
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)
                tile = image.crop((left, top, right, bottom))
                tile_detections = self._predict(tile, cuda_device)
                tile_count += 1

                for det in tile_detections:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    mask = self._shift_mask_origin(det.mask, left, top) if det.mask is not None else None
                    all_detections.append(
                        det.model_copy(
                            update={
                                "bbox_xyxy": [x1 + left, y1 + top, x2 + left, y2 + top],
                                "tile": YoloTile(left=left, top=top, right=right, bottom=bottom),
                                "mask": mask,
                            }
                        )
                    )

        merged = self._nms(all_detections, self.settings.iou)
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

    def _letterbox(self, image: Any) -> tuple[Any, float, tuple[int, int]]:
        from PIL import Image

        img_size = self._make_divisible(self.settings.imgsz, self.settings.stride)
        width, height = image.size
        gain = min(img_size / width, img_size / height)
        resized_width = int(round(width * gain))
        resized_height = int(round(height * gain))

        pad_w = img_size - resized_width
        pad_h = img_size - resized_height
        pad_left = int(round(pad_w / 2 - 0.1))
        pad_top = int(round(pad_h / 2 - 0.1))

        if (resized_width, resized_height) != image.size:
            image = image.resize((resized_width, resized_height), Image.BILINEAR)

        canvas = Image.new("RGB", (img_size, img_size), (114, 114, 114))
        canvas.paste(image, (pad_left, pad_top))
        return canvas, gain, (pad_left, pad_top)

    @staticmethod
    def _make_divisible(value: int, divisor: int) -> int:
        divisor = max(int(divisor), 1)
        return int(math.ceil(int(value) / divisor) * divisor)

    @staticmethod
    def _tile_positions(length: int, tile_size: int, step: int) -> list[int]:
        if length <= tile_size:
            return [0]

        positions = list(range(0, max(length - tile_size + 1, 1), step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _unwrap_model_output(self, output: Any) -> tuple[Any, Any | None]:
        """Return raw prediction tensor and optional segmentation prototypes.

        Ultralytics segment models may return ``(pred, proto)`` when exported,
        or ``(pred, (features, mask_coefficients, proto))`` from the native
        PyTorch model. Detection models usually return ``(pred, feature_maps)``.
        """
        if not isinstance(output, (tuple, list)):
            return output, None

        predictions = output[0]
        proto = None
        target_mask_dim = int(self._mask_coeff_count or 0)

        def find_proto(value: Any) -> Any | None:
            # Segmentation prototypes are usually [B, mask_dim, H, W], where
            # mask_dim is small (for example 32). Prefer the known mask_dim
            # from the model head, then choose the largest spatial canvas.
            candidates: list[Any] = []
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, (tuple, list)):
                    stack.extend(item)
                    continue
                shape = getattr(item, "shape", None)
                ndim = getattr(item, "ndim", None)
                if shape is not None and ndim == 4 and len(shape) == 4:
                    channels = int(shape[1])
                    if target_mask_dim and channels == target_mask_dim:
                        candidates.append(item)
                    elif not target_mask_dim and channels <= 128:
                        candidates.append(item)
            if not candidates:
                return None
            return max(candidates, key=lambda item: int(item.shape[-2]) * int(item.shape[-1]))

        if len(output) >= 2:
            proto = find_proto(output[1])
        if proto is None:
            proto = find_proto(output[1:])

        return predictions, proto

    def _decode_predictions(
        self,
        predictions: Any,
        original_width: int,
        original_height: int,
        gain: float,
        pad: tuple[int, int],
        proto: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        import torch

        pred = predictions[0] if predictions.ndim == 3 else predictions

        # Ultralytics raw detection/segmentation output is usually
        # [B, 4 + nc (+ mask_dim), N]. Convert it to [N, C]. If a model/export
        # already returns [B, N, C], this leaves it alone.
        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO output shape: {tuple(predictions.shape)}")

        nc = int(self._nc or len(self._names) or 0)
        proto_mask_dim = int(proto.shape[1]) if proto is not None and getattr(proto, "ndim", 0) == 4 else 0
        mask_dim = int(self._mask_coeff_count or proto_mask_dim or 0)

        possible_channels = {6}
        if nc:
            possible_channels.update({4 + nc, 5 + nc})
            if mask_dim:
                possible_channels.update({4 + nc + mask_dim, 5 + nc + mask_dim})

        channels_first_raw = pred.shape[0] in possible_channels and pred.shape[1] not in possible_channels
        unknown_names_channels_first = (
            nc == 0 and pred.shape[0] < pred.shape[1] and pred.shape[0] <= 512
        )
        if channels_first_raw or unknown_names_channels_first:
            pred = pred.transpose(0, 1).contiguous()

        channels = pred.shape[-1]
        if channels < 6:
            empty = torch.empty((0,), device=pred.device)
            return torch.empty((0, 4), device=pred.device), empty, empty.long(), None

        boxes: Any
        scores: Any
        class_ids: Any
        mask_coefficients = None

        if nc and mask_dim and channels == 4 + nc + mask_dim:
            # YOLOv8/YOLO11 segment style: cx, cy, w, h, class scores..., mask coeffs...
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4 : 4 + nc].max(dim=1)
            mask_coefficients = pred[:, 4 + nc : 4 + nc + mask_dim]
        elif nc and mask_dim and channels == 5 + nc + mask_dim:
            # YOLOv5-like segment: cx, cy, w, h, objectness, class scores..., mask coeffs...
            boxes = self._xywh_to_xyxy(pred[:, :4])
            class_scores, class_ids = pred[:, 5 : 5 + nc].max(dim=1)
            scores = pred[:, 4] * class_scores
            mask_coefficients = pred[:, 5 + nc : 5 + nc + mask_dim]
        elif nc and channels == 4 + nc:
            # YOLOv8/YOLO11 detect style: cx, cy, w, h, class scores...
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4:].max(dim=1)
        elif nc and channels == 5 + nc:
            # YOLOv5-like detect: cx, cy, w, h, objectness, class scores...
            boxes = self._xywh_to_xyxy(pred[:, :4])
            class_scores, class_ids = pred[:, 5:].max(dim=1)
            scores = pred[:, 4] * class_scores
        elif channels == 6:
            # Already post-processed-like: x1, y1, x2, y2, confidence, class.
            boxes = pred[:, :4]
            scores = pred[:, 4]
            class_ids = pred[:, 5].long()
        elif mask_dim and channels > 4 + mask_dim:
            # Fallback for segment models when class-name metadata is missing.
            inferred_nc = channels - 4 - mask_dim
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4 : 4 + inferred_nc].max(dim=1)
            mask_coefficients = pred[:, 4 + inferred_nc : 4 + inferred_nc + mask_dim]
        else:
            # Fallback for raw detect models when names are unavailable.
            boxes = self._xywh_to_xyxy(pred[:, :4])
            scores, class_ids = pred[:, 4:].max(dim=1)

        keep = scores >= self.settings.confidence
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep].long()
        if mask_coefficients is not None:
            mask_coefficients = mask_coefficients[keep]

        if boxes.numel() == 0:
            return boxes.reshape(0, 4), scores, class_ids, mask_coefficients

        boxes = self._scale_boxes_from_letterbox(boxes, original_width, original_height, gain, pad)
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
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height)
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
        """Convert YOLO segment prototypes and coefficients into RLE masks."""
        import torch
        import torch.nn.functional as F

        if mask_coefficients.numel() == 0:
            return []

        if proto.ndim == 4:
            proto = proto[0]
        if proto.ndim != 3:
            logger.warning("Unexpected YOLO segmentation proto shape: %s", tuple(proto.shape))
            return [None] * int(mask_coefficients.shape[0])

        proto = proto.float()
        mask_coefficients = mask_coefficients.float()
        mask_dim, proto_height, proto_width = proto.shape
        if mask_coefficients.shape[1] != mask_dim:
            logger.warning(
                "Mask coefficient/prototype mismatch: coeff=%s proto=%s",
                tuple(mask_coefficients.shape),
                tuple(proto.shape),
            )
            return [None] * int(mask_coefficients.shape[0])

        masks = mask_coefficients @ proto.reshape(mask_dim, -1)
        masks = masks.reshape(-1, proto_height, proto_width).sigmoid()

        # Upscale from prototype resolution to the letterboxed network input,
        # remove letterbox padding, then resize to original image/tile size.
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

        masks = masks[:, crop_top:crop_bottom, crop_left:crop_right]
        if masks.shape[-2:] != (original_height, original_width):
            masks = F.interpolate(
                masks[:, None],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        binary_masks = masks >= self.settings.mask_threshold
        binary_masks = self._crop_masks_to_boxes(binary_masks, boxes)

        results: list[YoloInstanceMask | None] = []
        for mask in binary_masks.detach().cpu():
            results.append(self._binary_mask_to_rle(mask))
        return results

    @staticmethod
    def _crop_masks_to_boxes(masks: Any, boxes: Any) -> Any:
        """Zero out mask pixels outside each detection box."""
        import torch

        if masks.numel() == 0:
            return masks

        n, height, width = masks.shape
        x = torch.arange(width, device=masks.device).view(1, 1, width)
        y = torch.arange(height, device=masks.device).view(1, height, 1)
        x1 = boxes[:, 0].floor().view(n, 1, 1)
        y1 = boxes[:, 1].floor().view(n, 1, 1)
        x2 = boxes[:, 2].ceil().view(n, 1, 1)
        y2 = boxes[:, 3].ceil().view(n, 1, 1)
        inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        return masks & inside

    def _binary_mask_to_rle(self, mask: Any) -> YoloInstanceMask:
        """Encode a boolean mask as row-major uncompressed RLE."""
        flat = mask.reshape(-1)
        # Convert to Python ints. This keeps the output dependency-free and
        # easy to decode in C++, Python, Rust, or JavaScript.
        values = [1 if bool(v) else 0 for v in flat.tolist()]

        counts: list[int] = []
        current_value = 0
        run_length = 0
        for value in values:
            if value == current_value:
                run_length += 1
            else:
                counts.append(run_length)
                current_value = value
                run_length = 1
        counts.append(run_length)

        height, width = int(mask.shape[0]), int(mask.shape[1])
        area = int(mask.sum().item())
        return YoloInstanceMask(
            size=[height, width],
            origin_xy=[0, 0],
            counts=counts,
            area=area,
            threshold=float(self.settings.mask_threshold),
        )

    @staticmethod
    def _shift_mask_origin(mask: YoloInstanceMask, dx: int, dy: int) -> YoloInstanceMask:
        """Shift a tile-local mask origin into original-image coordinates."""
        ox, oy = mask.origin_xy
        return mask.model_copy(update={"origin_xy": [int(ox + dx), int(oy + dy)]})

    @staticmethod
    def _torch_class_aware_nms(
        boxes: Any,
        scores: Any,
        class_ids: Any,
        iou_threshold: float,
        max_detections: int,
    ) -> Any:
        import torch

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        try:
            from torchvision.ops import batched_nms

            keep = batched_nms(boxes, scores, class_ids, iou_threshold)
        except Exception:
            # Torch-only fallback for deployments where torchvision.ops was not
            # compiled with NMS support. The inner loop still works on tensors.
            keep_parts = []
            for cls in class_ids.unique():
                cls_indices = torch.where(class_ids == cls)[0]
                cls_keep_local = YoloDetector._torch_nms_single_class(
                    boxes[cls_indices], scores[cls_indices], iou_threshold
                )
                keep_parts.append(cls_indices[cls_keep_local])

            if not keep_parts:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            keep = torch.cat(keep_parts)
            keep = keep[scores[keep].argsort(descending=True)]

        return keep[:max_detections]

    @staticmethod
    def _torch_nms_single_class(boxes: Any, scores: Any, iou_threshold: float) -> Any:
        import torch

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
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

        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)

    @staticmethod
    def _nms(detections: list[YoloDetection], iou_threshold: float) -> list[YoloDetection]:
        """Class-aware torch NMS for merged tiled detections."""
        if not detections:
            return []

        import torch

        boxes = torch.tensor([det.bbox_xyxy for det in detections], dtype=torch.float32)
        scores = torch.tensor([det.confidence for det in detections], dtype=torch.float32)
        class_ids = torch.tensor([det.class_id for det in detections], dtype=torch.long)
        keep = YoloDetector._torch_class_aware_nms(
            boxes, scores, class_ids, iou_threshold, max_detections=len(detections)
        )
        return [detections[idx] for idx in keep.cpu().tolist()]


class YoloRunner:
    """Background worker that consumes StartYoloParams from a queue."""

    def __init__(
        self,
        input_queue: Queue[StartYoloParams] | None = None,
        detector: YoloDetector | None = None,
    ) -> None:
        self.input_queue: Queue[StartYoloParams] = input_queue or Queue()
        self.detector = detector or YoloDetector()
        self.exception: Exception | None = None
        self.last_output_json_path: str | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="YoloRunner", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def change_detector(self, settings: YoloSettings) -> None:
        self.detector.change_settings(settings)

    def _run(self) -> None:
        logger.info("YOLO runner started")
        while not self._stop_event.is_set():
            try:
                params = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                result = self.detector.detect(params)
                self.last_output_json_path = result.output_json_path
                self.exception = None
            except Exception as exc:  # Keep the worker alive after bad requests.
                self.exception = exc
                logger.error("YOLO inference failed: %s\n%s", exc, traceback.format_exc())
            finally:
                self.input_queue.task_done()

        logger.info("YOLO runner stopped")


@dataclass
class YoloController:
    running: bool = False
    yolo_runner: YoloRunner | None = None
    yolo_input_queue: Queue[StartYoloParams] = field(default_factory=Queue)
    service_name: str = "jsonrpc"
    controller_name: str = "yolo"

    def __post_init__(self) -> None:
        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(input_queue=self.yolo_input_queue)
        else:
            self.yolo_input_queue = self.yolo_runner.input_queue

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
        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(input_queue=self.yolo_input_queue)

        if not self.yolo_runner.is_running:
            self.yolo_runner.start()

        self.running = self.yolo_runner.is_running
        self.yolo_input_queue.put(params)

        result = self._result(params=params)
        result.queued = True
        return result

    def stop(self, params: EmptyParams) -> YoloResult:
        if self.yolo_runner is not None:
            self.yolo_runner.stop()

        self.running = False
        return self._result()

    def status(self, params: EmptyParams) -> YoloResult:
        return self._result()

    def set_model(self, params: YoloSettings) -> YoloResult:
        if self.yolo_runner is None:
            self.yolo_runner = YoloRunner(input_queue=self.yolo_input_queue)

        try:
            self.yolo_runner.change_detector(settings=params)
        except Exception as exc:
            return self._result(err=str(exc))

        return self._result()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(YoloController())
    server.run_forever()


if __name__ == "__main__":
    main()
