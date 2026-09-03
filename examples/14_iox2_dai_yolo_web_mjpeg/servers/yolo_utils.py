"""Reusable Torch/image geometry helpers for YOLO-style inference pipelines.

The functions in this module deliberately avoid project-specific settings,
logging, and Pydantic models.  They operate on tensors and plain Python data so
they can be reused by detectors, tests, converters, and offline tooling.
"""
from __future__ import annotations

# Standard library imports
import ast
import colorsys
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

# Third-party imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class PolygonContour:
    """One contour extracted from a binary mask.

    ``parent_index`` refers to another item in the returned contour list, not to
    OpenCV's original contour index.  Holes are determined from contour nesting
    depth, so an island inside a hole is treated as a filled exterior contour.
    """

    points_xy: list[list[float]]
    is_hole: bool
    parent_index: int | None


def make_divisible(value: int, divisor: int) -> int:
    """Round ``value`` up to the nearest positive multiple of ``divisor``."""
    safe_divisor = max(int(divisor), 1)
    return int(math.ceil(int(value) / safe_divisor) * safe_divisor)


def tile_positions(length: int, tile_size: int, step: int) -> list[int]:
    """Return tile starts that cover an axis and always include its far edge."""
    if length <= tile_size:
        return [0]

    positions = list(range(0, max(length - tile_size + 1, 1), step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def letterbox_tensor(
    image: Any,
    *,
    image_size: int,
    dtype: Any,
    padding_value: float = 114.0 / 255.0,
) -> tuple[Any, float, tuple[int, int]]:
    """Resize, normalize, and pad a CHW tensor to a square network input."""
    if image.ndim != 3:
        raise ValueError(
            f"Expected CHW image tensor, got shape {tuple(image.shape)}"
        )

    _, height, width = image.shape
    height = int(height)
    width = int(width)
    image_size = int(image_size)

    gain = min(image_size / width, image_size / height)
    resized_width = int(round(width * gain))
    resized_height = int(round(height * gain))

    pad_w = image_size - resized_width
    pad_h = image_size - resized_height
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
            value=float(padding_value),
        )

    return tensor.contiguous(), gain, (pad_left, pad_top)


def split_prediction_batch(
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
            f"{tuple(getattr(predictions, 'shape', ()))}"
        )

    if int(predictions.shape[0]) != expected_batch_size:
        raise RuntimeError(
            "YOLO output batch mismatch: "
            f"expected={expected_batch_size}, "
            f"actual={int(predictions.shape[0])}, "
            f"shape={tuple(predictions.shape)}"
        )

    return [predictions[index] for index in range(expected_batch_size)]


def split_proto_batch(
    proto: Any | None,
    expected_batch_size: int,
) -> list[Any | None]:
    """Split a batched mask-prototype tensor, preserving absent prototypes."""
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


def xywh_to_xyxy(boxes: Any) -> Any:
    """Convert center-width-height boxes to corner coordinates."""
    converted = boxes.clone()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def scale_boxes_from_letterbox(
    boxes: Any,
    *,
    original_width: int,
    original_height: int,
    gain: float,
    pad: tuple[int, int],
) -> Any:
    """Map corner-coordinate boxes from letterboxed input to source pixels."""
    pad_left, pad_top = pad
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height)
    return boxes


def crop_masks_to_boxes(masks: Any, boxes: Any) -> Any:
    """Zero mask pixels outside each corresponding detection box."""
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


def decode_instance_masks(
    *,
    proto: Any,
    mask_coefficients: Any,
    boxes: Any,
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
    gain: float,
    pad: tuple[int, int],
    threshold: float,
) -> Any:
    """Decode prototype masks and return one CPU boolean mask per detection."""
    if proto.ndim == 4:
        proto = proto[0]
    if proto.ndim != 3:
        raise ValueError(f"Unexpected prototype shape: {tuple(proto.shape)}")

    proto = proto.float()
    mask_coefficients = mask_coefficients.float()
    mask_dim, proto_height, proto_width = proto.shape

    if int(mask_coefficients.shape[1]) != int(mask_dim):
        raise ValueError(
            "Mask coefficient/prototype mismatch: "
            f"coefficients={tuple(mask_coefficients.shape)}, "
            f"proto={tuple(proto.shape)}"
        )

    masks = mask_coefficients @ proto.reshape(mask_dim, -1)
    masks = masks.reshape(-1, proto_height, proto_width).sigmoid()
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

    detection_count = int(mask_coefficients.shape[0])
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return torch.zeros(
            (detection_count, original_height, original_width),
            dtype=torch.bool,
        )

    masks = masks[:, crop_top:crop_bottom, crop_left:crop_right]
    if masks.shape[-2:] != (original_height, original_width):
        masks = F.interpolate(
            masks[:, None],
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    binary_masks = masks >= float(threshold)
    binary_masks = crop_masks_to_boxes(binary_masks, boxes)
    return binary_masks.detach().to("cpu")


def binary_mask_to_uncompressed_rle(mask: Any) -> tuple[list[int], int]:
    """Encode a boolean CPU mask as row-major uncompressed RLE."""
    flat = mask.reshape(-1).to(dtype=torch.uint8, device="cpu")
    pixel_count = int(flat.numel())

    if pixel_count == 0:
        return [0], 0

    changes = torch.nonzero(flat[1:] != flat[:-1], as_tuple=False).flatten() + 1
    boundaries = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            changes.to(dtype=torch.long),
            torch.tensor([pixel_count], dtype=torch.long),
        )
    )
    counts = (boundaries[1:] - boundaries[:-1]).tolist()
    if int(flat[0].item()) == 1:
        counts.insert(0, 0)

    return [int(value) for value in counts], int(flat.sum().item())


def uncompressed_rle_to_binary_mask(
    counts: list[int],
    size: tuple[int, int] | list[int],
) -> Any:
    """Decode row-major uncompressed RLE to a CPU boolean tensor."""
    height, width = int(size[0]), int(size[1])
    pixel_count = height * width
    run_counts = torch.as_tensor(counts, dtype=torch.long)

    if run_counts.numel() == 0:
        return torch.zeros((height, width), dtype=torch.bool)
    if bool((run_counts < 0).any()):
        raise ValueError("RLE counts must be non-negative")
    if int(run_counts.sum().item()) != pixel_count:
        raise ValueError(
            "RLE count total does not match mask size: "
            f"sum={int(run_counts.sum().item())}, expected={pixel_count}"
        )

    values = torch.arange(run_counts.numel(), dtype=torch.long).remainder(2)
    flat = torch.repeat_interleave(values, run_counts).to(torch.bool)
    return flat.reshape(height, width)


def paste_binary_mask(
    canvas: Any,
    local_mask: Any,
    *,
    origin_xy: tuple[int, int] | list[int],
) -> None:
    """OR a local boolean mask into a larger canvas, clipping at the edges."""
    canvas_height, canvas_width = canvas.shape
    origin_x, origin_y = int(origin_xy[0]), int(origin_xy[1])
    mask_height, mask_width = local_mask.shape

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
    canvas[dst_top:dst_bottom, dst_left:dst_right] |= local_mask[
        src_top:src_bottom, src_left:src_right
    ]


def binary_mask_to_polygon_contours(
    mask: Any,
    *,
    simplify_epsilon: float = 1.0,
    minimum_area: float = 1.0,
) -> list[PolygonContour]:
    """Extract simplified exterior and hole contours from a binary mask.

    OpenCV is used only on a CPU uint8 array.  Returned points are pixel-space
    XY coordinates local to ``mask``.  Callers can add an origin offset when a
    compact tile-local mask is being converted.
    """
    if getattr(mask, "ndim", None) != 2:
        raise ValueError(
            f"Expected a 2-D binary mask, got shape {tuple(mask.shape)}"
        )

    mask_array = (
        mask.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
    )
    mask_array = np.ascontiguousarray(mask_array)

    contours, hierarchy = cv2.findContours(
        mask_array,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None or not contours:
        return []

    hierarchy_rows = hierarchy[0]
    retained: list[tuple[int, list[list[float]], bool, int | None]] = []

    def contour_depth(contour_index: int) -> int:
        depth = 0
        parent = int(hierarchy_rows[contour_index][3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy_rows[parent][3])
        return depth

    for original_index, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        if abs(float(cv2.contourArea(contour))) < float(minimum_area):
            continue

        if simplify_epsilon > 0:
            contour = cv2.approxPolyDP(
                contour,
                epsilon=float(simplify_epsilon),
                closed=True,
            )
        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2)
        parent_original_index = int(hierarchy_rows[original_index][3])
        retained.append(
            (
                original_index,
                [[float(x), float(y)] for x, y in points],
                contour_depth(original_index) % 2 == 1,
                parent_original_index if parent_original_index >= 0 else None,
            )
        )

    original_to_output = {
        original_index: output_index
        for output_index, (original_index, _, _, _) in enumerate(retained)
    }

    result: list[PolygonContour] = []
    for _, points_xy, is_hole, parent_original_index in retained:
        parent_index = None
        ancestor = parent_original_index
        while ancestor is not None:
            if ancestor in original_to_output:
                parent_index = original_to_output[ancestor]
                break
            next_parent = int(hierarchy_rows[ancestor][3])
            ancestor = next_parent if next_parent >= 0 else None

        result.append(
            PolygonContour(
                points_xy=points_xy,
                is_hole=is_hole,
                parent_index=parent_index,
            )
        )

    return result


def box_iou_and_iom(box: Any, boxes: Any) -> tuple[Any, Any]:
    """Return IoU and intersection-over-smaller-area for one box vs many."""
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])
    intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    union = box_area + boxes_area - intersection
    smaller = torch.minimum(box_area.expand_as(boxes_area), boxes_area)
    iou = intersection / union.clamp(min=1e-7)
    iom = intersection / smaller.clamp(min=1e-7)
    return iou, iom


def torch_nms_single_class(
    boxes: Any,
    scores: Any,
    iou_threshold: float,
) -> Any:
    """Pure-Torch NMS fallback for one class."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[Any] = []

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

        intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[current] + areas[rest] - intersection
        iou = intersection / union.clamp(min=1e-7)
        order = rest[iou <= iou_threshold]

    return (
        torch.stack(keep)
        if keep
        else torch.empty((0,), dtype=torch.long, device=boxes.device)
    )


def torch_class_aware_nms(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    iou_threshold: float,
    max_detections: int,
) -> Any:
    """Run class-aware NMS, preferring torchvision's optimized implementation."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    try:
        from torchvision.ops import batched_nms

        keep = batched_nms(boxes, scores, class_ids, iou_threshold)
    except Exception:
        keep_parts = []
        for cls in class_ids.unique():
            cls_indices = torch.where(class_ids == cls)[0]
            cls_keep_local = torch_nms_single_class(
                boxes[cls_indices],
                scores[cls_indices],
                iou_threshold,
            )
            keep_parts.append(cls_indices[cls_keep_local])

        if not keep_parts:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        keep = torch.cat(keep_parts)
        keep = keep[scores[keep].argsort(descending=True)]

    return keep[:max_detections]


def unwrap_yolo_model_output(
    output: Any,
    *,
    target_mask_dim: int = 0,
) -> tuple[Any, Any | None]:
    """Locate YOLO prediction and optional prototype tensors across layouts."""
    if not isinstance(output, (tuple, list)):
        return output, None

    target_mask_dim = int(target_mask_dim or 0)

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

        return max(
            candidates,
            key=lambda item: int(item.shape[-2]) * int(item.shape[-1]),
        )

    first = output[0]
    proto = None

    if isinstance(first, (tuple, list)):
        if not first:
            raise RuntimeError("YOLO returned an empty primary output")
        predictions = first[0]
        if len(first) > 1:
            proto = find_proto(first[1:])
    else:
        predictions = first

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


def decode_yolo_predictions(
    *,
    predictions: Any,
    class_count: int,
    class_name_count: int,
    mask_coeff_count: int,
    proto: Any | None,
    confidence_threshold: float,
    original_width: int,
    original_height: int,
    gain: float,
    pad: tuple[int, int],
) -> tuple[Any, Any, Any, Any | None]:
    """Decode YOLO raw and YOLO26 end-to-end detection/segmentation outputs."""

    pred = predictions[0] if predictions.ndim == 3 else predictions

    if pred.ndim != 2:
        raise RuntimeError(
            f"Unexpected YOLO output shape: {tuple(predictions.shape)}"
        )

    nc = int(class_count or class_name_count or 0)

    proto_mask_dim = (
        int(proto.shape[0])
        if proto is not None and getattr(proto, "ndim", 0) == 3
        else int(proto.shape[1])
        if proto is not None and getattr(proto, "ndim", 0) == 4
        else 0
    )

    mask_dim = int(mask_coeff_count or proto_mask_dim or 0)

    rows = int(pred.shape[0])
    cols = int(pred.shape[1])

    # -------------------------------------------------------------
    # YOLO26 END-TO-END
    #
    # Detection:
    #   [300, 6]
    #   x1, y1, x2, y2, conf, cls
    #
    # Segmentation:
    #   [300, 6 + nm]
    #   x1, y1, x2, y2, conf, cls, mask_coeffs...
    #
    # Important: determine this BEFORE doing any transpose.
    # -------------------------------------------------------------

    end2end_detect = (
        rows > cols
        and cols == 6
    )

    end2end_segment = (
        rows > cols
        and mask_dim > 0
        and cols == 6 + mask_dim
    )

    if end2end_detect or end2end_segment:
        boxes = pred[:, :4].clone()       # already xyxy
        scores = pred[:, 4]
        class_ids = pred[:, 5].long()

        mask_coefficients = None

        if end2end_segment:
            mask_coefficients = pred[:, 6 : 6 + mask_dim]

        keep = torch.isfinite(scores)
        keep &= scores >= float(confidence_threshold)

        if nc:
            keep &= class_ids >= 0
            keep &= class_ids < nc

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if mask_coefficients is not None:
            mask_coefficients = mask_coefficients[keep]

        if boxes.numel() == 0:
            return (
                boxes.reshape(0, 4),
                scores,
                class_ids,
                mask_coefficients,
            )

        boxes = scale_boxes_from_letterbox(
            boxes,
            original_width=original_width,
            original_height=original_height,
            gain=gain,
            pad=pad,
        )

        return boxes, scores, class_ids, mask_coefficients

    # -------------------------------------------------------------
    # TRADITIONAL / RAW YOLO
    #
    # Typical YOLO11:
    #
    #   [4 + nc, N]
    #   [4 + nc + nm, N]
    #
    # or older objectness-based layouts:
    #
    #   [5 + nc, N]
    # -------------------------------------------------------------

    possible_channels: set[int] = set()

    if nc:
        possible_channels.update({
            4 + nc,
            5 + nc,
        })

        if mask_dim:
            possible_channels.update({
                4 + nc + mask_dim,
                5 + nc + mask_dim,
            })

    channels_first_raw = (
        rows in possible_channels
        and cols not in possible_channels
    )

    unknown_names_channels_first = (
        not possible_channels
        and rows < cols
        and rows <= 512
    )

    if channels_first_raw or unknown_names_channels_first:
        pred = pred.transpose(0, 1).contiguous()

    channels = int(pred.shape[1])

    if channels < 5:
        empty = torch.empty((0,), device=pred.device)
        return (
            torch.empty((0, 4), device=pred.device),
            empty,
            empty.long(),
            None,
        )

    mask_coefficients = None

    # YOLO11 / modern raw segmentation:
    #
    # cx, cy, w, h, class scores..., mask coeffs...
    if nc and mask_dim and channels == 4 + nc + mask_dim:
        boxes = xywh_to_xyxy(pred[:, :4])

        scores, class_ids = pred[
            :, 4 : 4 + nc
        ].max(dim=1)

        mask_coefficients = pred[
            :, 4 + nc : 4 + nc + mask_dim
        ]

    # Older objectness-based segmentation:
    #
    # cx, cy, w, h, obj, classes..., masks...
    elif nc and mask_dim and channels == 5 + nc + mask_dim:
        boxes = xywh_to_xyxy(pred[:, :4])

        class_scores, class_ids = pred[
            :, 5 : 5 + nc
        ].max(dim=1)

        scores = pred[:, 4] * class_scores

        mask_coefficients = pred[
            :, 5 + nc : 5 + nc + mask_dim
        ]

    # YOLO11 / modern raw detection
    elif nc and channels == 4 + nc:
        boxes = xywh_to_xyxy(pred[:, :4])

        scores, class_ids = pred[
            :, 4 : 4 + nc
        ].max(dim=1)

    # Older objectness-based detection
    elif nc and channels == 5 + nc:
        boxes = xywh_to_xyxy(pred[:, :4])

        class_scores, class_ids = pred[
            :, 5 : 5 + nc
        ].max(dim=1)

        scores = pred[:, 4] * class_scores

    # Infer class count only as a fallback.
    elif mask_dim and channels > 4 + mask_dim:
        inferred_nc = channels - 4 - mask_dim

        boxes = xywh_to_xyxy(pred[:, :4])

        scores, class_ids = pred[
            :, 4 : 4 + inferred_nc
        ].max(dim=1)

        mask_coefficients = pred[
            :,
            4 + inferred_nc : 4 + inferred_nc + mask_dim,
        ]

    else:
        boxes = xywh_to_xyxy(pred[:, :4])
        scores, class_ids = pred[:, 4:].max(dim=1)

    # Raw predictions need confidence filtering.
    keep = torch.isfinite(scores)
    keep &= scores >= float(confidence_threshold)

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

    boxes = scale_boxes_from_letterbox(
        boxes,
        original_width=original_width,
        original_height=original_height,
        gain=gain,
        pad=pad,
    )

    return boxes, scores, class_ids, mask_coefficients


def greedy_class_aware_clusters(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    *,
    iou_threshold: float,
    iom_threshold: float,
) -> list[list[int]]:
    """Cluster overlapping boxes using the detector's confidence-seeded rule."""
    if boxes.numel() == 0:
        return []

    remaining = scores.argsort(descending=True)
    clusters: list[list[int]] = []

    while remaining.numel() > 0:
        seed_index = int(remaining[0].item())
        candidate_indices = remaining
        iou, iom = box_iou_and_iom(
            boxes[seed_index],
            boxes[candidate_indices],
        )
        same_class = class_ids[candidate_indices] == class_ids[seed_index]
        belongs = same_class & (
            (iou >= float(iou_threshold))
            | (iom >= float(iom_threshold))
        )
        belongs[0] = True

        clusters.append(
            [int(index.item()) for index in candidate_indices[belongs]]
        )
        remaining = candidate_indices[~belongs]

    return clusters


def show_yolo_results_pil(
    results: str | Path | Mapping[str, Any],
    *,
    confidence_threshold: float = 0.25,
    draw_boxes: bool = True,
    draw_masks: bool = True,
    draw_labels: bool = True,
    mask_alpha: int = 90,
    box_width: int = 4,
    font_path: str | Path | None = None,
    font_size: int = 24,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Image.Image:
    """
    Draw YOLO detection and segmentation results using Pillow.

    Expected detection format:
        {
            "detections": [
                {
                    "class_id": 39,
                    "class_name": "bottle",
                    "confidence": 0.95,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "mask": {
                        "format": "polygon",
                        "polygons": [
                            {
                                "points_xy": [[x, y], ...],
                                "is_hole": False
                            }
                        ]
                    }
                }
            ]
        }

    Args:
        image:
            Input image path or PIL image.

        results:
            YOLO result dictionary or path to a JSON/text file.

        confidence_threshold:
            Ignore detections below this confidence.

        mask_alpha:
            Segmentation opacity from 0 to 255.

        output_path:
            Optional path for saving the annotated image.

        show:
            Open the result with PIL.Image.show().

    Returns:
        Annotated RGB PIL image.
    """
    base_image = _load_image(results['input_jpg_path'])
    result_data = _load_results(results)

    # RGBA is needed for transparent segmentation overlays.
    annotated = base_image.convert("RGBA")
    image_width, image_height = annotated.size
    font = _load_font(font_path, font_size)

    detections = result_data.get("detections", [])

    # Draw lower-confidence detections first, so stronger detections remain visible.
    detections = sorted(
        detections,
        key=lambda item: float(item.get("confidence", 0.0)),
    )

    for detection in detections:
        confidence = float(detection.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue

        class_id = int(detection.get("class_id", 0))
        class_name = str(detection.get("class_name", class_id))
        color = _class_color(class_id)

        if draw_masks:
            _draw_detection_mask(
                annotated,
                detection,
                color=color,
                alpha=mask_alpha,
                image_size=(image_width, image_height),
            )

        draw = ImageDraw.Draw(annotated)

        bbox = detection.get("bbox_xyxy")
        if draw_boxes and bbox and len(bbox) == 4:
            x1, y1, x2, y2 = _clamp_bbox(
                bbox,
                image_width=image_width,
                image_height=image_height,
            )

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(*color, 255),
                width=box_width,
            )

            if draw_labels:
                label = f"{class_name} {confidence:.2f}"
                _draw_label(
                    draw,
                    position=(x1, y1),
                    text=label,
                    font=font,
                    color=color,
                    image_size=(image_width, image_height),
                )

    output = annotated.convert("RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)

    if show:
        output.show()

    return output


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy().convert("RGB")

    image_path = Path(image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def _load_results(
    results: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(results, Mapping):
        return dict(results)

    results_path = Path(results)
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    text = results_path.read_text(encoding="utf-8")

    try:
        # Normal JSON files.
        loaded = json.loads(text)
    except json.JSONDecodeError:
        # Also accepts a pasted Python dictionary using single quotes.
        loaded = ast.literal_eval(text)

    if not isinstance(loaded, dict):
        raise ValueError("The YOLO results must contain a dictionary/object.")

    return loaded


def _draw_detection_mask(
    image: Image.Image,
    detection: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    alpha: int,
    image_size: tuple[int, int],
) -> None:
    mask_data = detection.get("mask")
    if not isinstance(mask_data, Mapping):
        return

    if mask_data.get("format") != "polygon":
        return

    polygons = mask_data.get("polygons", [])
    if not polygons:
        return

    width, height = image_size

    # One grayscale mask per detection.
    instance_mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(instance_mask)

    # Draw regular polygons first.
    for polygon in polygons:
        if polygon.get("is_hole", False):
            continue

        points = _prepare_polygon_points(
            polygon.get("points_xy", []),
            width=width,
            height=height,
        )

        if len(points) >= 3:
            mask_draw.polygon(points, fill=255)

    # Remove hole polygons after the outer polygons have been filled.
    for polygon in polygons:
        if not polygon.get("is_hole", False):
            continue

        points = _prepare_polygon_points(
            polygon.get("points_xy", []),
            width=width,
            height=height,
        )

        if len(points) >= 3:
            mask_draw.polygon(points, fill=0)

    # Apply the requested opacity to the binary mask.
    alpha = max(0, min(255, int(alpha)))
    instance_mask = instance_mask.point(
        lambda value: alpha if value > 0 else 0
    )

    overlay = Image.new("RGBA", image.size, (*color, 0))
    overlay.putalpha(instance_mask)
    image.alpha_composite(overlay)


def _prepare_polygon_points(
    points: list[list[float]],
    *,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    prepared: list[tuple[int, int]] = []

    for point in points:
        if len(point) < 2:
            continue

        x = max(0, min(width - 1, round(float(point[0]))))
        y = max(0, min(height - 1, round(float(point[1]))))
        prepared.append((x, y))

    return prepared


def _clamp_bbox(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)

    x1 = max(0, min(image_width - 1, round(x1)))
    y1 = max(0, min(image_height - 1, round(y1)))
    x2 = max(0, min(image_width - 1, round(x2)))
    y2 = max(0, min(image_height - 1, round(y2)))

    # Protect against incorrectly ordered coordinates.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    x, y = position
    image_width, image_height = image_size
    padding = 4

    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    label_width = text_width + padding * 2
    label_height = text_height + padding * 2

    # Prefer drawing above the box. Move below it when near the top edge.
    label_y = y - label_height
    if label_y < 0:
        label_y = y

    label_x = min(x, max(0, image_width - label_width))
    label_y = min(label_y, max(0, image_height - label_height))

    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height,
        ),
        fill=(*color, 230),
    )

    # Choose black or white text based on the background brightness.
    brightness = (
        0.299 * color[0]
        + 0.587 * color[1]
        + 0.114 * color[2]
    )
    text_color = (0, 0, 0, 255) if brightness > 150 else (255, 255, 255, 255)

    draw.text(
        (label_x + padding, label_y + padding),
        text,
        fill=text_color,
        font=font,
    )


def _load_font(
    font_path: str | Path | None,
    font_size: int,
) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)

    # DejaVu Sans is commonly included with Pillow/Linux installations.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _class_color(class_id: int) -> tuple[int, int, int]:
    """
    Produce a stable, visually distinct color from a class ID.
    """
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)

    return (
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )

def plot_yolo_mixed_predictions(
    result,
    *,
    draw_masks: bool = True,
    draw_mask_outlines: bool = True,
    draw_boxes: bool = True,
    mask_opacity: float = 0.28,
    mask_outline_opacity: float = 0.85,
    box_opacity: float = 0.95,
    label_opacity: float = 0.58,
    mask_outline_width: int = 2,
    box_width: int = 3,
    font_size: int = 18,
    font_path: Optional[str] = None,
    show_confidence: bool = True,
    group_overlap: float = 0.25,
    padding: int = 7,
    line_spacing: int = 4,
    color_by: Literal["class", "instance"] = "instance",
    orig_is_bgr: bool = True,
) -> Image.Image:
    """
    Render overlapping Ultralytics YOLO predictions as a PIL image.

    Drawing order:
        1. All segmentation fills, combined without front/back ordering.
        2. All segmentation outlines, combined without front/back ordering.
        3. All bounding-box outlines, combined without front/back ordering.
        4. Grouped translucent label panels.

    At pixels shared by multiple predictions:
        - The contributing colors are averaged.
        - Opacity is accumulated.
        - No later prediction erases an earlier prediction.
        results = model("image.jpg")

    pretty_image = plot_yolo_mixed_predictions(
        results[0],

        # Every instance gets its own color.
        color_by="instance",

        # Transparent segmentation.
        mask_opacity=0.22,
        mask_outline_opacity=0.80,
        mask_outline_width=2,

        # All boxes remain visible.
        box_opacity=0.90,
        box_width=3,

        # Existing grouped-label style.
        label_opacity=0.55,
        group_overlap=0.20,
    )

    pretty_image.show()
    pretty_image.save("mixed_predictions.png")

    Args:
        result:
            One Ultralytics Results object, such as results[0].

        draw_masks:
            Draw segmentation fills when result.masks is available.

        draw_mask_outlines:
            Draw segmentation polygon boundaries.

        draw_boxes:
            Draw every bounding box.

        mask_opacity:
            Per-mask fill opacity from 0.0 to 1.0.

        mask_outline_opacity:
            Per-mask boundary opacity from 0.0 to 1.0.

        box_opacity:
            Bounding-box opacity from 0.0 to 1.0.

        label_opacity:
            Label-panel background opacity from 0.0 to 1.0.

        mask_outline_width:
            Segmentation-boundary width in pixels.

        box_width:
            Bounding-box width in pixels.

        font_size:
            Label font size.

        font_path:
            Optional path to a TrueType or OpenType font.

        show_confidence:
            Include confidence values in labels.

        group_overlap:
            Group labels when:

                intersection_area / smaller_box_area

            is greater than or equal to this value.

        padding:
            Label-panel inner padding.

        line_spacing:
            Spacing between rows in grouped labels.

        color_by:
            "class":
                All detections of the same class use the same color.

            "instance":
                Every prediction receives a different color. This is usually
                clearer for strongly overlapping predictions.

        orig_is_bgr:
            True for normal Ultralytics/OpenCV result.orig_img arrays.
            Ignored when result.orig_img is already a PIL image.

    Returns:
        PIL.Image.Image in RGB mode.
    """

    def clamp_opacity(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def original_to_pil() -> Image.Image:
        original = result.orig_img

        if isinstance(original, Image.Image):
            return original.convert("RGB")

        array = np.asarray(original)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")

        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                "result.orig_img must be a PIL image or an HxW, "
                "HxWx3, or HxWx4 NumPy array."
            )

        if array.shape[2] == 4:
            if orig_is_bgr:
                array = array[..., [2, 1, 0, 3]]

            return Image.fromarray(array).convert("RGB")

        if orig_is_bgr:
            array = array[..., ::-1]

        return Image.fromarray(array).convert("RGB")

    def prediction_color(key: int) -> tuple[int, int, int]:
        """Return a stable, visually separated RGB color."""

        hue = (int(key) * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(
            hue,
            0.68,
            0.96,
        )

        return (
            round(red * 255),
            round(green * 255),
            round(blue * 255),
        )

    def get_class_name(class_id: int) -> str:
        names = result.names

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def overlap_over_smaller(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))

        intersection = (
            max(0.0, right - left)
            * max(0.0, bottom - top)
        )

        first_area = (
            max(0.0, float(first[2] - first[0]))
            * max(0.0, float(first[3] - first[1]))
        )

        second_area = (
            max(0.0, float(second[2] - second[0]))
            * max(0.0, float(second[3] - second[1]))
        )

        smaller_area = min(first_area, second_area)

        if smaller_area <= 0:
            return 0.0

        return intersection / smaller_area

    def composite_all_layers(
        base: Image.Image,
        coverage_layers: list[np.ndarray],
        layer_colors: list[tuple[int, int, int]],
        opacity: float,
    ) -> Image.Image:
        """
        Composite all layers simultaneously.

        For a pixel covered by k predictions:

            color = average of all contributing colors
            alpha = 1 - (1 - opacity) ** k

        This makes the operation independent of prediction order.
        """

        opacity = clamp_opacity(opacity)

        if opacity <= 0 or not coverage_layers:
            return base

        height, width = coverage_layers[0].shape

        color_sum = np.zeros(
            (height, width, 3),
            dtype=np.float32,
        )

        coverage_count = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        for coverage, color in zip(
            coverage_layers,
            layer_colors,
        ):
            active = np.asarray(coverage, dtype=bool)

            if active.shape != (height, width):
                raise ValueError(
                    "All coverage layers must match the image dimensions."
                )

            coverage_count[active] += 1.0
            color_sum[active] += np.asarray(
                color,
                dtype=np.float32,
            )

        active_pixels = coverage_count > 0

        if not np.any(active_pixels):
            return base

        mixed_colors = np.zeros_like(color_sum)

        mixed_colors[active_pixels] = (
            color_sum[active_pixels]
            / coverage_count[active_pixels, None]
        )

        combined_alpha = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        combined_alpha[active_pixels] = (
            1.0
            - np.power(
                1.0 - opacity,
                coverage_count[active_pixels],
            )
        )

        base_array = np.asarray(
            base.convert("RGB"),
            dtype=np.float32,
        )

        output = (
            base_array
            * (1.0 - combined_alpha[..., None])
            + mixed_colors
            * combined_alpha[..., None]
        )

        return Image.fromarray(
            np.clip(output, 0, 255).astype(np.uint8)
        )

    def build_polygon_layers(
        polygons,
        image_size: tuple[int, int],
        outline_width: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        image_width, image_height = image_size

        fill_layers: list[np.ndarray] = []
        outline_layers: list[np.ndarray] = []

        for polygon in polygons:
            fill_image = Image.new(
                "1",
                (image_width, image_height),
                0,
            )

            outline_image = Image.new(
                "1",
                (image_width, image_height),
                0,
            )

            polygon_array = np.asarray(
                polygon,
                dtype=np.float32,
            )

            if (
                polygon_array.ndim != 2
                or polygon_array.shape[0] < 3
                or polygon_array.shape[1] != 2
            ):
                empty = np.zeros(
                    (image_height, image_width),
                    dtype=bool,
                )

                fill_layers.append(empty)
                outline_layers.append(empty.copy())
                continue

            points = [
                (
                    int(np.clip(round(x), 0, image_width - 1)),
                    int(np.clip(round(y), 0, image_height - 1)),
                )
                for x, y in polygon_array
            ]

            ImageDraw.Draw(fill_image).polygon(
                points,
                fill=1,
            )

            if outline_width > 0:
                closed_points = points + [points[0]]

                ImageDraw.Draw(outline_image).line(
                    closed_points,
                    fill=1,
                    width=outline_width,
                    joint="curve",
                )

            fill_layers.append(
                np.asarray(fill_image, dtype=bool)
            )

            outline_layers.append(
                np.asarray(outline_image, dtype=bool)
            )

        return fill_layers, outline_layers

    def build_box_layers(
        boxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
        width: int,
    ) -> list[np.ndarray]:
        image_width, image_height = image_size
        layers: list[np.ndarray] = []

        for box in boxes:
            layer_image = Image.new(
                "1",
                (image_width, image_height),
                0,
            )

            ImageDraw.Draw(layer_image).rectangle(
                box,
                outline=1,
                width=max(1, width),
            )

            layers.append(
                np.asarray(layer_image, dtype=bool)
            )

        return layers

    def rectangles_intersect(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        margin: int = 5,
    ) -> bool:
        return not (
            first[2] + margin <= second[0]
            or second[2] + margin <= first[0]
            or first[3] + margin <= second[1]
            or second[3] + margin <= first[1]
        )

    def place_label_panel(
        group_box: tuple[int, int, int, int],
        panel_width: int,
        panel_height: int,
        occupied: list[tuple[int, int, int, int]],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        group_x1, group_y1, group_x2, group_y2 = group_box

        max_x = max(0, image_width - panel_width)
        max_y = max(0, image_height - panel_height)

        def position_is_free(x: int, y: int) -> bool:
            candidate = (
                x,
                y,
                x + panel_width,
                y + panel_height,
            )

            return not any(
                rectangles_intersect(candidate, previous)
                for previous in occupied
            )

        candidates = [
            # Above the group.
            (
                group_x1,
                group_y1 - panel_height - 5,
            ),
            (
                group_x2 - panel_width,
                group_y1 - panel_height - 5,
            ),

            # Below the group.
            (
                group_x1,
                group_y2 + 5,
            ),
            (
                group_x2 - panel_width,
                group_y2 + 5,
            ),

            # Fallback inside the group.
            (
                group_x1 + 5,
                group_y1 + 5,
            ),
        ]

        for x, y in candidates:
            x = int(np.clip(x, 0, max_x))
            y = int(np.clip(y, 0, max_y))

            if position_is_free(x, y):
                return x, y

        search_x = int(np.clip(group_x1, 0, max_x))
        search_step = max(12, panel_height // 3)

        for search_y in range(
            0,
            max_y + 1,
            search_step,
        ):
            if position_is_free(search_x, search_y):
                return search_x, search_y

        return (
            int(np.clip(group_x1, 0, max_x)),
            int(np.clip(group_y1, 0, max_y)),
        )

    image = original_to_pil()
    image_width, image_height = image.size

    boxes_object = getattr(result, "boxes", None)

    if boxes_object is None or len(boxes_object) == 0:
        return image

    boxes = (
        boxes_object.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    classes = (
        boxes_object.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    if boxes_object.conf is None:
        confidences = np.ones(
            len(boxes),
            dtype=np.float32,
        )
    else:
        confidences = (
            boxes_object.conf
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    normalized_boxes: list[
        tuple[int, int, int, int]
    ] = []

    for x1, y1, x2, y2 in boxes:
        normalized_boxes.append(
            (
                int(np.clip(round(x1), 0, image_width - 1)),
                int(np.clip(round(y1), 0, image_height - 1)),
                int(np.clip(round(x2), 0, image_width - 1)),
                int(np.clip(round(y2), 0, image_height - 1)),
            )
        )

    colors: list[tuple[int, int, int]] = []

    for index, class_id in enumerate(classes):
        color_key = (
            int(class_id)
            if color_by == "class"
            else index
        )

        colors.append(
            prediction_color(color_key)
        )

    # ------------------------------------------------------------
    # 1. Draw all segmentation fills simultaneously.
    # ------------------------------------------------------------

    masks_object = getattr(result, "masks", None)

    if masks_object is not None and len(masks_object) > 0:
        polygons = list(masks_object.xy)

        usable_count = min(
            len(polygons),
            len(colors),
        )

        polygons = polygons[:usable_count]
        mask_colors = colors[:usable_count]

        mask_fill_layers, mask_boundary_layers = (
            build_polygon_layers(
                polygons,
                image.size,
                max(1, mask_outline_width),
            )
        )

        if draw_masks:
            image = composite_all_layers(
                image,
                mask_fill_layers,
                mask_colors,
                mask_opacity,
            )

        # --------------------------------------------------------
        # 2. Draw all mask boundaries simultaneously.
        # --------------------------------------------------------

        if (
            draw_mask_outlines
            and mask_outline_width > 0
        ):
            image = composite_all_layers(
                image,
                mask_boundary_layers,
                mask_colors,
                mask_outline_opacity,
            )

    # ------------------------------------------------------------
    # 3. Draw all bounding boxes simultaneously.
    # ------------------------------------------------------------

    if draw_boxes and box_width > 0:
        box_layers = build_box_layers(
            normalized_boxes,
            image.size,
            box_width,
        )

        image = composite_all_layers(
            image,
            box_layers,
            colors,
            box_opacity,
        )

    # ------------------------------------------------------------
    # 4. Draw the grouped translucent labels.
    # ------------------------------------------------------------

    try:
        font = ImageFont.truetype(
            font_path or "DejaVuSans.ttf",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    # Union-find grouping for overlapping detections.
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]

        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)

        if first_root != second_root:
            parents[second_root] = first_root

    overlap_threshold = float(
        np.clip(group_overlap, 0.0, 1.0)
    )

    for first_index in range(len(boxes)):
        for second_index in range(
            first_index + 1,
            len(boxes),
        ):
            overlap = overlap_over_smaller(
                boxes[first_index],
                boxes[second_index],
            )

            if overlap >= overlap_threshold:
                union(first_index, second_index)

    groups: dict[int, list[int]] = defaultdict(list)

    for index in range(len(boxes)):
        groups[find(index)].append(index)

    grouped_indices = sorted(
        groups.values(),
        key=lambda group: min(
            boxes[index][1]
            for index in group
        ),
    )

    label_layer = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    label_draw = ImageDraw.Draw(label_layer)

    occupied_panels: list[
        tuple[int, int, int, int]
    ] = []

    panel_alpha = round(
        clamp_opacity(label_opacity) * 255
    )

    for group in grouped_indices:
        # Highest confidence first.
        group = sorted(
            group,
            key=lambda index: float(confidences[index]),
            reverse=True,
        )

        rows = []

        for index in group:
            text = get_class_name(
                int(classes[index])
            )

            if show_confidence:
                text += (
                    f"  {float(confidences[index]):.2f}"
                )

            text_box = label_draw.textbbox(
                (0, 0),
                text,
                font=font,
            )

            rows.append(
                {
                    "index": index,
                    "text": text,
                    "bbox": text_box,
                    "width": text_box[2] - text_box[0],
                    "height": text_box[3] - text_box[1],
                }
            )

        swatch_size = max(9, font_size // 2)
        swatch_gap = 7

        panel_width = (
            max(
                row["width"]
                + swatch_size
                + swatch_gap
                for row in rows
            )
            + padding * 2
        )

        panel_height = (
            padding * 2
            + sum(
                row["height"]
                for row in rows
            )
            + line_spacing
            * max(0, len(rows) - 1)
        )

        panel_width = min(
            panel_width,
            image_width,
        )

        panel_height = min(
            panel_height,
            image_height,
        )

        group_box = (
            round(
                min(
                    boxes[index][0]
                    for index in group
                )
            ),
            round(
                min(
                    boxes[index][1]
                    for index in group
                )
            ),
            round(
                max(
                    boxes[index][2]
                    for index in group
                )
            ),
            round(
                max(
                    boxes[index][3]
                    for index in group
                )
            ),
        )

        panel_x, panel_y = place_label_panel(
            group_box=group_box,
            panel_width=panel_width,
            panel_height=panel_height,
            occupied=occupied_panels,
            image_width=image_width,
            image_height=image_height,
        )

        panel_box = (
            panel_x,
            panel_y,
            panel_x + panel_width,
            panel_y + panel_height,
        )

        occupied_panels.append(panel_box)

        label_draw.rounded_rectangle(
            panel_box,
            radius=7,
            fill=(10, 10, 12, panel_alpha),
            outline=(255, 255, 255, 55),
            width=1,
        )

        current_y = panel_y + padding

        for row in rows:
            row_height = row["height"]
            row_center_y = (
                current_y + row_height // 2
            )

            swatch_y = (
                row_center_y
                - swatch_size // 2
            )

            label_draw.rounded_rectangle(
                (
                    panel_x + padding,
                    swatch_y,
                    panel_x + padding + swatch_size,
                    swatch_y + swatch_size,
                ),
                radius=2,
                fill=(
                    *colors[row["index"]],
                    245,
                ),
            )

            label_draw.text(
                (
                    panel_x
                    + padding
                    + swatch_size
                    + swatch_gap,
                    current_y - row["bbox"][1],
                ),
                row["text"],
                font=font,
                fill=(255, 255, 255, 255),
            )

            current_y += (
                row_height + line_spacing
            )

    return Image.alpha_composite(
        image.convert("RGBA"),
        label_layer,
    ).convert("RGB")




