#!/usr/bin/env python3
"""Build an RGB-indexed VPI/CuPy stereo cloud and split it by YOLO masks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import cupy as cp
import numpy as np

from pcd_vpi_utils import (
    StereoRgbCalibration,
    VPIStereoDisparityGPU,
    points_left_to_rgb_depth,
    read_image,
    rectified_left_to_original_left,
    rgb8,
    rgb_depth_to_points_rgb,
    save_point_cloud,
    transform_points,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "object"


def detection_mask(detection: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Rasterize one YOLO detection into an RGB-image-sized boolean mask.

    Polygon masks are preferred. If a polygon mask is unavailable, bbox_xyxy is
    used as a fallback, matching pcd_yolo_utils.py.
    """
    mask = np.zeros((height, width), np.uint8)
    mask_info = detection.get("mask") or {}

    if mask_info.get("format") != "polygon":
        bbox = detection.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            return mask.astype(bool)

        x1, y1 = np.floor(np.asarray(bbox[:2], np.float32)).astype(int)
        x2, y2 = np.ceil(np.asarray(bbox[2:], np.float32)).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width)
        y1, y2 = np.clip((y1, y2), 0, height)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask.astype(bool)

    polygons = mask_info.get("polygons", [])
    for is_hole, value in ((False, 1), (True, 0)):
        contours = [
            np.rint(points).astype(np.int32)
            for polygon in polygons
            if bool(polygon.get("is_hole", False)) == is_hole
            if len(points := np.asarray(polygon.get("points_xy", []), np.float32)) >= 3
        ]
        if contours:
            cv2.fillPoly(mask, contours, value)

    return mask.astype(bool)


def build_rgb_indexed_cloud_vpi(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    predictor: VPIStereoDisparityGPU,
    *,
    min_disparity: float = 0.5,
    min_depth_m: float | None = 0.01,
    max_depth_m: float | None = 5.0,
    stride: int = 1,
    alpha: float = 0.0,
    splat_px: int = 0,
    rgb_image_is_undistorted: bool = False,
    stereo_input_color_order: str = "BGR",
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, Any]:
    """Build an RGB-pixel-indexed point cloud with VPI stereo + CuPy geometry.

    The returned point coordinates, RGB colors, and RGB pixel indices remain on
    the GPU as CuPy arrays. ``disparity`` is the VPI disparity image returned by
    ``VPIStereoDisparityGPU``.
    """
    rectifier = calibration.get_rectifier(alpha)
    left_rect, right_rect, rect = rectifier.rectify(left_image, right_image)

    disparity, confidence_u16 = predictor.predict(left_rect, right_rect)
    points_rect, _ = rect.disparity_to_points_rectified(
        disparity,
        confidence_u16,
        min_disparity=max(0.5, float(min_disparity)),
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        stride=stride,
    )
    if len(points_rect) == 0:
        raise RuntimeError("VPI stereo disparity produced no valid 3D points")

    # Q/reprojection gives points in the rectified-left frame. Convert back to
    # the original left-camera frame before projecting into the RGB camera.
    points_left = rectified_left_to_original_left(points_rect, rect)

    # Z-buffer the projected stereo points in RGB image coordinates. Rebuilding
    # from this depth map makes pixel_xy exactly match the YOLO/RGB image grid.
    depth_rgb, _ = points_left_to_rgb_depth(
        points_left,
        rgb_image,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
        splat_px=splat_px,
    )
    points_rgb, pixel_xy = rgb_depth_to_points_rgb(
        depth_rgb,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )
    if len(points_rgb) == 0:
        raise RuntimeError("No VPI stereo points projected into the RGB image")

    # Return geometry in the original left-camera frame, like pcd_yolo_utils.
    left_to_rgb_inv = cp.linalg.inv(cp.asarray(calibration.left_to_rgb, dtype=cp.float32))
    points_left = transform_points(points_rgb, left_to_rgb_inv)

    rgb_gpu = cp.asarray(rgb_image)
    x, y = pixel_xy.T
    colors_rgb = rgb8(rgb_gpu[y, x, :3], order=stereo_input_color_order)

    return points_left, colors_rgb, pixel_xy, disparity


def _save_gpu_cloud(
    path: str | Path,
    points_left: cp.ndarray,
    colors_rgb: cp.ndarray,
    *,
    binary_pcd: bool,
) -> None:
    """Copy only the cloud being written to CPU and use the existing writer."""
    save_point_cloud(
        path,
        cp.asnumpy(points_left),
        cp.asnumpy(colors_rgb),
        binary_pcd=binary_pcd,
    )


def split_cloud_vpi(
    points_left: cp.ndarray,
    colors_rgb: cp.ndarray,
    pixel_xy: cp.ndarray,
    detections_json: dict[str, Any],
    output_dir: str | Path,
    *,
    min_points: int = 30,
    erode_pixels: int = 0,
    exclusive: bool = False,
    save_background: bool = False,
    save_full_cloud: bool = False,
    binary_pcd: bool = True,
) -> list[dict[str, Any]]:
    """Split a GPU point cloud using YOLO masks defined in RGB image pixels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_left = cp.asarray(points_left)
    colors_rgb = cp.asarray(colors_rgb)
    pixel_xy = cp.asarray(pixel_xy, dtype=cp.int32)

    if points_left.ndim != 2 or points_left.shape[1] != 3:
        raise ValueError(f"points_left must be Nx3, got {points_left.shape}")
    if colors_rgb.ndim != 2 or colors_rgb.shape[1] < 3:
        raise ValueError(f"colors_rgb must be Nx3, got {colors_rgb.shape}")
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"pixel_xy must be Nx2, got {pixel_xy.shape}")
    if not (len(points_left) == len(colors_rgb) == len(pixel_xy)):
        raise ValueError("points_left, colors_rgb, and pixel_xy must have equal length")

    width = int(detections_json["image_width"])
    height = int(detections_json["image_height"])
    x, y = pixel_xy.T

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not bool(cp.all(valid).item()):
        points_left = points_left[valid]
        colors_rgb = colors_rgb[valid]
        x, y = x[valid], y[valid]

    detections = list(enumerate(detections_json.get("detections", [])))
    if exclusive:
        detections.sort(
            key=lambda item: float(item[1].get("confidence", 0)),
            reverse=True,
        )

    if erode_pixels < 0:
        raise ValueError("erode_pixels must be >= 0")
    kernel = (
        np.ones((2 * erode_pixels + 1,) * 2, np.uint8)
        if erode_pixels
        else None
    )

    claimed = cp.zeros(len(points_left), dtype=cp.bool_)
    union = claimed.copy()
    manifest: list[dict[str, Any]] = []

    for index, detection in detections:
        mask = detection_mask(detection, height, width)
        if kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

        # Upload the YOLO mask, then select points directly with their RGB pixel
        # coordinates. Point data itself remains on GPU until a PCD is saved.
        mask_gpu = cp.asarray(mask)
        keep = mask_gpu[y, x]
        union |= keep

        if exclusive:
            keep &= ~claimed

        count = int(cp.count_nonzero(keep).item())
        if count < min_points:
            print(
                f"skip detection {index}: {count} points "
                f"({detection.get('class_name', 'unknown')})"
            )
            continue

        if exclusive:
            claimed |= keep

        class_id = int(detection.get("class_id", -1))
        class_name = str(detection.get("class_name", "object"))
        confidence = float(detection.get("confidence", 0))
        filename = (
            f"{index:03d}_class{class_id}_{safe_name(class_name)}_"
            f"{confidence:.3f}_{count}pts.pcd"
        )

        _save_gpu_cloud(
            output_dir / filename,
            points_left[keep],
            colors_rgb[keep],
            binary_pcd=binary_pcd,
        )
        manifest.append(
            {
                "detection_index": index,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "point_count": count,
                "pcd": filename,
            }
        )
        print(f"saved {output_dir / filename} ({count} points)")

    if save_full_cloud:
        _save_gpu_cloud(
            output_dir / "full.pcd",
            points_left,
            colors_rgb,
            binary_pcd=binary_pcd,
        )

    if save_background:
        background = ~(claimed if exclusive else union)
        count = int(cp.count_nonzero(background).item())
        if count:
            filename = f"background_{count}pts.pcd"
            _save_gpu_cloud(
                output_dir / filename,
                points_left[background],
                colors_rgb[background],
                binary_pcd=binary_pcd,
            )
            print(f"saved background ({count} points)")

    return manifest


if __name__ == "__main__":
    pass