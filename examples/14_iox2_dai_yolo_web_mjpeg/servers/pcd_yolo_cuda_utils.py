#!/usr/bin/env python3
"""CUDA-first RGB-indexed StereoSGBM cloud builder and YOLO mask splitter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from pcd_cuda_utils import (
    StereoRgbCalibrationCpu,
    StereoRgbCalibrationCuda as StereoRgbCalibration,
    SGBMDisparityPredictorCuda as SGBMDisparityPredictor,
    _image_cuda,
    points_left_to_rgb_depth,
    read_image,
    rectified_left_to_original_left,
    rgb_depth_to_points_rgb,
    sample_rgb_colors,
    save_point_cloud,
    transform_points,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "object"


def detection_mask(detection: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Rasterize one YOLO polygon/bbox mask on CPU; upload only when it is used."""
    mask = np.zeros((height, width), np.uint8)
    mask_info = detection.get("mask") or {}
    if mask_info.get("format") != "polygon":
        bbox = detection.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            return mask.astype(bool)
        x1, y1 = np.floor(bbox[:2]).astype(int)
        x2, y2 = np.ceil(bbox[2:]).astype(int)
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


@torch.no_grad()
def build_rgb_indexed_cloud_sgbm(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    predictor: SGBMDisparityPredictor,
    *,
    device="cuda",
    min_disparity=0,
    min_depth_m=.01,
    max_depth_m=5.0,
    stride=1,
    alpha=0.0,
    splat_px=0,
    rgb_image_is_undistorted=False,
    stereo_input_color_order="BGR",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an RGB-resolution cloud; SGBM stays CPU, geometry stays on CUDA."""
    left_cuda_u8, right_cuda_u8 = _image_cuda(left_image), _image_cuda(right_image)
    rgb_cuda_u8 = _image_cuda(rgb_image)

    rectifier = calibration.get_rectifier(alpha)
    left, right, rect_cuda = rectifier.rectify(left_cuda_u8, right_cuda_u8)
    disparity = predictor.predict(left, right)
    cal = calibration.to_cuda(device)

    points, _ = rect_cuda.disparity_to_points_rectified(
        disparity,
        min_disparity=max(.5, float(min_disparity)),
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        stride=stride,
    )
    if points.shape[0] == 0:
        raise RuntimeError("StereoSGBM produced no valid 3D points")

    points_left = rectified_left_to_original_left(points, rect_cuda)
    depth_rgb, _ = points_left_to_rgb_depth(
        points_left, rgb_cuda_u8, cal,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
        splat_px=splat_px,
    )
    points_rgb, pixel_xy = rgb_depth_to_points_rgb(
        depth_rgb, cal,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )
    if points_rgb.shape[0] == 0:
        raise RuntimeError("No SGBM points projected into the RGB image")

    points_left = transform_points(points_rgb, torch.linalg.inv(cal.left_to_rgb))
    colors_rgb, valid = sample_rgb_colors(
        rgb_cuda_u8, pixel_xy,
        input_color_order=stereo_input_color_order,
    )
    # pixel_xy comes from valid depth pixels, but keep this robust if sampling rules change.
    if not bool(valid.all().item()):
        points_left, pixel_xy = points_left[valid], pixel_xy[valid]

    return points_left, colors_rgb, pixel_xy, disparity


@torch.no_grad()
def split_cloud_cuda(
    points_left: torch.Tensor,
    colors_rgb: torch.Tensor,
    pixel_xy: torch.Tensor,
    detections_json: dict[str, Any],
    output_dir: str | Path,
    *,
    min_points=30,
    erode_pixels=0,
    exclusive=False,
    save_background=False,
    save_full_cloud=False,
    binary_pcd=True,
) -> list[dict[str, Any]]:
    """Split a CUDA point cloud by YOLO masks; only mask rasterization/file I/O is CPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = points_left.device
    width, height = int(detections_json["image_width"]), int(detections_json["image_height"])

    x, y = pixel_xy[:, 0].long(), pixel_xy[:, 1].long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    points_left, colors_rgb, x, y = points_left[valid], colors_rgb[valid], x[valid], y[valid]

    detections = list(enumerate(detections_json.get("detections", [])))
    if exclusive:
        detections.sort(key=lambda item: float(item[1].get("confidence", 0)), reverse=True)

    kernel = np.ones((2 * erode_pixels + 1,) * 2, np.uint8) if erode_pixels else None
    claimed = torch.zeros(len(points_left), dtype=torch.bool, device=device)
    union = torch.zeros_like(claimed)
    manifest = []

    for index, detection in detections:
        mask = detection_mask(detection, height, width)
        if kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)

        keep = mask[y, x]
        union |= keep
        if exclusive:
            keep &= ~claimed
        count = int(keep.sum().item())
        if count < min_points:
            print(f"skip detection {index}: {count} points ({detection.get('class_name', 'unknown')})")
            continue
        if exclusive:
            claimed |= keep

        class_id = int(detection.get("class_id", -1))
        class_name = str(detection.get("class_name", "object"))
        confidence = float(detection.get("confidence", 0))
        filename = f"{index:03d}_class{class_id}_{safe_name(class_name)}_{confidence:.3f}_{count}pts.pcd"
        save_point_cloud(output_dir / filename, points_left[keep], colors_rgb[keep], binary_pcd=binary_pcd)
        manifest.append({
            "detection_index": index,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "point_count": count,
            "pcd": filename,
        })
        print(f"saved {output_dir / filename} ({count} points)")

    if save_full_cloud:
        save_point_cloud(output_dir / "full.pcd", points_left, colors_rgb, binary_pcd=binary_pcd)

    if save_background:
        background = ~(claimed if exclusive else union)
        count = int(background.sum().item())
        if count:
            filename = f"background_{count}pts.pcd"
            save_point_cloud(output_dir / filename, points_left[background], colors_rgb[background], binary_pcd=binary_pcd)
            print(f"saved background ({count} points)")

    return manifest


if __name__ == "__main__":
    pass