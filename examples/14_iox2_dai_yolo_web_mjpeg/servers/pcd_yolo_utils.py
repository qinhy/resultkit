#!/usr/bin/env python3
"""Build an RGB-indexed StereoSGBM cloud and split it by YOLO masks."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pcd_utils import (
    StereoRgbCalibration,
    points_left_to_rgb_depth,
    read_image,
    SGBMDisparityPredictor,
    stereo_to_point_cloud,
    rectified_left_to_original_left,
    rgb8,
    rgb_depth_to_points_rgb,
    save_point_cloud,
    transform_points,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "object"


def detection_mask(detection: dict[str, Any], height: int, width: int) -> np.ndarray:
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


def build_rgb_indexed_cloud_sgbm(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    predictor:SGBMDisparityPredictor,
    *,
    min_disparity=0,
    max_depth_m=5.0,
    stride=1,
    alpha=0.0,
    splat_px=0,
    rgb_image_is_undistorted=False,
    stereo_input_color_order="BGR",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rectifier = calibration.get_rectifier(alpha)
    left, right, rect = rectifier.rectify(left_image, right_image)
    disparity = predictor.predict(left, right)
    points, _ = rect.disparity_to_points_rectified(disparity,
                                              min_disparity=max(0.5, float(min_disparity)),
                                              max_depth_m=max_depth_m, stride=stride)
    if not len(points):
        raise RuntimeError("StereoSGBM produced no valid 3D points")

    points = rectified_left_to_original_left(points, rect)
    depth_rgb, _ = points_left_to_rgb_depth(
        points, rgb_image, calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted, splat_px=splat_px,
    )
    points_rgb, pixel_xy = rgb_depth_to_points_rgb(
        depth_rgb, calibration, rgb_image_is_undistorted=rgb_image_is_undistorted
    )
    if not len(points_rgb):
        raise RuntimeError("No SGBM points projected into the RGB image")

    points_left = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))
    x, y = pixel_xy.T
    return points_left, rgb8(rgb_image[y, x, :3], order=stereo_input_color_order), pixel_xy, disparity


def split_cloud(
    points_left: np.ndarray,
    colors_rgb: np.ndarray,
    pixel_xy: np.ndarray,
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = int(detections_json["image_width"]), int(detections_json["image_height"])
    x, y = pixel_xy.T
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not valid.all():
        points_left, colors_rgb, x, y = points_left[valid], colors_rgb[valid], x[valid], y[valid]

    detections = list(enumerate(detections_json.get("detections", [])))
    if exclusive:
        detections.sort(key=lambda item: float(item[1].get("confidence", 0)), reverse=True)
    kernel = np.ones((2 * erode_pixels + 1,) * 2, np.uint8) if erode_pixels else None
    claimed = np.zeros(len(points_left), bool)
    union = claimed.copy()
    manifest = []

    for index, detection in detections:
        mask = detection_mask(detection, height, width)
        if kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        keep = mask[y, x]
        union |= keep
        if exclusive:
            keep &= ~claimed
        count = int(keep.sum())
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
            "detection_index": index, "class_id": class_id, "class_name": class_name,
            "confidence": confidence, "point_count": count, "pcd": filename,
        })
        print(f"saved {output_dir / filename} ({count} points)")

    if save_full_cloud:
        save_point_cloud(output_dir / "full.pcd", points_left, colors_rgb, binary_pcd=binary_pcd)

    if save_background:
        background = ~(claimed if exclusive else union)
        if count := int(background.sum()):
            filename = f"background_{count}pts.pcd"
            save_point_cloud(output_dir / filename, points_left[background], colors_rgb[background], binary_pcd=binary_pcd)
            print(f"saved background ({count} points)")
    return manifest


def split_cloud_uv(
    points_left: Any,
    uv: Any,
    rgb_image: Any,
    detections_json: dict[str, Any],
    output_dir: str | Path,
    *,
    rgb_image_color_order: str = "BGR",
    min_points: int = 30,
    erode_pixels: int = 0,
    exclusive: bool = False,
    save_background: bool = False,
    save_full_cloud: bool = False,
    binary_pcd: bool = True,
) -> list[dict[str, Any]]:
    """
    Split a stereo-derived 3D cloud using detections from the RGB image.

    Parameters
    ----------
    points_left:
        Nx3 3D points in the original LEFT-camera coordinate frame.

    uv:
        Nx2 floating-point RGB image coordinates corresponding one-to-one
        with points_left.

        uv[i] == [u, v] is the RGB projection of points_left[i].

    rgb_image:
        RGB-camera image, HxWx3.
        Usually BGR if loaded by OpenCV.

    detections_json:
        Detection/segmentation result using the same RGB image coordinate
        system.

    rgb_image_color_order:
        "BGR" for normal OpenCV images.
        "RGB" when rgb_image is already RGB.

    Notes
    -----
    This function intentionally converts to NumPy once because:
      - detection_mask() is CPU/NumPy
      - cv2.erode() is CPU
      - PCD writing is CPU/file I/O

    No RGB-depth rasterization or RGB-depth backprojection is required.
    """

    # ------------------------------------------------------------------
    # Validate input arrays
    # ------------------------------------------------------------------
    if points_left.ndim != 2 or points_left.shape[1] != 3:
        raise ValueError(
            f"points_left must be Nx3, got {points_left.shape}"
        )

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(
            f"uv must be Nx2, got {uv.shape}"
        )

    if len(points_left) != len(uv):
        raise ValueError(
            f"points_left and uv must have same length, got "
            f"{len(points_left)} and {len(uv)}"
        )

    if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
        raise ValueError(
            f"rgb_image must be HxWx3, got {rgb_image.shape}"
        )

    rgb_h, rgb_w = rgb_image.shape[:2]

    detection_width = int(detections_json["image_width"])
    detection_height = int(detections_json["image_height"])

    if (rgb_w, rgb_h) != (detection_width, detection_height):
        raise ValueError(
            f"RGB image size {(rgb_w, rgb_h)} differs from detection "
            f"size {(detection_width, detection_height)}"
        )

    # ------------------------------------------------------------------
    # Floating projection coordinates:
    #
    #     uv[:,0] = u / image x
    #     uv[:,1] = v / image y
    #
    # Convert to nearest RGB pixel.
    # ------------------------------------------------------------------
    uv_finite = np.isfinite(uv).all(axis=1)

    # Avoid converting NaN/Inf directly to integer.
    safe_uv = np.where(np.isfinite(uv), uv, 0)

    u = np.rint(safe_uv[:, 0]).astype(np.int64)
    v = np.rint(safe_uv[:, 1]).astype(np.int64)

    # ------------------------------------------------------------------
    # Keep only points that actually project inside the RGB image.
    # ------------------------------------------------------------------
    inside = (
        uv_finite
        & (u >= 0)
        & (u < rgb_w)
        & (v >= 0)
        & (v < rgb_h)
    )

    points_left = points_left[inside]
    u = u[inside]
    v = v[inside]

    if len(points_left) == 0:
        raise RuntimeError(
            "No 3D points project inside the RGB image"
        )

    # ------------------------------------------------------------------
    # Get RGB color belonging to every surviving 3D point.
    #
    # One-to-one relationship:
    #
    # points_left[i]
    #      ↕
    # (u[i], v[i])
    #      ↕
    # colors_rgb[i]
    # ------------------------------------------------------------------
    sampled_colors = rgb_image[v, u, :3]

    colors_rgb = rgb8(
        sampled_colors,
        order=rgb_image_color_order,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Detection ordering
    # ------------------------------------------------------------------
    detections = list(
        enumerate(
            detections_json.get("detections", [])
        )
    )

    if exclusive:
        # Highest-confidence detections get first ownership of
        # overlapping 3D points.
        detections.sort(
            key=lambda item: float(
                item[1].get("confidence", 0)
            ),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Optional erosion
    # ------------------------------------------------------------------
    kernel = None

    if erode_pixels > 0:
        radius = int(erode_pixels)

        kernel = np.ones(
            (2 * radius + 1, 2 * radius + 1),
            dtype=np.uint8,
        )

    # claimed:
    #   points assigned to successful detections when exclusive=True
    #
    # union:
    #   points falling inside any detection mask
    claimed = np.zeros(
        len(points_left),
        dtype=bool,
    )

    union = np.zeros(
        len(points_left),
        dtype=bool,
    )

    manifest: list[dict[str, Any]] = []

    # ==================================================================
    # Detection splitting
    # ==================================================================
    for detection_index, detection in detections:

        # --------------------------------------------------------------
        # Build HxW detection segmentation mask.
        # --------------------------------------------------------------
        mask = detection_mask(
            detection,
            rgb_h,
            rgb_w,
        )

        if mask.shape != (rgb_h, rgb_w):
            raise ValueError(
                f"Detection {detection_index} mask has shape "
                f"{mask.shape}, expected {(rgb_h, rgb_w)}"
            )

        # --------------------------------------------------------------
        # Optional erosion removes boundary pixels.
        # --------------------------------------------------------------
        if kernel is not None:
            mask = cv2.erode(
                mask.astype(np.uint8),
                kernel,
            ).astype(bool)

        # --------------------------------------------------------------
        # KEY OPERATION
        #
        # Look up ALL projected cloud points in the mask at once.
        #
        # No Python loop over points.
        # --------------------------------------------------------------
        keep = mask[v, u]

        # Any detection coverage.
        union |= keep

        # --------------------------------------------------------------
        # If exclusive:
        # don't allow a point already assigned to a higher-confidence
        # detection to belong to this detection.
        # --------------------------------------------------------------
        if exclusive:
            keep &= ~claimed

        count = int(keep.sum())

        if count < min_points:
            print(
                f"skip detection {detection_index}: "
                f"{count} points "
                f"({detection.get('class_name', 'unknown')})"
            )
            continue

        if exclusive:
            claimed |= keep

        # --------------------------------------------------------------
        # Detection metadata
        # --------------------------------------------------------------
        class_id = int(
            detection.get("class_id", -1)
        )

        class_name = str(
            detection.get("class_name", "object")
        )

        confidence = float(
            detection.get("confidence", 0)
        )

        filename = (
            f"{detection_index:03d}_"
            f"class{class_id}_"
            f"{safe_name(class_name)}_"
            f"{confidence:.3f}_"
            f"{count}pts.pcd"
        )

        # --------------------------------------------------------------
        # Detection point cloud
        # --------------------------------------------------------------
        object_points = points_left[keep]
        object_colors = colors_rgb[keep]

        save_point_cloud(
            output_dir / filename,
            object_points,
            object_colors,
            binary_pcd=binary_pcd,
        )

        manifest.append({
            "detection_index": detection_index,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "point_count": count,
            "pcd": filename,
        })

        print(
            f"saved {output_dir / filename} "
            f"({count} points)"
        )

    # ==================================================================
    # Full colored point cloud
    # ==================================================================
    if save_full_cloud:
        save_point_cloud(
            output_dir / "full.pcd",
            points_left,
            colors_rgb,
            binary_pcd=binary_pcd,
        )

        print(
            f"saved {output_dir / 'full.pcd'} "
            f"({len(points_left)} points)"
        )

    # ==================================================================
    # Background cloud
    # ==================================================================
    if save_background:

        if exclusive:
            background = ~claimed
        else:
            background = ~union

        count = int(background.sum())

        if count:
            filename = (
                f"background_{count}pts.pcd"
            )

            save_point_cloud(
                output_dir / filename,
                points_left[background],
                colors_rgb[background],
                binary_pcd=binary_pcd,
            )

            print(
                f"saved background "
                f"({count} points)"
            )

    return manifest

if __name__ == "__main__":
    # rectifier = calibration.get_rectifier(alpha)

    # # 1. Stereo -> disparity
    # left_rect, right_rect, rect = rectifier.rectify(
    #     left_image,
    #     right_image,
    # )

    # disparity = disparity_predictor.predict(
    #     left_rect,
    #     right_rect,
    # )

    # # 2. disparity -> rectified-left XYZ
    # points_rect, _ = rect.disparity_to_points_rectified(
    #     disparity,
    #     min_disparity=0.5,
    #     max_depth_m=5.0,
    # )

    # # 3. rectified-left -> original-left XYZ
    # points_left = rectified_left_to_original_left(
    #     points_rect,
    #     rect,
    # )

    # # 4. LEFT XYZ -> RGB image coordinates
    # uv, _ = project_points_to_rgb_pixels(
    #     points_left,
    #     rgb_image,
    #     calibration,
    # )

    # # 5. Split directly with RGB detections
    # manifest = split_cloud(
    #     points_left,
    #     uv,
    #     rgb_image,
    #     detections_json,
    #     output_dir,
    #     ops=calibration.ops,
    #     rgb_image_color_order="BGR",
    #     save_full_cloud=True,
    # )
    pass