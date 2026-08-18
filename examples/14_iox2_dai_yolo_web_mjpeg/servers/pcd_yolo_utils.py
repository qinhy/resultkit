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


if __name__ == "__main__":
    root = "recording/rgb_stereo/2026-07-22/field_all/111737.603022000JST/"
    left = read_image(root+"imgs/rgbd_left/left.jpg", color=False)
    right = read_image(root+"imgs/rgbd_left/right.jpg", color=False)
    rgb = read_image(root+"imgs/rgbd_left/rgb.jpg", color=True)  # cv2 gives BGR
    with open(root+"calib/rgbd_left.json") as f:
        calibration = StereoRgbCalibration.from_dict(json.load(f))

    detections = json.loads(Path(root+"yolo/rgbd_left/rgb.json").read_text(encoding="utf-8"))
    from yolo_utils import show_yolo_results_pil
    show_yolo_results_pil(detections)
    
    predictor = SGBMDisparityPredictor(num_disparities=160,
                                        block_size=5,
                                        uniqueness_ratio=8,
                                        speckle_window_size=80,
                                        speckle_range=2,
                                        disp12_max_diff=1,
                                        pre_filter_cap=31)

    points_left, colors_rgb, pixel_xy, disparity = build_rgb_indexed_cloud_sgbm(
        left,right,rgb,calibration,
        predictor=predictor,
        min_disparity=0.5,
        max_depth_m=5.0,
        stride=1,
        alpha=0.0,
        splat_px=0,
        rgb_image_is_undistorted=False,
        stereo_input_color_order="BGR",
    )

    output_dir = Path(root+"tmp/")
    manifest = split_cloud(
        points_left,
        colors_rgb,
        pixel_xy,
        detections,
        output_dir,
        min_points=1,
        erode_pixels=0,
        exclusive=False,
        save_background=True,
        save_full_cloud=True,
        binary_pcd=False,
    )
