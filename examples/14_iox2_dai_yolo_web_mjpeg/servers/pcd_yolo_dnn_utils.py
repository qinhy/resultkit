#!/usr/bin/env python3
"""Build an RGB-indexed Fast-FoundationStereo cloud and split it by YOLO masks."""

import argparse
import json
from pathlib import Path

import numpy as np

from pcd_dnn_utils import (
    FastFoundationStereoDisparity,
    StereoRgbCalibration,
    points_left_to_rgb_depth,
    read_image,
    rectified_left_to_original_left,
    rgb8,
    rgb_depth_to_points_rgb,
    transform_points,
)
from pcd_yolo_utils import split_cloud


def build_rgb_indexed_cloud_dnn(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    predictor: FastFoundationStereoDisparity,
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
    disparity = predictor.predict(
        left, right, input_color_order=stereo_input_color_order)
    points, _ = rect.disparity_to_points_rectified(disparity,
                                              min_disparity=max(0.5, float(min_disparity)),
                                              max_depth_m=max_depth_m, stride=stride)
    if not len(points):
        raise RuntimeError("DNN disparity produced no valid 3D points")

    points = rectified_left_to_original_left(points, rect)
    depth_rgb, _ = points_left_to_rgb_depth(
        points, rgb_image, calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted, splat_px=splat_px,
    )
    points_rgb, pixel_xy = rgb_depth_to_points_rgb(
        depth_rgb, calibration, rgb_image_is_undistorted=rgb_image_is_undistorted
    )
    if not len(points_rgb):
        raise RuntimeError("No DNN points projected into the RGB image")

    points_left = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))
    x, y = pixel_xy.T
    return points_left, rgb8(rgb_image[y, x, :3], order=stereo_input_color_order), pixel_xy, disparity

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

    predictor = FastFoundationStereoDisparity(
        repo_dir="./examples/14_iox2_dai_yolo_web_mjpeg/fast-foundationstereo",
        model_path="weights/23-36-37/model_best_bp2_serialize.pth",
        valid_iters=8,
        max_disp=192,
    )

    points_left, colors_rgb, pixel_xy, disparity = build_rgb_indexed_cloud_dnn(
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
