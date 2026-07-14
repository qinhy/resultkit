#!/usr/bin/env python3
"""
Split a Fast-FoundationStereo DNN point cloud into one PCD per YOLO detection.

Required files in the same Python environment:
  - pcd_utils.py
  - pcd_dnn_utils.py

Example:

python pcd_yolo_dnn_utils.py \
  --left left.jpg \
  --right right.jpg \
  --rgb rgb.jpg \
  --calibration rgbd_left.json \
  --yolo rgb.json \
  --output-dir split_pcd_dnn \
  --repo-dir /path/to/Fast-FoundationStereo \
  --model-path weights/23-36-37/model_best_bp2_serialize.pth \
  --device cuda \
  --model-scale 0.5

split_pcd_dnn/
├── 000_class0_person_0.540_1234pts.pcd
├── 001_class15_cat_0.523_456pts.pcd
├── 002_class16_dog_0.424_2189pts.pcd
├── manifest.json
├── full_cloud_dnn.pcd       # with --save-full-cloud
└── background_12345pts.pcd  # with --save-background
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pcd_dnn_utils import (
    FastFoundationStereoDisparity,
    StereoRgbCalibration,
    disparity_to_points_rectified,
    points_left_to_rgb_depth,
    read_image,
    rectify_stereo_pair,
    rectified_left_to_original_left,
    rgb8,
    rgb_depth_to_points_rgb,
    save_point_cloud,
    transform_points,
)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value or "object"


def detection_mask(detection: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Rasterize one YOLO polygon mask. Falls back to bbox if needed."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_info = detection.get("mask")

    if not mask_info or mask_info.get("format") != "polygon":
        bbox = detection.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            return mask.astype(bool)
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(np.floor(x1), 0, width))
        y1 = int(np.clip(np.floor(y1), 0, height))
        x2 = int(np.clip(np.ceil(x2), 0, width))
        y2 = int(np.clip(np.ceil(y2), 0, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask.astype(bool)

    polygons = mask_info.get("polygons", [])

    # Draw outer contours first.
    for polygon in polygons:
        if polygon.get("is_hole", False):
            continue
        points = np.asarray(polygon.get("points_xy", []), dtype=np.float32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)

    # Remove holes second.
    for polygon in polygons:
        if not polygon.get("is_hole", False):
            continue
        points = np.asarray(polygon.get("points_xy", []), dtype=np.float32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 0)

    return mask.astype(bool)


def build_rgb_indexed_cloud_dnn(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    predictor: FastFoundationStereoDisparity,
    *,
    model_scale: float,
    stereo_input_color_order: str,
    min_disparity: float,
    max_depth_m: float | None,
    alpha: float,
    splat_px: int,
    rgb_image_is_undistorted: bool,
    remove_invisible: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a cloud with a one-to-one RGB-pixel index.

    Returns:
      points_left: N x 3 points in original left-camera coordinates, meters
      colors_rgb: N x 3 uint8 colors
      pixel_xy: N x 2 integer coordinates in the YOLO/RGB image
      disparity: H x W DNN disparity in rectified stereo pixel units
    """
    left_rect, right_rect, rectification = rectify_stereo_pair(
        left_image,
        right_image,
        calibration,
        alpha=alpha,
    )

    disparity = predictor.predict(
        left_rect,
        right_rect,
        input_color_order=stereo_input_color_order,
        model_scale=model_scale,
        remove_invisible=remove_invisible,
    )

    points_rectified, _ = disparity_to_points_rectified(
        disparity,
        rectification,
        min_disparity=min_disparity,
        max_depth_m=max_depth_m,
        stride=1,
    )

    if len(points_rectified) == 0:
        raise RuntimeError(
            "The DNN disparity produced no valid 3D points. Check calibration, "
            "min disparity, maximum depth, model, and left/right image order."
        )

    points_left_sparse = rectified_left_to_original_left(
        points_rectified,
        rectification,
    )

    # Z-buffer the left-camera points into the RGB/YOLO image. This gives
    # each retained point an exact RGB pixel, making YOLO-mask selection fast.
    depth_rgb, _ = points_left_to_rgb_depth(
        points_left_sparse,
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
        raise RuntimeError(
            "No DNN points projected into the RGB image. Check RGB calibration, "
            "extrinsic translation units, and image synchronization."
        )

    # Keep saved PCD coordinates in the original left/depth camera frame.
    points_left = transform_points(
        points_rgb,
        np.linalg.inv(calibration.left_to_rgb),
    )

    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]
    colors_rgb = rgb8(rgb_image[y, x, :3], order="BGR")

    return points_left, colors_rgb, pixel_xy, disparity


def split_cloud(
    points_left: np.ndarray,
    colors_rgb: np.ndarray,
    pixel_xy: np.ndarray,
    detections_json: dict[str, Any],
    output_dir: Path,
    *,
    min_points: int,
    erode_pixels: int,
    exclusive: bool,
    save_background: bool,
    save_full_cloud: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_width = int(detections_json["image_width"])
    image_height = int(detections_json["image_height"])
    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]

    in_bounds = (
        (x >= 0)
        & (x < image_width)
        & (y >= 0)
        & (y < image_height)
    )
    if not np.all(in_bounds):
        points_left = points_left[in_bounds]
        colors_rgb = colors_rgb[in_bounds]
        x = x[in_bounds]
        y = y[in_bounds]

    if save_full_cloud:
        save_point_cloud(
            output_dir / "full_cloud_dnn.pcd",
            points_left,
            colors_rgb,
            binary_pcd=True,
        )

    detections = list(enumerate(detections_json.get("detections", [])))
    if exclusive:
        # Highest-confidence detection claims an overlapping point first.
        detections.sort(
            key=lambda item: float(item[1].get("confidence", 0.0)),
            reverse=True,
        )

    claimed = np.zeros(len(points_left), dtype=bool)
    union = np.zeros(len(points_left), dtype=bool)
    manifest: list[dict[str, Any]] = []

    for detection_index, detection in detections:
        mask = detection_mask(detection, image_height, image_width)

        if erode_pixels > 0:
            size = 2 * erode_pixels + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            mask = cv2.erode(
                mask.astype(np.uint8),
                kernel,
                iterations=1,
            ).astype(bool)

        keep = mask[y, x]
        union |= keep
        if exclusive:
            keep &= ~claimed

        count = int(np.count_nonzero(keep))
        if count < min_points:
            print(
                f"skip detection {detection_index}: {count} points "
                f"({detection.get('class_name', 'unknown')})"
            )
            continue

        if exclusive:
            claimed |= keep

        class_name = safe_name(str(detection.get("class_name", "object")))
        class_id = int(detection.get("class_id", -1))
        confidence = float(detection.get("confidence", 0.0))
        filename = (
            f"{detection_index:03d}_class{class_id}_{class_name}_"
            f"{confidence:.3f}_{count}pts.pcd"
        )
        output_path = output_dir / filename

        save_point_cloud(
            output_path,
            points_left[keep],
            colors_rgb[keep],
            binary_pcd=True,
        )

        manifest.append(
            {
                "detection_index": detection_index,
                "class_id": class_id,
                "class_name": str(detection.get("class_name", "object")),
                "confidence": confidence,
                "point_count": count,
                "pcd": filename,
            }
        )
        print(f"saved {output_path} ({count} points)")

    if save_background:
        used = claimed if exclusive else union
        background = ~used
        count = int(np.count_nonzero(background))
        if count:
            save_point_cloud(
                output_dir / f"background_{count}pts.pcd",
                points_left[background],
                colors_rgb[background],
                binary_pcd=True,
            )
            print(f"saved background ({count} points)")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fast-FoundationStereo once, project its cloud into the RGB/YOLO "
            "image, and save one PCD for each YOLO mask."
        )
    )

    parser.add_argument("--left", required=True, help="Raw left stereo image")
    parser.add_argument("--right", required=True, help="Raw right stereo image")
    parser.add_argument("--rgb", required=True, help="RGB image used by YOLO")
    parser.add_argument("--calibration", required=True, help="Camera calibration JSON")
    parser.add_argument("--yolo", required=True, help="YOLO detection JSON")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--repo-dir", default=None, help="Fast-FoundationStereo repository")
    parser.add_argument("--model-path", default=None, help="Checkpoint path; absolute or relative to repo")
    parser.add_argument("--model-dir", default=None, help="Directory containing the checkpoint")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--hiera", action="store_true")
    parser.add_argument(
        "--stereo-input-color-order",
        choices=("RGB", "BGR"),
        default="RGB",
        help="Use BGR only when left/right were read as color with OpenCV",
    )
    parser.add_argument(
        "--keep-invisible",
        action="store_true",
        help="Do not remove left-view pixels whose disparity points outside the right image",
    )

    parser.add_argument("--min-disparity", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--splat-px",
        type=int,
        default=0,
        help="0 is safest at mask edges; 1 fills more RGB pixels but may bleed across boundaries",
    )
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument(
        "--erode-pixels",
        type=int,
        default=0,
        help="Use 1 or 2 to remove uncertain YOLO silhouette-edge points",
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Assign overlapping points only to the highest-confidence detection",
    )
    parser.add_argument("--save-background", action="store_true")
    parser.add_argument("--save-full-cloud", action="store_true")
    parser.add_argument(
        "--rgb-undistorted",
        action="store_true",
        help="Set only when the RGB image used by YOLO was explicitly undistorted",
    )
    parser.add_argument(
        "--calibration-translation-unit",
        choices=("m", "cm", "mm"),
        default="cm",
        help="Unit used by translations in the calibration JSON",
    )
    args = parser.parse_args()

    if args.model_scale <= 0:
        parser.error("--model-scale must be positive")
    if args.splat_px < 0:
        parser.error("--splat-px must be >= 0")
    if args.erode_pixels < 0:
        parser.error("--erode-pixels must be >= 0")

    left = read_image(args.left, color=False)
    right = read_image(args.right, color=False)
    rgb = read_image(args.rgb, color=True)  # OpenCV BGR

    calibration_data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = StereoRgbCalibration.from_dict(
        calibration_data,
        source_translation_unit=args.calibration_translation_unit,
    )
    detections = json.loads(Path(args.yolo).read_text(encoding="utf-8"))

    expected_width = int(detections["image_width"])
    expected_height = int(detections["image_height"])
    if rgb.shape[1] != expected_width or rgb.shape[0] != expected_height:
        raise ValueError(
            f"RGB is {rgb.shape[1]}x{rgb.shape[0]}, but YOLO JSON is "
            f"{expected_width}x{expected_height}. Use the exact RGB image sent to YOLO."
        )

    predictor = FastFoundationStereoDisparity(
        repo_dir=args.repo_dir,
        model_path=args.model_path,
        model_dir=args.model_dir,
        device=args.device,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        hiera=args.hiera,
    )

    points_left, colors_rgb, pixel_xy, disparity = build_rgb_indexed_cloud_dnn(
        left,
        right,
        rgb,
        calibration,
        predictor,
        model_scale=args.model_scale,
        stereo_input_color_order=args.stereo_input_color_order,
        min_disparity=args.min_disparity,
        max_depth_m=args.max_depth_m,
        alpha=args.alpha,
        splat_px=args.splat_px,
        rgb_image_is_undistorted=args.rgb_undistorted,
        remove_invisible=not args.keep_invisible,
    )

    output_dir = Path(args.output_dir)
    manifest = split_cloud(
        points_left,
        colors_rgb,
        pixel_xy,
        detections,
        output_dir,
        min_points=args.min_points,
        erode_pixels=args.erode_pixels,
        exclusive=args.exclusive,
        save_background=args.save_background,
        save_full_cloud=args.save_full_cloud,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": "Fast-FoundationStereo + YOLO mask split",
        "valid_disparity_pixels": int(np.count_nonzero(np.isfinite(disparity))),
        "rgb_indexed_cloud_points": int(len(points_left)),
        "detections_written": len(manifest),
        "exclusive": bool(args.exclusive),
        "splat_px": int(args.splat_px),
        "erode_pixels": int(args.erode_pixels),
        "outputs": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"done: {len(points_left):,} RGB-indexed DNN points; "
        f"wrote {len(manifest)} detection clouds"
    )


if __name__ == "__main__":
    main()
