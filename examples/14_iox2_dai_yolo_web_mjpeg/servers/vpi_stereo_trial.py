"""Minimal NVIDIA VPI CUDA trial for the existing pcd_utils.py pipeline.

This uses VPI CUDA for stereo disparity and pcd_utils_optimized.py for RGB
projection/colorization. The optimized colorizer implements OpenCV's full
4/5/8/12/14-coefficient distortion model directly, avoiding the very large
Jacobian allocated by Python cv2.projectPoints().
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pcd_utils_optimized import (
    ColoredPointCloud,
    StereoRgbCalibration,
    colorize_points_from_rgb,
    gray8,
    read_image,
    save_point_cloud,
)

try:
    import vpi
except ImportError as exc:
    raise SystemExit(
        "Could not import NVIDIA VPI. Install the VPI Python binding that "
        "matches your Python version, then run this script with that interpreter."
    ) from exc


class VPIStereoDisparityPredictor:
    """VPI CUDA SGM disparity predictor returning float disparity pixels."""

    def __init__(
        self,
        *,
        min_disparity: int = 0,
        max_disparity: int = 256,
        window_size: int = 5,
        confidence_threshold: int = 32767,
        p1: int = 3,
        p2: int = 48,
        uniqueness: float = -1.0,
        include_diagonals: bool = True,
        quality: int = 6,
        invalid_to_nan: bool = True,
    ) -> None:
        if not 1 <= max_disparity <= 256:
            raise ValueError("VPI CUDA max_disparity must be in [1, 256]")
        self.min_disparity = int(min_disparity)
        self.max_disparity = int(max_disparity)
        self.window_size = int(window_size)
        self.confidence_threshold = int(confidence_threshold)
        self.p1 = int(p1)
        self.p2 = int(p2)
        self.uniqueness = float(uniqueness)
        self.include_diagonals = bool(include_diagonals)
        self.quality = int(quality)
        self.invalid_to_nan = bool(invalid_to_nan)

    def predict(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left8 = np.ascontiguousarray(gray8(left))
        right8 = np.ascontiguousarray(gray8(right))
        if left8.shape != right8.shape:
            raise ValueError("Rectified left/right images must have the same shape")

        height, width = left8.shape
        confidence = vpi.Image((width, height), vpi.Format.U16)

        # NVIDIA's official sample converts 8-bit images to Y16_ER with scale=1
        # before running CUDA stereo disparity.
        with vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left8).convert(vpi.Format.Y16_ER, scale=1)
            right_vpi = vpi.asimage(right8).convert(vpi.Format.Y16_ER, scale=1)

            disparity_s16 = vpi.stereodisp(
                left_vpi,
                right_vpi,
                out_confmap=confidence,
                window=self.window_size,
                maxdisp=self.max_disparity,
                confthreshold=self.confidence_threshold,
                conftype=vpi.ConfidenceType.ABSOLUTE,
                quality=self.quality,
                mindisp=self.min_disparity,
                p1=self.p1,
                p2=self.p2,
                uniqueness=self.uniqueness,
                includediagonals=self.include_diagonals,
            )

        # VPI disparity is signed 16-bit Q10.5 fixed point: divide by 32.
        disparity = np.squeeze(np.asarray(disparity_s16.cpu(), dtype=np.float32)) / 32.0
        confidence_cpu = np.squeeze(np.asarray(confidence.cpu()))
        if disparity.ndim != 2 or confidence_cpu.shape != disparity.shape:
            raise RuntimeError(
                f"Unexpected VPI output shapes: disparity={disparity.shape}, "
                f"confidence={confidence_cpu.shape}"
            )

        if self.invalid_to_nan:
            invalid = (
                ~np.isfinite(disparity)
                | (disparity <= self.min_disparity)
                | (confidence_cpu == 0)
            )
            disparity = disparity.copy()
            disparity[invalid] = np.nan

        return disparity


def run(args: argparse.Namespace) -> ColoredPointCloud:
    root = Path(args.root)
    image_dir = root / "imgs" / args.camera
    calibration_path = root / "calib" / f"{args.camera}.json"

    left = read_image(image_dir / "left.jpg", color=False)
    right = read_image(image_dir / "right.jpg", color=False)
    rgb = read_image(image_dir / "rgb.jpg", color=True)

    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = StereoRgbCalibration.from_dict(json.load(file))

    # Keep the existing CPU rectification for this first controlled trial.
    rectifier = calibration.get_rectifier(args.alpha)
    t0 = time.perf_counter()
    left_rect, right_rect, rect = rectifier.rectify(left, right)
    t1 = time.perf_counter()

    predictor = VPIStereoDisparityPredictor(
        min_disparity=args.min_disparity,
        max_disparity=args.max_disparity,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        include_diagonals=not args.skip_diagonals,
    )

    # Warm up CUDA/VPI before measuring. Initial invocation includes setup cost.
    predictor.predict(left_rect, right_rect)
    t2 = time.perf_counter()

    timings = []
    disparity = None
    for _ in range(args.runs):
        start = time.perf_counter()
        disparity = predictor.predict(left_rect, right_rect)
        timings.append(time.perf_counter() - start)
    assert disparity is not None
    t3 = time.perf_counter()

    points_rectified, _ = rect.disparity_to_points_rectified(
        disparity,
        min_disparity=max(0.5, float(args.min_disparity)),
        max_depth_m=args.max_depth,
        stride=args.stride,
    )
    t4 = time.perf_counter()

    # Warm up the optimized NumPy colorizer, then report a stable median.
    colorize_points_from_rgb(
        points_rectified,
        rgb,
        calibration,
        rectification=rect,
        output_frame=args.output_frame,
        input_color_order="BGR",
    )
    color_timings = []
    points = colors = None
    for _ in range(args.color_runs):
        start = time.perf_counter()
        points, colors = colorize_points_from_rgb(
            points_rectified,
            rgb,
            calibration,
            rectification=rect,
            output_frame=args.output_frame,
            input_color_order="BGR",
        )
        color_timings.append(time.perf_counter() - start)
    assert points is not None and colors is not None
    t5 = time.perf_counter()

    if not args.skip_write:
        save_point_cloud(args.output, points, colors, binary_pcd=True)
    t6 = time.perf_counter()

    finite = np.isfinite(disparity)
    print(f"VPI version: {getattr(vpi, '__version__', 'unknown')}")
    print(f"Rectification CPU: {(t1 - t0) * 1000:.2f} ms")
    print(f"VPI warm-up:       {(t2 - t1) * 1000:.2f} ms")
    print(
        "VPI disparity:     "
        f"median={np.median(timings) * 1000:.2f} ms, "
        f"mean={np.mean(timings) * 1000:.2f} ms over {args.runs} runs"
    )
    print(f"3-D reprojection:  {(t4 - t3) * 1000:.2f} ms")
    print(
        "RGB colorization:  "
        f"median={np.median(color_timings) * 1000:.2f} ms, "
        f"mean={np.mean(color_timings) * 1000:.2f} ms over {args.color_runs} runs"
    )
    print(
        "PCD writing:       "
        + (f"{(t6 - t5) * 1000:.2f} ms" if not args.skip_write else "skipped")
    )
    print(f"Valid disparity:   {finite.sum():,}/{finite.size:,} pixels")
    print(f"Saved points:      {len(points):,}")
    if not args.skip_write:
        print(f"Output:            {Path(args.output).resolve()}")

    return ColoredPointCloud(points, colors, disparity, rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Recording timestamp directory")
    parser.add_argument("--camera", default="rgbd_left")
    parser.add_argument("--output", default="colored_cloud_vpi.pcd")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--max-disparity", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=int, default=32767)
    parser.add_argument("--skip-diagonals", action="store_true")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--color-runs", type=int, default=5)
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--output-frame", choices=("left", "left_rectified"), default="left")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
