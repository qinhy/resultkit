"""VPI CUDA stereo + CuPy fused disparity-to-colored-point-cloud trial.

This version keeps VPI disparity on the GPU through VPI's CUDA-buffer
interoperability, then uses CuPy for:
  * disparity reprojection with Q,
  * rectified-left -> original-left -> RGB transforms,
  * OpenCV-compatible 14-term RGB distortion,
  * nearest-neighbor RGB sampling,
  * validity/depth filtering.

Only the final compact points/colors are downloaded for PCD writing.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pcd_utils import (
    ColoredPointCloud,
    StereoRgbCalibration,
    gray8,
    read_image,
    save_point_cloud,
    scale_K,
)

try:
    import vpi
except ImportError as exc:
    raise SystemExit("NVIDIA VPI Python bindings are required") from exc

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "CuPy is required. For CUDA 12: pip install cupy-cuda12x; "
        "for CUDA 13: pip install cupy-cuda13x"
    ) from exc


class VPIStereoDisparityGPU:
    """Run VPI CUDA stereo and return GPU-resident VPI images."""

    def __init__(
        self,
        *,
        min_disparity: int = 0,
        max_disparity: int = 128,
        window_size: int = 5,
        confidence_threshold: int = 32767,
        p1: int = 3,
        p2: int = 48,
        uniqueness: float = -1.0,
        include_diagonals: bool = True,
        quality: int = 6,
    ) -> None:
        if max_disparity not in (64, 128, 256):
            raise ValueError("max_disparity must be 64, 128, or 256")
        self.min_disparity = int(min_disparity)
        self.max_disparity = int(max_disparity)
        self.window_size = int(window_size)
        self.confidence_threshold = int(confidence_threshold)
        self.p1 = int(p1)
        self.p2 = int(p2)
        self.uniqueness = float(uniqueness)
        self.include_diagonals = bool(include_diagonals)
        self.quality = int(quality)
        self._confidence = None
        self._confidence_shape = None

    def predict(self, left: np.ndarray, right: np.ndarray):
        left8 = np.ascontiguousarray(gray8(left))
        right8 = np.ascontiguousarray(gray8(right))
        if left8.shape != right8.shape:
            raise ValueError("Rectified left/right images must have matching shape")

        height, width = left8.shape
        shape = (width, height)
        if self._confidence is None or self._confidence_shape != shape:
            self._confidence = vpi.Image(shape, vpi.Format.U16)
            self._confidence_shape = shape

        with vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left8).convert(vpi.Format.Y16_ER, scale=1)
            right_vpi = vpi.asimage(right8).convert(vpi.Format.Y16_ER, scale=1)
            disparity_s16 = vpi.stereodisp(
                left_vpi,
                right_vpi,
                out_confmap=self._confidence,
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
        return disparity_s16, self._confidence

    @staticmethod
    def synchronize(image) -> None:
        # A read lock waits for producers and exposes the existing CUDA memory.
        with image.rlock_cuda():
            pass


def _tilt_projection_matrix(tau_x: float, tau_y: float) -> np.ndarray:
    cx, sx = math.cos(float(tau_x)), math.sin(float(tau_x))
    cy, sy = math.cos(float(tau_y)), math.sin(float(tau_y))
    rot_x = np.array(((1, 0, 0), (0, cx, sx), (0, -sx, cx)), np.float32)
    rot_y = np.array(((cy, 0, -sy), (0, 1, 0), (sy, 0, cy)), np.float32)
    rot_xy = rot_y @ rot_x
    proj_z = np.array(
        (
            (rot_xy[2, 2], 0, -rot_xy[0, 2]),
            (0, rot_xy[2, 2], -rot_xy[1, 2]),
            (0, 0, 1),
        ),
        np.float32,
    )
    return proj_z @ rot_xy


class CuPyColoredCloudBuilder:
    """Fused CUDA reprojection, camera projection, and color lookup."""

    def __init__(
        self,
        calibration: StereoRgbCalibration,
        rectification,
        rgb_shape: tuple[int, ...],
        *,
        min_disparity: float = 0.5,
        max_depth_m: float | None = 5.0,
        stride: int = 1,
        output_frame: str = "left",
        input_color_order: str = "BGR",
        rgb_image_is_undistorted: bool = False,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if output_frame not in ("left", "left_rectified"):
            raise ValueError("output_frame must be 'left' or 'left_rectified'")
        if input_color_order not in ("RGB", "BGR"):
            raise ValueError("input_color_order must be RGB or BGR")

        self.min_disparity = float(min_disparity)
        self.max_depth_m = max_depth_m
        self.stride = int(stride)
        self.output_frame = output_frame
        self.input_color_order = input_color_order
        self.rgb_image_is_undistorted = bool(rgb_image_is_undistorted)

        rgb_h, rgb_w = rgb_shape[:2]
        K = scale_K(
            calibration.rgb_intrinsics,
            calibration.rgb_resolution,
            (rgb_w, rgb_h),
        ).astype(np.float32)

        self.Q = cp.asarray(np.asarray(rectification.Q, np.float32))
        self.R1 = cp.asarray(np.asarray(rectification.R1, np.float32))
        self.T = cp.asarray(np.asarray(calibration.left_to_rgb, np.float32))
        self.K = cp.asarray(K)

        distortion = np.zeros(14, np.float32)
        if not self.rgb_image_is_undistorted:
            source = np.asarray(calibration.rgb_distortion, np.float32).reshape(-1)
            if len(source) not in (4, 5, 8, 12, 14):
                raise ValueError("RGB distortion must contain 4, 5, 8, 12, or 14 values")
            distortion[: len(source)] = source
        self.distortion = cp.asarray(distortion)
        self.has_distortion = bool(np.any(distortion != 0))
        self.tilt = cp.asarray(_tilt_projection_matrix(distortion[12], distortion[13]))
        self.has_tilt = bool(distortion[12] != 0 or distortion[13] != 0)

    def _project_rgb(self, points_rgb):
        z = points_rgb[:, 2]
        x = points_rgb[:, 0] / z
        y = points_rgb[:, 1] / z

        if self.has_distortion:
            d = self.distortion
            k1, k2, p1, p2, k3, k4, k5, k6 = [d[i] for i in range(8)]
            s1, s2, s3, s4 = [d[i] for i in range(8, 12)]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
                1 + k4 * r2 + k5 * r4 + k6 * r6
            )
            xy = x * y
            xd = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy + s3 * r2 + s4 * r4
            if self.has_tilt:
                tilt = self.tilt
                tx = tilt[0, 0] * xd + tilt[0, 1] * yd + tilt[0, 2]
                ty = tilt[1, 0] * xd + tilt[1, 1] * yd + tilt[1, 2]
                tz = tilt[2, 0] * xd + tilt[2, 1] * yd + tilt[2, 2]
                x, y = tx / tz, ty / tz
            else:
                x, y = xd, yd

        K = self.K
        u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
        v = K[1, 0] * x + K[1, 1] * y + K[1, 2]
        return u, v

    def build(self, disparity_s16, confidence_u16, rgb_image: np.ndarray):
        image = np.ascontiguousarray(rgb_image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("rgb_image must be HxWx3/4")

        # Include one RGB upload per call to keep the benchmark frame-realistic.
        rgb_gpu = cp.asarray(image[:, :, :3])

        with disparity_s16.rlock_cuda() as disparity_buffer, confidence_u16.rlock_cuda() as confidence_buffer:
            # cp.asarray uses the VPI CUDA buffer's CUDA Array Interface; no
            # disparity/confidence device-to-host copy is performed here.
            disparity_raw = cp.asarray(disparity_buffer).squeeze()
            confidence = cp.asarray(confidence_buffer).squeeze()
            if disparity_raw.ndim != 2 or confidence.shape != disparity_raw.shape:
                raise RuntimeError(
                    f"Unexpected VPI CUDA shapes: disparity={disparity_raw.shape}, "
                    f"confidence={confidence.shape}"
                )

            h, w = disparity_raw.shape
            valid2d = (disparity_raw > int(round(self.min_disparity * 32))) & (confidence > 0)

            if self.stride == 1:
                y, x = cp.nonzero(valid2d)
            else:
                ys, xs = cp.nonzero(valid2d[:: self.stride, :: self.stride])
                y, x = ys * self.stride, xs * self.stride

            d = disparity_raw[y, x].astype(cp.float32) * (1.0 / 32.0)
            xf, yf = x.astype(cp.float32), y.astype(cp.float32)
            Q = self.Q
            W = Q[3, 0] * xf + Q[3, 1] * yf + Q[3, 2] * d + Q[3, 3]
            X = (Q[0, 0] * xf + Q[0, 1] * yf + Q[0, 2] * d + Q[0, 3]) / W
            Y = (Q[1, 0] * xf + Q[1, 1] * yf + Q[1, 2] * d + Q[1, 3]) / W
            Z = (Q[2, 0] * xf + Q[2, 1] * yf + Q[2, 2] * d + Q[2, 3]) / W
            points_rect = cp.stack((X, Y, Z), axis=1)

            keep = cp.isfinite(points_rect).all(axis=1) & (Z > 0)
            if self.max_depth_m is not None:
                keep &= Z <= float(self.max_depth_m)
            points_rect = points_rect[keep]

            # Row-vector equivalent of p_left = R1.T @ p_rectified.
            points_left = points_rect @ self.R1
            points_rgb = points_left @ self.T[:3, :3].T + self.T[:3, 3]
            front = cp.isfinite(points_rgb).all(axis=1) & (points_rgb[:, 2] > 0)
            points_rgb = points_rgb[front]
            points_left = points_left[front]
            points_rect = points_rect[front]

            u_float, v_float = self._project_rgb(points_rgb)
            finite = cp.isfinite(u_float) & cp.isfinite(v_float)
            u = cp.rint(u_float[finite]).astype(cp.int32)
            v = cp.rint(v_float[finite]).astype(cp.int32)
            rgb_h, rgb_w = image.shape[:2]
            inside = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
            u, v = u[inside], v[inside]

            source_points = points_left if self.output_frame == "left" else points_rect
            source_points = source_points[finite][inside]
            colors = rgb_gpu[v, u]
            if self.input_color_order == "BGR":
                colors = colors[:, ::-1]

            # These transfers synchronize the CuPy stream before the VPI locks
            # are released. Only compact final output is copied to the host.
            points_cpu = cp.asnumpy(source_points).astype(np.float32, copy=False)
            colors_cpu = cp.asnumpy(colors).astype(np.uint8, copy=False)
            valid_count = int(cp.count_nonzero(valid2d).get())

        return points_cpu, colors_cpu, valid_count


def _timed_vpi_predict(predictor, left_rect, right_rect):
    start = time.perf_counter()
    disparity, confidence = predictor.predict(left_rect, right_rect)
    predictor.synchronize(disparity)
    return disparity, confidence, time.perf_counter() - start


def run(args: argparse.Namespace) -> ColoredPointCloud:
    root = Path(args.root)
    image_dir = root / "imgs" / args.camera
    calibration_path = root / "calib" / f"{args.camera}.json"

    left = read_image(image_dir / "left.jpg", color=False)
    right = read_image(image_dir / "right.jpg", color=False)
    rgb = read_image(image_dir / "rgb.jpg", color=True)
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = StereoRgbCalibration.from_dict(json.load(file))

    rectifier = calibration.get_rectifier(args.alpha)
    start = time.perf_counter()
    left_rect, right_rect, rect = rectifier.rectify(left, right)
    rectification_time = time.perf_counter() - start

    predictor = VPIStereoDisparityGPU(
        min_disparity=args.min_disparity,
        max_disparity=args.max_disparity,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        include_diagonals=not args.skip_diagonals,
    )

    _, _, warmup_time = _timed_vpi_predict(predictor, left_rect, right_rect)
    disparity = confidence = None
    disparity_timings = []
    for _ in range(args.runs):
        disparity, confidence, elapsed = _timed_vpi_predict(predictor, left_rect, right_rect)
        disparity_timings.append(elapsed)
    assert disparity is not None and confidence is not None

    builder = CuPyColoredCloudBuilder(
        calibration,
        rect,
        rgb.shape,
        min_disparity=max(0.5, float(args.min_disparity)),
        max_depth_m=args.max_depth,
        stride=args.stride,
        output_frame=args.output_frame,
        input_color_order="BGR",
    )

    # Warm up CuPy kernels/memory pools.
    builder.build(disparity, confidence, rgb)
    cloud_timings = []
    points = colors = None
    valid_count = 0
    for _ in range(args.cloud_runs):
        start = time.perf_counter()
        points, colors, valid_count = builder.build(disparity, confidence, rgb)
        cloud_timings.append(time.perf_counter() - start)
    assert points is not None and colors is not None

    start = time.perf_counter()
    if not args.skip_write:
        save_point_cloud(args.output, points, colors, binary_pcd=True)
    writing_time = time.perf_counter() - start

    print(f"VPI version:             {getattr(vpi, '__version__', 'unknown')}")
    print(f"CuPy version:            {cp.__version__}")
    print(f"Rectification CPU:       {rectification_time * 1000:.2f} ms")
    print(f"VPI warm-up:             {warmup_time * 1000:.2f} ms")
    print(
        "VPI disparity GPU-only: "
        f"median={np.median(disparity_timings) * 1000:.2f} ms, "
        f"mean={np.mean(disparity_timings) * 1000:.2f} ms over {args.runs} runs"
    )
    print(
        "GPU cloud + download:   "
        f"median={np.median(cloud_timings) * 1000:.2f} ms, "
        f"mean={np.mean(cloud_timings) * 1000:.2f} ms over {args.cloud_runs} runs"
    )
    print(
        "PCD writing:            "
        + (f"{writing_time * 1000:.2f} ms" if not args.skip_write else "skipped")
    )
    estimated_total = (
        rectification_time
        + float(np.median(disparity_timings))
        + float(np.median(cloud_timings))
        + (writing_time if not args.skip_write else 0.0)
    )
    print(f"Estimated median total:  {estimated_total * 1000:.2f} ms")
    print(f"Valid disparity:         {valid_count:,}/{left_rect.size:,} pixels")
    print(f"Saved points:            {len(points):,}")
    if not args.skip_write:
        print(f"Output:                  {Path(args.output).resolve()}")

    # Returning CPU compact output; disparity intentionally remains GPU-resident.
    return ColoredPointCloud(points, colors, None, rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera", default="rgbd_left")
    parser.add_argument("--output", default="colored_cloud_vpi_cupy.pcd")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--max-disparity", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=int, default=32767)
    parser.add_argument("--skip-diagonals", action="store_true")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cloud-runs", type=int, default=10)
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--output-frame", choices=("left", "left_rectified"), default="left")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
