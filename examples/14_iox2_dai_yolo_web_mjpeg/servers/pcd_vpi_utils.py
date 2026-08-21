"""Compact VPI/CuPy stereo -> colored point-cloud pipeline."""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

try:
    import vpi
except ImportError as exc:
    raise SystemExit("NVIDIA VPI Python bindings are required") from exc

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit("CuPy is required (for example: cupy-cuda12x or cupy-cuda13x)") from exc

from pcd_utils import read_image, save_point_cloud

ColorOrder = Literal["RGB", "BGR"]
PointsFrame = Literal["left_rectified", "left"]
OutputFrame = Literal["left", "left_rectified"]


def cv2():
    try:
        import cv2 as _cv2
        return _cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required: uv pip install opencv-python") from exc


def arr(x: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = np.asarray(x, np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value.copy()


def resolution(x: Any, name: str) -> tuple[int, int]:
    if len(x) != 2:
        raise ValueError(f"{name} must be [width, height]")
    wh = tuple(map(int, x))
    if min(wh) <= 0:
        raise ValueError(f"{name} must contain positive values")
    return wh


def scale_K(K: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    out = K.astype(np.float64, copy=True)
    if from_wh != to_wh:
        sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
        out[0, [0, 2]] *= sx
        out[1, [1, 2]] *= sy
    return out


@dataclass(frozen=True)
class StereoRgbCalibration:
    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]
    rgb_intrinsics: np.ndarray | cp.ndarray
    left_intrinsics: np.ndarray | cp.ndarray
    right_intrinsics: np.ndarray | cp.ndarray
    rgb_distortion: np.ndarray | cp.ndarray
    left_distortion: np.ndarray | cp.ndarray
    right_distortion: np.ndarray | cp.ndarray
    left_to_right: np.ndarray | cp.ndarray
    left_to_rgb: np.ndarray | cp.ndarray
    source_translation_unit: Literal["m", "cm", "mm"] = "cm"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_translation_unit: Literal["m", "cm", "mm"] = "cm"):
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[source_translation_unit]
        lr = arr(data["left_to_right_extrinsics"], "left_to_right_extrinsics", (4, 4))
        lrgb = arr(data["left_to_rgb_extrinsics"], "left_to_rgb_extrinsics", (4, 4))
        lr[:3, 3] *= scale
        lrgb[:3, 3] *= scale
        return cls(
            resolution(data["rgb_resolution"], "rgb_resolution"),
            resolution(data["left_resolution"], "left_resolution"),
            resolution(data["right_resolution"], "right_resolution"),
            arr(data["rgb_intrinsics"], "rgb_intrinsics", (3, 3)),
            arr(data["left_intrinsics"], "left_intrinsics", (3, 3)),
            arr(data["right_intrinsics"], "right_intrinsics", (3, 3)),
            arr(data["rgb_distortion"], "rgb_distortion").reshape(-1, 1),
            arr(data["left_distortion"], "left_distortion").reshape(-1, 1),
            arr(data["right_distortion"], "right_distortion").reshape(-1, 1),
            lr,
            lrgb,
            source_translation_unit,
        )

    def get_rectifier(self, alpha: float = 0.0, zero_disparity: bool = True):
        return StereoRectifier(self, alpha, zero_disparity)

    @property
    def stereo_baseline_m(self) -> float: return abs(float(self.left_to_right[0, 3]))
    @property
    def stereo_baseline_cm(self) -> float: return self.stereo_baseline_m * 100
    @property
    def stereo_translation_norm_m(self) -> float: 
        if isinstance(self.left_to_right,cp.ndarray):
            return float(cp.linalg.norm(self.left_to_right[:3, 3]))    
        return float(np.linalg.norm(self.left_to_right[:3, 3]))
    @property
    def left_to_right_rotation(self): return self.left_to_right[:3, :3].copy()
    @property
    def left_to_right_translation_m(self): return self.left_to_right[:3, 3:4].copy()
    @property
    def left_to_rgb_rotation(self): return self.left_to_rgb[:3, :3].copy()
    @property
    def left_to_rgb_translation_m(self): return self.left_to_rgb[:3, 3:4].copy()


    def to_cupy(self):
        return StereoRgbCalibration(
            rgb_resolution=self.rgb_resolution,
            left_resolution=self.left_resolution,
            right_resolution=self.right_resolution,
            rgb_intrinsics=cp.asarray(self.rgb_intrinsics, dtype=cp.float32),
            left_intrinsics=cp.asarray(self.left_intrinsics, dtype=cp.float32),
            right_intrinsics=cp.asarray(self.right_intrinsics, dtype=cp.float32),
            rgb_distortion=cp.asarray(self.rgb_distortion, dtype=cp.float32),
            left_distortion=cp.asarray(self.left_distortion, dtype=cp.float32),
            right_distortion=cp.asarray(self.right_distortion, dtype=cp.float32),
            left_to_right=cp.asarray(self.left_to_right, dtype=cp.float32),
            left_to_rgb=cp.asarray(self.left_to_rgb, dtype=cp.float32),
            source_translation_unit=self.source_translation_unit,
        )


@dataclass(frozen=True)
class StereoRectification:
    image_size: tuple[int, int]
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    valid_roi_left: tuple[int, int, int, int]
    valid_roi_right: tuple[int, int, int, int]

    def disparity_to_points_rectified(
        self,
        disparity,
        confidence_u16,
        *,
        min_disparity: float = 0.5,
        min_depth_m: float | None = 0.01,
        max_depth_m: float | None = 10.0,
        stride: int = 1,
    ):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        Q = cp.asarray(self.Q, dtype=cp.float32)

        with disparity.rlock_cuda() as dbuf, confidence_u16.rlock_cuda() as cbuf:
            disparity = cp.asarray(dbuf).squeeze()
            confidence = cp.asarray(cbuf).squeeze()
            if disparity.ndim != 2 or confidence.shape != disparity.shape:
                raise RuntimeError(f"Unexpected disparity/confidence shapes: {disparity.shape}, {confidence.shape}")

            valid = (disparity > math.floor(float(min_disparity) * 32.0)) & (confidence > 0)
            if stride == 1:
                y, x = cp.nonzero(valid)
            else:
                y, x = cp.nonzero(valid[::stride, ::stride])
                y, x = y * stride, x * stride

            d = disparity[y, x].astype(cp.float32) / 32.0
            xf, yf = x.astype(cp.float32), y.astype(cp.float32)
            W = Q[3, 0] * xf + Q[3, 1] * yf + Q[3, 2] * d + Q[3, 3]
            X = (Q[0, 0] * xf + Q[0, 1] * yf + Q[0, 2] * d + Q[0, 3]) / W
            Y = (Q[1, 0] * xf + Q[1, 1] * yf + Q[1, 2] * d + Q[1, 3]) / W
            Z = (Q[2, 0] * xf + Q[2, 1] * yf + Q[2, 2] * d + Q[2, 3]) / W
            points = cp.stack((X, Y, Z), axis=1)

            keep = cp.isfinite(points).all(axis=1) & (Z > 0)
            if min_depth_m is not None:
                keep &= Z >= float(min_depth_m)
            if max_depth_m is not None:
                keep &= Z <= float(max_depth_m)
            points, xy = points[keep], cp.stack((x, y), axis=1)[keep]
            cp.cuda.get_current_stream().synchronize()
        return points, xy


class StereoRectifier:
    def __init__(self, calibration: StereoRgbCalibration, alpha: float = 0.0, zero_disparity: bool = True):
        self.calibration = calibration
        self.alpha = alpha
        self.zero_disparity = zero_disparity
        self.remapper = None

    def make(self, image_size: tuple[int, int] | None = None) -> StereoRectification:
        c, cal = cv2(), self.calibration
        size = image_size or cal.left_resolution
        if size != cal.left_resolution:
            warnings.warn("Input size differs from calibration; intrinsics are scaled.", RuntimeWarning, stacklevel=2)

        K1 = scale_K(cal.left_intrinsics, cal.left_resolution, size)
        K2 = scale_K(cal.right_intrinsics, cal.right_resolution, size)
        flags = c.CALIB_ZERO_DISPARITY if self.zero_disparity else 0
        R1, R2, P1, P2, Q, roi1, roi2 = c.stereoRectify(
            K1, cal.left_distortion, K2, cal.right_distortion, size,
            cal.left_to_right_rotation, cal.left_to_right_translation_m,
            flags=flags, alpha=float(self.alpha),
        )
        left_maps = c.initUndistortRectifyMap(K1, cal.left_distortion, R1, P1, size, c.CV_32FC1)
        right_maps = c.initUndistortRectifyMap(K2, cal.right_distortion, R2, P2, size, c.CV_32FC1)
        rect = StereoRectification(
            size, *left_maps, *right_maps, R1, R2, P1, P2, Q,
            tuple(map(int, roi1)), tuple(map(int, roi2)),
        )
        self.remapper = VPIStereoRemapper(rect)
        return rect

    def rectify(self, left: Any, right: Any, rectification: StereoRectification | None = None):
        if left is None or right is None or left.shape[:2] != right.shape[:2]:
            raise ValueError("Left/right images are required and must have matching sizes.")
        h, w = left.shape[:2]
        rect = rectification or self.make((w, h))
        if self.remapper is None:
            self.remapper = VPIStereoRemapper(rect)
        left_rect, right_rect = self.remapper.rectify(left, right)
        self.remapper.synchronize(left_rect, right_rect)
        return left_rect, right_rect, rect


def _image_hw(image: Any) -> tuple[int, int]:
    if isinstance(image, vpi.Image):
        w, h = image.size
        return int(h), int(w)
    shape = tuple(map(int, image.shape))
    if len(shape) < 2:
        raise ValueError(f"Image must have at least two dimensions, got {shape}")
    return shape[:2]


def _as_vpi_u8_gray(image: Any) -> vpi.Image:
    if isinstance(image, vpi.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 3:
        c = cv2()
        array = c.cvtColor(array[:, :, :3], c.COLOR_BGR2GRAY) if array.shape[2] >= 3 else array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale HxW image, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1:
            array = array * 255
        if array.dtype == np.uint16 and array.size and array.max() > 0:
            array = array.astype(np.float32) * (255 / array.max())
        array = np.clip(array, 0, 255).astype(np.uint8)
    return vpi.asimage(np.ascontiguousarray(array), format=vpi.Format.U8)


def _cv_maps_to_vpi_warpmap(map_x: np.ndarray, map_y: np.ndarray) -> vpi.WarpMap:
    map_x, map_y = np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)
    if map_x.ndim != 2 or map_x.shape != map_y.shape:
        raise ValueError("map_x and map_y must be matching HxW arrays")
    h, w = map_x.shape
    warp = vpi.WarpMap((w, h), interval=1)
    view = np.asarray(warp)
    if view.shape != (h, w, 2):
        raise RuntimeError(f"Unexpected VPI WarpMap shape {view.shape}; expected {(h, w, 2)}")
    view[..., 0], view[..., 1] = map_x, map_y
    return warp


class VPIStereoRemapper:
    def __init__(self, rectification: StereoRectification):
        self.rectification = rectification
        self.left_warp = _cv_maps_to_vpi_warpmap(rectification.left_map_x, rectification.left_map_y)
        self.right_warp = _cv_maps_to_vpi_warpmap(rectification.right_map_x, rectification.right_map_y)
        self._left_output = self._right_output = None
        self._input_refs = ()

    def rectify(self, left: Any, right: Any):
        lh, lw = _image_hw(left)
        rh, rw = _image_hw(right)
        if (lh, lw) != (rh, rw):
            raise ValueError("Left/right images must have matching sizes")
        if (lw, lh) != self.rectification.image_size:
            raise ValueError(f"Input size {(lw, lh)} differs from rectification size {self.rectification.image_size}")

        left_vpi, right_vpi = _as_vpi_u8_gray(left), _as_vpi_u8_gray(right)
        self._input_refs = (left_vpi, right_vpi, left, right)
        with vpi.Backend.CUDA:
            if self._left_output is None:
                self._left_output = left_vpi.remap(self.left_warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                self._right_output = right_vpi.remap(self.right_warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            else:
                left_vpi.remap(self.left_warp, out=self._left_output, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                right_vpi.remap(self.right_warp, out=self._right_output, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
        return self._left_output, self._right_output

    @staticmethod
    def synchronize(left_rect, right_rect) -> None:
        with left_rect.rlock_cuda(), right_rect.rlock_cuda():
            pass


class VPIStereoDisparityGPU:
    def __init__(
        self,
        *,
        min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        confidence_threshold: int = 32767,
        p1: int = 3,
        p2: int = 48,
        uniqueness: float = -1.0,
        include_diagonals: bool = True,
        quality: int = 6,
        width: int=-1,height: int=-1
    ):
        if num_disparities not in (64, 128, 256):
            raise ValueError("num_disparities must be 64, 128, or 256")
        self.min_disparity = int(min_disparity)
        self.max_disparity = int(num_disparities)
        self.window_size = int(block_size)
        self.confidence_threshold = int(confidence_threshold)
        self.p1, self.p2 = int(p1), int(p2)
        self.uniqueness = float(uniqueness)
        self.include_diagonals = bool(include_diagonals)
        self.quality = int(quality)
        self._shape = self._left_y16 = self._right_y16 = self._disparity = self._confidence = None
        self._input_refs = ()

    def _reset_for_shape(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self._left_y16 = self._right_y16 = self._disparity = None
        self._confidence = vpi.Image(shape, vpi.Format.U16)

    def predict(self, left: Any, right: Any):
        left_vpi, right_vpi = _as_vpi_u8_gray(left), _as_vpi_u8_gray(right)
        if left_vpi.size != right_vpi.size:
            raise ValueError("Rectified left/right images must have matching size")
        shape = tuple(map(int, left_vpi.size))
        if self._shape != shape:
            self._reset_for_shape(shape)
        self._input_refs = (left_vpi, right_vpi, left, right)

        with vpi.Backend.CUDA:
            if self._left_y16 is None:
                self._left_y16 = left_vpi.convert(vpi.Format.Y16_ER, scale=1)
                self._right_y16 = right_vpi.convert(vpi.Format.Y16_ER, scale=1)
            else:
                left_vpi.convert(self._left_y16, scale=1)
                right_vpi.convert(self._right_y16, scale=1)

            kwargs = dict(
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
            if self._disparity is None:
                self._disparity = vpi.stereodisp(self._left_y16, self._right_y16, **kwargs)
            else:
                vpi.stereodisp(self._left_y16, self._right_y16, out=self._disparity, **kwargs)

        self.synchronize(self._disparity)
        return self._disparity, self._confidence

    @staticmethod
    def synchronize(image) -> None:
        with image.rlock_cuda():
            pass


def _tilt_projection_matrix(tau_x: float, tau_y: float) -> np.ndarray:
    cx, sx = math.cos(float(tau_x)), math.sin(float(tau_x))
    cy, sy = math.cos(float(tau_y)), math.sin(float(tau_y))
    rot = np.array(((cy, sy * sx, -sy * cx), (0, cx, sx), (sy, -cy * sx, cy * cx)), np.float32)
    proj = np.array(((rot[2, 2], 0, -rot[0, 2]), (0, rot[2, 2], -rot[1, 2]), (0, 0, 1)), np.float32)
    return proj @ rot


def _project_camera_points_opencv_model_gpu(points_camera, K, distortion=None, tilt=None):
    p = cp.asarray(points_camera, dtype=cp.float32)
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    if distortion is not None:
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = distortion[:12]
        r2 = x * x + y * y
        r4, r6, xy = r2 * r2, r2 * r2 * r2, x * y
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        xd = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy + s3 * r2 + s4 * r4
        if tilt is not None:
            tx = tilt[0, 0] * xd + tilt[0, 1] * yd + tilt[0, 2]
            ty = tilt[1, 0] * xd + tilt[1, 1] * yd + tilt[1, 2]
            tz = tilt[2, 0] * xd + tilt[2, 1] * yd + tilt[2, 2]
            x, y = tx / tz, ty / tz
        else:
            x, y = xd, yd
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 0] * x + K[1, 1] * y + K[1, 2]
    return cp.stack((u, v), axis=1)


def colorize_points_from_rgb_gpu(
    points_m,
    rgb_image,
    calibration: StereoRgbCalibration,
    *,
    rectification=None,
    points_frame: PointsFrame = "left_rectified",
    output_frame: OutputFrame = "left",
    input_color_order: ColorOrder = "RGB",
    rgb_image_is_undistorted: bool = False,
):
    image = rgb_image if isinstance(rgb_image, cp.ndarray) else cp.asarray(rgb_image if hasattr(rgb_image, "__cuda_array_interface__") else np.ascontiguousarray(rgb_image))
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"rgb_image must be HxWx3/4, got {image.shape}")
    if input_color_order not in ("RGB", "BGR"):
        raise ValueError("input_color_order must be 'RGB' or 'BGR'")
    if points_frame not in ("left_rectified", "left") or output_frame not in ("left", "left_rectified"):
        raise ValueError(f"Unsupported frame: points={points_frame}, output={output_frame}")

    points = cp.asarray(points_m, dtype=cp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must be Nx3, got {points.shape}")

    if points_frame == "left_rectified":
        if rectification is None:
            raise ValueError("rectification is required for left_rectified points")
        points_left = points @ cp.asarray(rectification.R1, dtype=cp.float32)
    else:
        points_left = points
    source_points = points_left if output_frame == "left" else points

    T = cp.asarray(calibration.left_to_rgb, dtype=cp.float32)
    points_rgb = points_left @ T[:3, :3].T + T[:3, 3]
    front = cp.isfinite(points_rgb).all(axis=1) & (points_rgb[:, 2] > 0)
    points_rgb, source_points = points_rgb[front], source_points[front]

    rgb_h, rgb_w = map(int, image.shape[:2])
    K = cp.asarray(scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (rgb_w, rgb_h)), dtype=cp.float32)
    distortion = tilt = None
    if not rgb_image_is_undistorted:
        source = np.asarray(calibration.rgb_distortion, np.float32).reshape(-1)
        if len(source) not in (4, 5, 8, 12, 14):
            raise ValueError("RGB distortion must contain 4, 5, 8, 12, or 14 values")
        d = np.zeros(14, np.float32)
        d[: len(source)] = source
        distortion = cp.asarray(d)
        if d[12] != 0 or d[13] != 0:
            tilt = cp.asarray(_tilt_projection_matrix(d[12], d[13]))

    pixels = _project_camera_points_opencv_model_gpu(points_rgb, K, distortion, tilt)
    finite = cp.isfinite(pixels).all(axis=1)
    pixels, source_points = pixels[finite], source_points[finite]
    u, v = cp.rint(pixels[:, 0]).astype(cp.int32), cp.rint(pixels[:, 1]).astype(cp.int32)
    inside = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
    u, v, out_points = u[inside], v[inside], source_points[inside]

    colors = image[v, u, :3]
    if input_color_order == "BGR":
        colors = colors[:, ::-1]
    if cp.issubdtype(colors.dtype, cp.floating) and colors.size:
        colors = colors * cp.where(cp.nanmax(colors) <= 1, cp.float32(255), cp.float32(1))
    return out_points, cp.clip(cp.rint(colors), 0, 255).astype(cp.uint8)


def rectified_left_to_original_left(
    points_rectified_m: cp.ndarray,
    rectification: StereoRectification,
) -> cp.ndarray:
    return cp.asarray(points_rectified_m, dtype=cp.float32) @ cp.asarray(
        rectification.R1, dtype=cp.float32
    )


def transform_points(points_m: cp.ndarray, transform_4x4) -> cp.ndarray:
    p = cp.asarray(points_m, dtype=cp.float32)
    T = cp.asarray(transform_4x4, dtype=cp.float32)

    if p.ndim != 2 or p.shape[1] != 3 or T.shape != (4, 4):
        raise ValueError("points must be Nx3 and transform must be 4x4")

    return p @ T[:3, :3].T + T[:3, 3]


def _image_cupy(image) -> cp.ndarray:
    if isinstance(image, cp.ndarray):
        a = image
    elif torch.is_tensor(image):
        a = cp.from_dlpack(image.cuda())
    else:
        a = cp.asarray(image)

    if a.ndim not in (2, 3):
        raise ValueError(f"image must be HxW or HxWxC, got {a.shape}")

    return a


def rgb8(colors: cp.ndarray, order: ColorOrder = "RGB") -> cp.ndarray:
    a = cp.asarray(colors)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"colors must be Nx3, got {a.shape}")
    a = a[:, :3]
    if cp.issubdtype(a.dtype, cp.floating) and a.size and cp.nanmax(a) <= 1:
        a = a * 255
    a = cp.clip(cp.rint(a), 0, 255).astype(cp.uint8)
    return a[:, ::-1] if order == "BGR" else a

def _undistort_points_opencv_model(pix, K, distortion, iterations=5):
    pix = cp.asarray(pix, dtype=cp.float32)
    K = cp.asarray(K, dtype=cp.float32)
    d = cp.zeros(14, dtype=cp.float32)

    src = cp.asarray(distortion, dtype=cp.float32).ravel()
    d[:src.size] = src

    # Pixel -> normalized distorted coordinates
    x0 = (pix[:, 0] - K[0, 2]) / K[0, 0]
    y0 = (pix[:, 1] - K[1, 2]) / K[1, 1]

    # Undo sensor tilt for 14-coefficient OpenCV model
    if src.size >= 14:
        tilt = cp.asarray(
            _tilt_projection_matrix(float(d[12]), float(d[13])),
        )
        inv_tilt = cp.linalg.inv(tilt)

        qx = inv_tilt[0, 0] * x0 + inv_tilt[0, 1] * y0 + inv_tilt[0, 2]
        qy = inv_tilt[1, 0] * x0 + inv_tilt[1, 1] * y0 + inv_tilt[1, 2]
        qz = inv_tilt[2, 0] * x0 + inv_tilt[2, 1] * y0 + inv_tilt[2, 2]
        x0, y0 = qx / qz, qy / qz

    x, y = x0.copy(), y0.copy()

    k1, k2, p1, p2, k3, k4, k5, k6 = d[:8]
    s1, s2, s3, s4 = d[8:12]

    for _ in range(iterations):
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2

        inv_radial = (
            1 + k4 * r2 + k5 * r4 + k6 * r6
        ) / (
            1 + k1 * r2 + k2 * r4 + k3 * r6
        )

        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4

        x = (x0 - dx) * inv_radial
        y = (y0 - dy) * inv_radial

    return cp.stack((
        K[0, 0] * x + K[0, 2],
        K[1, 1] * y + K[1, 2],
    ), axis=1)


def rgb_depth_to_points_rgb(
    depth_rgb_m: cp.ndarray,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted=False,
):
    d = cp.asarray(depth_rgb_m, dtype=cp.float32)

    if d.ndim != 2:
        raise ValueError(f"depth_rgb_m must be HxW, got {d.shape}")

    h, w = d.shape
    y, x = cp.nonzero(cp.isfinite(d) & (d > 0))
    z = d[y, x]

    pix = cp.stack((x, y), axis=1).astype(cp.float32)

    K = cp.asarray(
        scale_K(
            calibration.rgb_intrinsics,
            calibration.rgb_resolution,
            (w, h),
        ),
        dtype=cp.float32,
    )

    if not rgb_image_is_undistorted:
        pix = _undistort_points_opencv_model(
            pix,
            K,
            calibration.rgb_distortion,
        )

    X = (pix[:, 0] - K[0, 2]) * z / K[0, 0]
    Y = (pix[:, 1] - K[1, 2]) * z / K[1, 1]

    return (
        cp.stack((X, Y, z), axis=1),
        cp.stack((x, y), axis=1).astype(cp.int32),
    )


def project_points_to_rgb_pixels(
    points_left_m: cp.ndarray,
    rgb_image,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted=False,
):
    rgb_h, rgb_w = rgb_image.shape[:2]

    K = cp.asarray(
        scale_K(
            calibration.rgb_intrinsics,
            calibration.rgb_resolution,
            (rgb_w, rgb_h),
        ),
        dtype=cp.float32,
    )

    points_rgb = transform_points(
        points_left_m,
        calibration.left_to_rgb,
    )

    distortion = (
        None
        if rgb_image_is_undistorted
        else cp.asarray(calibration.rgb_distortion, dtype=cp.float32)
    )

    pixels = _project_camera_points_opencv_model_gpu(
        points_rgb,
        K,
        distortion,
    )

    return pixels, points_rgb


def points_left_to_rgb_depth(
    points_left_m: cp.ndarray,
    rgb_image,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted=False,
    splat_px=0,
):
    rgb_h, rgb_w = rgb_image.shape[:2]

    pixels, points_rgb = project_points_to_rgb_pixels(
        points_left_m,
        rgb_image,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )

    pixels = cp.asarray(pixels)
    points_rgb = cp.asarray(points_rgb)

    u0 = cp.rint(pixels[:, 0]).astype(cp.int32)
    v0 = cp.rint(pixels[:, 1]).astype(cp.int32)
    z = points_rgb[:, 2].astype(cp.float32)

    depth = cp.full((rgb_h, rgb_w), cp.inf, cp.float32)
    base_valid = cp.isfinite(pixels).all(axis=1) & cp.isfinite(z) & (z > 0)

    r = max(0, int(splat_px))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            u, v = u0 + dx, v0 + dy
            valid = (
                base_valid
                & (u >= 0) & (u < rgb_w)
                & (v >= 0) & (v < rgb_h)
            )
            cp.minimum.at(depth, (v[valid], u[valid]), z[valid])

    valid = cp.isfinite(depth)
    depth[~valid] = cp.nan
    return depth, valid


def stereo_to_point_cloud(left_image, right_image, calibration:StereoRgbCalibration, *, alpha=0.0, min_disparity=0,
                          num_disparities=128, block_size=5, min_depth_m=0.01, max_depth_m=10.0, stride=1):
    
    rectifier = calibration.get_rectifier(alpha)
    left, right, rect = rectifier.rectify(left_image, right_image)
    predictor = VPIStereoDisparityGPU(num_disparities=num_disparities, block_size=block_size)
    disparity,confidence_u16 = predictor.predict(left, right)
    points, _ = rect.disparity_to_points_rectified(disparity,
    confidence_u16=confidence_u16,
                                              min_disparity=max(0.5, float(min_disparity)),
                                              min_depth_m=min_depth_m, max_depth_m=max_depth_m, stride=stride)
    return disparity, rect, points


@dataclass(frozen=True)
class ColoredPointCloud:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    disparity: np.ndarray | None = None
    rectification: StereoRectification | None = None

def stereo_rgb_to_colored_point_cloud(left_image, right_image, rgb_image, *, calibration=None,
                                      output_path=None, input_color_order: ColorOrder = "RGB",
                                      rgb_image_is_undistorted=False, alpha=0.0, min_disparity=0,
                                      num_disparities=128, block_size=5, min_depth_m=0.01, max_depth_m=10.0, stride=1,
                                      output_frame: OutputFrame = "left", save_binary_pcd=True):
    disparity, rect, points = stereo_to_point_cloud(
        left_image, right_image, calibration, alpha=alpha, min_disparity=min_disparity,
        num_disparities=num_disparities, block_size=block_size,
        min_depth_m=min_depth_m,max_depth_m=max_depth_m, stride=stride
    )
    print(type(points))
    points, colors = colorize_points_from_rgb_gpu(
        points, rgb_image, calibration, rectification=rect, output_frame=output_frame,
        input_color_order=input_color_order, rgb_image_is_undistorted=rgb_image_is_undistorted
    )
    if output_path:
        points = points.get()
        colors = colors.get()
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)
    return ColoredPointCloud(points, colors, disparity, rect)


if __name__ == "__main__":
    pass