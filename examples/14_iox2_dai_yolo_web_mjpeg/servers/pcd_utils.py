"""Compact helpers for stereo+RGB images -> colored point clouds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json, math, warnings

import numpy as np

ColorOrder = Literal["RGB", "BGR"]
PointsFrame = Literal["left", "left_rectified"]
OutputFrame = Literal["left", "left_rectified"]

DEFAULT_CALIBRATION: dict[str, Any] = {
    "rgb_resolution": [4056, 3040],
    "left_resolution": [1280, 800],
    "right_resolution": [1280, 800],
    "stereo_translation_units_hint": (
        "Calibration extrinsics units from device; Luxonis stereo baseline "
        "override API uses centimeters."
    ),
    "rgb_intrinsics": [
        [2430.31884765625, 0.0, 2063.196044921875],
        [0.0, 2429.41748046875, 1490.1956787109375],
        [0.0, 0.0, 1.0],
    ],
    "left_intrinsics": [
        [570.8507690429688, 0.0, 653.754150390625],
        [0.0, 570.580810546875, 390.99169921875],
        [0.0, 0.0, 1.0],
    ],
    "right_intrinsics": [
        [567.8758544921875, 0.0, 655.560546875],
        [0.0, 567.7424926757812, 393.97039794921875],
        [0.0, 0.0, 1.0],
    ],
    "left_to_right_extrinsics": [
        [
            0.9998766183853149,
            0.0023975009098649025,
            -0.015519456937909126,
            -7.537897109985352,
        ],
        [
            -0.0024368134327232838,
            0.9999938607215881,
            -0.002514647087082267,
            0.09707357734441757,
        ],
        [
            0.01551333349198103,
            0.0025521547067910433,
            0.9998764395713806,
            -0.08006280660629272,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "left_to_rgb_extrinsics": [
        [
            0.999745786190033,
            -0.010174884460866451,
            -0.020119857043027878,
            -3.7557337284088135,
        ],
        [
            0.010090984404087067,
            0.9999399781227112,
            -0.004267154261469841,
            -0.004705727566033602,
        ],
        [
            0.02016206830739975,
            0.004063040018081665,
            0.9997884631156921,
            -0.04603101313114166,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "rgb_distortion": [
        11.808209419250488,
        11.02328872680664,
        0.0005683265044353902,
        -0.0014364976668730378,
        -1.831695795059204,
        11.769153594970703,
        14.672675132751465,
        -1.088363766670227,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.00932407472282648,
        -0.015433108434081078,
    ],
    "left_distortion": [
        5.454817771911621,
        1.694711446762085,
        8.319105836562812e-05,
        -5.4938958783168346e-05,
        0.029059873893857002,
        5.82321310043335,
        3.369436502456665,
        0.23804199695587158,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.004699581768363714,
        -0.0014164879685267806,
    ],
    "right_distortion": [
        5.091114521026611,
        1.5919005870819092,
        -7.720104804320727e-06,
        2.0027317077619955e-05,
        0.029687780886888504,
        5.4577412605285645,
        3.150493621826172,
        0.22900323569774628,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0026084419805556536,
        -0.002354657743126154,
    ],
}



def cv2():
    try:
        import cv2 as _cv2  # type: ignore
        return _cv2
    except ImportError as e:
        raise ImportError("OpenCV is required: pip install opencv-python") from e


def arr(x: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    a = np.asarray(x, np.float64)
    if shape and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    return a.copy()


def resolution(x: Any, name: str) -> tuple[int, int]:
    if len(x) != 2:
        raise ValueError(f"{name} must be [width, height]")
    w, h = map(int, x)
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} must contain positive values")
    return w, h


def scale_K(K: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    K = K.astype(np.float64, copy=True)
    if from_wh != to_wh:
        sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
        K[0, [0, 2]] *= sx
        K[1, [1, 2]] *= sy
    return K


def gray8(image: np.ndarray) -> np.ndarray:
    c = cv2()
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = c.cvtColor(a[:, :, :3], c.COLOR_BGR2GRAY)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    elif a.ndim != 2:
        raise ValueError(f"Unsupported image shape: {a.shape}")
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating) and np.nanmax(a) <= 1:
        a = a * 255
    if a.dtype == np.uint16 and a.max() > 0:
        a = a.astype(np.float32) * (255 / a.max())
    return np.clip(a, 0, 255).astype(np.uint8)


def rgb8(colors: np.ndarray, order: ColorOrder = "RGB") -> np.ndarray:
    a = np.asarray(colors)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"colors must be Nx3, got {a.shape}")
    a = a[:, :3]
    if np.issubdtype(a.dtype, np.floating) and a.size and np.nanmax(a) <= 1:
        a = a * 255
    a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return a[:, ::-1] if order == "BGR" else a


@dataclass(frozen=True)
class StereoRgbCalibration:
    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]
    rgb_intrinsics: np.ndarray
    left_intrinsics: np.ndarray
    right_intrinsics: np.ndarray
    rgb_distortion: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    left_to_right: np.ndarray
    left_to_rgb: np.ndarray
    source_translation_unit: Literal["m", "cm", "mm"] = "cm"

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, source_translation_unit: Literal["m", "cm", "mm"] = "cm"):
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[source_translation_unit]
        l2r = arr(d["left_to_right_extrinsics"], "left_to_right_extrinsics", (4, 4))
        l2rgb = arr(d["left_to_rgb_extrinsics"], "left_to_rgb_extrinsics", (4, 4))
        l2r[:3, 3] *= scale
        l2rgb[:3, 3] *= scale
        return cls(
            resolution(d["rgb_resolution"], "rgb_resolution"),
            resolution(d["left_resolution"], "left_resolution"),
            resolution(d["right_resolution"], "right_resolution"),
            arr(d["rgb_intrinsics"], "rgb_intrinsics", (3, 3)),
            arr(d["left_intrinsics"], "left_intrinsics", (3, 3)),
            arr(d["right_intrinsics"], "right_intrinsics", (3, 3)),
            arr(d["rgb_distortion"], "rgb_distortion").reshape(-1, 1),
            arr(d["left_distortion"], "left_distortion").reshape(-1, 1),
            arr(d["right_distortion"], "right_distortion").reshape(-1, 1),
            l2r, l2rgb, source_translation_unit,
        )

    @classmethod
    def default(cls):
        return cls.from_dict(DEFAULT_CALIBRATION)

    @property
    def stereo_baseline_m(self) -> float: return abs(float(self.left_to_right[0, 3]))
    @property
    def stereo_baseline_cm(self) -> float: return self.stereo_baseline_m * 100
    @property
    def stereo_translation_norm_m(self) -> float: return float(np.linalg.norm(self.left_to_right[:3, 3]))
    @property
    def left_to_right_rotation(self) -> np.ndarray: return self.left_to_right[:3, :3].copy()
    @property
    def left_to_right_translation_m(self) -> np.ndarray: return self.left_to_right[:3, 3:4].copy()
    @property
    def left_to_rgb_rotation(self) -> np.ndarray: return self.left_to_rgb[:3, :3].copy()
    @property
    def left_to_rgb_translation_m(self) -> np.ndarray: return self.left_to_rgb[:3, 3:4].copy()


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


@dataclass(frozen=True)
class ColoredPointCloud:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    disparity: np.ndarray | None = None
    rectification: StereoRectification | None = None


def load_calibration_json(path: str | Path, *, source_translation_unit: Literal["m", "cm", "mm"] = "cm"):
    return StereoRgbCalibration.from_dict(json.loads(Path(path).read_text()), source_translation_unit=source_translation_unit)


def make_stereo_rectification(calibration: StereoRgbCalibration, *, image_size=None, alpha=0.0, zero_disparity=True):
    c = cv2()
    image_size = image_size or calibration.left_resolution
    if image_size != calibration.left_resolution:
        warnings.warn("Input size differs from calibration; intrinsics are scaled.", RuntimeWarning, stacklevel=2)
    K1 = scale_K(calibration.left_intrinsics, calibration.left_resolution, image_size)
    K2 = scale_K(calibration.right_intrinsics, calibration.right_resolution, image_size)
    flags = c.CALIB_ZERO_DISPARITY if zero_disparity else 0
    R1, R2, P1, P2, Q, roi1, roi2 = c.stereoRectify(
        K1, calibration.left_distortion, K2, calibration.right_distortion, image_size,
        calibration.left_to_right_rotation, calibration.left_to_right_translation_m,
        flags=flags, alpha=float(alpha),
    )
    maps = [c.initUndistortRectifyMap(K, D, R, P, image_size, c.CV_32FC1)
            for K, D, R, P in ((K1, calibration.left_distortion, R1, P1),
                               (K2, calibration.right_distortion, R2, P2))]
    return StereoRectification(image_size, *maps[0], *maps[1], R1, R2, P1, P2, Q, tuple(map(int, roi1)), tuple(map(int, roi2)))


def rectify_stereo_pair(left_image, right_image, calibration=None, *, rectification=None, alpha=0.0):
    c = cv2()
    if left_image is None or right_image is None or left_image.shape[:2] != right_image.shape[:2]:
        raise ValueError("left/right images are required and must have matching sizes")
    h, w = left_image.shape[:2]
    rectification = rectification or make_stereo_rectification(
        calibration or StereoRgbCalibration.default(), image_size=(w, h), alpha=alpha
    )
    left = c.remap(left_image, rectification.left_map_x, rectification.left_map_y, c.INTER_LINEAR, borderMode=c.BORDER_CONSTANT)
    right = c.remap(right_image, rectification.right_map_x, rectification.right_map_y, c.INTER_LINEAR, borderMode=c.BORDER_CONSTANT)
    return left, right, rectification


def compute_disparity_sgbm(left_rectified, right_rectified, *, min_disparity=0, num_disparities=128, block_size=5,
                           uniqueness_ratio=8, speckle_window_size=80, speckle_range=2,
                           disp12_max_diff=1, pre_filter_cap=31, invalid_to_nan=True):
    c = cv2()
    left, right = gray8(left_rectified), gray8(right_rectified)
    if left.shape != right.shape:
        raise ValueError("Rectified left/right images must have the same shape")
    num_disparities = math.ceil(max(16, int(num_disparities)) / 16) * 16
    block_size = max(3, int(block_size)) | 1
    p1, p2 = 8 * block_size ** 2, 32 * block_size ** 2
    matcher = c.StereoSGBM_create(
        minDisparity=int(min_disparity), numDisparities=num_disparities, blockSize=block_size,
        P1=p1, P2=p2, disp12MaxDiff=int(disp12_max_diff), uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size), speckleRange=int(speckle_range),
        preFilterCap=int(pre_filter_cap), mode=c.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(left, right).astype(np.float32) / 16
    if invalid_to_nan:
        disparity[disparity <= float(min_disparity)] = np.nan
    return disparity


def disparity_to_points_rectified(disparity, rectification: StereoRectification, *, min_disparity=0.5,
                                  max_depth_m=10.0, stride=1, mask=None):
    c = cv2()
    d = np.asarray(disparity, np.float32)
    if d.ndim != 2:
        raise ValueError(f"disparity must be HxW, got {d.shape}")
    valid = np.isfinite(d) & (d > float(min_disparity))
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape != d.shape:
            raise ValueError(f"mask shape {m.shape} does not match disparity shape {d.shape}")
        valid &= m
    if stride > 1:
        s = np.zeros_like(valid); s[::stride, ::stride] = True; valid &= s
    points = c.reprojectImageTo3D(np.nan_to_num(d), rectification.Q)
    valid &= np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    if max_depth_m is not None:
        valid &= points[:, :, 2] <= float(max_depth_m)
    y, x = np.nonzero(valid)
    return points[y, x].astype(np.float64), np.column_stack([x, y]).astype(np.int32)


def rectified_left_to_original_left(points_rectified_m, rectification: StereoRectification):
    return np.asarray(points_rectified_m, np.float64) @ rectification.R1


def transform_points(points_m, transform_4x4):
    p, T = np.asarray(points_m, np.float64), np.asarray(transform_4x4, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or T.shape != (4, 4):
        raise ValueError("points must be Nx3 and transform must be 4x4")
    return p @ T[:3, :3].T + T[:3, 3]


def project_points_to_rgb_pixels(points_left_m, rgb_image, calibration: StereoRgbCalibration, *, rgb_image_is_undistorted=False):
    c = cv2()
    rgb_h, rgb_w = np.asarray(rgb_image).shape[:2]
    K = scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (rgb_w, rgb_h))
    points_rgb = transform_points(points_left_m, calibration.left_to_rgb)
    pixels, _ = c.projectPoints(points_rgb.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                K, None if rgb_image_is_undistorted else calibration.rgb_distortion)
    return pixels.reshape(-1, 2).astype(np.float64), points_rgb


def sample_rgb_colors(rgb_image, pixel_xy, *, input_color_order: ColorOrder = "RGB", interpolation: Literal["nearest"] = "nearest"):
    if interpolation != "nearest":
        raise NotImplementedError("Only nearest-neighbor sampling is implemented")
    image, pix = np.asarray(rgb_image), np.asarray(pixel_xy, np.float64)
    if image.ndim != 3 or image.shape[2] < 3 or pix.ndim != 2 or pix.shape[1] != 2:
        raise ValueError("rgb_image must be HxWx3/4 and pixel_xy must be Nx2")
    h, w = image.shape[:2]
    u, v = np.rint(pix[:, 0]).astype(int), np.rint(pix[:, 1]).astype(int)
    valid = np.isfinite(pix).all(axis=1) & (0 <= u) & (u < w) & (0 <= v) & (v < h)
    return rgb8(image[v[valid], u[valid], :3], input_color_order), valid


def colorize_points_from_rgb(points_m, rgb_image, calibration: StereoRgbCalibration, *, rectification=None,
                             points_frame: PointsFrame = "left_rectified", output_frame: OutputFrame = "left",
                             input_color_order: ColorOrder = "RGB", rgb_image_is_undistorted=False):
    points = np.asarray(points_m, np.float64)
    if points_frame == "left_rectified":
        if rectification is None:
            raise ValueError("rectification is required for rectified-left points")
        points_left = rectified_left_to_original_left(points, rectification)
    elif points_frame == "left":
        points_left = points
    else:
        raise ValueError(f"Unsupported points_frame: {points_frame}")
    pixels, points_rgb = project_points_to_rgb_pixels(points_left, rgb_image, calibration,
                                                      rgb_image_is_undistorted=rgb_image_is_undistorted)
    front = points_rgb[:, 2] > 0
    colors, valid_front = sample_rgb_colors(rgb_image, pixels[front], input_color_order=input_color_order)
    valid = np.zeros(len(points), bool)
    valid[np.flatnonzero(front)[valid_front]] = True
    if output_frame not in ("left", "left_rectified"):
        raise ValueError(f"Unsupported output_frame: {output_frame}")
    out_points = points_left[valid] if output_frame == "left" else points[valid]
    return out_points.astype(np.float64), colors


def _rgb_float(colors_rgb):
    c = rgb8(colors_rgb)
    packed = (c[:, 0].astype(np.uint32) << 16) | (c[:, 1].astype(np.uint32) << 8) | c[:, 2].astype(np.uint32)
    return packed.astype("<u4").view("<f4")


def _clean_cloud(points_m, colors_rgb):
    p, c = np.asarray(points_m, np.float32), rgb8(colors_rgb)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(c):
        raise ValueError("points must be Nx3 and match colors length")
    finite = np.isfinite(p).all(axis=1)
    return p[finite], c[finite]


def save_pcd(path: str | Path, points_m, colors_rgb, *, binary=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    p, c = _clean_cloud(points_m, colors_rgb)
    rgb, n = _rgb_float(c), len(p)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA {'binary' if binary else 'ascii'}\n")
    if binary:
        data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        data["x"], data["y"], data["z"], data["rgb"] = p[:, 0], p[:, 1], p[:, 2], rgb
        with path.open("wb") as f:
            f.write(header.encode("ascii")); data.tofile(f)
    else:
        with path.open("w", encoding="ascii") as f:
            f.write(header)
            f.writelines(f"{x:.8f} {y:.8f} {z:.8f} {float(r):.9e}\n" for (x, y, z), r in zip(p, rgb))
    return path


def save_ply_ascii(path: str | Path, points_m, colors_rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    p, c = _clean_cloud(points_m, colors_rgb)
    header = ("ply\nformat ascii 1.0\n" f"element vertex {len(p)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("w", encoding="ascii") as f:
        f.write(header)
        f.writelines(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n" for (x, y, z), (r, g, b) in zip(p, c))
    return path


def save_point_cloud(path: str | Path, points_m, colors_rgb, *, binary_pcd=True):
    suffix = Path(path).suffix.lower()
    if suffix == ".pcd": return save_pcd(path, points_m, colors_rgb, binary=binary_pcd)
    if suffix == ".ply": return save_ply_ascii(path, points_m, colors_rgb)
    raise ValueError("Use .pcd or .ply")


def npz_to_pcd(input_npz: str | Path, output_path: str | Path, *, binary_pcd=True):
    with np.load(input_npz, allow_pickle=False) as z:
        save_point_cloud(output_path, z["points_m"], z["colors_rgb"], binary_pcd=binary_pcd)
    return Path(output_path)


def stereo_rgb_to_colored_point_cloud(left_image, right_image, rgb_image, *, calibration=None, output_path=None,
                                      input_color_order: ColorOrder = "RGB", rgb_image_is_undistorted=False,
                                      alpha=0.0, min_disparity=0, num_disparities=128, block_size=5,
                                      max_depth_m=10.0, stride=1, output_frame: OutputFrame = "left",
                                      save_binary_pcd=True):
    calibration = calibration or StereoRgbCalibration.default()
    left, right, rect = rectify_stereo_pair(left_image, right_image, calibration, alpha=alpha)
    disparity = compute_disparity_sgbm(left, right, min_disparity=min_disparity,
                                       num_disparities=num_disparities, block_size=block_size)
    points, _ = disparity_to_points_rectified(disparity, rect, min_disparity=max(0.5, float(min_disparity)),
                                              max_depth_m=max_depth_m, stride=stride)
    points, colors = colorize_points_from_rgb(points, rgb_image, calibration, rectification=rect,
                                              output_frame=output_frame, input_color_order=input_color_order,
                                              rgb_image_is_undistorted=rgb_image_is_undistorted)
    if output_path:
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)
    return ColoredPointCloud(points, colors, disparity, rect)


def read_image(path: str | Path, *, color=True):
    c = cv2()
    image = c.imread(str(path), c.IMREAD_COLOR if color else c.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def points_left_to_rgb_depth(points_left_m, rgb_image, calibration: StereoRgbCalibration, *,
                             rgb_image_is_undistorted=False, splat_px=0):
    rgb_h, rgb_w = np.asarray(rgb_image).shape[:2]
    pixels, points_rgb = project_points_to_rgb_pixels(points_left_m, rgb_image, calibration,
                                                      rgb_image_is_undistorted=rgb_image_is_undistorted)
    u0, v0 = np.rint(pixels[:, 0]).astype(np.int32), np.rint(pixels[:, 1]).astype(np.int32)
    z = points_rgb[:, 2].astype(np.float64)
    depth = np.full((rgb_h, rgb_w), np.inf, np.float64)
    r = max(0, int(splat_px))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            u, v = u0 + dx, v0 + dy
            ok = (np.isfinite(pixels).all(axis=1) & np.isfinite(z) & (z > 0) &
                  (0 <= u) & (u < rgb_w) & (0 <= v) & (v < rgb_h))
            np.minimum.at(depth, (v[ok], u[ok]), z[ok])
    valid = np.isfinite(depth)
    depth[~valid] = np.nan
    return depth, valid


def rgb_depth_to_points_rgb(depth_rgb_m, calibration: StereoRgbCalibration, *,
                            rgb_image_is_undistorted=False):
    c = cv2()
    d = np.asarray(depth_rgb_m, np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth_rgb_m must be HxW, got {d.shape}")
    h, w = d.shape
    y, x = np.nonzero(np.isfinite(d) & (d > 0))
    z = d[y, x]
    pix = np.column_stack([x, y]).astype(np.float64)
    K = scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (w, h))
    if not rgb_image_is_undistorted:
        pix = c.undistortPoints(pix.reshape(-1, 1, 2), K, calibration.rgb_distortion, P=K).reshape(-1, 2)
    X = (pix[:, 0] - K[0, 2]) * z / K[0, 0]
    Y = (pix[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.column_stack([X, Y, z]).astype(np.float64), np.column_stack([x, y]).astype(np.int32)


RgbOutputFrame = Literal["left", "rgb"]
def stereo_rgb_to_colored_point_cloud_rgb_res(left_image, right_image, rgb_image, *, calibration=None, output_path=None,
                                      input_color_order: ColorOrder = "RGB", rgb_image_is_undistorted=False,
                                      alpha=0.0, min_disparity=0, num_disparities=128, block_size=5,
                                      max_depth_m=10.0, splat_px=1, output_frame: RgbOutputFrame = "left",
                                      save_binary_pcd=True):
    calibration = calibration or StereoRgbCalibration.default()
    left, right, rect = rectify_stereo_pair(left_image, right_image, calibration, alpha=alpha)
    disparity = compute_disparity_sgbm(left, right, min_disparity=min_disparity,
                                       num_disparities=num_disparities, block_size=block_size)
    points_rect, _ = disparity_to_points_rectified(disparity, rect, min_disparity=max(0.5, float(min_disparity)),
                                                   max_depth_m=max_depth_m, stride=1)
    points_left = rectified_left_to_original_left(points_rect, rect)
    depth_rgb, _ = points_left_to_rgb_depth(points_left, rgb_image, calibration,
                                            rgb_image_is_undistorted=rgb_image_is_undistorted,
                                            splat_px=splat_px)
    points_rgb, xy = rgb_depth_to_points_rgb(depth_rgb, calibration,
                                             rgb_image_is_undistorted=rgb_image_is_undistorted)
    x, y = xy[:, 0], xy[:, 1]
    colors = rgb8(np.asarray(rgb_image)[y, x, :3], input_color_order)
    if output_frame == "rgb":
        points = points_rgb
    elif output_frame == "left":
        points = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))
    else:
        raise ValueError("output_frame must be 'left' or 'rgb'")
    if output_path:
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)
    return ColoredPointCloud(points, colors, disparity, rect)


__all__ = [name for name in globals() if not name.startswith("_") and name not in {"Any", "json", "math", "warnings", "Path", "np"}]


if __name__ == "__main__":
    # Example:
    #
    # left = read_image("test/left.png", color=False)
    # right = read_image("test/right.png", color=False)
    # rgb = read_image("test/rgb.jpg", color=True)  # cv2 gives BGR
    
    # cloud = stereo_rgb_to_colored_point_cloud(
    #     left,
    #     right,
    #     rgb,
    #     output_path="colored_cloud.pcd",
    #     input_color_order="BGR",
    #     num_disparities=160,
    #     block_size=5,
    #     max_depth_m=2.0,
    #     stride=1,
    # )    

    # cloud = stereo_rgb_to_colored_point_cloud_rgb_res(
    #     left,
    #     right,
    #     rgb,
    #     output_path="rgb_res_cloud.pcd",
    #     input_color_order="BGR",  # because read_image() uses cv2.imread()
    #     splat_px=1,
    #     output_frame="left",
    #     max_depth_m=2.0,
    # )

    # print(f"Saved {len(cloud.points_m):,} colored points")
    pass
