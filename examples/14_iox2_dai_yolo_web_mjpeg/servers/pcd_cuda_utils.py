"""Compact helpers for stereo+RGB images -> colored point clouds."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json, math, warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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


def _device(device=None):
    device = torch.device(device or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build/runtime")
    return device


def _tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _image_hw(image):
    s = tuple(image.shape)
    if len(s) == 2:
        return s
    if len(s) == 3:
        return s[:2]
    raise ValueError(f"image must be HxW or HxWxC, got {s}")


def _image_cuda(image, device="cuda"):
    t = image.to(device) if torch.is_tensor(image) else torch.as_tensor(np.asarray(image), device=device)
    if t.ndim not in (2, 3):
        raise ValueError(f"image must be HxW or HxWxC, got {tuple(t.shape)}")
    return t


def _numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


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


def scale_K(K, from_wh: tuple[int, int], to_wh: tuple[int, int]):
    if torch.is_tensor(K):
        out = K.clone()
        if from_wh != to_wh:
            sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
            out[0, 0] *= sx; out[0, 2] *= sx
            out[1, 1] *= sy; out[1, 2] *= sy
        return out
    out = np.asarray(K, np.float64).copy()
    if from_wh != to_wh:
        sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
        out[0, [0, 2]] *= sx
        out[1, [1, 2]] *= sy
    return out


def gray8(image) -> np.ndarray:
    a = _numpy(image)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    elif a.ndim != 2:
        raise ValueError(f"Unsupported image shape: {a.shape}")
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating) and a.size and np.nanmax(a) <= 1:
        a = a * 255
    if a.dtype == np.uint16 and a.size and a.max() > 0:
        a = a.astype(np.float32) * (255 / a.max())
    return np.clip(a, 0, 255).astype(np.uint8)


def rgb8(colors, order: ColorOrder = "RGB") -> np.ndarray:
    a = _numpy(colors)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"colors must be Nx3, got {a.shape}")
    a = a[:, :3]
    if np.issubdtype(a.dtype, np.floating) and a.size and np.nanmax(a) <= 1:
        a = a * 255
    a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return a[:, ::-1] if order == "BGR" else a


def rgb8_cuda(colors:torch.Tensor, order: ColorOrder = "RGB"):
    a = colors[:, :3]
    if a.dtype.is_floating_point:
        # Camera images are normally uint8; this also supports [0,1] float images.
        if a.numel() and bool((torch.nan_to_num(a).amax() <= 1).item()):
            a = a * 255
        a = a.round().clamp(0, 255).to(torch.uint8)
    else:
        a = a.clamp(0, 255).to(torch.uint8)
    return a.flip(1) if order == "BGR" else a


@dataclass(frozen=True)
class StereoRgbCalibrationCpu:
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
    def from_dict(cls, d: dict[str, Any],
                  source_translation_unit: Literal["m", "cm", "mm"] = "cm"):
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
            arr(d["rgb_distortion"], "rgb_distortion").reshape(-1),
            arr(d["left_distortion"], "left_distortion").reshape(-1),
            arr(d["right_distortion"], "right_distortion").reshape(-1),
            l2r, l2rgb, source_translation_unit,
        )

    @classmethod
    def default(cls):
        return cls.from_dict(DEFAULT_CALIBRATION)

    def get_rectifier(self, alpha=0.0, zero_disparity=True):
        return StereoRectifierCuda(self, alpha, zero_disparity)

    def to_cuda(self, device="cuda"):
        device = _device(device)
        t = lambda x: _tensor(x, device)
        return StereoRgbCalibrationCuda(
            self.rgb_resolution, self.left_resolution, self.right_resolution,
            t(self.rgb_intrinsics), t(self.left_intrinsics), t(self.right_intrinsics),
            t(self.rgb_distortion), t(self.left_distortion), t(self.right_distortion),
            t(self.left_to_right), t(self.left_to_rgb), self.source_translation_unit,
        )

    @property
    def stereo_baseline_m(self): return abs(float(self.left_to_right[0, 3]))
    @property
    def stereo_baseline_cm(self): return self.stereo_baseline_m * 100
    # @property
    # def stereo_translation_norm_m(self): return float(np.linalg.norm(self.left_to_right[:3, 3]))
    @property
    def left_to_right_rotation(self): return self.left_to_right[:3, :3].copy()
    @property
    def left_to_right_translation_m(self): return self.left_to_right[:3, 3:4].copy()


@dataclass(frozen=True)
class StereoRgbCalibrationCuda:
    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]
    rgb_intrinsics: torch.Tensor
    left_intrinsics: torch.Tensor
    right_intrinsics: torch.Tensor
    rgb_distortion: torch.Tensor
    left_distortion: torch.Tensor
    right_distortion: torch.Tensor
    left_to_right: torch.Tensor
    left_to_rgb: torch.Tensor
    source_translation_unit: Literal["m", "cm", "mm"] = "cm"

    @classmethod
    def from_dict(cls, d: dict[str, Any], source_translation_unit: Literal["m", "cm", "mm"] = "cm"):
        res:StereoRgbCalibrationCpu = StereoRgbCalibrationCpu.from_dict(d,source_translation_unit)
        return res.to_cuda()
            
    @classmethod
    def from_cpu(cls,calib:StereoRgbCalibrationCpu):
        return StereoRgbCalibrationCpu(
            calib.rgb_resolution, calib.left_resolution, calib.right_resolution,
            calib.rgb_intrinsics, calib.left_intrinsics, calib.right_intrinsics,
            calib.rgb_distortion, calib.left_distortion, calib.right_distortion,
            calib.left_to_right, calib.left_to_rgb, calib.source_translation_unit,
        ).to_cuda()

    @classmethod
    def default(cls):
        return cls.from_dict(DEFAULT_CALIBRATION)

    def get_rectifier(self, alpha=0.0, zero_disparity=True):
        return StereoRectifierCuda(self, alpha, zero_disparity)

    def to_cuda(self, device="cuda"):
        return self

    @property
    def device(self): return self.rgb_intrinsics.device
    @property
    def stereo_baseline_m(self): return abs(float(self.left_to_right[0, 3]))
    @property
    def stereo_baseline_cm(self): return self.stereo_baseline_m * 100
    # @property
    # def stereo_translation_norm_m(self): return float(np.linalg.norm(self.left_to_right[:3, 3]))
    @property
    def left_to_right_rotation(self): return self.left_to_right[:3, :3]
    @property
    def left_to_right_translation_m(self): return self.left_to_right[:3, 3:4]


@dataclass(frozen=True)
class _StereoRectification:
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

    def to_cuda(self, device="cuda"):
        device = _device(device)
        t = lambda x: _tensor(x, device)
        return StereoRectificationCuda(
            self.image_size, t(self.left_map_x), t(self.left_map_y),
            t(self.right_map_x), t(self.right_map_y), t(self.R1), t(self.R2),
            t(self.P1), t(self.P2), t(self.Q), self.valid_roi_left, self.valid_roi_right,
        )


@dataclass(frozen=True)
class StereoRectificationCuda:
    image_size: tuple[int, int]
    left_map_x: torch.Tensor
    left_map_y: torch.Tensor
    right_map_x: torch.Tensor
    right_map_y: torch.Tensor
    R1: torch.Tensor
    R2: torch.Tensor
    P1: torch.Tensor
    P2: torch.Tensor
    Q: torch.Tensor
    valid_roi_left: tuple[int, int, int, int]
    valid_roi_right: tuple[int, int, int, int]

    @property
    def device(self): return self.Q.device

    @torch.no_grad()
    def disparity_to_points_rectified(self, disparity, *, min_disparity=.5,
                                      min_depth_m=.01, max_depth_m=10., stride=1, mask=None):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        d0 = _tensor(disparity, self.device)
        if d0.ndim != 2:
            raise ValueError(f"disparity must be HxW, got {tuple(d0.shape)}")
        d = d0[::stride, ::stride]
        valid = torch.isfinite(d) & (d > min_disparity)
        if mask is not None:
            m = _tensor(mask, self.device, torch.bool)
            if m.shape != d0.shape:
                raise ValueError(f"mask shape {tuple(m.shape)} does not match disparity shape {tuple(d0.shape)}")
            valid &= m[::stride, ::stride]
        y, x = torch.where(valid)
        disp = d[y, x]
        x, y = x * stride, y * stride
        q = self.Q
        w = q[3, 2] * disp + q[3, 3]
        z = q[2, 3] / w
        good = torch.isfinite(z) & torch.isfinite(w) & (w != 0) & (z > 0)
        if min_depth_m is not None: good &= z >= min_depth_m
        if max_depth_m is not None: good &= z <= max_depth_m
        x, y, z, w = x[good], y[good], z[good], w[good]
        points = torch.stack(((x.float() + q[0, 3]) / w,
                              (y.float() + q[1, 3]) / w, z), 1)
        return points, torch.stack((x, y), 1).to(torch.int32)


@dataclass(frozen=True)
class ColoredPointCloudCuda:
    points_m: torch.Tensor
    colors_rgb: torch.Tensor
    disparity: torch.Tensor | None = None
    rectification: StereoRectificationCuda | None = None

    def cpu(self):
        return ColoredPointCloud(
            self.points_m.detach().cpu().numpy(),
            self.colors_rgb.detach().cpu().numpy(),
            None if self.disparity is None else self.disparity.detach().cpu().numpy(),
            self.rectification,
        )


@dataclass(frozen=True)
class ColoredPointCloud:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    disparity: np.ndarray | None = None
    rectification: Any = None


class StereoRectifierCuda:
    """OpenCV computes calibration maps once; maps can then be moved to CUDA."""
    def __init__(self, calibration: StereoRgbCalibrationCpu, alpha=0.0, zero_disparity=True):
        self.calibration = calibration
        self.alpha, self.zero_disparity = alpha, zero_disparity
        self.rectification = None
        self.image_size = None

    def make(self, image_size=None):
        if self.rectification is not None and image_size == self.image_size:
            return self.rectification.to_cuda()
        cal = self.calibration
        self.image_size = size = image_size or cal.left_resolution
        if size != cal.left_resolution:
            warnings.warn("Input size differs from calibration; intrinsics are scaled.", RuntimeWarning, stacklevel=2)
        K1 = scale_K(cal.left_intrinsics, cal.left_resolution, size)
        K2 = scale_K(cal.right_intrinsics, cal.right_resolution, size)
        flags = cv2.CALIB_ZERO_DISPARITY if self.zero_disparity else 0
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, cal.left_distortion, K2, cal.right_distortion, size,
            cal.left_to_right_rotation, cal.left_to_right_translation_m,
            flags=flags, alpha=float(self.alpha),
        )
        maps = [cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
                for K, D, R, P in ((K1, cal.left_distortion, R1, P1),
                                   (K2, cal.right_distortion, R2, P2))]
        self.rectification = _StereoRectification(
            size, *maps[0], *maps[1], R1, R2, P1, P2, Q,
            tuple(map(int, roi1)), tuple(map(int, roi2)),
        )
        return self.rectification.to_cuda()

    @staticmethod
    def _grid(mx:torch.Tensor, my:torch.Tensor):
        h, w = mx.shape
        return torch.stack((2 * mx / (w - 1) - 1,
                            2 * my / (h - 1) - 1), -1)[None]

    @staticmethod
    @torch.no_grad()
    def remap(gray, img_map_x, img_map_y):
        grid = StereoRectifierCuda._grid
        dtype, is_gray = gray.dtype, gray.ndim == 2
        x = gray[None, None] if is_gray else gray.permute(2, 0, 1)[None]
        grid = grid(img_map_x, img_map_y)
        out = F.grid_sample(x.float(), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)[0]
        if not dtype.is_floating_point:
            out = out.round().clamp(0, torch.iinfo(dtype).max).to(dtype)
        else:
            out = out.to(dtype)
        return out[0] if is_gray else out.permute(1, 2, 0)
    
    def rectify(self, left, right, rectification=None, device="cuda"):
        if isinstance(left,np.ndarray):            
            left, right = _image_cuda(left), _image_cuda(right)

        h, w = _image_hw(left)
        if _image_hw(right) != (h, w):
            raise ValueError("Left/right images must have matching sizes")
        r = rectification or self.make((w, h))
        if isinstance(r, _StereoRectification):
            r = r.to_cuda(device)
        return (
            self.remap(left, r.left_map_x, r.left_map_y),
            self.remap(right, r.right_map_x, r.right_map_y),
            r,
        )


class SGBMDisparityPredictorCuda:
    """libSGM CUDA stereo matcher."""

    def __init__(
        self,
        width: int,
        height: int,
        num_disparities: int = 128,
        p1: int = 10,
        p2: int = 120,
        uniqueness: float = 0.95,
        paths: int = 4,
        min_disparity: int = 0,
        lr_max_diff: int = 1,
        device: str | torch.device = "cuda",
        dll: str | Path = "build/Release/sgm_py.dll",
        block_size=None,
    ) -> None:
        if num_disparities not in (64, 128, 256):
            raise ValueError("num_disparities must be 64, 128, or 256")

        self.device = torch.device(device)
        self.shape = (height, width)

        self.lib = ctypes.CDLL(str(Path(dll).resolve()))

        self.lib.sgm_create.restype = ctypes.c_void_p
        self.lib.sgm_execute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.sgm_invalid.argtypes = [ctypes.c_void_p]
        self.lib.sgm_invalid.restype = ctypes.c_int
        self.lib.sgm_destroy.argtypes = [ctypes.c_void_p]

        with torch.cuda.device(self.device):
            self.handle = self.lib.sgm_create(
                width,
                height,
                num_disparities,
                p1,
                p2,
                ctypes.c_float(uniqueness),
                paths,
                min_disparity,
                lr_max_diff,
            )

        self.invalid = self.lib.sgm_invalid(self.handle)

    @torch.inference_mode()
    def predict(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != self.shape or right.shape != self.shape:
            raise ValueError(f"Expected images with shape {self.shape}")
        if left.dtype != torch.uint8 or right.dtype != torch.uint8:
            raise ValueError("Expected uint8 images")
        # if left.device != self.device or right.device != self.device:
        #     raise ValueError(f"Images must be on {self.device}")

        left = left.contiguous()
        right = right.contiguous()
        output = torch.empty(
            self.shape,
            dtype=torch.int16,
            device=self.device,
        )

        with torch.cuda.device(self.device):
            torch.cuda.current_stream().synchronize()

            self.lib.sgm_execute(
                self.handle,
                ctypes.c_void_p(left.data_ptr()),
                ctypes.c_void_p(right.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
            )

            torch.cuda.synchronize()

        disparity = output.float().div_(16)
        disparity[output == self.invalid] = torch.nan
        return disparity

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.sgm_destroy(handle)

@torch.no_grad()
def rectified_left_to_original_left(points_rectified_m, rectification: StereoRectificationCuda):
    return _tensor(points_rectified_m, rectification.device) @ rectification.R1


@torch.no_grad()
def transform_points(points_m, transform_4x4):
    if not torch.is_tensor(transform_4x4):
        device = points_m.device if torch.is_tensor(points_m) else _device()
        T = _tensor(transform_4x4, device)
    else:
        T = transform_4x4
    p = _tensor(points_m, T.device)
    if p.ndim != 2 or p.shape[1] != 3 or T.shape != (4, 4):
        raise ValueError("points must be Nx3 and transform must be 4x4")
    return p @ T[:3, :3].T + T[:3, 3]


def _tilt_projection_matrix_cuda(tau_x, tau_y):
    one, zero = torch.ones_like(tau_x), torch.zeros_like(tau_x)
    cx, sx, cy, sy = torch.cos(tau_x), torch.sin(tau_x), torch.cos(tau_y), torch.sin(tau_y)
    rx = torch.stack((one,zero,zero, zero,cx,sx, zero,-sx,cx)).reshape(3,3)
    ry = torch.stack((cy,zero,-sy, zero,one,zero, sy,zero,cy)).reshape(3,3)
    r = ry @ rx
    pz = torch.stack((r[2,2],zero,-r[0,2], zero,r[2,2],-r[1,2], zero,zero,one)).reshape(3,3)
    return pz @ r


@torch.no_grad()
def _project_camera_points_opencv_model(
        points_camera: torch.Tensor,K: torch.Tensor,distortion: torch.Tensor | None = None
    ):
    p = points_camera
    if p.ndim != 2 or p.shape[1] != 3 or K.shape != (3, 3):
        raise ValueError("points_camera must be Nx3 and K must be 3x3")
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    if distortion is not None and distortion.numel():
        src = distortion.flatten()
        if src.numel() not in (4, 5, 8, 12, 14):
            raise ValueError("distortion must contain 4, 5, 8, 12, or 14 coefficients")
        d = torch.zeros(14, device=p.device, dtype=p.dtype); d[:src.numel()] = src
        k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tx,ty = d
        r2 = x*x + y*y; r4 = r2*r2; r6 = r4*r2
        radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
        xy = x*y
        xd = x*radial + 2*p1*xy + p2*(r2 + 2*x*x) + s1*r2 + s2*r4
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*xy + s3*r2 + s4*r4
        if src.numel() == 14:
            tilt = _tilt_projection_matrix_cuda(tx, ty)
            h = torch.stack((xd, yd, torch.ones_like(xd)), 1) @ tilt.T
            x, y = h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]
        else:
            x, y = xd, yd
    u = K[0,0]*x + K[0,1]*y + K[0,2]
    v = K[1,0]*x + K[1,1]*y + K[1,2]
    return torch.stack((u, v), 1)


@torch.no_grad()
def project_points_to_rgb_pixels(points_left_m, rgb_image, calibration: StereoRgbCalibrationCuda, *,
                                 rgb_image_is_undistorted=False):
    h, w = _image_hw(rgb_image)
    K = scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (w, h))
    points_rgb = transform_points(points_left_m, calibration.left_to_rgb)
    distortion = None if rgb_image_is_undistorted else calibration.rgb_distortion
    return _project_camera_points_opencv_model(points_rgb, K, distortion), points_rgb


@torch.no_grad()
def sample_rgb_colors(rgb_cuda_u8, pixel_xy, *,
                      input_color_order: ColorOrder = "RGB", interpolation: Literal["nearest"] = "nearest"):
    if interpolation != "nearest":
        raise NotImplementedError("Only nearest-neighbor sampling is implemented")
    device = pixel_xy.device if torch.is_tensor(pixel_xy) else _device()
    image, pix = _image_cuda(rgb_cuda_u8, device), _tensor(pixel_xy, device)
    if image.ndim != 3 or image.shape[2] < 3 or pix.ndim != 2 or pix.shape[1] != 2:
        raise ValueError("rgb_image must be HxWx3/4 and pixel_xy must be Nx2")
    h, w = image.shape[:2]
    finite = torch.isfinite(pix).all(1)
    safe = torch.where(finite[:, None], pix, torch.zeros_like(pix))
    u, v = safe.round().long().T
    valid = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return rgb8_cuda(image[v[valid], u[valid], :3], input_color_order), valid


@torch.no_grad()
def colorize_points_from_rgb(points_cuda:torch.Tensor, rgb_cuda_u8:torch.Tensor,
                             calibration: StereoRgbCalibrationCuda, *, 
                             rectification:StereoRectificationCuda=None,
                             points_frame: PointsFrame = "left_rectified", output_frame: OutputFrame = "left",
                             input_color_order: ColorOrder = "RGB", rgb_image_is_undistorted=False):
    if rgb_cuda_u8.ndim != 3 or rgb_cuda_u8.shape[2] < 3 or points_cuda.ndim != 2 or points_cuda.shape[1] != 3:
        raise ValueError("rgb_image must be HxWx3/4 and points_m must be Nx3")
    if points_frame == "left_rectified":
        if rectification is None: raise ValueError("rectification is required for rectified-left points")
        points_left = points_cuda @ rectification.R1
    elif points_frame == "left":
        points_left = points_cuda
    else: raise ValueError(f"Unsupported points_frame: {points_frame}")
    points_rgb = transform_points(points_left, calibration.left_to_rgb)
    front = torch.isfinite(points_rgb).all(1) & (points_rgb[:, 2] > 0)
    idx = torch.where(front)[0]
    if not idx.numel():
        return points_cuda.new_empty((0,3)), torch.empty((0,3), device=points_cuda.device, dtype=torch.uint8)
    h, w = rgb_cuda_u8.shape[:2]
    K = scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (w, h))
    distortion = None if rgb_image_is_undistorted else calibration.rgb_distortion
    pix = _project_camera_points_opencv_model(points_rgb[idx], K, distortion)
    finite = torch.isfinite(pix).all(1)
    safe = torch.where(finite[:,None], pix, torch.zeros_like(pix))
    u, v = safe.round().long().T
    inside = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    selected = idx[inside]; u, v = u[inside], v[inside]
    colors = rgb8_cuda(rgb_cuda_u8[v, u, :3], input_color_order)
    out = points_left[selected] if output_frame == "left" else points_cuda[selected]
    return out, colors


@torch.no_grad()
def points_left_to_rgb_depth(points_left_m, rgb_image, calibration: StereoRgbCalibrationCuda, *,
                             rgb_image_is_undistorted=False, splat_px=0):
    h, w = _image_hw(rgb_image)
    if isinstance(rgb_image,np.ndarray):            
        rgb_image = _image_cuda(rgb_image)

    pixels, p = project_points_to_rgb_pixels(points_left_m, rgb_image, calibration,
                                             rgb_image_is_undistorted=rgb_image_is_undistorted)
    finite = torch.isfinite(pixels).all(1)
    safe = torch.where(finite[:,None], pixels, torch.zeros_like(pixels))
    u0, v0 = safe.round().long().T
    z = p[:, 2]
    base = finite & torch.isfinite(z) & (z > 0)
    depth = torch.full((h*w,), torch.inf, device=calibration.device, dtype=torch.float32)
    r = max(0, int(splat_px))
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            u, v = u0 + dx, v0 + dy
            ok = base & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            depth.scatter_reduce_(0, v[ok]*w + u[ok], z[ok], reduce="amin", include_self=True)
    depth = depth.view(h, w)
    valid = torch.isfinite(depth)
    return depth.masked_fill(~valid, torch.nan), valid


def _undistort_normalized_cuda(xd: torch.Tensor,yd: torch.Tensor,
        distortion: torch.Tensor,iterations: int = 5,) -> tuple[torch.Tensor, torch.Tensor]:
    src = distortion.flatten()
    if not src.numel(): return xd, yd
    if src.numel() not in (4, 5, 8, 12, 14):
        raise ValueError("distortion must contain 4, 5, 8, 12, or 14 coefficients")
    d = torch.zeros(14, device=xd.device, dtype=xd.dtype); d[:src.numel()] = src
    k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tx,ty = d
    if src.numel() == 14:
        inv_tilt = torch.linalg.inv(_tilt_projection_matrix_cuda(tx, ty))
        h = torch.stack((xd, yd, torch.ones_like(xd)), 1) @ inv_tilt.T
        x0, y0 = h[:,0]/h[:,2], h[:,1]/h[:,2]
    else:
        x0, y0 = xd, yd
    x, y = x0.clone(), y0.clone()
    for _ in range(iterations):
        r2 = x*x + y*y; r4 = r2*r2; r6 = r4*r2
        icdist = (1 + k4*r2 + k5*r4 + k6*r6) / (1 + k1*r2 + k2*r4 + k3*r6)
        xy = x*y
        dx = 2*p1*xy + p2*(r2 + 2*x*x) + s1*r2 + s2*r4
        dy = p1*(r2 + 2*y*y) + 2*p2*xy + s3*r2 + s4*r4
        x, y = (x0 - dx)*icdist, (y0 - dy)*icdist
    return x, y


@torch.no_grad()
def rgb_depth_to_points_rgb(depth_rgb_m, calibration: StereoRgbCalibrationCuda, *, rgb_image_is_undistorted=False):
    d = _tensor(depth_rgb_m, calibration.device)
    if d.ndim != 2: raise ValueError(f"depth_rgb_m must be HxW, got {tuple(d.shape)}")
    h, w = d.shape
    y, x = torch.where(torch.isfinite(d) & (d > 0)); z = d[y, x]
    K = scale_K(calibration.rgb_intrinsics, calibration.rgb_resolution, (w, h))
    yn = (y.float() - K[1,2]) / K[1,1]
    xn = (x.float() - K[0,2] - K[0,1]*yn) / K[0,0]
    if not rgb_image_is_undistorted:
        xn, yn = _undistort_normalized_cuda(xn, yn, calibration.rgb_distortion)
    return torch.stack((xn*z, yn*z, z), 1), torch.stack((x, y), 1).to(torch.int32)


def _rgb_float(colors_rgb):
    c = rgb8(colors_rgb)
    packed = (c[:,0].astype(np.uint32)<<16) | (c[:,1].astype(np.uint32)<<8) | c[:,2].astype(np.uint32)
    return packed.astype("<u4").view("<f4")


def _clean_cloud(points_m, colors_rgb):
    p, c = _numpy(points_m).astype(np.float32, copy=False), rgb8(colors_rgb)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(c):
        raise ValueError("points must be Nx3 and match colors length")
    finite = np.isfinite(p).all(1)
    return p[finite], c[finite]


def save_pcd(path: str | Path, points_m, colors_rgb, *, binary=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    p, c = _clean_cloud(points_m, colors_rgb); rgb, n = _rgb_float(c), len(p)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA {'binary' if binary else 'ascii'}\n")
    if binary:
        data = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4")])
        data["x"],data["y"],data["z"],data["rgb"] = p[:,0],p[:,1],p[:,2],rgb
        with path.open("wb") as f: f.write(header.encode("ascii")); data.tofile(f)
    else:
        with path.open("w", encoding="ascii") as f:
            f.write(header); f.writelines(f"{x:.8f} {y:.8f} {z:.8f} {float(r):.9e}\n" for (x,y,z),r in zip(p,rgb))
    return path


def save_ply_ascii(path: str | Path, points_m, colors_rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    p, c = _clean_cloud(points_m, colors_rgb)
    header = ("ply\nformat ascii 1.0\n" f"element vertex {len(p)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("w", encoding="ascii") as f:
        f.write(header); f.writelines(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n" for (x,y,z),(r,g,b) in zip(p,c))
    return path


def save_point_cloud(path: str | Path, points_m, colors_rgb, *, binary_pcd=True,
                     prepend_center_of_gravity=False, center_color_rgb=None):
    points, colors = _numpy(points_m), _numpy(colors_rgb)
    if points.ndim != 2 or points.shape[1] != 3 or colors.ndim != 2 or colors.shape[1] != 3 or len(points) != len(colors):
        raise ValueError("points_m/colors_rgb must both be Nx3 with matching length")
    if prepend_center_of_gravity:
        if not len(points): raise ValueError("Cannot calculate the center of gravity of an empty point cloud")
        center = points.mean(0, keepdims=True)
        cc = colors.mean(0, keepdims=True).astype(colors.dtype) if center_color_rgb is None else np.asarray(center_color_rgb, dtype=colors.dtype).reshape(1,3)
        points, colors = np.concatenate((center,points)), np.concatenate((cc,colors))
    suffix = Path(path).suffix.lower()
    if suffix == ".pcd": return save_pcd(path, points, colors, binary=binary_pcd)
    if suffix == ".ply": return save_ply_ascii(path, points, colors)
    raise ValueError("Use .pcd or .ply")


def npz_to_pcd(input_npz: str | Path, output_path: str | Path, *, binary_pcd=True):
    with np.load(input_npz, allow_pickle=False) as z:
        save_point_cloud(output_path, z["points_m"], z["colors_rgb"], binary_pcd=binary_pcd)
    return Path(output_path)


def read_image(path: str | Path, *, color=True):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if image is None: raise FileNotFoundError(f"Could not read image: {path}")
    if not color and len(image.shape)!=2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image


@torch.no_grad()
def stereo_to_point_cloud_cuda(left_img_u8:np.ndarray, right_img_u8:np.ndarray,
                               calibration: StereoRgbCalibrationCpu, *, device="cuda",
                               alpha=0.0, min_disparity=0, min_depth_m=.01, num_disparities=128,
                               block_size=5, max_depth_m=10.0, stride=1):
    """SGBM remains CPU; disparity -> points is CUDA."""
    device = _device(device)
    left_cuda_u8, right_cuda_u8 = _image_cuda(left_img_u8), _image_cuda(right_img_u8)

    rectifier = calibration.get_rectifier(alpha)
    left_cuda_u8, right_cuda_u8, rect_cuda = rectifier.rectify(left_cuda_u8, right_cuda_u8)
    height, width = left_cuda_u8.shape
    disparity_cuda = SGBMDisparityPredictorCuda(width, height,
                                        num_disparities=num_disparities, block_size=block_size
                                          ).predict(left_cuda_u8, right_cuda_u8)
    points_cuda, _ = rect_cuda.disparity_to_points_rectified(
        disparity_cuda, min_disparity=max(.5, float(min_disparity)), min_depth_m=min_depth_m,
        max_depth_m=max_depth_m, stride=stride)
    return disparity_cuda, rect_cuda, points_cuda


@torch.no_grad()
def stereo_rgb_to_colored_point_cloud_cuda(left_image, right_image, rgb_image, *, calibration=None, device="cuda",
                                           output_path=None, input_color_order: ColorOrder="RGB",
                                           rgb_image_is_undistorted=False, alpha=0.0, min_disparity=0,
                                           min_depth_m=.01, num_disparities=128, block_size=5,
                                           max_depth_m=10.0, stride=1, output_frame: OutputFrame="left",
                                           save_binary_pcd=True):
    calibration = calibration or StereoRgbCalibrationCpu.default()
    cal = calibration.to_cuda(device)
    rgb_cuda_u8 = _image_cuda(rgb_image, cal.device)

    disparity_cuda, rect_cuda, points_cuda = stereo_to_point_cloud_cuda(
        left_image, right_image, calibration, device=device, alpha=alpha,
        min_disparity=min_disparity, min_depth_m=min_depth_m,
        num_disparities=num_disparities, block_size=block_size,
        max_depth_m=max_depth_m, stride=stride)
    
    points_cuda, colors = colorize_points_from_rgb(
        points_cuda, rgb_cuda_u8, cal, rectification=rect_cuda, output_frame=output_frame,
        input_color_order=input_color_order, rgb_image_is_undistorted=rgb_image_is_undistorted)
    if output_path: save_point_cloud(output_path, points_cuda, colors, binary_pcd=save_binary_pcd)
    return ColoredPointCloudCuda(points_cuda, colors, disparity_cuda, rect_cuda)


RgbOutputFrame = Literal["left", "rgb"]
@torch.no_grad()
def stereo_rgb_to_colored_point_cloud_rgb_res_cuda(left_image, right_image, rgb_image, *, calibration=None,
                                                   device="cuda", output_path=None,
                                                   input_color_order: ColorOrder="RGB",
                                                   rgb_image_is_undistorted=False, alpha=0.0,
                                                   min_disparity=0, min_depth_m=.01,
                                                   num_disparities=128, block_size=5,
                                                   max_depth_m=10.0, splat_px=1,
                                                   output_frame: RgbOutputFrame="left", save_binary_pcd=True):
    calibration = calibration or StereoRgbCalibrationCpu.default(); cal = calibration.to_cuda(device)
    disparity, rect, points_rect = stereo_to_point_cloud_cuda(
        left_image, right_image, calibration, device=device, alpha=alpha,
        min_disparity=min_disparity, min_depth_m=min_depth_m,
        num_disparities=num_disparities, block_size=block_size,
        max_depth_m=max_depth_m, stride=1)
    points_left = rectified_left_to_original_left(points_rect, rect)
    depth, _ = points_left_to_rgb_depth(points_left, rgb_image, cal,
                                        rgb_image_is_undistorted=rgb_image_is_undistorted,
                                        splat_px=splat_px)
    points_rgb, xy = rgb_depth_to_points_rgb(depth, cal,
                                             rgb_image_is_undistorted=rgb_image_is_undistorted)
    image = _image_cuda(rgb_image, cal.device); x, y = xy[:,0].long(), xy[:,1].long()
    colors = rgb8_cuda(image[y, x, :3], input_color_order)
    if output_frame == "rgb": points = points_rgb
    elif output_frame == "left": points = transform_points(points_rgb, torch.linalg.inv(cal.left_to_rgb))
    else: raise ValueError("output_frame must be 'left' or 'rgb'")
    if output_path: save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)
    return ColoredPointCloudCuda(points, colors, disparity, rect)


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
    pass