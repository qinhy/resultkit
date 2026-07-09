"""Compact DNN stereo helpers extending pcd_utils with Fast-FoundationStereo disparity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import contextlib, os, sys, warnings

import numpy as np

import pcd_utils as _pcd_utils
from pcd_utils import *  # noqa: F401,F403

ColorOrder = Literal["RGB", "BGR"]
DeviceLike = str | Any


def _require_cv2():
    try:
        import cv2 as _cv2  # type: ignore
        return _cv2
    except ImportError as e:
        raise ImportError("OpenCV is required: pip install opencv-python") from e


def _require_torch():
    try:
        import torch as _torch  # type: ignore
        return _torch
    except ImportError as e:
        raise ImportError("PyTorch is required for Fast-FoundationStereo inference; install a build matching your system.") from e


def _resolve_repo_dir(repo_dir: str | Path | None, model_path: str | Path | None) -> Path:
    candidates: list[Path] = []
    if repo_dir is not None:
        candidates.append(Path(repo_dir).expanduser())
    if os.environ.get("FAST_FOUNDATIONSTEREO_REPO"):
        candidates.append(Path(os.environ["FAST_FOUNDATIONSTEREO_REPO"]).expanduser())
    if model_path is not None:
        p = Path(model_path).expanduser().resolve()
        candidates += [q for q in [p.parent, *p.parents] if q.name == "Fast-FoundationStereo"]
        if len(p.parents) >= 3:
            candidates.append(p.parents[2])
    candidates.append(Path.cwd())
    for c in candidates:
        c = c.resolve()
        if (c / "scripts" / "run_demo.py").exists() and (c / "core").exists():
            return c
    raise FileNotFoundError(
        "Could not locate Fast-FoundationStereo. Pass repo_dir=... or set FAST_FOUNDATIONSTEREO_REPO; "
        "expected scripts/run_demo.py and core/ under repo_dir."
    )


def _resolve_model_path(repo_dir: Path, model_path: str | Path | None, model_dir: str | Path | None) -> Path:
    if model_path is not None:
        p = Path(model_path).expanduser()
        p = p if p.is_absolute() else (repo_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo model_path does not exist: {p}")
        return p
    dirs: list[Path] = []
    if model_dir is not None:
        d = Path(model_dir).expanduser()
        dirs.append(d if d.is_absolute() else (repo_dir / d).resolve())
    dirs += [repo_dir / "weights" / "23-36-37", repo_dir / "weights"]
    for d in dirs:
        if d.is_file():
            return d.resolve()
        if d.exists():
            for pat in ("model_best_bp2_serialize.pth", "model_best*.pth", "*.pth", "*.pt"):
                m = sorted(d.glob(pat))
                if m:
                    return m[0].resolve()
    raise FileNotFoundError("Could not find a Fast-FoundationStereo checkpoint. Pass model_path=... or model_dir=...")


def _ensure_uint8_rgb_image(image: np.ndarray, *, input_color_order: ColorOrder = "RGB") -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
        if input_color_order == "BGR":
            a = a[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image shape: {a.shape}")
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if np.issubdtype(a.dtype, np.floating):
        finite = a[np.isfinite(a)]
        if finite.size and float(finite.max()) <= 1.0:
            a = a * 255
    elif a.dtype == np.uint16 and int(a.max()) > 0:
        a = a.astype(np.float32) * (255.0 / int(a.max()))
    return np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))


def _resize_pair_for_model(left_rgb: np.ndarray, right_rgb: np.ndarray, *, model_scale: float):
    if model_scale <= 0:
        raise ValueError(f"model_scale must be positive, got {model_scale}")
    if model_scale == 1.0:
        return left_rgb, right_rgb
    c = _require_cv2(); interp = c.INTER_AREA if model_scale < 1.0 else c.INTER_LINEAR
    left = c.resize(left_rgb, None, fx=float(model_scale), fy=float(model_scale), interpolation=interp)
    right = c.resize(right_rgb, (left.shape[1], left.shape[0]), interpolation=interp)
    return left, right


def _restore_disparity_to_original_geometry(disparity_model_pixels: np.ndarray, *, original_hw: tuple[int, int], model_scale: float):
    c = _require_cv2(); d = np.asarray(disparity_model_pixels, np.float32); h, w = original_hw
    if d.shape[:2] != (h, w):
        d = c.resize(d, (w, h), interpolation=c.INTER_LINEAR).astype(np.float32)
    if model_scale != 1.0:
        d /= float(model_scale)
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


@dataclass
class FastFoundationStereoDisparity:
    """Small PyTorch wrapper returning Fast-FoundationStereo disparity in original pixel units."""
    repo_dir: str | Path | None = None
    model_path: str | Path | None = None
    model_dir: str | Path | None = None
    device: DeviceLike = "cuda"
    valid_iters: int = 8
    max_disp: int = 192
    hiera: bool = False
    autocast: bool = True
    amp_dtype: Any | None = None
    optimize_build_volume: str = "pytorch1"

    def __post_init__(self) -> None:
        self.repo_path = _resolve_repo_dir(self.repo_dir, self.model_path)
        self.checkpoint_path = _resolve_model_path(self.repo_path, self.model_path, self.model_dir)
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        torch = self.torch = _require_torch()
        try:
            from core.utils.utils import InputPadder  # type: ignore
        except Exception as e:
            raise ImportError(f"Could not import Fast-FoundationStereo InputPadder from {self.repo_path}") from e
        self.InputPadder = InputPadder
        if self.amp_dtype is None:
            try:
                from Utils import AMP_DTYPE  # type: ignore
                self.amp_dtype = AMP_DTYPE
            except Exception:
                self.amp_dtype = torch.float16
        if isinstance(self.device, str) and self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("device='cuda' requested but CUDA is unavailable; falling back to CPU.", RuntimeWarning, stacklevel=2)
            self.device = "cpu"
        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch
        model = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        if hasattr(model, "args"):
            for k, v in {"valid_iters": self.valid_iters, "max_disp": self.max_disp}.items():
                try:
                    setattr(model.args, k, int(v))
                except Exception:
                    pass
        model.to(self.device); model.eval()
        return model

    @property
    def device_type(self) -> str:
        return ("cuda" if self.device.startswith("cuda") else self.device) if isinstance(self.device, str) else getattr(self.device, "type", "cuda")

    def predict(self, left_rectified: np.ndarray, right_rectified: np.ndarray, *, input_color_order: ColorOrder = "RGB",
                model_scale: float = 1.0, remove_invisible: bool = True) -> np.ndarray:
        """Predict HxW float32 disparity; invalid pixels are NaN."""
        torch = self.torch
        if left_rectified is None or right_rectified is None:
            raise ValueError("left_rectified and right_rectified are required")
        if np.asarray(left_rectified).shape[:2] != np.asarray(right_rectified).shape[:2]:
            raise ValueError("left_rectified and right_rectified must have the same height/width")
        h0, w0 = np.asarray(left_rectified).shape[:2]
        left = _ensure_uint8_rgb_image(left_rectified, input_color_order=input_color_order)
        right = _ensure_uint8_rgb_image(right_rectified, input_color_order=input_color_order)
        left, right = _resize_pair_for_model(left, right, model_scale=float(model_scale))
        hm, wm = left.shape[:2]
        img0 = torch.as_tensor(left, device=self.device).float()[None].permute(0, 3, 1, 2).contiguous()
        img1 = torch.as_tensor(right, device=self.device).float()[None].permute(0, 3, 1, 2).contiguous()
        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        amp = torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype) if self.autocast and self.device_type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            if self.hiera:
                if not hasattr(self.model, "run_hierachical"):
                    raise AttributeError("This model does not expose run_hierachical(); set hiera=False.")
                disp = self.model.run_hierachical(img0, img1, iters=int(self.valid_iters), test_mode=True, small_ratio=0.5)
            else:
                kwargs = dict(iters=int(self.valid_iters), test_mode=True, optimize_build_volume=self.optimize_build_volume)
                try:
                    disp = self.model.forward(img0, img1, **kwargs)
                except TypeError:
                    kwargs.pop("optimize_build_volume")
                    disp = self.model.forward(img0, img1, **kwargs)
        d = np.clip(padder.unpad(disp.float()).detach().cpu().numpy().reshape(hm, wm).astype(np.float32), 0, None)
        if remove_invisible:
            d[np.arange(d.shape[1], dtype=np.float32)[None, :] - d < 0] = np.nan
        return _restore_disparity_to_original_geometry(d, original_hw=(h0, w0), model_scale=float(model_scale))


def compute_disparity_fast_foundationstereo(left_rectified: np.ndarray, right_rectified: np.ndarray, *,
                                            predictor: FastFoundationStereoDisparity | None = None,
                                            repo_dir: str | Path | None = None, model_path: str | Path | None = None,
                                            model_dir: str | Path | None = None, device: DeviceLike = "cuda",
                                            valid_iters: int = 8, max_disp: int = 192, hiera: bool = False,
                                            model_scale: float = 1.0, input_color_order: ColorOrder = "RGB",
                                            remove_invisible: bool = True) -> np.ndarray:
    """One-off Fast-FoundationStereo disparity; pass predictor=... for repeated/live use."""
    predictor = predictor or FastFoundationStereoDisparity(repo_dir, model_path, model_dir, device, valid_iters, max_disp, hiera)
    return predictor.predict(left_rectified, right_rectified, input_color_order=input_color_order,
                             model_scale=model_scale, remove_invisible=remove_invisible)


def stereo_rgb_to_colored_point_cloud_dnn(left_image: np.ndarray, right_image: np.ndarray, rgb_image: np.ndarray, *,
                                          calibration: StereoRgbCalibration | None = None,  # noqa: F405
                                          disparity_predictor: FastFoundationStereoDisparity | None = None,
                                          repo_dir: str | Path | None = None, model_path: str | Path | None = None,
                                          model_dir: str | Path | None = None, device: DeviceLike = "cuda",
                                          valid_iters: int = 8, max_disp: int = 192, hiera: bool = False,
                                          model_scale: float = 1.0, output_path: str | Path | None = None,
                                          input_color_order: ColorOrder = "RGB", stereo_input_color_order: ColorOrder = "RGB",
                                          rgb_image_is_undistorted: bool = False, alpha: float = 0.0,
                                          min_disparity: float = 0.5, max_depth_m: float | None = 10.0,
                                          stride: int = 1, output_frame: Literal["left", "left_rectified"] = "left",
                                          save_binary_pcd: bool = True, remove_invisible: bool = True) -> ColoredPointCloud:  # noqa: F405
    """Full DNN pipeline: raw stereo+RGB -> colored point cloud, optionally saved as PCD/PLY."""
    calibration = calibration or StereoRgbCalibration.default()  # noqa: F405
    left, right, rect = rectify_stereo_pair(left_image, right_image, calibration, alpha=alpha)  # noqa: F405
    disparity_predictor = disparity_predictor or FastFoundationStereoDisparity(
        repo_dir, model_path, model_dir, device, valid_iters, max_disp, hiera
    )
    disparity = disparity_predictor.predict(left, right, input_color_order=stereo_input_color_order,
                                            model_scale=model_scale, remove_invisible=remove_invisible)
    points_rect, _ = disparity_to_points_rectified(disparity, rect, min_disparity=float(min_disparity),  # noqa: F405
                                                   max_depth_m=max_depth_m, stride=stride)
    points, colors = colorize_points_from_rgb(points_rect, rgb_image, calibration, rectification=rect,  # noqa: F405
                                              points_frame="left_rectified", output_frame=output_frame,
                                              input_color_order=input_color_order,
                                              rgb_image_is_undistorted=rgb_image_is_undistorted)
    if output_path is not None:
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)  # noqa: F405
    return ColoredPointCloud(points, colors, disparity, rect)  # noqa: F405


def stereo_rgb_to_colored_point_cloud_rgb_res_dnn(left_image: np.ndarray, right_image: np.ndarray, rgb_image: np.ndarray, *,
                                                  calibration: StereoRgbCalibration | None = None,  # noqa: F405
                                                  disparity_predictor: FastFoundationStereoDisparity | None = None,
                                                  repo_dir: str | Path | None = None, model_path: str | Path | None = None,
                                                  model_dir: str | Path | None = None, device: DeviceLike = "cuda",
                                                  valid_iters: int = 8, max_disp: int = 192, hiera: bool = False,
                                                  model_scale: float = 1.0, output_path: str | Path | None = None,
                                                  input_color_order: ColorOrder = "RGB", stereo_input_color_order: ColorOrder = "RGB",
                                                  rgb_image_is_undistorted: bool = False, alpha: float = 0.0,
                                                  min_disparity: float = 0.5, max_depth_m: float | None = 10.0,
                                                  splat_px: int = 1, output_frame: Literal["left", "rgb"] = "left",
                                                  save_binary_pcd: bool = True, remove_invisible: bool = True) -> ColoredPointCloud:  # noqa: F405
    """DNN sibling of stereo_rgb_to_colored_point_cloud_rgb_res(): output at RGB-image pixel density."""
    calibration = calibration or StereoRgbCalibration.default()  # noqa: F405
    left, right, rect = rectify_stereo_pair(left_image, right_image, calibration, alpha=alpha)  # noqa: F405
    disparity_predictor = disparity_predictor or FastFoundationStereoDisparity(
        repo_dir, model_path, model_dir, device, valid_iters, max_disp, hiera
    )
    disparity = disparity_predictor.predict(left, right, input_color_order=stereo_input_color_order,
                                            model_scale=model_scale, remove_invisible=remove_invisible)
    points_rect, _ = disparity_to_points_rectified(disparity, rect, min_disparity=float(min_disparity),  # noqa: F405
                                                   max_depth_m=max_depth_m, stride=1)
    points_left = rectified_left_to_original_left(points_rect, rect)  # noqa: F405
    depth_rgb, _ = points_left_to_rgb_depth(points_left, rgb_image, calibration,  # noqa: F405
                                            rgb_image_is_undistorted=rgb_image_is_undistorted,
                                            splat_px=splat_px)
    points_rgb, xy = rgb_depth_to_points_rgb(depth_rgb, calibration,  # noqa: F405
                                             rgb_image_is_undistorted=rgb_image_is_undistorted)
    x, y = xy[:, 0], xy[:, 1]
    colors = rgb8(np.asarray(rgb_image)[y, x, :3], input_color_order)  # noqa: F405
    if output_frame == "rgb":
        points = points_rgb
    elif output_frame == "left":
        points = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))  # noqa: F405
    else:
        raise ValueError("output_frame must be 'left' or 'rgb'")
    if output_path is not None:
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)  # noqa: F405
    return ColoredPointCloud(points, colors, disparity, rect)  # noqa: F405


def write_fast_foundationstereo_intrinsic_file(path: str | Path, rectification: StereoRectification) -> Path:  # noqa: F405
    """Write Fast-FoundationStereo demo K.txt: flattened rectified K, then baseline in meters."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    K = np.asarray(rectification.P1[:3, :3], np.float64); fx = float(rectification.P1[0, 0])
    if fx == 0:
        raise ValueError("Invalid rectification.P1: fx is zero")
    baseline_m = abs(float(rectification.P2[0, 3]) / fx)
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.12g}" for v in K.reshape(-1)) + "\n")
        f.write(f"{baseline_m:.12g}\n")
    return path


_DNN_PUBLIC_NAMES = [
    "FastFoundationStereoDisparity",
    "compute_disparity_fast_foundationstereo",
    "stereo_rgb_to_colored_point_cloud_dnn",
    "stereo_rgb_to_colored_point_cloud_rgb_res_dnn",
    "write_fast_foundationstereo_intrinsic_file",
]
__all__ = list(getattr(_pcd_utils, "__all__", [])) + _DNN_PUBLIC_NAMES


if __name__ == "__main__":
    # Example:
    # left = read_image("test/left.png", color=False)
    # right = read_image("test/right.png", color=False)
    # rgb = read_image("test/rgb.jpg", color=True)  # cv2 gives BGR
    # predictor = FastFoundationStereoDisparity(
    #     repo_dir="./fast-foundationstereo",
    #     model_path="weights/23-36-37/model_best_bp2_serialize.pth",
    #     valid_iters=8,
    #     max_disp=192,
    # )
    # cloud = stereo_rgb_to_colored_point_cloud_dnn(
    #     left, right, rgb, disparity_predictor=predictor, output_path="colored_cloud.pcd",
    #     input_color_order="BGR", model_scale=0.5, max_depth_m=2.0, stride=1,
    # )
    # cloud = stereo_rgb_to_colored_point_cloud_rgb_res_dnn(
    #     left, right, rgb, disparity_predictor=predictor, output_path="rgb_res_cloud.pcd",
    #     input_color_order="BGR", model_scale=0.5, splat_px=1, output_frame="left", max_depth_m=2.0,
    # )
    pass
