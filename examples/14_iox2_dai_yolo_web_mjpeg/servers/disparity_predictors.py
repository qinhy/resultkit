"""Stereo disparity predictors with a strict Torch prediction interface.

Every ``predict(left, right)`` accepts Torch tensors and returns an HxW float32
Torch tensor. OpenCV SGBM returns CPU Torch; GPU/model backends return Torch on
their native device. ``predict_numpy`` remains available as a CPU convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
import ctypes
import inspect
import math
from pathlib import Path
import sys
from typing import Any, ClassVar, Literal
import warnings

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

try:
    import vpi
except ImportError:
    vpi = None  # type: ignore[assignment]

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]


ColorOrder = Literal["RGB", "BGR"]
ImageLayout = Literal["HWC", "CHW"]
ValueRange = Literal["auto", "0_1", "0_255"]
OutputBackend = Literal["torch"]
DeviceLike = str | Any


def _unexpected(name: str, kwargs: dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(f"Unexpected {name} predict keyword(s): {', '.join(sorted(kwargs))}")


def _require(condition: bool, message: str, exc: type[Exception] = ValueError) -> None:
    if not condition:
        raise exc(message)


def _resize_4d(x: torch.Tensor, size: tuple[int, int], down: bool = False) -> torch.Tensor:
    if tuple(x.shape[-2:]) == size:
        return x
    if down:
        return interpolate(x, size=size, mode="area")
    return interpolate(x, size=size, mode="bilinear", align_corners=False)


class DisparityPredictor(ABC):
    """Base class shared by all stereo backends."""

    output_backend: ClassVar[OutputBackend]

    def __init__(
        self,
        *,
        # Common stereo controls
        min_disparity: int = 0,
        num_disparities: int = 256,
        block_size: int = 5,
        # VPI / SGM-style controls
        confidence_threshold: int = 32767,
        p1: int | None = None,
        p2: int | None = None,
        uniqueness: float | None = None,
        include_diagonals: bool = True,
        quality: int = 6,
        width: int = -1,
        height: int = -1,
        # OpenCV SGBM controls
        uniqueness_ratio: int = 8,
        speckle_window_size: int = 80,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 31,
        invalid_to_nan: bool = True,
        # libSGM controls
        paths: int = 4,
        lr_max_diff: int = 1,
        device: DeviceLike = "cuda",
        dll: str | Path = "build/Release/sgm_py.dll",
        # FoundationStereo controls
        repo_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        valid_iters: int = 8,
        max_disp: int = 192,
        hiera: bool = False,
        autocast: bool = True,
        amp_dtype: Any | None = None,
        optimize_build_volume: str = "pytorch1",
        allow_tf32: bool = True,
        cudnn_benchmark: bool = True,
        compile_model: bool = False,
    ) -> None:
        ints = {
            "min_disparity": min_disparity, "num_disparities": num_disparities,
            "block_size": block_size, "confidence_threshold": confidence_threshold,
            "quality": quality, "width": width, "height": height,
            "uniqueness_ratio": uniqueness_ratio,
            "speckle_window_size": speckle_window_size, "speckle_range": speckle_range,
            "disp12_max_diff": disp12_max_diff, "pre_filter_cap": pre_filter_cap,
            "paths": paths, "lr_max_diff": lr_max_diff, "valid_iters": valid_iters,
            "max_disp": max_disp,
        }
        bools = {
            "include_diagonals": include_diagonals, "invalid_to_nan": invalid_to_nan,
            "hiera": hiera, "autocast": autocast, "allow_tf32": allow_tf32,
            "cudnn_benchmark": cudnn_benchmark, "compile_model": compile_model,
        }
        for name, value in ints.items():
            setattr(self, name, int(value))
        for name, value in bools.items():
            setattr(self, name, bool(value))

        self.p1, self.p2, self.uniqueness = p1, p2, uniqueness
        self.device, self.dll = device, dll
        self.repo_dir, self.model_path, self.model_dir = repo_dir, model_path, model_dir
        self.amp_dtype = amp_dtype
        self.optimize_build_volume = optimize_build_volume
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        pass

    @abstractmethod
    def predict(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.predict(left, right, **kwargs)

    def predict_numpy(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> np.ndarray:
        return _to_numpy_disparity(self.predict(left, right, **kwargs))

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _require_tensor(image: Any, name: str = "image") -> torch.Tensor:
    _require(isinstance(image, torch.Tensor), f"{name} must be a torch.Tensor, got {type(image)!r}", TypeError)
    return image


def _require_tensor_pair(left: Any, right: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _require_tensor(left, "left"), _require_tensor(right, "right")


def _to_numpy_disparity(disparity: torch.Tensor) -> np.ndarray:
    disparity = _require_tensor(disparity, "disparity")
    _require(disparity.ndim == 2, f"Expected HxW disparity output, got shape {tuple(disparity.shape)}")
    return disparity.detach().cpu().numpy().astype(np.float32, copy=False)


def _as_numpy_u8_gray(image: torch.Tensor) -> np.ndarray:
    """Convert a Torch HxW/HWC/CHW image to contiguous CPU uint8 grayscale."""
    x = _require_tensor(image).detach()
    if x.ndim == 3 and x.shape[0] <= 4 < x.shape[-1]:
        x = x.permute(1, 2, 0)
    array = x.cpu().numpy()

    if array.ndim == 3 and array.shape[2] >= 3:
        array = cv2.cvtColor(array[..., :3], cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    elif array.ndim != 2:
        raise ValueError(f"Expected grayscale/HWC/CHW image, got shape {tuple(image.shape)}")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1:
        array = array * 255
    elif array.dtype == np.uint16 and array.size:
        maximum = array.max()
        if maximum > 0:
            array = array.astype(np.float32) * (255.0 / float(maximum))

    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _as_vpi_u8_gray(image: torch.Tensor):
    _require(vpi is not None, "NVIDIA VPI Python bindings are required for VPIStereoDisparityGPU", ImportError)
    return vpi.asimage(_as_numpy_u8_gray(image), format=vpi.Format.U8)


class VPIStereoDisparityGPU(DisparityPredictor):
    """NVIDIA VPI CUDA stereo matcher returning Torch float32 disparity."""

    output_backend: ClassVar[OutputBackend] = "torch"

    def _initialize_backend(self) -> None:
        _require(vpi is not None, "NVIDIA VPI Python bindings are required for VPIStereoDisparityGPU", ImportError)
        _require(cp is not None, "CuPy is required for VPIStereoDisparityGPU", ImportError)
        _require(self.min_disparity >= 0, "VPI CUDA requires min_disparity >= 0")
        _require(self.num_disparities > 0, "num_disparities must be positive")
        _require(self.block_size > 0, "block_size must be positive")

        self.max_disparity = self.min_disparity + self.num_disparities
        _require(
            self.max_disparity in (64, 128, 256),
            "For VPI CUDA, min_disparity + num_disparities must be 64, 128, or 256",
        )
        self.p1 = 3 if self.p1 is None else int(self.p1)
        self.p2 = 48 if self.p2 is None else int(self.p2)
        self.uniqueness = -1.0 if self.uniqueness is None else float(self.uniqueness)
        _require(self.p1 > 0, "VPI p1 must be positive")
        _require(self.p1 <= self.p2 < 256, "VPI p2 must satisfy p1 <= p2 < 256")
        _require(self.uniqueness == -1.0 or 0 <= self.uniqueness <= 1, "VPI uniqueness must be -1 or in [0, 1]")

        self.window_size = self.block_size
        self._shape = None
        self._left_y16 = self._right_y16 = self._disparity = None
        self._input_refs = ()

    def _reset_for_shape(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self._left_y16 = self._right_y16 = self._disparity = None

    def predict(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        _unexpected("VPI", kwargs)
        left, right = _require_tensor_pair(left, right)
        left_vpi, right_vpi = _as_vpi_u8_gray(left), _as_vpi_u8_gray(right)
        _require(left_vpi.size == right_vpi.size, "Rectified left/right images must have matching size")

        size = tuple(map(int, left_vpi.size))
        if self._shape != size:
            self._reset_for_shape(size)
        self._input_refs = (left_vpi, right_vpi, left, right)

        with vpi.Backend.CUDA:
            if self._left_y16 is None:
                self._left_y16 = left_vpi.convert(vpi.Format.Y16_ER, scale=1)
                self._right_y16 = right_vpi.convert(vpi.Format.Y16_ER, scale=1)
            else:
                left_vpi.convert(self._left_y16, scale=1)
                right_vpi.convert(self._right_y16, scale=1)

            args = dict(
                window=self.window_size, maxdisp=self.max_disparity,
                confthreshold=self.confidence_threshold,
                conftype=vpi.ConfidenceType.ABSOLUTE, quality=self.quality,
                mindisp=self.min_disparity, p1=self.p1, p2=self.p2,
                uniqueness=self.uniqueness, includediagonals=self.include_diagonals,
            )
            if self._disparity is None:
                self._disparity = vpi.stereodisp(self._left_y16, self._right_y16, **args)
            else:
                vpi.stereodisp(self._left_y16, self._right_y16, out=self._disparity, **args)

        assert self._disparity is not None
        with self._disparity.rlock_cuda() as buffer:
            result = cp.asarray(buffer).astype(cp.float32)
            result *= 1.0 / 32.0
            cp.cuda.get_current_stream().synchronize()
        return torch.from_dlpack(result)


class SGBMDisparityPredictor(DisparityPredictor):
    """OpenCV StereoSGBM predictor returning CPU Torch float32 disparity."""

    output_backend: ClassVar[OutputBackend] = "torch"

    def _initialize_backend(self) -> None:
        _require(self.num_disparities > 0, "num_disparities must be positive")
        _require(self.block_size > 0, "block_size must be positive")
        _require(0 <= self.uniqueness_ratio < 100, "uniqueness_ratio must be in [0, 100)")
        _require(self.speckle_window_size >= 0, "speckle_window_size must be non-negative")
        _require(self.speckle_range >= 0, "speckle_range must be non-negative")

        ndisp = math.ceil(max(16, self.num_disparities) / 16) * 16
        block = max(3, self.block_size) | 1
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity, numDisparities=ndisp, blockSize=block,
            P1=8 * block**2, P2=32 * block**2,
            disp12MaxDiff=self.disp12_max_diff, uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size, speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def predict(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        _unexpected("SGBM", kwargs)
        left, right = _require_tensor_pair(left, right)
        left_np, right_np = _as_numpy_u8_gray(left), _as_numpy_u8_gray(right)
        _require(left_np.shape == right_np.shape, "Rectified left/right images must have the same shape")
        disparity = self.matcher.compute(left_np, right_np).astype(np.float32) / 16.0
        if self.invalid_to_nan:
            disparity[disparity < self.min_disparity] = np.nan
        return torch.from_numpy(disparity)


class SGBMDisparityPredictorCuda(DisparityPredictor):
    """libSGM CUDA stereo matcher returning a CUDA Torch float32 tensor."""

    output_backend: ClassVar[OutputBackend] = "torch"

    def _initialize_backend(self) -> None:
        _require(self.num_disparities in (64, 128, 256), "num_disparities must be 64, 128, or 256")
        _require(self.width > 0 and self.height > 0, "width and height must be positive for libSGM")
        self.p1 = 10 if self.p1 is None else int(self.p1)
        self.p2 = 120 if self.p2 is None else int(self.p2)
        self.uniqueness = 0.95 if self.uniqueness is None else float(self.uniqueness)

        self.device = torch.device(self.device)
        _require(self.device.type == "cuda", "libSGM predictor requires a CUDA device")
        _require(torch.cuda.is_available(), "CUDA is not available", RuntimeError)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.shape = (self.height, self.width)
        dll = Path(self.dll).expanduser().resolve()
        self.lib = ctypes.CDLL(str(dll))
        self.lib.sgm_create.restype = ctypes.c_void_p
        self.lib.sgm_execute.argtypes = [ctypes.c_void_p] * 4
        self.lib.sgm_invalid.argtypes = [ctypes.c_void_p]
        self.lib.sgm_invalid.restype = ctypes.c_int
        self.lib.sgm_destroy.argtypes = [ctypes.c_void_p]

        with torch.cuda.device(self.device):
            self.handle = self.lib.sgm_create(
                self.width, self.height, self.num_disparities, self.p1, self.p2,
                ctypes.c_float(self.uniqueness), self.paths,
                self.min_disparity, self.lr_max_diff,
            )
        _require(bool(self.handle), f"sgm_create failed for library {dll}", RuntimeError)
        self.invalid = int(self.lib.sgm_invalid(self.handle))

    @torch.inference_mode()
    def predict(self, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        _unexpected("libSGM", kwargs)
        left, right = _require_tensor_pair(left, right)
        _require(tuple(left.shape) == self.shape and tuple(right.shape) == self.shape, f"Expected images with shape {self.shape}")
        _require(left.dtype == right.dtype == torch.uint8, "Expected uint8 images")
        _require(left.device == right.device == self.device, f"Images must be on {self.device}; got {left.device} and {right.device}")

        left, right = left.contiguous(), right.contiguous()
        output = torch.empty(self.shape, dtype=torch.int16, device=self.device)
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).synchronize()
            self.lib.sgm_execute(
                self.handle,
                ctypes.c_void_p(left.data_ptr()), ctypes.c_void_p(right.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
            )
            torch.cuda.synchronize(self.device)

        disparity = output.float().mul_(1.0 / 16.0)
        disparity[output == self.invalid] = torch.nan
        return disparity

    def close(self) -> None:
        if handle := getattr(self, "handle", None):
            self.lib.sgm_destroy(handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _infer_layout(shape: tuple[int, ...], requested: ImageLayout | None) -> ImageLayout:
    if requested is not None:
        return requested
    _require(len(shape) == 3, f"Cannot infer layout for shape {shape}")
    if shape[0] <= 4 < shape[2]:
        return "CHW"
    if shape[2] <= 4:
        return "HWC"
    raise ValueError(f"Ambiguous three-dimensional image layout for shape {shape}")


def _to_torch_chw(
    image: torch.Tensor,
    *,
    device: Any,
    color: ColorOrder,
    layout: ImageLayout | None,
    value_range: ValueRange,
) -> torch.Tensor:
    x = _require_tensor(image)

    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        if _infer_layout(tuple(map(int, x.shape)), layout) == "HWC":
            x = x.permute(2, 0, 1)
    else:
        raise ValueError(f"Image must be HxW, HxWxC, or CxHxW; got {tuple(x.shape)}")

    channels = int(x.shape[0])
    if channels == 1:
        x = x.expand(3, -1, -1)
    elif channels >= 3:
        x = x[:3]
    else:
        raise ValueError(f"Image needs one or at least three channels, got {channels}")
    if color == "BGR":
        x = x[[2, 1, 0]]

    _require(value_range in ("auto", "0_1", "0_255"), f"Unsupported value_range: {value_range}")
    x = x.to(device=device, non_blocking=True)
    if x.is_floating_point():
        x = torch.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
        if value_range == "0_1" or (value_range == "auto" and float(x.amax().item()) <= 1.0):
            x = x * 255.0
        x = x.clamp_(0.0, 255.0)
    elif x.dtype == torch.uint16:
        maximum = int(x.to(torch.int32).max().item())
        x = x.float()
        if maximum:
            x *= 255.0 / maximum
    elif x.dtype != torch.uint8:
        x = x.clamp(0, 255)
    return x if x.is_contiguous() else x.contiguous()


def _extract_model_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        preferred = ("disp", "disparity", "flow_up", "prediction", "pred")
        values = [output[k] for k in preferred if k in output] + list(reversed(output.values()))
    elif isinstance(output, (tuple, list)):
        values = reversed(output)
    else:
        raise TypeError(f"Could not find a disparity tensor in model output {type(output)!r}")

    for value in values:
        try:
            return _extract_model_tensor(value)
        except TypeError:
            pass
    raise TypeError(f"Could not find a disparity tensor in model output {type(output)!r}")


def _supports_kwarg(func: Any, name: str) -> bool | None:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class FastFoundationStereoDisparity(DisparityPredictor):
    """Fast-FoundationStereo predictor returning Torch disparity on the model device."""

    output_backend: ClassVar[OutputBackend] = "torch"

    def _initialize_backend(self) -> None:
        _require(self.repo_dir is not None, "repo_dir is required")
        _require(self.model_path is not None or self.model_dir is not None, "model_path (or model_dir) is required")

        self.repo_path = Path(self.repo_dir).expanduser().resolve()
        _require(self.repo_path.is_dir(), f"Fast-FoundationStereo repo_dir does not exist: {self.repo_path}", FileNotFoundError)
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        checkpoint = Path(self.model_path if self.model_path is not None else self.model_dir).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = self.repo_path / checkpoint
        self.checkpoint_path = checkpoint.resolve()
        _require(self.checkpoint_path.is_file(), f"Fast-FoundationStereo model checkpoint does not exist: {self.checkpoint_path}", FileNotFoundError)

        try:
            from core.utils.utils import InputPadder  # type: ignore
        except Exception as exc:
            raise ImportError(f"Could not import Fast-FoundationStereo InputPadder from {self.repo_path}") from exc
        self.InputPadder = InputPadder

        if self.amp_dtype is None:
            try:
                from Utils import AMP_DTYPE  # type: ignore
                self.amp_dtype = AMP_DTYPE
            except Exception:
                self.amp_dtype = torch.float16

        self.device = torch.device(self.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(f"device={str(self.device)!r} requested but CUDA is unavailable; falling back to CPU.", RuntimeWarning, stacklevel=2)
            self.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = self.allow_tf32
        self.model = self._load_model()

    def _load_model(self):
        model = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        if hasattr(model, "args"):
            for name, value in (("valid_iters", self.valid_iters), ("max_disp", self.max_disp)):
                try:
                    setattr(model.args, name, int(value))
                except Exception:
                    pass

        model.to(self.device).eval()
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            pass

        if self.compile_model and self.hiera:
            warnings.warn("compile_model=True is ignored when hiera=True.", RuntimeWarning, stacklevel=2)
        elif self.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                warnings.warn(f"torch.compile failed; continuing eagerly: {exc}", RuntimeWarning, stacklevel=2)
        return model

    @property
    def device_type(self) -> str:
        return self.device.type

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor):
        if self.hiera:
            _require(hasattr(self.model, "run_hierachical"), "This model does not expose run_hierachical(); set hiera=False.", AttributeError)
            return self.model.run_hierachical(img0, img1, iters=self.valid_iters, test_mode=True, small_ratio=0.5)

        kwargs = {"iters": self.valid_iters, "test_mode": True}
        supports = _supports_kwarg(self.model.forward, "optimize_build_volume")
        if supports is not False:
            kwargs["optimize_build_volume"] = self.optimize_build_volume
        try:
            return self.model.forward(img0, img1, **kwargs)
        except TypeError as exc:
            message = str(exc).lower()
            if "optimize_build_volume" not in kwargs or "optimize_build_volume" not in message or "keyword" not in message:
                raise
            kwargs.pop("optimize_build_volume")
            return self.model.forward(img0, img1, **kwargs)

    def predict_cuda(
        self,
        left_rectified: torch.Tensor,
        right_rectified: torch.Tensor,
        *,
        input_color_order: ColorOrder = "RGB",
        input_layout: ImageLayout | None = None,
        input_value_range: ValueRange = "auto",
        model_scale: float = 1.0,
        remove_invisible: bool = True,
    ) -> torch.Tensor:
        left_rectified, right_rectified = _require_tensor_pair(left_rectified, right_rectified)
        _require(input_color_order in ("RGB", "BGR"), f"input_color_order must be 'RGB' or 'BGR', got {input_color_order!r}")
        _require(input_layout in (None, "HWC", "CHW"), f"input_layout must be None, 'HWC', or 'CHW', got {input_layout!r}")
        _require(model_scale > 0, f"model_scale must be positive, got {model_scale}")

        convert = lambda image: _to_torch_chw(
            image, device=self.device, color=input_color_order,
            layout=input_layout, value_range=input_value_range,
        )
        left, right = convert(left_rectified), convert(right_rectified)
        _require(tuple(left.shape[1:]) == tuple(right.shape[1:]), "left_rectified and right_rectified must have matching height/width")

        original_h, original_w = map(int, left.shape[1:])
        img0, img1 = left[None].float(), right[None].float()
        if model_scale != 1.0:
            size = tuple(max(1, round(v * float(model_scale))) for v in (original_h, original_w))
            down = model_scale < 1.0
            img0, img1 = _resize_4d(img0, size, down), _resize_4d(img1, size, down)

        model_h, model_w = map(int, img0.shape[-2:])
        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        amp = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.autocast and self.device_type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            disparity = padder.unpad(_extract_model_tensor(self._forward(img0, img1)).float())

        while disparity.ndim > 2 and disparity.shape[0] == 1:
            disparity = disparity.squeeze(0)
        _require(disparity.ndim == 2, f"Expected one disparity map, got {tuple(disparity.shape)}", RuntimeError)

        disparity = _resize_4d(disparity[None, None], (model_h, model_w))[0, 0]
        disparity = _resize_4d(disparity[None, None], (original_h, original_w))[0, 0]
        if model_w != original_w:
            disparity *= float(original_w) / model_w

        invalid = ~torch.isfinite(disparity) | (disparity <= 0)
        if remove_invisible:
            x = torch.arange(original_w, device=disparity.device, dtype=disparity.dtype)
            invalid |= x[None] - disparity < 0
        return disparity.masked_fill(invalid, float("nan")).contiguous()

    def predict(
        self,
        left_rectified: torch.Tensor,
        right_rectified: torch.Tensor,
        *,
        input_color_order: ColorOrder = "RGB",
        input_layout: ImageLayout | None = None,
        input_value_range: ValueRange = "auto",
        model_scale: float = 1.0,
        remove_invisible: bool = True,
    ) -> torch.Tensor:
        return self.predict_cuda(
            left_rectified, right_rectified,
            input_color_order=input_color_order, input_layout=input_layout,
            input_value_range=input_value_range, model_scale=model_scale,
            remove_invisible=remove_invisible,
        )


__all__ = [
    "DisparityPredictor",
    "VPIStereoDisparityGPU",
    "SGBMDisparityPredictor",
    "SGBMDisparityPredictorCuda",
    "FastFoundationStereoDisparity",
]
