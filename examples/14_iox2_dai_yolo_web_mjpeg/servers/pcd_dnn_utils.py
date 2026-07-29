"""Fast-FoundationStereo helpers with a Torch-only CUDA fast path.

Compatibility path
------------------
The original NumPy/OpenCV helpers remain available and return CPU disparity.

Torch CUDA path
---------------
``FastFoundationStereoTorchPipeline`` keeps all per-frame heavy stages in
PyTorch:

* Torchvision/nvJPEG decodes JPEGs directly to CUDA,
* ``torch.nn.functional.grid_sample`` applies cached OpenCV rectification maps,
* Fast-FoundationStereo returns disparity as a CUDA ``torch.Tensor``,
* Torch performs reprojection, RGB-camera projection, lens distortion, and
  color sampling on CUDA,
* only compact final points/colors are copied to CPU for optional writing.

OpenCV is used only during setup to calculate calibrated stereo rectification
geometry and maps. NVIDIA VPI and CuPy are not required.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import json

import torch
from torch.nn.functional import interpolate
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from pcd_utils import ColoredPointCloud, StereoRectification, StereoRgbCalibration, read_image, save_point_cloud

ColorOrder = Literal["RGB", "BGR"]
ImageLayout = Literal["HWC", "CHW"]
DeviceLike = str | Any
ValueRange = Literal["auto", "0_1", "0_255"]


def _resolve_repo_dir(repo_dir: str | Path | None, model_path: str | Path | None) -> Path:
    candidates: list[Path] = []
    if repo_dir is not None:
        candidates.append(Path(repo_dir).expanduser())
    if os.environ.get("FAST_FOUNDATIONSTEREO_REPO"):
        candidates.append(Path(os.environ["FAST_FOUNDATIONSTEREO_REPO"]).expanduser())
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        candidates += [p for p in [path.parent, *path.parents] if p.name == "Fast-FoundationStereo"]
        if len(path.parents) >= 3:
            candidates.append(path.parents[2])
    candidates.append(Path.cwd())
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "scripts" / "run_demo.py").exists() and (candidate / "core").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Fast-FoundationStereo. Pass repo_dir=... or set "
        "FAST_FOUNDATIONSTEREO_REPO; expected scripts/run_demo.py and core/."
    )


def _resolve_model_path(
    repo_dir: Path,
    model_path: str | Path | None,
    model_dir: str | Path | None,
) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser()
        path = path if path.is_absolute() else (repo_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo model_path does not exist: {path}")
        return path

    directories: list[Path] = []
    if model_dir is not None:
        directory = Path(model_dir).expanduser()
        directories.append(
            directory if directory.is_absolute() else (repo_dir / directory).resolve()
        )
    directories += [repo_dir / "weights" / "23-36-37", repo_dir / "weights"]
    for directory in directories:
        if directory.is_file():
            return directory.resolve()
        if directory.exists():
            for pattern in (
                "model_best_bp2_serialize.pth",
                "model_best*.pth",
                "*.pth",
                "*.pt",
            ):
                matches = sorted(directory.glob(pattern))
                if matches:
                    return matches[0].resolve()
    raise FileNotFoundError(
        "Could not find a Fast-FoundationStereo checkpoint. "
        "Pass model_path=... or model_dir=..."
    )


def _infer_layout(shape: tuple[int, ...], requested: ImageLayout | None) -> ImageLayout:
    if requested is not None:
        return requested
    if len(shape) != 3:
        raise ValueError(f"Cannot infer layout for shape {shape}")
    if shape[0] <= 4 and shape[2] > 4:
        return "CHW"
    if shape[2] <= 4:
        return "HWC"
    raise ValueError(f"Ambiguous three-dimensional image layout for shape {shape}")


def _extract_model_tensor(output: Any):
    """Extract a disparity tensor from common model output containers."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("disp", "disparity", "flow_up", "prediction", "pred"):
            if key in output:
                try:
                    return _extract_model_tensor(output[key])
                except TypeError:
                    pass
        for value in reversed(tuple(output.values())):
            try:
                return _extract_model_tensor(value)
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        for value in reversed(output):
            try:
                return _extract_model_tensor(value)
            except TypeError:
                continue
    raise TypeError(f"Could not find a disparity tensor in model output {type(output)!r}")


def _image_to_torch_chw(
    image: Any,
    *,
    device: Any,
    input_color_order: ColorOrder,
    input_layout: ImageLayout | None,
    value_range: ValueRange = "auto",
):
    """Convert NumPy/Torch/DLPack image to a device CHW tensor.

    ``value_range='0_255'`` is recommended for the Torch rectifier output; it
    avoids a GPU scalar read used by automatic range detection.
    """
    if isinstance(image, torch.Tensor):
        tensor = image
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
    elif hasattr(image, "__dlpack__"):
        tensor = torch.from_dlpack(image)
    else:
        tensor = torch.as_tensor(image)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        layout = _infer_layout(tuple(map(int, tensor.shape)), input_layout)
        if layout == "HWC":
            tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Image must be HxW, HxWxC, or CxHxW; got {tuple(tensor.shape)}")

    channels = int(tensor.shape[0])
    if channels == 1:
        tensor = tensor.expand(3, -1, -1)
    elif channels >= 3:
        tensor = tensor[:3]
    else:
        raise ValueError(f"Image needs one or at least three channels, got {channels}")

    if input_color_order == "BGR":
        tensor = tensor[[2, 1, 0]]

    tensor = tensor.to(device=device, non_blocking=True)
    if value_range not in ("auto", "0_1", "0_255"):
        raise ValueError(f"Unsupported value_range: {value_range}")
    if tensor.is_floating_point():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)
        scale_from_unit = value_range == "0_1"
        if value_range == "auto":
            # Automatic detection is kept for compatibility. The optimized
            # pipeline passes value_range="0_255" and avoids this synchronization.
            scale_from_unit = float(tensor.amax().item()) <= 1.0
        if scale_from_unit:
            tensor = tensor * 255.0
        tensor = tensor.clamp_(0.0, 255.0)
    elif tensor.dtype == torch.uint16:
        maximum = int(tensor.to(torch.int32).max().item())
        tensor = tensor.float()
        if maximum > 0:
            tensor = tensor * (255.0 / maximum)
    elif tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

@dataclass
class FastFoundationStereoDisparity:
    """Fast-FoundationStereo wrapper with both CUDA and NumPy result APIs."""

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
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.repo_path = _resolve_repo_dir(self.repo_dir, self.model_path)
        self.checkpoint_path = _resolve_model_path(
            self.repo_path,
            self.model_path,
            self.model_dir,
        )
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        try:
            from core.utils.utils import InputPadder  # type: ignore
        except Exception as exc:
            raise ImportError(
                f"Could not import Fast-FoundationStereo InputPadder from {self.repo_path}"
            ) from exc
        self.InputPadder = InputPadder

        if self.amp_dtype is None:
            try:
                from Utils import AMP_DTYPE  # type: ignore

                self.amp_dtype = AMP_DTYPE
            except Exception:
                self.amp_dtype = torch.float16

        if isinstance(self.device, str) and self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "device='cuda' requested but CUDA is unavailable; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device = "cpu"

        self.device = torch.device(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(self.cudnn_benchmark)
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = bool(self.allow_tf32)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = bool(self.allow_tf32)
        self.model = self._load_model()

    def _load_model(self):
        model = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if hasattr(model, "args"):
            for key, value in {
                "valid_iters": self.valid_iters,
                "max_disp": self.max_disp,
            }.items():
                try:
                    setattr(model.args, key, int(value))
                except Exception:
                    pass
        model.to(self.device)
        model.eval()
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            pass
        if self.compile_model and self.hiera:
            warnings.warn(
                "compile_model=True is ignored when hiera=True because the hierarchical "
                "entry point is not model.forward().",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                warnings.warn(
                    f"torch.compile failed; continuing eagerly: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return model

    @property
    def device_type(self) -> str:
        return self.device.type

    def _forward(self, img0, img1):
        if self.hiera:
            if not hasattr(self.model, "run_hierachical"):
                raise AttributeError(
                    "This model does not expose run_hierachical(); set hiera=False."
                )
            return self.model.run_hierachical(
                img0,
                img1,
                iters=int(self.valid_iters),
                test_mode=True,
                small_ratio=0.5,
            )

        kwargs = {
            "iters": int(self.valid_iters),
            "test_mode": True,
            "optimize_build_volume": self.optimize_build_volume,
        }
        try:
            return self.model.forward(img0, img1, **kwargs)
        except TypeError:
            kwargs.pop("optimize_build_volume")
            return self.model.forward(img0, img1, **kwargs)

    def predict_cuda(
        self,
        left_rectified: Any,
        right_rectified: Any,
        *,
        input_color_order: ColorOrder = "RGB",
        input_layout: ImageLayout | None = None,
        input_value_range: ValueRange = "auto",
        model_scale: float = 1.0,
        remove_invisible: bool = True,
    ):
        """Return HxW float32 disparity on the model device.

        The disparity is restored to the original rectified image dimensions and
        original-pixel units entirely on the GPU. Invalid pixels are NaN.
        """
        if left_rectified is None or right_rectified is None:
            raise ValueError("left_rectified and right_rectified are required")
        if model_scale <= 0:
            raise ValueError(f"model_scale must be positive, got {model_scale}")

        left = _image_to_torch_chw(
            left_rectified,
            device=self.device,
            input_color_order=input_color_order,
            input_layout=input_layout,
            value_range=input_value_range,
        )
        right = _image_to_torch_chw(
            right_rectified,
            device=self.device,
            input_color_order=input_color_order,
            input_layout=input_layout,
            value_range=input_value_range,
        )
        if tuple(left.shape[1:]) != tuple(right.shape[1:]):
            raise ValueError(
                "left_rectified and right_rectified must have matching height/width"
            )

        original_h, original_w = map(int, left.shape[1:])
        img0 = left.unsqueeze(0).float()
        img1 = right.unsqueeze(0).float()

        if model_scale != 1.0:
            model_h = max(1, int(round(original_h * float(model_scale))))
            model_w = max(1, int(round(original_w * float(model_scale))))
            if model_scale < 1.0:
                img0 = interpolate(img0, size=(model_h, model_w), mode="area")
                img1 = interpolate(img1, size=(model_h, model_w), mode="area")
            else:
                img0 = interpolate(
                    img0,
                    size=(model_h, model_w),
                    mode="bilinear",
                    align_corners=False,
                )
                img1 = interpolate(
                    img1,
                    size=(model_h, model_w),
                    mode="bilinear",
                    align_corners=False,
                )

        model_h, model_w = map(int, img0.shape[-2:])
        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        amp = (
            torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype)
            if self.autocast and self.device_type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), amp:
            output = self._forward(img0, img1)

        disparity = _extract_model_tensor(output).float()
        disparity = padder.unpad(disparity)
        disparity = disparity.squeeze()
        if disparity.ndim != 2:
            raise RuntimeError(
                f"Expected a single disparity map after unpadding, got {tuple(disparity.shape)}"
            )
        if tuple(disparity.shape) != (model_h, model_w):
            disparity = interpolate(
                disparity[None, None],
                size=(model_h, model_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if (model_h, model_w) != (original_h, original_w):
            disparity = interpolate(
                disparity[None, None],
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        if model_scale != 1.0:
            disparity = disparity / float(model_scale)

        invalid = ~torch.isfinite(disparity) | (disparity <= 0)
        if remove_invisible:
            x = torch.arange(original_w, device=disparity.device, dtype=disparity.dtype)
            invalid |= x.unsqueeze(0) - disparity < 0
        return disparity.masked_fill(invalid, float("nan")).contiguous()

    def predict(
        self,
        left_rectified: Any,
        right_rectified: Any,
        *,
        input_color_order: ColorOrder = "RGB",
        input_layout: ImageLayout | None = None,
        input_value_range: ValueRange = "auto",
        model_scale: float = 1.0,
        remove_invisible: bool = True,
    ) -> np.ndarray:
        """Compatibility API returning HxW float32 NumPy disparity."""
        disparity = self.predict_cuda(
            left_rectified,
            right_rectified,
            input_color_order=input_color_order,
            input_layout=input_layout,
            input_value_range=input_value_range,
            model_scale=model_scale,
            remove_invisible=remove_invisible,
        )
        return disparity.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_disparity_fast_foundationstereo(
    left_rectified: Any,
    right_rectified: Any,
    *,
    predictor: FastFoundationStereoDisparity | None = None,
    repo_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    device: DeviceLike = "cuda",
    valid_iters: int = 8,
    max_disp: int = 192,
    hiera: bool = False,
    model_scale: float = 1.0,
    input_color_order: ColorOrder = "RGB",
    input_layout: ImageLayout | None = None,
    input_value_range: ValueRange = "auto",
    remove_invisible: bool = True,
) -> np.ndarray:
    """One-off CPU-result API; reuse ``predictor`` for repeated/live use."""
    predictor = predictor or FastFoundationStereoDisparity(
        repo_dir=repo_dir,
        model_path=model_path,
        model_dir=model_dir,
        device=device,
        valid_iters=valid_iters,
        max_disp=max_disp,
        hiera=hiera,
    )
    return predictor.predict(
        left_rectified,
        right_rectified,
        input_color_order=input_color_order,
        input_layout=input_layout,
        input_value_range=input_value_range,
        model_scale=model_scale,
        remove_invisible=remove_invisible,
    )


def compute_disparity_fast_foundationstereo_cuda(
    left_rectified: Any,
    right_rectified: Any,
    *,
    predictor: FastFoundationStereoDisparity,
    model_scale: float = 1.0,
    input_color_order: ColorOrder = "RGB",
    input_layout: ImageLayout | None = None,
    input_value_range: ValueRange = "auto",
    remove_invisible: bool = True,
):
    """Return the restored disparity as a CUDA torch.Tensor."""
    return predictor.predict_cuda(
        left_rectified,
        right_rectified,
        input_color_order=input_color_order,
        input_layout=input_layout,
        input_value_range=input_value_range,
        model_scale=model_scale,
        remove_invisible=remove_invisible,
    )



# ---------------------------------------------------------------------------
# Torch CUDA rectification, nvJPEG decoding, and point-cloud construction
# ---------------------------------------------------------------------------


def _image_hw(image: Any) -> tuple[int, int]:
    shape = tuple(map(int, image.shape))
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        layout = _infer_layout(shape, None)
        return (shape[1], shape[2]) if layout == "CHW" else (shape[0], shape[1])
    raise ValueError(f"Unsupported image shape {shape}")


def _image_to_torch_gray(
    image: Any,
    *,
    device: Any,
    input_layout: ImageLayout | None = None,
    value_range: ValueRange = "auto",
):
    """Return a contiguous 1xHxW float32 tensor in the 0..255 range."""
    if isinstance(image, torch.Tensor):
        tensor = image
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
    elif hasattr(image, "__dlpack__"):
        tensor = torch.from_dlpack(image)
    else:
        tensor = torch.as_tensor(image)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        layout = _infer_layout(tuple(map(int, tensor.shape)), input_layout)
        tensor = tensor[:1] if layout == "CHW" else tensor[..., :1].permute(2, 0, 1)
    else:
        raise ValueError(f"Stereo image must be HxW, HxWxC, or CxHxW; got {tuple(tensor.shape)}")

    tensor = tensor.to(device=device, non_blocking=True)
    if value_range not in ("auto", "0_1", "0_255"):
        raise ValueError(f"Unsupported value_range: {value_range}")
    if tensor.is_floating_point():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)
        scale_from_unit = value_range == "0_1"
        if value_range == "auto":
            scale_from_unit = float(tensor.amax().item()) <= 1.0
        if scale_from_unit:
            tensor = tensor * 255.0
        tensor = tensor.clamp(0.0, 255.0).to(torch.float32)
    elif tensor.dtype == torch.uint16:
        maximum = int(tensor.to(torch.int32).max().item())
        tensor = tensor.to(torch.float32)
        if maximum > 0:
            tensor = tensor * (255.0 / maximum)
    else:
        tensor = tensor.clamp(0, 255).to(torch.float32)
    return tensor.contiguous()


class TorchCudaStereoRectifier:
    """Apply cached OpenCV output-to-input maps with Torch ``grid_sample``.

    OpenCV still computes the stereo geometry once. Per-frame rectification is
    pure Torch CUDA and can feed the DNN without framework handoffs.
    """

    def __init__(
        self,
        rectification: StereoRectification,
        *,
        device: DeviceLike = "cuda",
        align_corners: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.rectification = rectification
        self.align_corners = bool(align_corners)
        self.image_size = tuple(map(int, rectification.image_size))
        left = self._map_to_grid(rectification.left_map_x, rectification.left_map_y)
        right = self._map_to_grid(rectification.right_map_x, rectification.right_map_y)
        self.grid = torch.stack((left, right), dim=0).contiguous()

    def _map_to_grid(self, map_x: Any, map_y: Any):
        x = torch.as_tensor(np.asarray(map_x, np.float32), device=self.device)
        y = torch.as_tensor(np.asarray(map_y, np.float32), device=self.device)
        if x.ndim != 2 or x.shape != y.shape:
            raise ValueError("Rectification maps must be matching HxW arrays")
        height, width = map(int, x.shape)
        if self.align_corners:
            grid_x = 2.0 * x / max(width - 1, 1) - 1.0
            grid_y = 2.0 * y / max(height - 1, 1) - 1.0
        else:
            # OpenCV maps reference source pixel centers. This conversion maps
            # source x=0 to the center of the first grid_sample pixel.
            grid_x = (2.0 * x + 1.0) / width - 1.0
            grid_y = (2.0 * y + 1.0) / height - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def rectify(
        self,
        left: Any,
        right: Any,
        *,
        input_layout: ImageLayout | None = None,
        input_value_range: ValueRange = "auto",
    ):
        left_t = _image_to_torch_gray(
            left,
            device=self.device,
            input_layout=input_layout,
            value_range=input_value_range,
        )
        right_t = _image_to_torch_gray(
            right,
            device=self.device,
            input_layout=input_layout,
            value_range=input_value_range,
        )
        if tuple(left_t.shape) != tuple(right_t.shape):
            raise ValueError("Left/right images must have matching shape")
        height, width = map(int, left_t.shape[-2:])
        if (width, height) != self.image_size:
            raise ValueError(
                f"Input size {(width, height)} differs from rectification size {self.image_size}"
            )
        batch = torch.stack((left_t, right_t), dim=0)  # 2x1xHxW
        rectified = torch.nn.functional.grid_sample(
            batch,
            self.grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return rectified[0, 0].contiguous(), rectified[1, 0].contiguous()

@dataclass
class CudaDecodedInputs:
    left: Any
    right: Any
    rgb: Any
    rgb_layout: ImageLayout = "CHW"
    input_color_order: ColorOrder = "RGB"
    owners: tuple[Any, ...] = ()


class TorchvisionCudaJpegDecoder:
    """Batch nvJPEG decoder with optional cached compressed bitstreams."""

    def __init__(self, *, device_id: int = 0, cache_bitstreams: bool = True) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.cache_bitstreams = bool(cache_bitstreams)
        self._paths: tuple[Path, Path, Path] | None = None
        self._encoded: list[Any] | None = None

    def _read_encoded(self, paths: tuple[Path, Path, Path]) -> list[Any]:
        return [read_file(str(path)) for path in paths]

    def preload(self, paths: tuple[str | Path, str | Path, str | Path]) -> None:
        normalized = tuple(Path(path) for path in paths)
        self._encoded = self._read_encoded(normalized)
        self._paths = normalized

    def decode(
        self,
        left_path: str | Path,
        right_path: str | Path,
        rgb_path: str | Path,
    ) -> CudaDecodedInputs:
        paths = (Path(left_path), Path(right_path), Path(rgb_path))
        if self._encoded is None or self._paths != paths or not self.cache_bitstreams:
            encoded = self._read_encoded(paths)
            if self.cache_bitstreams:
                self._paths, self._encoded = paths, encoded
        else:
            encoded = self._encoded

        with torch.cuda.device(self.device):
            images = decode_jpeg(encoded, mode=ImageReadMode.RGB, device=self.device)
        if not isinstance(images, (tuple, list)) or len(images) != 3:
            raise RuntimeError(f"Expected three decoded images, got {type(images)!r}")
        left_rgb, right_rgb, rgb = images
        for name, image in (("left", left_rgb), ("right", right_rgb), ("rgb", rgb)):
            if image.ndim != 3 or image.shape[0] < 3 or image.dtype != torch.uint8:
                raise RuntimeError(
                    f"Expected {name} as CUDA uint8 CHW with >=3 channels; "
                    f"got shape={tuple(image.shape)}, dtype={image.dtype}"
                )
            if not image.is_cuda:
                raise RuntimeError(f"Torchvision decoded {name} on CPU")

        left = left_rgb[0].contiguous()
        right = right_rgb[0].contiguous()
        return CudaDecodedInputs(
            left=left,
            right=right,
            rgb=rgb,
            owners=(left_rgb, right_rgb, rgb, left, right),
        )


def _scale_K(
    intrinsic: np.ndarray,
    from_wh: tuple[int, int],
    to_wh: tuple[int, int],
) -> np.ndarray:
    result = np.asarray(intrinsic, np.float64).copy()
    if from_wh != to_wh:
        sx = to_wh[0] / from_wh[0]
        sy = to_wh[1] / from_wh[1]
        result[0, [0, 2]] *= sx
        result[1, [1, 2]] *= sy
    return result


def _tilt_projection_matrix(tau_x: float, tau_y: float) -> np.ndarray:
    cx, sx = math.cos(float(tau_x)), math.sin(float(tau_x))
    cy, sy = math.cos(float(tau_y)), math.sin(float(tau_y))
    rot_x = np.array(((1, 0, 0), (0, cx, sx), (0, -sx, cx)), np.float32)
    rot_y = np.array(((cy, 0, -sy), (0, 1, 0), (sy, 0, cy)), np.float32)
    rot_xy = rot_y @ rot_x
    project_z = np.array(
        (
            (rot_xy[2, 2], 0, -rot_xy[0, 2]),
            (0, rot_xy[2, 2], -rot_xy[1, 2]),
            (0, 0, 1),
        ),
        np.float32,
    )
    return project_z @ rot_xy


class TorchDnnColoredCloudBuilder:
    """Pure-Torch CUDA reprojection, RGB projection, and color lookup."""

    def __init__(
        self,
        calibration: StereoRgbCalibration,
        rectification: StereoRectification,
        rgb_shape: tuple[int, ...],
        *,
        device: DeviceLike = "cuda",
        rgb_layout: ImageLayout = "HWC",
        min_disparity: float = 0.5,
        max_depth_m: float | None = 5.0,
        stride: int = 1,
        output_frame: Literal["left", "left_rectified"] = "left",
        input_color_order: ColorOrder = "BGR",
        rgb_image_is_undistorted: bool = False,
        rgb_value_range: ValueRange = "auto",
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if output_frame not in ("left", "left_rectified"):
            raise ValueError("output_frame must be 'left' or 'left_rectified'")
        if rgb_layout not in ("HWC", "CHW"):
            raise ValueError("rgb_layout must be HWC or CHW")
        if input_color_order not in ("RGB", "BGR"):
            raise ValueError("input_color_order must be RGB or BGR")
        if rgb_value_range not in ("auto", "0_1", "0_255"):
            raise ValueError(f"Unsupported rgb_value_range: {rgb_value_range}")

        self.device = torch.device(device)
        self.min_disparity = float(min_disparity)
        self.max_depth_m = max_depth_m
        self.stride = int(stride)
        self.output_frame = output_frame
        self.input_color_order = input_color_order
        self.rgb_layout = rgb_layout
        self.rgb_value_range = rgb_value_range

        if len(rgb_shape) != 3:
            raise ValueError(f"RGB image must be three-dimensional, got {rgb_shape}")
        if rgb_layout == "HWC":
            rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
        else:
            rgb_h, rgb_w = int(rgb_shape[1]), int(rgb_shape[2])

        intrinsic = _scale_K(
            calibration.rgb_intrinsics,
            tuple(calibration.rgb_resolution),
            (rgb_w, rgb_h),
        ).astype(np.float32)
        as_device = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)
        self.Q = as_device(rectification.Q)
        self.R1 = as_device(rectification.R1)
        self.T = as_device(calibration.left_to_rgb)
        self.K = as_device(intrinsic)

        distortion = np.zeros(14, np.float32)
        if not rgb_image_is_undistorted:
            source = np.asarray(calibration.rgb_distortion, np.float32).reshape(-1)
            if len(source) not in (4, 5, 8, 12, 14):
                raise ValueError("RGB distortion must contain 4, 5, 8, 12, or 14 values")
            distortion[: len(source)] = source
        self.distortion = as_device(distortion)
        self.has_distortion = bool(np.any(distortion != 0))
        self.tilt = as_device(_tilt_projection_matrix(distortion[12], distortion[13]))
        self.has_tilt = bool(distortion[12] != 0 or distortion[13] != 0)

    def _as_torch_rgb(self, value: Any):
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
        elif hasattr(value, "__dlpack__"):
            tensor = torch.from_dlpack(value)
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.to(device=self.device, non_blocking=True)
        if tensor.ndim != 3:
            raise ValueError(f"rgb_image must be three-dimensional, got {tuple(tensor.shape)}")
        if self.rgb_layout == "HWC":
            if tensor.shape[2] < 3:
                raise ValueError(f"HWC RGB image needs >=3 channels, got {tuple(tensor.shape)}")
            tensor = tensor[:, :, :3]
        else:
            if tensor.shape[0] < 3:
                raise ValueError(f"CHW RGB image needs >=3 channels, got {tuple(tensor.shape)}")
            tensor = tensor[:3]

        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)
            scale_from_unit = self.rgb_value_range == "0_1"
            if self.rgb_value_range == "auto":
                scale_from_unit = float(tensor.amax().item()) <= 1.0
            if scale_from_unit:
                tensor = tensor * 255.0
            tensor = tensor.round().clamp(0.0, 255.0).to(torch.uint8)
        elif tensor.dtype == torch.uint16:
            maximum = int(tensor.to(torch.int32).max().item())
            tensor = tensor.to(torch.float32)
            if maximum > 0:
                tensor = tensor * (255.0 / maximum)
            tensor = tensor.round().clamp(0.0, 255.0).to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        return tensor.contiguous()

    def _project_rgb(self, points_rgb: torch.Tensor):
        z = points_rgb[:, 2]
        x = points_rgb[:, 0] / z
        y = points_rgb[:, 1] / z

        if self.has_distortion:
            d = self.distortion
            k1, k2, p1, p2, k3, k4, k5, k6 = (d[i] for i in range(8))
            s1, s2, s3, s4 = (d[i] for i in range(8, 12))
            r2 = x.square() + y.square()
            r4 = r2.square()
            r6 = r4 * r2
            radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
                1 + k4 * r2 + k5 * r4 + k6 * r6
            )
            xy = x * y
            xd = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x.square()) + s1 * r2 + s2 * r4
            yd = y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * xy + s3 * r2 + s4 * r4
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

    def build_cuda(
        self,
        disparity_cuda: Any,
        rgb_image: Any,
        *,
        return_pixel_xy: bool = False,
    ):
        """Return compact CUDA points/colors and a CUDA valid-pixel count.

        When ``return_pixel_xy`` is true, also return the integer RGB pixel
        coordinate ``(x, y)`` associated with every returned point/color pair.
        The default return shape is unchanged for backward compatibility.
        """
        if isinstance(disparity_cuda, torch.Tensor):
            disparity = disparity_cuda.to(device=self.device, non_blocking=True)
        elif hasattr(disparity_cuda, "__dlpack__"):
            disparity = torch.from_dlpack(disparity_cuda).to(self.device)
        else:
            disparity = torch.as_tensor(disparity_cuda, device=self.device)
        disparity = disparity.squeeze()
        rgb_gpu = self._as_torch_rgb(rgb_image)
        if disparity.ndim != 2:
            raise ValueError(f"disparity must be HxW, got {tuple(disparity.shape)}")

        if self.rgb_layout == "HWC":
            rgb_h, rgb_w = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
        else:
            rgb_h, rgb_w = int(rgb_gpu.shape[1]), int(rgb_gpu.shape[2])

        valid2d = torch.isfinite(disparity) & (disparity > self.min_disparity)
        if self.stride == 1:
            y, x = torch.nonzero(valid2d, as_tuple=True)
        else:
            ys, xs = torch.nonzero(valid2d[:: self.stride, :: self.stride], as_tuple=True)
            y, x = ys * self.stride, xs * self.stride

        d = disparity[y, x].to(torch.float32)
        xf = x.to(torch.float32)
        yf = y.to(torch.float32)
        Q = self.Q
        W = Q[3, 0] * xf + Q[3, 1] * yf + Q[3, 2] * d + Q[3, 3]
        X = (Q[0, 0] * xf + Q[0, 1] * yf + Q[0, 2] * d + Q[0, 3]) / W
        Y = (Q[1, 0] * xf + Q[1, 1] * yf + Q[1, 2] * d + Q[1, 3]) / W
        Z = (Q[2, 0] * xf + Q[2, 1] * yf + Q[2, 2] * d + Q[2, 3]) / W
        points_rect = torch.stack((X, Y, Z), dim=1)

        keep = torch.isfinite(points_rect).all(dim=1) & (Z > 0)
        if self.max_depth_m is not None:
            keep &= Z <= float(self.max_depth_m)
        points_rect = points_rect[keep]

        points_left = points_rect @ self.R1
        points_rgb = points_left @ self.T[:3, :3].T + self.T[:3, 3]
        front = torch.isfinite(points_rgb).all(dim=1) & (points_rgb[:, 2] > 0)
        points_rgb = points_rgb[front]
        points_left = points_left[front]
        points_rect = points_rect[front]

        u_float, v_float = self._project_rgb(points_rgb)
        finite = torch.isfinite(u_float) & torch.isfinite(v_float)
        u = torch.round(u_float[finite]).to(torch.long)
        v = torch.round(v_float[finite]).to(torch.long)
        inside = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
        u, v = u[inside], v[inside]

        source_points = points_left if self.output_frame == "left" else points_rect
        source_points = source_points[finite][inside].contiguous()
        if self.rgb_layout == "HWC":
            colors = rgb_gpu[v, u]
        else:
            colors = rgb_gpu[:3, v, u].T
        if self.input_color_order == "BGR":
            colors = colors.flip(1)

        source_points = source_points.contiguous()
        colors = colors.contiguous()
        valid_count = valid2d.count_nonzero()
        if return_pixel_xy:
            pixel_xy = torch.stack((u, v), dim=1).contiguous()
            return source_points, colors, pixel_xy, valid_count
        return source_points, colors, valid_count

    def build(
        self,
        disparity_cuda: Any,
        rgb_image: Any,
        *,
        return_pixel_xy: bool = False,
    ):
        """Return compact NumPy outputs; only final results leave CUDA."""
        outputs = self.build_cuda(
            disparity_cuda,
            rgb_image,
            return_pixel_xy=return_pixel_xy,
        )
        if return_pixel_xy:
            points_t, colors_t, pixel_xy_t, valid_count_t = outputs
        else:
            points_t, colors_t, valid_count_t = outputs

        points = points_t.detach().cpu().numpy().astype(np.float32, copy=False)
        colors = colors_t.detach().cpu().numpy().astype(np.uint8, copy=False)
        valid_count = int(valid_count_t.item())
        if return_pixel_xy:
            pixel_xy = pixel_xy_t.detach().cpu().numpy().astype(np.int64, copy=False)
            return points, colors, pixel_xy, valid_count
        return points, colors, valid_count


class FastFoundationStereoTorchPipeline:
    """Reusable Torch rectification + DNN disparity + Torch cloud pipeline."""

    def __init__(
        self,
        calibration: Any,
        predictor: FastFoundationStereoDisparity,
        *,
        alpha: float = 0.0,
        model_scale: float = 1.0,
        min_disparity: float = 0.5,
        max_depth_m: float | None = 5.0,
        stride: int = 1,
        output_frame: Literal["left", "left_rectified"] = "left",
        rgb_image_is_undistorted: bool = False,
        device_id: int | None = None,
        align_corners: bool = False,
    ) -> None:
        if predictor.device_type != "cuda":
            raise ValueError("Fast Torch pipeline requires predictor.device on CUDA")
        self.calibration:StereoRgbCalibration = calibration
        self.predictor = predictor
        self.alpha = float(alpha)
        self.model_scale = float(model_scale)
        self.min_disparity = float(min_disparity)
        self.max_depth_m = max_depth_m
        self.stride = int(stride)
        self.output_frame = output_frame
        self.rgb_image_is_undistorted = bool(rgb_image_is_undistorted)
        self.align_corners = bool(align_corners)
        inferred_device_id = predictor.device.index
        selected_device_id = inferred_device_id if device_id is None else device_id
        self.device_id = int(0 if selected_device_id is None else selected_device_id)
        if predictor.device.index not in (None, self.device_id):
            raise ValueError(
                f"predictor is on cuda:{predictor.device.index}, but device_id={self.device_id}"
            )
        self.device = predictor.device
        self.rectification = None
        self.rectifier = None
        self.cloud_builder = None
        self._setup_key = None
        self.decoder = TorchvisionCudaJpegDecoder(device_id=self.device_id)

    def _ensure_setup(
        self,
        stereo_hw: tuple[int, int],
        rgb_shape: tuple[int, ...],
        rgb_layout: ImageLayout,
        input_color_order: ColorOrder,
        rgb_value_range: ValueRange,
    ) -> None:
        key = (stereo_hw, rgb_shape, rgb_layout, input_color_order, rgb_value_range)
        if self._setup_key == key:
            return
        height, width = stereo_hw
        rectifier = self.calibration.get_rectifier(self.alpha)
        self.rectification = rectifier.make((width, height))
        self.rectifier = TorchCudaStereoRectifier(
            self.rectification,
            device=self.device,
            align_corners=self.align_corners,
        )
        self.cloud_builder = TorchDnnColoredCloudBuilder(
            self.calibration,
            self.rectification,
            rgb_shape,
            device=self.device,
            rgb_layout=rgb_layout,
            min_disparity=self.min_disparity,
            max_depth_m=self.max_depth_m,
            stride=self.stride,
            output_frame=self.output_frame,
            input_color_order=input_color_order,
            rgb_image_is_undistorted=self.rgb_image_is_undistorted,
            rgb_value_range=rgb_value_range,
        )
        self._setup_key = key

    def process(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        rgb_image: torch.Tensor,
        *,
        stereo_layout: ImageLayout | None = None,
        stereo_value_range: ValueRange = "auto",
        rgb_layout: ImageLayout = "HWC",
        rgb_value_range: ValueRange = "auto",
        input_color_order: ColorOrder = "BGR",
        output_path: str | Path | None = None,
        save_binary_pcd: bool = True,
        remove_invisible: bool = True,
        download_disparity: bool = False,
    ):
        """Process CPU or CUDA images with no VPI or CuPy dependency."""
        torch.cuda.set_device(self.device_id)

        stereo_hw = _image_hw(left_image)
        if stereo_hw != _image_hw(right_image):
            raise ValueError("Left/right images must have matching dimensions")
        rgb_shape = tuple(map(int, rgb_image.shape))
        self._ensure_setup(
            stereo_hw,
            rgb_shape,
            rgb_layout,
            input_color_order,
            rgb_value_range,
        )
        assert self.rectifier is not None
        assert self.cloud_builder is not None
        assert self.rectification is not None

        left_rect, right_rect = self.rectifier.rectify(
            left_image,
            right_image,
            input_layout=stereo_layout,
            input_value_range=stereo_value_range,
        )
        disparity_cuda = self.predictor.predict_cuda(
            left_rect,
            right_rect,
            input_color_order="RGB",
            input_layout=None,
            input_value_range="0_255",
            model_scale=self.model_scale,
            remove_invisible=remove_invisible,
        )
        points, colors, _ = self.cloud_builder.build(disparity_cuda, rgb_image)
        disparity_cpu = (
            disparity_cuda.detach().cpu().numpy().astype(np.float32, copy=False)
            if download_disparity
            else None
        )

        if output_path is not None:
            save_point_cloud(  # noqa: F405
                output_path,
                points,
                colors,
                binary_pcd=save_binary_pcd,
            )
        return ColoredPointCloud(  # noqa: F405
            points,
            colors,
            disparity_cpu,
            self.rectification,
        )

    def process_jpegs(
        self,
        left_path: str | Path,
        right_path: str | Path,
        rgb_path: str | Path,
        *,
        output_path: str | Path | None = None,
        save_binary_pcd: bool = True,
        remove_invisible: bool = True,
        download_disparity: bool = False,
    ):
        """nvJPEG decode three files and run the full Torch CUDA pipeline."""
        decoded = self.decoder.decode(left_path, right_path, rgb_path)
        cloud = self.process(
            decoded.left,
            decoded.right,
            decoded.rgb,
            stereo_layout=None,
            stereo_value_range="0_255",
            rgb_layout=decoded.rgb_layout,
            rgb_value_range="0_255",
            input_color_order=decoded.input_color_order,
            output_path=output_path,
            save_binary_pcd=save_binary_pcd,
            remove_invisible=remove_invisible,
            download_disparity=download_disparity,
        )
        _ = decoded.owners
        return cloud


def stereo_rgb_to_colored_point_cloud_dnn_cuda(
    left_image: Any,
    right_image: Any,
    rgb_image: Any,
    *,
    calibration: Any,
    disparity_predictor: FastFoundationStereoDisparity,
    pipeline: FastFoundationStereoTorchPipeline | None = None,
    model_scale: float = 1.0,
    output_path: str | Path | None = None,
    input_color_order: ColorOrder = "BGR",
    rgb_layout: ImageLayout = "HWC",
    rgb_image_is_undistorted: bool = False,
    alpha: float = 0.0,
    min_disparity: float = 0.5,
    max_depth_m: float | None = 10.0,
    stride: int = 1,
    output_frame: Literal["left", "left_rectified"] = "left",
    save_binary_pcd: bool = True,
    remove_invisible: bool = True,
    download_disparity: bool = False,
):
    """Functional wrapper around :class:`FastFoundationStereoTorchPipeline`."""
    pipeline = pipeline or FastFoundationStereoTorchPipeline(
        calibration,
        disparity_predictor,
        alpha=alpha,
        model_scale=model_scale,
        min_disparity=min_disparity,
        max_depth_m=max_depth_m,
        stride=stride,
        output_frame=output_frame,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )
    return pipeline.process(
        left_image,
        right_image,
        rgb_image,
        rgb_layout=rgb_layout,
        input_color_order=input_color_order,
        output_path=output_path,
        save_binary_pcd=save_binary_pcd,
        remove_invisible=remove_invisible,
        download_disparity=download_disparity,
    )


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
    
    root = "recording/rgb_stereo/2026-07-22/field_all/111737.603022000JST/"
    left = root+"imgs/rgbd_left/left.jpg"
    right = root+"imgs/rgbd_left/right.jpg"
    rgb = root+"imgs/rgbd_left/rgb.jpg"
    with open(root+"calib/rgbd_left.json") as f:
        calibration = StereoRgbCalibration.from_dict(json.load(f))

    predictor = FastFoundationStereoDisparity(
        repo_dir="./examples/14_iox2_dai_yolo_web_mjpeg/fast-foundationstereo",
        model_path="weights/23-36-37/model_best_bp2_serialize.pth",
        device="cuda:0",
        valid_iters=8,
        max_disp=320,
    )

    pipeline = FastFoundationStereoTorchPipeline(
        calibration,
        predictor,
        model_scale=0.5,
        max_depth_m=5.0,
        stride=1,
    )

    # cloud = pipeline.process(left,right,rgb,
    #     output_path="colored_cloud_torch.pcd",
    #     download_disparity=False,
    # )
    cloud = pipeline.process_jpegs(left,right,rgb,
        output_path="colored_cloud_torch.pcd",
        download_disparity=False,
    )