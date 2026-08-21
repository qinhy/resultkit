"""Standalone Torchvision nvJPEG + VPI + CuPy colored-cloud benchmark.

Default pipeline (``--decoder torchvision --remap vpi``):
  * compressed JPEG bytes are cached in CPU memory once,
  * Torchvision/nvJPEG batch-decodes left, right, and RGB directly to CUDA,
  * zero-copy DLPack exposes Torch CUDA tensors to CuPy,
  * left/right channel views remain CUDA-resident,
  * OpenCV computes calibrated rectification maps once during setup,
  * VPI CUDA remap applies those maps on the GPU,
  * VPI CUDA computes stereo disparity,
  * VPI CUDA disparity is exposed zero-copy to CuPy,
  * fused CUDA reprojection, RGB projection, distortion, and CHW color sampling,
  * only compact final points/colors are downloaded for optional PCD writing.

Use ``--decoder opencv`` and/or ``--remap opencv`` for controlled baselines.
Pass ``--decode-include-io`` to include JPEG file reads in each decode timing.
The file embeds calibration, rectification-map generation, CUDA cloud building,
and PCD writing and has no project-local imports.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ColorOrder = Literal["RGB", "BGR"]


def cv2():
    try:
        import cv2 as _cv2  # type: ignore
        return _cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required: uv pip install opencv-python") from exc


def arr(x: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(x, np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result.copy()


def resolution(x: Any, name: str) -> tuple[int, int]:
    if len(x) != 2:
        raise ValueError(f"{name} must be [width, height]")
    width, height = map(int, x)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive values")
    return width, height


def scale_K(
    K: np.ndarray,
    from_wh: tuple[int, int],
    to_wh: tuple[int, int],
) -> np.ndarray:
    result = K.astype(np.float64, copy=True)
    if from_wh != to_wh:
        sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
        result[0, [0, 2]] *= sx
        result[1, [1, 2]] *= sy
    return result


def gray8(image: np.ndarray) -> np.ndarray:
    c = cv2()
    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] >= 3:
        value = c.cvtColor(value[:, :, :3], c.COLOR_BGR2GRAY)
    elif value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    elif value.ndim != 2:
        raise ValueError(f"Unsupported image shape: {value.shape}")
    if value.dtype == np.uint8:
        return value
    if np.issubdtype(value.dtype, np.floating) and np.nanmax(value) <= 1:
        value = value * 255
    if value.dtype == np.uint16 and value.max() > 0:
        value = value.astype(np.float32) * (255 / value.max())
    return np.clip(value, 0, 255).astype(np.uint8)


def rgb8(colors: np.ndarray, order: ColorOrder = "RGB") -> np.ndarray:
    value = np.asarray(colors)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"colors must be Nx3, got {value.shape}")
    value = value[:, :3]
    if np.issubdtype(value.dtype, np.floating) and value.size and np.nanmax(value) <= 1:
        value = value * 255
    value = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return value[:, ::-1] if order == "BGR" else value


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
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_translation_unit: Literal["m", "cm", "mm"] = "cm",
    ) -> "StereoRgbCalibration":
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[source_translation_unit]
        left_to_right = arr(
            data["left_to_right_extrinsics"],
            "left_to_right_extrinsics",
            (4, 4),
        )
        left_to_rgb = arr(
            data["left_to_rgb_extrinsics"],
            "left_to_rgb_extrinsics",
            (4, 4),
        )
        left_to_right[:3, 3] *= scale
        left_to_rgb[:3, 3] *= scale
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
            left_to_right,
            left_to_rgb,
            source_translation_unit,
        )

    def get_rectifier(self, alpha: float = 0.0, zero_disparity: bool = True):
        return StereoRectifier(self, alpha, zero_disparity)

    @property
    def left_to_right_rotation(self) -> np.ndarray:
        return self.left_to_right[:3, :3].copy()

    @property
    def left_to_right_translation_m(self) -> np.ndarray:
        return self.left_to_right[:3, 3:4].copy()


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


class StereoRectifier:
    def __init__(
        self,
        calibration: StereoRgbCalibration,
        alpha: float = 0.0,
        zero_disparity: bool = True,
    ) -> None:
        self.calibration = calibration
        self.alpha = alpha
        self.zero_disparity = zero_disparity

    def make(self, image_size: tuple[int, int] | None = None) -> StereoRectification:
        c = cv2()
        calibration = self.calibration
        size = image_size or calibration.left_resolution
        if size != calibration.left_resolution:
            warnings.warn(
                "Input size differs from calibration; intrinsics are scaled.",
                RuntimeWarning,
                stacklevel=2,
            )

        K1 = scale_K(calibration.left_intrinsics, calibration.left_resolution, size)
        K2 = scale_K(calibration.right_intrinsics, calibration.right_resolution, size)
        flags = c.CALIB_ZERO_DISPARITY if self.zero_disparity else 0
        R1, R2, P1, P2, Q, roi1, roi2 = c.stereoRectify(
            K1,
            calibration.left_distortion,
            K2,
            calibration.right_distortion,
            size,
            calibration.left_to_right_rotation,
            calibration.left_to_right_translation_m,
            flags=flags,
            alpha=float(self.alpha),
        )
        maps = [
            c.initUndistortRectifyMap(K, D, R, P, size, c.CV_32FC1)
            for K, D, R, P in (
                (K1, calibration.left_distortion, R1, P1),
                (K2, calibration.right_distortion, R2, P2),
            )
        ]
        return StereoRectification(
            size,
            *maps[0],
            *maps[1],
            R1,
            R2,
            P1,
            P2,
            Q,
            tuple(map(int, roi1)),
            tuple(map(int, roi2)),
        )

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray,
        rectification: StereoRectification | None = None,
    ):
        c = cv2()
        if left is None or right is None or left.shape[:2] != right.shape[:2]:
            raise ValueError("Left/right images are required and must have matching sizes.")
        height, width = left.shape[:2]
        result = rectification or self.make((width, height))

        def remap(image, map_x, map_y):
            return c.remap(
                image,
                map_x,
                map_y,
                c.INTER_LINEAR,
                borderMode=c.BORDER_CONSTANT,
            )

        return (
            remap(left, result.left_map_x, result.left_map_y),
            remap(right, result.right_map_x, result.right_map_y),
            result,
        )


def _rgb_float(colors_rgb: np.ndarray) -> np.ndarray:
    colors = rgb8(colors_rgb)
    packed = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    return packed.astype("<u4").view("<f4")


def _clean_cloud(points_m, colors_rgb):
    points = np.asarray(points_m, np.float32)
    colors = rgb8(colors_rgb)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(colors):
        raise ValueError("points must be Nx3 and match colors length")
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def save_pcd(path: str | Path, points_m, colors_rgb, *, binary: bool = True) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points, colors = _clean_cloud(points_m, colors_rgb)
    rgb = _rgb_float(colors)
    count = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z rgb\n"
        "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {count}\nDATA {'binary' if binary else 'ascii'}\n"
    )
    if binary:
        data = np.empty(
            count,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")],
        )
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
        data["rgb"] = rgb
        with output.open("wb") as file:
            file.write(header.encode("ascii"))
            data.tofile(file)
    else:
        with output.open("w", encoding="ascii") as file:
            file.write(header)
            file.writelines(
                f"{x:.8f} {y:.8f} {z:.8f} {float(color):.9e}\n"
                for (x, y, z), color in zip(points, rgb)
            )
    return output


def save_point_cloud(
    path: str | Path,
    points_m,
    colors_rgb,
    *,
    binary_pcd: bool = True,
) -> Path:
    suffix = Path(path).suffix.lower()
    if suffix != ".pcd":
        raise ValueError("This standalone benchmark writes .pcd files")
    return save_pcd(path, points_m, colors_rgb, binary=binary_pcd)


def read_image(path: str | Path, *, color: bool = True) -> np.ndarray:
    c = cv2()
    image = c.imread(str(path), c.IMREAD_COLOR if color else c.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


try:
    import vpi
except ImportError as exc:
    raise SystemExit("NVIDIA VPI Python bindings are required") from exc

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "CuPy is required. For CUDA 12: uv pip install cupy-cuda12x; "
        "for CUDA 13: uv pip install cupy-cuda13x"
    ) from exc

try:
    import torch
    import torchvision
    from torchvision.io import ImageReadMode, decode_jpeg, read_file
except ImportError:
    torch = None
    torchvision = None
    ImageReadMode = None
    decode_jpeg = None
    read_file = None


DecoderName = Literal["torchvision", "opencv"]
RemapName = Literal["vpi", "opencv"]
RgbLayout = Literal["HWC", "CHW"]


@dataclass
class DecodedInputs:
    left: Any
    right: Any
    rgb: Any
    input_color_order: Literal["RGB", "BGR"]
    rgb_layout: RgbLayout
    stereo_on_cuda: bool
    owners: tuple[Any, ...] = ()


class TorchvisionInputDecoder:
    """Batch-decode three JPEGs with Torchvision's CUDA nvJPEG backend.

    Torchvision returns CHW CUDA tensors. Left/right are grayscale camera files,
    so channel 0 is used as a zero-copy 2-D view. The RGB image remains CHW,
    which avoids allocating a 3040x4032x3 HWC transpose.
    """

    def __init__(
        self,
        *,
        device_id: int = 0,
        stereo_to_cpu: bool = False,
        include_io: bool = False,
    ) -> None:
        if torch is None or torchvision is None or decode_jpeg is None:
            raise RuntimeError(
                "Torch and Torchvision are required. Install CUDA wheels, for example: "
                "uv pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu132"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.stereo_to_cpu = bool(stereo_to_cpu)
        self.include_io = bool(include_io)
        self._paths: tuple[Path, Path, Path] | None = None
        self._encoded: list[Any] | None = None

    def _load_encoded(self, paths: tuple[Path, Path, Path]) -> list[Any]:
        assert read_file is not None
        return [read_file(str(path)) for path in paths]

    def preload(self, paths: tuple[Path, Path, Path]) -> float:
        if self.include_io:
            return 0.0
        start = time.perf_counter()
        self._encoded = self._load_encoded(paths)
        self._paths = paths
        return time.perf_counter() - start

    def read(self, left_path: Path, right_path: Path, rgb_path: Path) -> DecodedInputs:
        paths = (left_path, right_path, rgb_path)
        if self.include_io or self._encoded is None or self._paths != paths:
            encoded = self._load_encoded(paths)
            if not self.include_io:
                self._encoded, self._paths = encoded, paths
        else:
            encoded = self._encoded

        assert decode_jpeg is not None and ImageReadMode is not None
        with torch.cuda.device(self.device):
            images = decode_jpeg(
                encoded,
                mode=ImageReadMode.RGB,
                device=self.device,
            )
            torch.cuda.synchronize(self.device)

        if not isinstance(images, (list, tuple)) or len(images) != 3:
            raise RuntimeError(f"Expected three decoded images, got {type(images)!r}")
        left_t, right_t, rgb_t = images
        for name, image in (("left", left_t), ("right", right_t), ("rgb", rgb_t)):
            if image.ndim != 3 or image.shape[0] < 3 or image.dtype != torch.uint8:
                raise RuntimeError(
                    f"Expected {name} CUDA image as uint8 CHW with >=3 channels, "
                    f"got shape={tuple(image.shape)}, dtype={image.dtype}"
                )
            if not image.is_cuda:
                raise RuntimeError(f"Torchvision decoded {name} on CPU, not CUDA")

        # Keep stereo inputs as CUDA PyTorch tensors. VPI 4.1 officially
        # supports wrapping CUDA PyTorch tensors with vpi.asimage(). Converting
        # them to CuPy first can make this VPI build select Python's CPU buffer
        # protocol, which raises "Accessing a CuPy ndarray on CPU is not allowed".
        left_gray_t = left_t[0]
        right_gray_t = right_t[0]
        if not left_gray_t.is_contiguous():
            left_gray_t = left_gray_t.contiguous()
        if not right_gray_t.is_contiguous():
            right_gray_t = right_gray_t.contiguous()

        if tuple(left_gray_t.shape) != tuple(right_gray_t.shape):
            raise RuntimeError(
                "Decoded stereo sizes differ: "
                f"{tuple(left_gray_t.shape)} vs {tuple(right_gray_t.shape)}"
            )

        # The large RGB image is consumed by CuPy. DLPack provides a zero-copy
        # CUDA view and leaves the tensor in CHW layout, avoiding a full HWC copy.
        rgb_chw = cp.from_dlpack(rgb_t)

        if self.stereo_to_cpu:
            left = left_gray_t.cpu().numpy().copy()
            right = right_gray_t.cpu().numpy().copy()
            stereo_on_cuda = False
        else:
            left, right = left_gray_t, right_gray_t
            stereo_on_cuda = True

        return DecodedInputs(
            left=left,
            right=right,
            rgb=rgb_chw,
            input_color_order="RGB",
            rgb_layout="CHW",
            stereo_on_cuda=stereo_on_cuda,
            owners=(left_t, right_t, rgb_t, left_gray_t, right_gray_t, rgb_chw),
        )


class OpenCVInputDecoder:
    """OpenCV JPEG baseline with optional cached compressed bitstreams."""

    def __init__(self, *, include_io: bool = False) -> None:
        self.include_io = bool(include_io)
        self._paths: tuple[Path, Path, Path] | None = None
        self._encoded: list[np.ndarray] | None = None

    @staticmethod
    def _load_encoded(paths: tuple[Path, Path, Path]) -> list[np.ndarray]:
        encoded = []
        for path in paths:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                raise RuntimeError(f"Could not read JPEG bytes: {path}")
            encoded.append(data)
        return encoded

    def preload(self, paths: tuple[Path, Path, Path]) -> float:
        if self.include_io:
            return 0.0
        start = time.perf_counter()
        self._encoded = self._load_encoded(paths)
        self._paths = paths
        return time.perf_counter() - start

    def read(self, left_path: Path, right_path: Path, rgb_path: Path) -> DecodedInputs:
        c = cv2()
        paths = (left_path, right_path, rgb_path)
        if self.include_io or self._encoded is None or self._paths != paths:
            encoded = self._load_encoded(paths)
            if not self.include_io:
                self._encoded, self._paths = encoded, paths
        else:
            encoded = self._encoded

        left = c.imdecode(encoded[0], c.IMREAD_GRAYSCALE)
        right = c.imdecode(encoded[1], c.IMREAD_GRAYSCALE)
        rgb = c.imdecode(encoded[2], c.IMREAD_COLOR)
        if left is None or right is None or rgb is None:
            raise RuntimeError("OpenCV failed to decode one or more JPEGs")
        return DecodedInputs(
            left=left,
            right=right,
            rgb=rgb,
            input_color_order="BGR",
            rgb_layout="HWC",
            stereo_on_cuda=False,
        )


def _make_input_decoder(args: argparse.Namespace):
    if args.decoder == "opencv":
        return OpenCVInputDecoder(include_io=args.decode_include_io)
    return TorchvisionInputDecoder(
        device_id=args.device_id,
        stereo_to_cpu=args.remap == "opencv",
        include_io=args.decode_include_io,
    )

def _timed_decode(decoder, paths: tuple[Path, Path, Path]):
    start = time.perf_counter()
    decoded = decoder.read(*paths)
    return decoded, time.perf_counter() - start


def _image_hw(image: Any) -> tuple[int, int]:
    if isinstance(image, vpi.Image):
        width, height = image.size
        return int(height), int(width)
    shape = tuple(map(int, image.shape))
    if len(shape) < 2:
        raise ValueError(f"Image must have at least two dimensions, got {shape}")
    return shape[0], shape[1]


def _as_vpi_u8_gray(image: Any) -> vpi.Image:
    """Wrap a NumPy or CUDA PyTorch grayscale buffer as a VPI image."""
    if isinstance(image, vpi.Image):
        return image

    # VPI 4.1 documents CUDA PyTorch tensors as supported CudaBuffer inputs.
    # Test this branch before checking __cuda_array_interface__, since CUDA
    # tensors also expose interoperability protocols.
    if torch is not None and isinstance(image, torch.Tensor):
        tensor = image
        if not tensor.is_cuda:
            raise ValueError("Expected a CUDA PyTorch tensor for VPI CUDA remap")
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected CUDA grayscale HxW tensor, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return vpi.asimage(tensor, format=vpi.Format.U8)

    if isinstance(image, cp.ndarray) or hasattr(image, "__cuda_array_interface__"):
        raise TypeError(
            "This VPI 4.1 Python build does not reliably wrap CuPy inputs. "
            "Pass a CUDA PyTorch tensor to VPI remap; keep CuPy for downstream "
            "point-cloud processing."
        )

    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        array = gray8(array)
    array = np.ascontiguousarray(array)
    return vpi.asimage(array, format=vpi.Format.U8)


def _cv_maps_to_vpi_warpmap(map_x: np.ndarray, map_y: np.ndarray) -> vpi.WarpMap:
    """Copy OpenCV output-to-input maps into a dense VPI WarpMap."""
    map_x = np.asarray(map_x, np.float32)
    map_y = np.asarray(map_y, np.float32)
    if map_x.ndim != 2 or map_x.shape != map_y.shape:
        raise ValueError("map_x and map_y must be matching HxW arrays")

    height, width = map_x.shape
    warp = vpi.WarpMap((width, height), interval=1)
    view = np.asarray(warp)
    if view.shape != (height, width, 2):
        raise RuntimeError(
            f"Unexpected dense VPI WarpMap shape {view.shape}; "
            f"expected {(height, width, 2)}"
        )
    view[..., 0] = map_x
    view[..., 1] = map_y
    return warp


class OpenCVStereoRemapper:
    """CPU cv2.remap baseline."""

    def __init__(self, rectification: StereoRectification) -> None:
        self.rectification = rectification

    def rectify(self, left: Any, right: Any):
        if isinstance(left, cp.ndarray) or isinstance(right, cp.ndarray):
            raise TypeError("OpenCV remap requires CPU left/right images")
        c = cv2()
        left_cpu = np.ascontiguousarray(gray8(np.asarray(left)))
        right_cpu = np.ascontiguousarray(gray8(np.asarray(right)))
        if left_cpu.shape != right_cpu.shape:
            raise ValueError("Left/right images must have matching sizes")
        r = self.rectification
        left_rect = c.remap(
            left_cpu,
            r.left_map_x,
            r.left_map_y,
            c.INTER_LINEAR,
            borderMode=c.BORDER_CONSTANT,
        )
        right_rect = c.remap(
            right_cpu,
            r.right_map_x,
            r.right_map_y,
            c.INTER_LINEAR,
            borderMode=c.BORDER_CONSTANT,
        )
        return left_rect, right_rect

    @staticmethod
    def synchronize(left_rect, right_rect) -> None:
        return None


class VPIStereoRemapper:
    """CUDA remap using dense VPI WarpMaps built from OpenCV maps."""

    def __init__(self, rectification: StereoRectification) -> None:
        self.rectification = rectification
        self.left_warp = _cv_maps_to_vpi_warpmap(
            rectification.left_map_x,
            rectification.left_map_y,
        )
        self.right_warp = _cv_maps_to_vpi_warpmap(
            rectification.right_map_x,
            rectification.right_map_y,
        )
        self._left_output = None
        self._right_output = None
        self._input_refs: tuple[Any, ...] = ()

    def rectify(self, left: Any, right: Any):
        left_h, left_w = _image_hw(left)
        right_h, right_w = _image_hw(right)
        if (left_h, left_w) != (right_h, right_w):
            raise ValueError("Left/right images must have matching sizes")
        if (left_w, left_h) != self.rectification.image_size:
            raise ValueError(
                f"Input size {(left_w, left_h)} differs from rectification "
                f"size {self.rectification.image_size}"
            )

        left_vpi = _as_vpi_u8_gray(left)
        right_vpi = _as_vpi_u8_gray(right)
        self._input_refs = (left_vpi, right_vpi, left, right)

        with vpi.Backend.CUDA:
            if self._left_output is None:
                self._left_output = left_vpi.remap(
                    self.left_warp,
                    interp=vpi.Interp.LINEAR,
                    border=vpi.Border.ZERO,
                )
                self._right_output = right_vpi.remap(
                    self.right_warp,
                    interp=vpi.Interp.LINEAR,
                    border=vpi.Border.ZERO,
                )
            else:
                left_vpi.remap(
                    self.left_warp,
                    out=self._left_output,
                    interp=vpi.Interp.LINEAR,
                    border=vpi.Border.ZERO,
                )
                right_vpi.remap(
                    self.right_warp,
                    out=self._right_output,
                    interp=vpi.Interp.LINEAR,
                    border=vpi.Border.ZERO,
                )
        return self._left_output, self._right_output

    @staticmethod
    def synchronize(left_rect, right_rect) -> None:
        with left_rect.rlock_cuda(), right_rect.rlock_cuda():
            pass


def _make_remapper(args: argparse.Namespace, rectification: StereoRectification):
    if args.remap == "opencv":
        return OpenCVStereoRemapper(rectification)
    return VPIStereoRemapper(rectification)


def _timed_remap(remapper, left, right):
    start = time.perf_counter()
    left_rect, right_rect = remapper.rectify(left, right)
    remapper.synchronize(left_rect, right_rect)
    return left_rect, right_rect, time.perf_counter() - start


class VPIStereoDisparityGPU:
    """Run VPI CUDA stereo and retain all intermediate images on CUDA."""

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
        self._shape = None
        self._left_y16 = None
        self._right_y16 = None
        self._disparity = None
        self._confidence = None
        self._input_refs: tuple[Any, ...] = ()

    def _reset_for_shape(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self._left_y16 = None
        self._right_y16 = None
        self._disparity = None
        self._confidence = vpi.Image(shape, vpi.Format.U16)

    def predict(self, left: Any, right: Any):
        left_vpi = _as_vpi_u8_gray(left)
        right_vpi = _as_vpi_u8_gray(right)
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
                self._disparity = vpi.stereodisp(
                    self._left_y16,
                    self._right_y16,
                    **kwargs,
                )
            else:
                vpi.stereodisp(
                    self._left_y16,
                    self._right_y16,
                    out=self._disparity,
                    **kwargs,
                )
        return self._disparity, self._confidence

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
        rgb_layout: RgbLayout = "HWC",
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
        if rgb_layout not in ("HWC", "CHW"):
            raise ValueError("rgb_layout must be HWC or CHW")

        self.min_disparity = float(min_disparity)
        self.max_depth_m = max_depth_m
        self.stride = int(stride)
        self.output_frame = output_frame
        self.input_color_order = input_color_order
        self.rgb_layout = rgb_layout
        self.rgb_image_is_undistorted = bool(rgb_image_is_undistorted)

        if len(rgb_shape) != 3:
            raise ValueError(f"RGB image must be three-dimensional, got {rgb_shape}")
        if rgb_layout == "HWC":
            rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
        else:
            rgb_h, rgb_w = int(rgb_shape[1]), int(rgb_shape[2])
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

    def build(self, disparity_s16, confidence_u16, rgb_image):
        """Build a colored cloud from a CPU or CUDA-resident RGB image.

        CUDA Array Interface and DLPack-backed CuPy arrays stay on the GPU.
        Torchvision RGB is sampled directly in CHW layout; OpenCV uses HWC.
        """
        if isinstance(rgb_image, cp.ndarray):
            rgb_gpu = rgb_image
        elif hasattr(rgb_image, "__cuda_array_interface__"):
            rgb_gpu = cp.asarray(rgb_image)
        else:
            image_cpu = np.ascontiguousarray(rgb_image)
            rgb_gpu = cp.asarray(image_cpu)

        if rgb_gpu.ndim != 3:
            raise ValueError(f"rgb_image must be three-dimensional, got {rgb_gpu.shape}")
        if self.rgb_layout == "HWC":
            if rgb_gpu.shape[2] < 3:
                raise ValueError(f"HWC rgb_image must have >=3 channels, got {rgb_gpu.shape}")
            rgb_gpu = rgb_gpu[:, :, :3]
            rgb_h, rgb_w = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
        else:
            if rgb_gpu.shape[0] < 3:
                raise ValueError(f"CHW rgb_image must have >=3 channels, got {rgb_gpu.shape}")
            rgb_gpu = rgb_gpu[:3]
            rgb_h, rgb_w = int(rgb_gpu.shape[1]), int(rgb_gpu.shape[2])

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
            inside = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
            u, v = u[inside], v[inside]

            source_points = points_left if self.output_frame == "left" else points_rect
            source_points = source_points[finite][inside]
            if self.rgb_layout == "HWC":
                colors = rgb_gpu[v, u]
            else:
                colors = rgb_gpu[:3, v, u].T
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
    paths = (
        image_dir / "left.jpg",
        image_dir / "right.jpg",
        image_dir / "rgb.jpg",
    )
    for path in (*paths, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep Torch, CuPy, VPI, and DLPack consumers on the requested CUDA device.
    cp.cuda.Device(args.device_id).use()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = StereoRgbCalibration.from_dict(json.load(file))

    decoder = _make_input_decoder(args)
    bitstream_read_time = decoder.preload(paths)
    decoded, decode_warmup = _timed_decode(decoder, paths)
    decode_timings = []
    for _ in range(args.decode_runs):
        decoded, elapsed = _timed_decode(decoder, paths)
        decode_timings.append(elapsed)

    # Keep Torch/DLPack owners reachable until all CUDA consumers finish.
    decode_owners = decoded.owners
    left, right, rgb = decoded.left, decoded.right, decoded.rgb
    left_h, left_w = _image_hw(left)
    right_h, right_w = _image_hw(right)
    if (left_h, left_w) != (right_h, right_w):
        raise ValueError("Decoded left/right images have different dimensions")

    # OpenCV computes the exact calibrated rectification geometry once. VPI
    # then executes the output-to-input maps on CUDA when --remap vpi is used.
    setup_start = time.perf_counter()
    rectifier = calibration.get_rectifier(args.alpha)
    rect = rectifier.make((left_w, left_h))
    remapper = _make_remapper(args, rect)
    rectification_setup_time = time.perf_counter() - setup_start

    left_rect, right_rect, remap_warmup = _timed_remap(remapper, left, right)
    remap_timings = []
    for _ in range(args.remap_runs):
        left_rect, right_rect, elapsed = _timed_remap(remapper, left, right)
        remap_timings.append(elapsed)

    predictor = VPIStereoDisparityGPU(
        min_disparity=args.min_disparity,
        max_disparity=args.max_disparity,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        include_diagonals=not args.skip_diagonals,
    )

    _, _, vpi_warmup = _timed_vpi_predict(predictor, left_rect, right_rect)
    disparity = confidence = None
    disparity_timings = []
    for _ in range(args.runs):
        disparity, confidence, elapsed = _timed_vpi_predict(
            predictor,
            left_rect,
            right_rect,
        )
        disparity_timings.append(elapsed)
    assert disparity is not None and confidence is not None

    builder = CuPyColoredCloudBuilder(
        calibration,
        rect,
        rgb.shape,
        rgb_layout=decoded.rgb_layout,
        min_disparity=max(0.5, float(args.min_disparity)),
        max_depth_m=args.max_depth,
        stride=args.stride,
        output_frame=args.output_frame,
        input_color_order=decoded.input_color_order,
    )

    # Warm up CuPy kernels and allocator.
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

    decode_median = float(np.median(decode_timings))
    remap_median = float(np.median(remap_timings))
    disparity_median = float(np.median(disparity_timings))
    cloud_median = float(np.median(cloud_timings))
    compute_total = decode_median + remap_median + disparity_median + cloud_median
    estimated_total = compute_total + (writing_time if not args.skip_write else 0.0)
    pixel_count = int(rect.image_size[0] * rect.image_size[1])

    print(f"Decoder:                 {args.decoder}")
    print(f"Remap backend:           {args.remap}")
    if args.decoder == "torchvision":
        print(f"Torch version:           {getattr(torch, '__version__', 'unknown')}")
        print(f"Torchvision version:     {getattr(torchvision, '__version__', 'unknown')}")
    print(f"VPI version:             {getattr(vpi, '__version__', 'unknown')}")
    print(f"CuPy version:            {cp.__version__}")
    if args.decode_include_io:
        print("JPEG bitstream input:    included in every decode timing")
    else:
        print(f"JPEG bitstream read:     {bitstream_read_time * 1000:.2f} ms (one-time cache)")
    print(f"Decode warm-up:          {decode_warmup * 1000:.2f} ms")
    print(
        "JPEG decode/input:      "
        f"median={decode_median * 1000:.2f} ms, "
        f"mean={np.mean(decode_timings) * 1000:.2f} ms over {args.decode_runs} runs"
    )
    if args.decoder == "torchvision" and args.remap == "vpi":
        print("  (Torch nvJPEG -> zero-copy CuPy; left/right/RGB remain on CUDA)")
    elif args.decoder == "torchvision":
        print("  (includes left/right GPU->CPU copies; CHW RGB remains on CUDA)")
    elif args.remap == "vpi":
        print("  (OpenCV-decoded left/right are uploaded by VPI remap)")

    print(
        f"Rectification setup:     {rectification_setup_time * 1000:.2f} ms "
        "(one-time maps/warp setup)"
    )
    print(f"Remap warm-up:           {remap_warmup * 1000:.2f} ms")
    print(
        ("VPI remap GPU-only:    " if args.remap == "vpi" else "OpenCV remap CPU:       ")
        + f"median={remap_median * 1000:.2f} ms, "
        f"mean={np.mean(remap_timings) * 1000:.2f} ms over {args.remap_runs} runs"
    )
    print(f"VPI stereo warm-up:      {vpi_warmup * 1000:.2f} ms")
    print(
        "VPI disparity GPU-only: "
        f"median={disparity_median * 1000:.2f} ms, "
        f"mean={np.mean(disparity_timings) * 1000:.2f} ms over {args.runs} runs"
    )
    print(
        "GPU cloud + download:   "
        f"median={cloud_median * 1000:.2f} ms, "
        f"mean={np.mean(cloud_timings) * 1000:.2f} ms over {args.cloud_runs} runs"
    )
    print(f"Compute median total:    {compute_total * 1000:.2f} ms")
    print(
        "PCD writing:            "
        + (f"{writing_time * 1000:.2f} ms" if not args.skip_write else "skipped")
    )
    print(f"Estimated median total:  {estimated_total * 1000:.2f} ms")
    print(f"Valid disparity:         {valid_count:,}/{pixel_count:,} pixels")
    print(f"Saved points:            {len(points):,}")
    print(f"Stereo input:            {left_w}x{left_h} " + ("CUDA" if decoded.stereo_on_cuda else "CPU"))
    print(
        f"RGB input:               {tuple(map(int, rgb.shape))} "
        f"{rgb.dtype} layout={decoded.rgb_layout}"
    )
    if not args.skip_write:
        print(f"Output:                  {Path(args.output).resolve()}")

    # Explicitly retain owners and VPI resources through final CUDA copies.
    _ = (decode_owners, remapper)
    return ColoredPointCloud(points, colors, None, rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torchvision nvJPEG + VPI CUDA remap/stereo + fused CuPy cloud"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera", default="rgbd_left")
    parser.add_argument("--output", default="colored_cloud_vpi_v4.pcd")

    parser.add_argument(
        "--decoder",
        choices=("torchvision", "opencv"),
        default="torchvision",
        help="Torchvision uses CUDA nvJPEG; OpenCV is a CPU decode baseline",
    )
    parser.add_argument(
        "--remap",
        choices=("vpi", "opencv"),
        default="vpi",
        help="rectification backend; calibrated geometry maps are generated by OpenCV",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--decode-runs", type=int, default=10)
    parser.add_argument("--remap-runs", type=int, default=20)
    parser.add_argument(
        "--decode-include-io",
        action="store_true",
        help="read JPEG bytes from disk for every decode run instead of caching them once",
    )

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
    args = parser.parse_args()
    if min(args.decode_runs, args.remap_runs, args.runs, args.cloud_runs) < 1:
        parser.error("--decode-runs, --remap-runs, --runs, and --cloud-runs must be >= 1")
    return args

