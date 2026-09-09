from __future__ import annotations

from abc import ABC
import ctypes
import enum
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor as tTensor

class ArrayLike(ABC): pass
ArrayLike.register(np.ndarray)
ArrayLike.register(tTensor)
try:
    import cupy as cp
    ArrayLike.register(cp.ndarray)
except ImportError:  # CuPy is an optional, CUDA-specific dependency.
    cp = None


@lru_cache(maxsize=None)
def _imp_cp():
    if cp is None:
        raise ImportError(
            "CuPy support is optional. Install a CUDA-matched CuPy package, "
            "for example `pip install matops[cupy-cuda12]`."
        )
    return cp


def _cupy_device_context(device: Optional[Union[str, int, Any]]):
    cupy = _imp_cp()
    if device is None or device == "cuda":
        return nullcontext()
    if isinstance(device, str):
        if not device.startswith("cuda:"):
            raise ValueError(f"Unsupported CuPy device: {device!r}")
        device = int(device.split(":", 1)[1])
    return cupy.cuda.Device(device)


class MatDevice(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"
    # MPS = "mps"
    UNKNOWN = "UNKNOWN"
    
    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray):
            return MatDevice.CPU
        if isinstance(data, tTensor):
            return MatDevice.CPU if data.device.type == "cpu" else MatDevice.CUDA
        if cp is not None and isinstance(data, cp.ndarray):
            return MatDevice.CUDA
        raise ValueError(f"Unsupported data type: {type(data)}")


class DataType(str, enum.Enum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"
    UINT16 = "uint16"
    INT32 = "int32"
    INT64 = "int64"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_dtype(cls, dtype: Any, *, strict: bool = False) -> "DataType | None":
        """Return the matching DataType for a NumPy/Torch/CuPy/Python dtype.

        Returns None for unsupported dtypes unless strict=True.
        """
        if isinstance(dtype, cls):
            return dtype

        # NumPy dtype, NumPy scalar type, or dtype string such as "uint8".
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            np_dtype = None

        if np_dtype is not None:
            result = {
                np.dtype(np.float64): cls.FLOAT64,
                np.dtype(np.float32): cls.FLOAT32,
                np.dtype(np.float16): cls.FLOAT16,
                np.dtype(np.uint8): cls.UINT8,
                np.dtype(np.uint16): cls.UINT16,
                np.dtype(np.int32): cls.INT32,
                np.dtype(np.int64): cls.INT64,
            }.get(np_dtype)
            if result is not None:
                return result

        # Torch dtype objects, for example torch.float32.
        result = {
            torch.float64: cls.FLOAT64,
            torch.float32: cls.FLOAT32,
            torch.float16: cls.FLOAT16,
            torch.bfloat16: cls.BFLOAT16,
            torch.uint8: cls.UINT8,
            **({torch.uint16: cls.UINT16} if hasattr(torch, "uint16") else {}),
            torch.int32: cls.INT32,
            torch.int64: cls.INT64,
        }.get(dtype)
        if result is not None:
            return result

        if strict:
            raise TypeError(f"Unsupported dtype: {dtype!r}")
        return None

    @classmethod
    def which(cls, data: Any, *, strict: bool = True) -> "DataType | None":
        """Infer DataType from an array, tensor, dtype object, enum, or dtype string.

        Existing calls like DataType.which(data) still work. Unsupported inputs return
        None by default; pass strict=True to raise a TypeError instead.
        """
        if isinstance(data, np.ndarray):
            return cls.from_dtype(data.dtype, strict=strict)
        if isinstance(data, tTensor):
            return cls.from_dtype(data.dtype, strict=strict)
        if cp is not None and isinstance(data, cp.ndarray):
            return cls.from_dtype(data.dtype, strict=strict)
        return cls.from_dtype(data, strict=strict)


class MatOps(BaseModel):
    """Small backend-neutral matrix/tensor operation interface.

    Subclasses implement the same common operations for NumPy arrays, PyTorch
    tensors, and CuPy arrays. This is intentionally lightweight; it is useful when
    result payloads may be produced on either backend but downstream code wants
    a consistent operation surface.
    """

    int32: ClassVar[Any] = None
    int64: ClassVar[Any] = None
    uint8: ClassVar[Any] = None
    uint16: ClassVar[Any] = None
    float64: ClassVar[Any] = None
    float32: ClassVar[Any] = None
    float16: ClassVar[Any] = None

    device: MatDevice = MatDevice.UNKNOWN

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def mat(self, data: Any, dtype: Any) -> ArrayLike: raise NotImplementedError
    def ndim(self, x: ArrayLike): return x.ndim
    def shape(self, x: ArrayLike): return x.shape
    def dtype(self, x: ArrayLike): return x.dtype
    def numel(self, x: ArrayLike) -> int: raise NotImplementedError
    def is_floating_point(self, x: ArrayLike) -> bool: raise NotImplementedError
    def sin(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def cos(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def eye(self, size: int, dtype: Any) -> ArrayLike: raise NotImplementedError
    def ones(self, shape: Sequence[int], dtype: Any) -> ArrayLike: raise NotImplementedError
    def zeros(self, shape: Sequence[int], dtype: Any) -> ArrayLike: raise NotImplementedError
    def hstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike: raise NotImplementedError
    def norm(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def dot(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: raise NotImplementedError
    def cross(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: raise NotImplementedError
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: raise NotImplementedError
    def inv(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def no_grad(self): return nullcontext()
    def from_numpy(self, x: np.ndarray) -> ArrayLike: raise NotImplementedError
    def to_numpy(self, x: ArrayLike) -> np.ndarray: raise NotImplementedError
    def mean(self, x: ArrayLike, dim: int = 0) -> ArrayLike: raise NotImplementedError
    def median(self, x: ArrayLike, dim: int = 0) -> ArrayLike: raise NotImplementedError
    def std(self, x: ArrayLike, dim: int = 0) -> ArrayLike: raise NotImplementedError
    def max(self, x: ArrayLike, dim: Optional[int] = 0) -> ArrayLike: raise NotImplementedError
    def min(self, x: ArrayLike, dim: Optional[int] = 0) -> ArrayLike: raise NotImplementedError
    def nanmax(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def all(self, x: ArrayLike, dim: Optional[int] = None) -> ArrayLike: raise NotImplementedError
    def isfinite(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def abs(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def floor(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def round(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def where(self, condition: ArrayLike, x: Any, y: Any) -> ArrayLike: raise NotImplementedError
    def flip(self, x: ArrayLike, dim: int) -> ArrayLike: raise NotImplementedError
    def stack(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike: raise NotImplementedError
    def cat(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike: raise NotImplementedError
    def reshape(self, x: ArrayLike, shape: Sequence[int]) -> ArrayLike: raise NotImplementedError
    def permute(self, x: ArrayLike, dims: Sequence[int]) -> ArrayLike: raise NotImplementedError
    def copy_mat(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def copyto(self, dst: ArrayLike, src: ArrayLike) -> ArrayLike: raise NotImplementedError
    def scatter_min(self, dst: ArrayLike, index: ArrayLike, src: ArrayLike) -> ArrayLike: raise NotImplementedError
    def logical_and(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: raise NotImplementedError
    def logical_or(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: raise NotImplementedError
    def clip(self, x: ArrayLike, min_val: Any, max_val: Any) -> ArrayLike: raise NotImplementedError
    def astype(self, x: ArrayLike, dtype: Any) -> ArrayLike: raise NotImplementedError
    def astype_int32(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def astype_int64(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def astype_uint8(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def astype_float64(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def astype_float32(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def astype_float16(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]: raise NotImplementedError
    def flatten(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError
    def reinterpret(self, x: ArrayLike, dtype: Any) -> ArrayLike: raise NotImplementedError

    @staticmethod
    def from_xyxy_to_xywh(data):
        x1,y1,x2,y2 = data.T
        w = x2 - x1
        h = y2 - y1
        return x1,y1,w,h
    
    @staticmethod
    def from_xywh_to_xyxy(data):
        x1,y1,w,h = data.T
        x2 = x1 + w
        y2 = y1 + h
        return x1,y1,x2,y2
    
    @staticmethod
    def from_cxcywh_to_xyxy(data):
        cx,cy,w,h = data.T
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return x1,y1,x2,y2
    
    @staticmethod
    def from_xyxy_to_cxcywh(data):
        x1,y1,x2,y2 = data.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        return cx,cy,w,h
    
    @staticmethod
    def from_xywh_to_cxcywh(data):
        x1,y1,w,h = data.T
        cx = x1 + w/2
        cy = y1 + h/2
        return cx,cy,w,h
    
    @staticmethod
    def from_cxcywh_to_xywh(data):
        cx,cy,w,h = data.T
        x1 = cx - w/2
        y1 = cy - h/2
        return x1,y1,w,h


class NumpyMatOps(MatOps):
    """NumPy implementation of :class:`MatOps`."""
    int32: ClassVar[Any] = np.int32
    int64: ClassVar[Any] = np.int64
    uint8: ClassVar[Any] = np.uint8
    uint16: ClassVar[Any] = np.uint16
    float64: ClassVar[Any] = np.float64
    float32: ClassVar[Any] = np.float32
    float16: ClassVar[Any] = np.float16
    device: MatDevice = MatDevice.CPU

    def mat(self, data: Any, dtype: Any) -> np.ndarray: return np.array(data, dtype=dtype)
    def numel(self, x: np.ndarray) -> int: return x.size
    def is_floating_point(self, x: np.ndarray) -> bool: return np.issubdtype(x.dtype, np.floating)
    def sin(self, x): return np.sin(x)
    def cos(self, x): return np.cos(x)
    def eye(self, size: int, dtype: Any) -> np.ndarray: return np.eye(size, dtype=dtype)
    def ones(self, shape: Sequence[int], dtype: Any) -> np.ndarray: return np.ones(shape, dtype=dtype)
    def zeros(self, shape: Sequence[int], dtype: Any) -> np.ndarray: return np.zeros(shape, dtype=dtype)
    def hstack(self, arrays: Sequence[np.ndarray]) -> np.ndarray: return np.hstack(arrays)
    def norm(self, x: np.ndarray) -> np.ndarray: return np.linalg.norm(x)
    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.dot(a, b)
    def cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.cross(a, b)
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return a @ b
    def inv(self, x: np.ndarray) -> np.ndarray: return np.linalg.inv(x)
    def from_numpy(self, x: np.ndarray) -> np.ndarray: return np.asarray(x)
    def to_numpy(self, x: np.ndarray) -> np.ndarray: return np.asarray(x)
    def mean(self, x: np.ndarray, dim: int = 0) -> np.ndarray: return np.mean(x, axis=dim)
    def median(self, x: np.ndarray, dim: int = 0) -> np.ndarray: return np.median(x, axis=dim)
    def std(self, x: np.ndarray, dim: int = 0) -> np.ndarray: return np.std(x, axis=dim)
    def max(self, x: np.ndarray, dim: Optional[int] = 0) -> np.ndarray: return np.max(x, axis=dim)
    def min(self, x: np.ndarray, dim: Optional[int] = 0) -> np.ndarray: return np.min(x, axis=dim)
    def nanmax(self, x: np.ndarray) -> np.ndarray: return np.nanmax(x)
    def all(self, x: np.ndarray, dim: Optional[int] = None) -> np.ndarray: return np.all(x, axis=dim)
    def isfinite(self, x: np.ndarray) -> np.ndarray: return np.isfinite(x)
    def abs(self, x: np.ndarray) -> np.ndarray: return np.abs(x)
    def floor(self, x: np.ndarray) -> np.ndarray: return np.floor(x)
    def round(self, x: np.ndarray) -> np.ndarray: return np.rint(x)
    def where(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray: return np.where(condition, x, y)
    def flip(self, x: np.ndarray, dim: int) -> np.ndarray: return np.flip(x, axis=dim)
    def stack(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray: return np.stack(xs, axis=dim)
    def cat(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray: return np.concatenate(xs, axis=dim)
    def reshape(self, x: np.ndarray, shape: Sequence[int]) -> np.ndarray: return np.reshape(x, shape)
    def permute(self, x: np.ndarray, dims: Sequence[int]) -> np.ndarray: return np.transpose(x, axes=tuple(dims))
    def copy_mat(self, x: np.ndarray) -> np.ndarray: return x.copy()
    def copyto(self, dst: np.ndarray, src: np.ndarray) -> np.ndarray:
        np.copyto(dst, src)
        return dst
    def scatter_min(self, dst: np.ndarray, index: np.ndarray, src: np.ndarray) -> np.ndarray:
        np.minimum.at(dst, index, src)
        return dst
    def logical_and(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.logical_and(a, b)
    def logical_or(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.logical_or(a, b)
    def clip(self, x: np.ndarray, min_val: Any, max_val: Any) -> np.ndarray: return np.clip(x, min_val, max_val)
    def astype(self, x: np.ndarray, dtype: Any) -> np.ndarray: return x.astype(dtype, copy=False)
    def astype_int32(self, x: np.ndarray) -> np.ndarray: return x.astype(np.int32)
    def astype_int64(self, x: np.ndarray) -> np.ndarray: return x.astype(np.int64)
    def astype_uint8(self, x: np.ndarray) -> np.ndarray: return x.astype(np.uint8)
    def astype_float64(self, x: np.ndarray) -> np.ndarray: return x.astype(np.float64)
    def astype_float32(self, x: np.ndarray) -> np.ndarray: return x.astype(np.float32)
    def astype_float16(self, x: np.ndarray) -> np.ndarray: return x.astype(np.float16)
    def nonzero(self, x: np.ndarray) -> tuple[np.ndarray, ...]: return np.nonzero(x)
    def flatten(self, x: np.ndarray) -> np.ndarray: return x.flatten()
    def reinterpret(self, x: np.ndarray, dtype: Any) -> np.ndarray: return x.view(dtype)


class TorchMatOps(MatOps):
    """PyTorch implementation of :class:`MatOps`."""
    int32: ClassVar[Any] = None if torch is None else torch.int32
    int64: ClassVar[Any] = None if torch is None else torch.int64
    uint8: ClassVar[Any] = None if torch is None else torch.uint8
    uint16: ClassVar[Any] = getattr(torch, "uint16", None)
    float64: ClassVar[Any] = None if torch is None else torch.float64
    float32: ClassVar[Any] = None if torch is None else torch.float32
    float16: ClassVar[Any] = None if torch is None else torch.float16
    device: MatDevice = MatDevice.CPU # MatDevice.CUDA0

    def mat(self, data: tTensor, dtype: Any) -> tTensor: return torch.tensor(data, dtype=dtype, device=self.device)
    def numel(self, x: tTensor) -> int: return x.numel()
    def is_floating_point(self, x: tTensor) -> bool: return x.is_floating_point()
    def sin(self, x): return torch.sin(x)
    def cos(self, x): return torch.cos(x)
    def eye(self, size: int, dtype: Any) -> tTensor: return torch.eye(size, dtype=dtype, device=self.device)
    def ones(self, shape: Sequence[int], dtype: Any) -> tTensor: return torch.ones(tuple(shape), dtype=dtype, device=self.device)
    def zeros(self, shape: Sequence[int], dtype: Any) -> tTensor: return torch.zeros(tuple(shape), dtype=dtype, device=self.device)
    def hstack(self, arrays: Sequence[Any]) -> tTensor: return torch.cat(tuple(arrays), dim=1)
    def norm(self, x: tTensor) -> tTensor: return torch.norm(x)
    def dot(self, a: tTensor, b: Any) -> tTensor: return torch.dot(a, b)
    def cross(self, a: tTensor, b: Any) -> tTensor: return torch.cross(a, b)
    def matmul(self, a: tTensor, b: Any) -> tTensor: return torch.matmul(a, b)
    def inv(self, x: tTensor) -> tTensor: return torch.linalg.inv(x)
    def no_grad(self): return torch.no_grad()
    def mean(self, x: tTensor, dim: int = 0) -> tTensor: return torch.mean(x, dim=dim)
    def median(self, x: tTensor, dim: int = 0) -> tTensor: return torch.median(x, dim=dim).values
    def std(self, x: tTensor, dim: int = 0) -> tTensor: return torch.std(x, dim=dim, unbiased=False)
    def max(self, x: tTensor, dim: Optional[int] = 0) -> tTensor:
        return torch.max(x) if dim is None else torch.max(x, dim=dim).values

    def min(self, x: tTensor, dim: Optional[int] = 0) -> tTensor:
        return torch.min(x) if dim is None else torch.min(x, dim=dim).values

    def nanmax(self, x: tTensor) -> tTensor:
        if not x.is_floating_point():
            return torch.max(x)
        if x.numel() == 0:
            return torch.max(x)  # Preserve reduction failure on empty tensors.
        valid = ~torch.isnan(x)
        if not torch.any(valid):
            return x.new_tensor(float("nan"))
        return torch.max(x[valid])

    def all(self, x: tTensor, dim: Optional[int] = None) -> tTensor:
        return torch.all(x) if dim is None else torch.all(x, dim=dim)

    def isfinite(self, x: tTensor) -> tTensor: return torch.isfinite(x)
    def abs(self, x: tTensor) -> tTensor: return torch.abs(x)
    def floor(self, x: tTensor) -> tTensor: return torch.floor(x)
    def round(self, x: tTensor) -> tTensor: return torch.round(x)
    def where(self, condition: tTensor, x: Any, y: Any) -> tTensor: return torch.where(condition, x, y)
    def flip(self, x: tTensor, dim: int) -> tTensor: return torch.flip(x, dims=(dim,))
    def stack(self, xs: Sequence[Any], dim: int = 0) -> tTensor: return torch.stack(tuple(xs), dim=dim)
    def cat(self, xs: Sequence[Any], dim: int = 0) -> tTensor: return torch.cat(tuple(xs), dim=dim)
    def reshape(self, x: tTensor, shape: Sequence[int]) -> tTensor: return x.reshape(tuple(shape))
    def permute(self, x: tTensor, dims: Sequence[int]) -> tTensor: return x.permute(*dims)
    def copy_mat(self, x: tTensor) -> tTensor: return x.clone()
    def copyto(self, dst: tTensor, src: tTensor) -> tTensor:
        dst.copy_(src)
        return dst
    def scatter_min(self, dst: tTensor, index: tTensor, src: tTensor) -> tTensor:
        dst.scatter_reduce_(0, index, src, reduce="amin", include_self=True)
        return dst
    def logical_and(self, a: tTensor, b: Any) -> tTensor: return torch.logical_and(a, b)
    def logical_or(self, a: tTensor, b: Any) -> tTensor: return torch.logical_or(a, b)
    def clip(self, x: tTensor, min_val: Any, max_val: Any) -> tTensor: return torch.clamp(x, min=min_val, max=max_val)
    def astype(self, x: tTensor, dtype: Any) -> tTensor: return x.to(dtype=dtype)
    def astype_int32(self, x: tTensor) -> tTensor: return x.to(dtype=torch.int32)
    def astype_int64(self, x: tTensor) -> tTensor: return x.to(dtype=torch.int64)
    def astype_uint8(self, x: tTensor) -> tTensor: return x.to(dtype=torch.uint8)
    def astype_float64(self, x: tTensor) -> tTensor: return x.to(dtype=torch.float64)
    def astype_float32(self, x: tTensor) -> tTensor: return x.to(dtype=torch.float32)
    def astype_float16(self, x: tTensor) -> tTensor: return x.to(dtype=torch.float16)
    def nonzero(self, x: tTensor) -> tuple[tTensor, ...]: return torch.nonzero(x, as_tuple=True)
    def flatten(self, x: tTensor) -> tTensor: return x.flatten()
    def reinterpret(self, x: tTensor, dtype: Any) -> tTensor: return x.view(dtype)
    def from_numpy(self, data: np.ndarray) -> tTensor: return torch.from_numpy(data).to(device=self.device)    
    def to_numpy(self, x: tTensor) -> np.ndarray: return x.detach().cpu().numpy()


class CupyMatOps(MatOps):
    """CuPy implementation of :class:`MatOps`.

    CuPy is optional. Instantiating this class is always safe, but calling an
    operation requires a CuPy installation compatible with the local CUDA
    runtime.
    """

    int32: ClassVar[Any] = None if cp is None else cp.int32
    int64: ClassVar[Any] = None if cp is None else cp.int64
    uint8: ClassVar[Any] = None if cp is None else cp.uint8
    uint16: ClassVar[Any] = None if cp is None else cp.uint16
    float64: ClassVar[Any] = None if cp is None else cp.float64
    float32: ClassVar[Any] = None if cp is None else cp.float32
    float16: ClassVar[Any] = None if cp is None else cp.float16
    device: MatDevice = MatDevice.CUDA # MatDevice.CUDA0

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def _context(self):
        return _cupy_device_context(self.device)

    def mat(self, data: Any, dtype: Any) -> Any:
        with self._context(): return _imp_cp().array(data, dtype=dtype)

    def numel(self, x: Any) -> int: return x.size
    def is_floating_point(self, x: Any) -> bool: return _imp_cp().issubdtype(x.dtype, _imp_cp().floating)
    def sin(self, x): return _imp_cp().sin(x)
    def cos(self, x): return _imp_cp().cos(x)
    
    def eye(self, size: int, dtype: Any) -> Any:
        with self._context(): return _imp_cp().eye(size, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Any) -> Any:
        with self._context(): return _imp_cp().ones(tuple(shape), dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Any) -> Any:
        with self._context(): return _imp_cp().zeros(tuple(shape), dtype=dtype)

    def hstack(self, arrays: Sequence[Any]) -> Any: return _imp_cp().hstack(tuple(arrays))
    def norm(self, x: Any) -> Any: return _imp_cp().linalg.norm(x)
    def dot(self, a: Any, b: Any) -> Any: return _imp_cp().dot(a, b)
    def cross(self, a: Any, b: Any) -> Any: return _imp_cp().cross(a, b)
    def matmul(self, a: Any, b: Any) -> Any: return _imp_cp().matmul(a, b)
    def inv(self, x: Any) -> Any: return _imp_cp().linalg.inv(x)

    def from_numpy(self, data: np.ndarray) -> Any:
        with self._context(): return _imp_cp().asarray(data)

    def to_numpy(self, x: Any) -> np.ndarray: return _imp_cp().asnumpy(x)
    def mean(self, x: Any, dim: int = 0) -> Any: return _imp_cp().mean(x, axis=dim)
    def median(self, x: Any, dim: int = 0) -> Any: return _imp_cp().median(x, axis=dim)
    def std(self, x: Any, dim: int = 0) -> Any: return _imp_cp().std(x, axis=dim, ddof=0)
    def max(self, x: Any, dim: Optional[int] = 0) -> Any: return _imp_cp().max(x, axis=dim)
    def min(self, x: Any, dim: Optional[int] = 0) -> Any: return _imp_cp().min(x, axis=dim)
    def nanmax(self, x: Any) -> Any: return _imp_cp().nanmax(x)
    def all(self, x: Any, dim: Optional[int] = None) -> Any: return _imp_cp().all(x, axis=dim)
    def isfinite(self, x: Any) -> Any: return _imp_cp().isfinite(x)
    def abs(self, x: Any) -> Any: return _imp_cp().abs(x)
    def floor(self, x: Any) -> Any: return _imp_cp().floor(x)
    def round(self, x: Any) -> Any: return _imp_cp().rint(x)
    def where(self, condition: Any, x: Any, y: Any) -> Any: return _imp_cp().where(condition, x, y)
    def flip(self, x: Any, dim: int) -> Any: return _imp_cp().flip(x, axis=dim)
    def stack(self, xs: Sequence[Any], dim: int = 0) -> Any: return _imp_cp().stack(tuple(xs), axis=dim)
    def cat(self, xs: Sequence[Any], dim: int = 0) -> Any: return _imp_cp().concatenate(tuple(xs), axis=dim)
    def reshape(self, x: Any, shape: Sequence[int]) -> Any: return _imp_cp().reshape(x, tuple(shape))
    def permute(self, x: Any, dims: Sequence[int]) -> Any: return _imp_cp().transpose(x, axes=tuple(dims))
    def copy_mat(self, x: Any) -> Any: return x.copy()
    def copyto(self, dst: Any, src: Any) -> Any:
        _imp_cp().copyto(dst, src)
        return dst
    def scatter_min(self, dst: Any, index: Any, src: Any) -> Any:
        _imp_cp().minimum.at(dst, index, src)
        return dst
    def logical_and(self, a: Any, b: Any) -> Any: return _imp_cp().logical_and(a, b)
    def logical_or(self, a: Any, b: Any) -> Any: return _imp_cp().logical_or(a, b)
    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any: return _imp_cp().clip(x, min_val, max_val)
    def astype(self, x: Any, dtype: Any) -> Any: return x.astype(dtype, copy=False)
    def astype_int32(self, x: Any) -> Any: return x.astype(_imp_cp().int32)
    def astype_int64(self, x: Any) -> Any: return x.astype(_imp_cp().int64)
    def astype_uint8(self, x: Any) -> Any: return x.astype(_imp_cp().uint8)
    def astype_float64(self, x: Any) -> Any: return x.astype(_imp_cp().float64)
    def astype_float32(self, x: Any) -> Any: return x.astype(_imp_cp().float32)
    def astype_float16(self, x: Any) -> Any: return x.astype(_imp_cp().float16)
    def nonzero(self, x: Any) -> tuple[Any, ...]: return _imp_cp().nonzero(x)
    def flatten(self, x: Any) -> Any: return x.flatten()
    def reinterpret(self, x: Any, dtype: Any) -> Any: return x.view(dtype)


class MatLib(str, enum.Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    CUPY = "cupy"

    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray): return MatLib.NUMPY
        if isinstance(data, tTensor): return MatLib.TORCH
        if cp is not None and isinstance(data, cp.ndarray): return MatLib.CUPY
        raise ValueError(f"Unsupported data type: {type(data)}")

TypeMap = {
    # Tuple order preserves the original public layout: NumPy, Torch, ctypes, CuPy.
    "uint8": (np.uint8, torch.uint8, ctypes.c_uint8, None if cp is None else cp.uint8),
    "int8": (np.int8, torch.int8, ctypes.c_int8, None if cp is None else cp.int8),
    "uint16": (
        np.uint16,
        getattr(torch, "uint16", None),
        ctypes.c_uint16,
        None if cp is None else cp.uint16,
    ),
    "int16": (np.int16, torch.int16, ctypes.c_int16, None if cp is None else cp.int16),
    "uint32": (
        np.uint32,
        getattr(torch, "uint32", None),
        ctypes.c_uint32,
        None if cp is None else cp.uint32,
    ),
    "int32": (np.int32, torch.int32, ctypes.c_int32, None if cp is None else cp.int32),
    "uint64": (
        np.uint64,
        getattr(torch, "uint64", None),
        ctypes.c_uint64,
        None if cp is None else cp.uint64,
    ),
    "int64": (np.int64, torch.int64, ctypes.c_int64, None if cp is None else cp.int64),
    "float16": (np.float16, torch.float16, None, None if cp is None else cp.float16),
    "float32": (np.float32, torch.float32, ctypes.c_float, None if cp is None else cp.float32),
    "float64": (np.float64, torch.float64, ctypes.c_double, None if cp is None else cp.float64),
    "bool": (np.bool_, torch.bool, ctypes.c_bool, None if cp is None else cp.bool_),
}

def to_type_name(dtype: Any) -> str:
    """
    Convert a NumPy / PyTorch / CuPy / ctypes dtype into the canonical string key
    used by TypeMap.

    Examples:
        np.uint8          -> "uint8"
        np.dtype("uint8") -> "uint8"
        torch.uint8       -> "uint8"
        cupy.uint8        -> "uint8"
        ctypes.c_uint8    -> "uint8"
        "uint8"           -> "uint8"
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    if isinstance(dtype, str):
        if dtype in TypeMap:
            return dtype
        raise KeyError(f"Unknown dtype name: {dtype!r}")

    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None

    for name, info in TypeMap.items():
        if np_dtype is not None and np_dtype == info[0]: return name
        if info[1] is not None and dtype is info[1]: return name
        if info[2] is not None and dtype is info[2]: return name
        if info[3] is not None and dtype is info[3]: return name
    raise TypeError(f"Unsupported dtype: {dtype!r}")


def to_np_type(dtype: Any) -> np.dtype:
    return TypeMap[to_type_name(dtype)][0]


def to_torch_type(dtype: Any) -> torch.dtype:
    result = TypeMap[to_type_name(dtype)][1]
    if result is None:
        raise TypeError(f"No PyTorch dtype mapping for {dtype!r}")
    return result


def to_cupy_type(dtype: Any) -> Any:
    _imp_cp()
    result = TypeMap[to_type_name(dtype)][3]
    if result is None:
        raise TypeError(f"No CuPy dtype mapping for {dtype!r}")
    return result


def to_ctypes_type(dtype: Any):
    result = TypeMap[to_type_name(dtype)][2]
    if result is None:
        raise TypeError(f"No ctypes dtype mapping for {dtype!r}")
    return result
