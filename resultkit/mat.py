from __future__ import annotations

import enum
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

ArrayLike = Union[np.ndarray,torch.Tensor]

class MatOps(BaseModel):
    """Small backend-neutral matrix/tensor operation interface.

    Subclasses implement the same common operations for NumPy arrays and
    PyTorch tensors. This is intentionally lightweight; it is useful when
    result payloads may be produced on either backend but downstream code wants
    a consistent operation surface.
    """

    int32: ClassVar[Any] = None
    uint8: ClassVar[Any] = None
    float32: ClassVar[Any] = None
    float16: ClassVar[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def mat(self, data: Any, dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def hstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        raise NotImplementedError

    def norm(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def dot(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def cross(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def to_numpy(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def mean(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def median(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def std(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def max(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def min(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def abs(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def stack(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def cat(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def reshape(self, x: ArrayLike, shape: Sequence[int]) -> ArrayLike:
        raise NotImplementedError

    def copy_mat(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def logical_and(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def logical_or(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def clip(self, x: ArrayLike, min_val: Any, max_val: Any) -> ArrayLike:
        raise NotImplementedError

    def astype_int32(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_uint8(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_float32(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_float16(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def nonzero(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError
    
    def flatten(self, x: ArrayLike) -> ArrayLike:
        return NotImplementedError

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
    uint8: ClassVar[Any] = np.uint8
    float32: ClassVar[Any] = np.float32
    float16: ClassVar[Any] = np.float16

    def mat(self, data: Any, dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.eye(size, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def hstack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.hstack(arrays)

    def norm(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x)

    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def mean(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.mean(x, axis=dim)

    def median(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.median(x, axis=dim)

    def std(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.std(x, axis=dim)

    def max(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.max(x, axis=dim)

    def min(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.min(x, axis=dim)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def stack(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.stack(xs, axis=dim)

    def cat(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.concatenate(xs, axis=dim)

    def reshape(self, x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(x, shape)

    def copy_mat(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def logical_and(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_and(a, b)

    def logical_or(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_or(a, b)

    def clip(self, x: np.ndarray, min_val: Any, max_val: Any) -> np.ndarray:
        return np.clip(x, min_val, max_val)

    def astype_int32(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.int32)

    def astype_uint8(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.uint8)

    def astype_float32(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def astype_float16(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float16)

    def nonzero(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        return np.nonzero(x)
    
    def flatten(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()

class TorchMatOps(MatOps):
    """PyTorch implementation of :class:`MatOps`."""

    int32: ClassVar[Any] = None if torch is None else torch.int32
    uint8: ClassVar[Any] = None if torch is None else torch.uint8
    float32: ClassVar[Any] = None if torch is None else torch.float32
    float16: ClassVar[Any] = None if torch is None else torch.float16

    def mat(self, data: torch.Tensor, dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=device)

    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.eye(size, dtype=dtype, device=device)

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.ones(tuple(shape), dtype=dtype, device=device)

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.zeros(tuple(shape), dtype=dtype, device=device)

    def hstack(self, arrays: Sequence[Any]) -> torch.Tensor:
        return torch.cat(tuple(arrays), dim=1)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x)

    def dot(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.dot(a, b)

    def cross(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.cross(a, b)

    def matmul(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.matmul(a, b)

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def mean(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.mean(x, dim=dim)

    def median(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.median(x, dim=dim).values

    def std(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.std(x, dim=dim, unbiased=False)

    def max(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.max(x, dim=dim).values

    def min(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.min(x, dim=dim).values

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def stack(self, xs: Sequence[Any], dim: int = 0) -> torch.Tensor:
        return torch.stack(tuple(xs), dim=dim)

    def cat(self, xs: Sequence[Any], dim: int = 0) -> torch.Tensor:
        return torch.cat(tuple(xs), dim=dim)

    def reshape(self, x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return x.reshape(tuple(shape))

    def copy_mat(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def logical_and(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.logical_and(a, b)

    def logical_or(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.logical_or(a, b)

    def clip(self, x: torch.Tensor, min_val: Any, max_val: Any) -> torch.Tensor:
        return torch.clamp(x, min=min_val, max=max_val)

    def astype_int32(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.int32)

    def astype_uint8(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.uint8)

    def astype_float32(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float32)

    def astype_float16(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float16)

    def nonzero(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x)

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()

class MatLib(str, enum.Enum):
    NUMPY = "numpy"
    TORCH = "torch"

    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray):
            return MatLib.NUMPY
        if isinstance(data, torch.Tensor):
            return MatLib.TORCH
        raise ValueError(f"Unsupported data type: {type(data)}")

class MatDevice(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"
    # MPS = "mps"
    
    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray):
            return MatDevice.CPU
        if isinstance(data, torch.Tensor):
            return MatDevice.CPU if data.device.type == "cpu" else MatDevice.CUDA
        raise ValueError(f"Unsupported data type: {type(data)}")

class DataType(str, enum.Enum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_dtype(cls, dtype: Any, *, strict: bool = False) -> "DataType | None":
        """Return the matching DataType for a NumPy/Torch/Python dtype.

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
        """Infer DataType from an ndarray, tensor, dtype object, enum, or dtype string.

        Existing calls like DataType.which(data) still work. Unsupported inputs return
        None by default; pass strict=True to raise a TypeError instead.
        """
        if isinstance(data, np.ndarray):
            return cls.from_dtype(data.dtype, strict=strict)

        if isinstance(data, torch.Tensor):
            return cls.from_dtype(data.dtype, strict=strict)

        return cls.from_dtype(data, strict=strict)