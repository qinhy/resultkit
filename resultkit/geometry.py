from __future__ import annotations

import enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from .arrays import as_box_array, as_points_array, to_jsonable


class BoxFormat(str, enum.Enum):
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"


class ScaleFormat(str, enum.Enum):
    ZERO_ONE = "01"
    RAW = "raw"


class BoundingBox(BaseModel):
    """One or more bounding boxes with format and scale metadata.

    Data is stored as an ``(N, 4)`` float32 array. A single ``(4,)`` input is
    accepted and normalized to ``(1, 4)``.
    """

    type: Literal["box"] = "box"
    data: np.ndarray = Field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    format: BoxFormat = BoxFormat.XYXY
    scale: ScaleFormat = ScaleFormat.RAW
    image_size: Optional[Tuple[int, int]] = None  # width, height
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Any) -> np.ndarray:
        return as_box_array(value)

    @field_serializer("data")
    def serialize_data(self, value: np.ndarray) -> list:
        return value.tolist()

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)

    @classmethod
    def xyxy(
        cls,
        data: Any,
        *,
        scale: ScaleFormat | str = ScaleFormat.RAW,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> "BoundingBox":
        return cls(data=data, format=BoxFormat.XYXY, scale=scale, image_size=image_size, **kwargs)

    @classmethod
    def xywh(
        cls,
        data: Any,
        *,
        scale: ScaleFormat | str = ScaleFormat.RAW,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> "BoundingBox":
        return cls(data=data, format=BoxFormat.XYWH, scale=scale, image_size=image_size, **kwargs)

    @classmethod
    def cxcywh(
        cls,
        data: Any,
        *,
        scale: ScaleFormat | str = ScaleFormat.RAW,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> "BoundingBox":
        return cls(data=data, format=BoxFormat.CXCYWH, scale=scale, image_size=image_size, **kwargs)

    def to_xyxy_array(self) -> np.ndarray:
        data = self.data.astype(np.float32, copy=True)
        if self.format == BoxFormat.XYXY:
            return data
        if self.format == BoxFormat.XYWH:
            x, y, w, h = data.T
            return np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)
        if self.format == BoxFormat.CXCYWH:
            cx, cy, w, h = data.T
            return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
        raise ValueError(f"unsupported box format: {self.format}")

    def to_array(self, format: BoxFormat | str) -> np.ndarray:
        """Return a copy of the box coordinates in the requested format."""
        target = BoxFormat(format)
        xyxy = self.to_xyxy_array()
        if target == BoxFormat.XYXY:
            return xyxy
        x1, y1, x2, y2 = xyxy.T
        w = x2 - x1
        h = y2 - y1
        if target == BoxFormat.XYWH:
            return np.stack([x1, y1, w, h], axis=1).astype(np.float32)
        if target == BoxFormat.CXCYWH:
            return np.stack([x1 + w / 2, y1 + h / 2, w, h], axis=1).astype(np.float32)
        raise ValueError(f"unsupported target format: {target}")

    def to(self, format: BoxFormat | str) -> "BoundingBox":
        """Return a new Box converted to the requested coordinate format."""
        target = BoxFormat(format)
        return self.model_copy(update={"data": self.to_array(target), "format": target})

    def area(self) -> np.ndarray:
        """Return the area of each box in the current scale."""
        x1, y1, x2, y2 = self.to_xyxy_array().T
        return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    def clip(self, width: Optional[float] = None, height: Optional[float] = None) -> "BoundingBox":
        """Clip boxes to an image or normalized range.

        If width/height are omitted and scale is ``01``, the range [0, 1] is used.
        If width/height are omitted and image_size exists, that size is used.
        """
        if width is None or height is None:
            if self.scale == ScaleFormat.ZERO_ONE:
                width, height = 1.0, 1.0
            elif self.image_size is not None:
                width, height = self.image_size
            else:
                raise ValueError("clip requires width/height, normalized scale, or image_size")

        xyxy = self.to_xyxy_array()
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
        converted = BoundingBox.xyxy(xyxy, scale=self.scale, image_size=self.image_size, metadata=self.metadata)
        return converted.to(self.format)

    def normalize(self, width: Optional[float] = None, height: Optional[float] = None) -> "BoundingBox":
        """Convert raw pixel coordinates to normalized [0, 1] coordinates."""
        if self.scale == ScaleFormat.ZERO_ONE:
            return self.model_copy(deep=True)
        if width is None or height is None:
            if self.image_size is None:
                raise ValueError("normalize requires width/height or image_size")
            width, height = self.image_size
        xyxy = self.to_xyxy_array()
        xyxy[:, [0, 2]] /= float(width)
        xyxy[:, [1, 3]] /= float(height)
        return BoundingBox.xyxy(xyxy, scale=ScaleFormat.ZERO_ONE, image_size=(int(width), int(height)), metadata=self.metadata).to(self.format)

    def denormalize(self, width: Optional[float] = None, height: Optional[float] = None) -> "BoundingBox":
        """Convert normalized [0, 1] coordinates to raw pixel coordinates."""
        if self.scale == ScaleFormat.RAW:
            return self.model_copy(deep=True)
        if width is None or height is None:
            if self.image_size is None:
                raise ValueError("denormalize requires width/height or image_size")
            width, height = self.image_size
        xyxy = self.to_xyxy_array()
        xyxy[:, [0, 2]] *= float(width)
        xyxy[:, [1, 3]] *= float(height)
        return BoundingBox.xyxy(xyxy, scale=ScaleFormat.RAW, image_size=(int(width), int(height)), metadata=self.metadata).to(self.format)

    def iou(self, other: "BoundingBox") -> np.ndarray:
        """Pairwise IoU matrix with another Box object."""
        a = self.to_xyxy_array()
        b = other.to_xyxy_array()
        if self.scale != other.scale:
            raise ValueError("cannot compute IoU between different scales")
        a_area = self.area()[:, None]
        b_area = other.area()[None, :]
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = a_area + b_area - inter
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class Polygon(BaseModel):
    """A polygon represented by an ``(N, 2)`` array of points."""

    type: Literal["polygon"] = "polygon"
    points: np.ndarray
    scale: ScaleFormat = ScaleFormat.RAW
    image_size: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, value: Any) -> np.ndarray:
        return as_points_array(value)

    @field_serializer("points")
    def serialize_points(self, value: np.ndarray) -> list:
        return value.tolist()

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)


class Keypoints(BaseModel):
    """Keypoints as ``(N, 2)`` or ``(N, 3)`` array: x, y, optional score/visibility."""

    type: Literal["keypoints"] = "keypoints"
    points: np.ndarray
    names: Optional[List[str]] = None
    scale: ScaleFormat = ScaleFormat.RAW
    image_size: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, value: Any) -> np.ndarray:
        return as_points_array(value, allow_score=True)

    @field_serializer("points")
    def serialize_points(self, value: np.ndarray) -> list:
        return value.tolist()

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)


class Mask(BaseModel):
    """A 2D binary or probabilistic mask."""

    type: Literal["mask"] = "mask"
    data: np.ndarray
    image_size: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim != 2:
            raise ValueError("mask data must be a 2D array")
        return arr

    @field_serializer("data")
    def serialize_data(self, value: np.ndarray) -> list:
        return value.tolist()

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)


class Vector(BaseModel):
    """A generic numeric vector, useful for embeddings or logits."""

    type: Literal["vector"] = "vector"
    data: np.ndarray
    names: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector data must be a 1D array")
        return arr

    @field_serializer("data")
    def serialize_data(self, value: np.ndarray) -> list:
        return value.tolist()

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)
