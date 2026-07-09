from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from common import *

from iox2_jsonrpc import EmptyParams, RpcModel
from store.custom_record_store import CustomRecord
from server_yolo import YoloDetectResult

_THIS_DIR = Path(__file__).absolute().parent
for path in (
    _THIS_DIR,
    _THIS_DIR.parent,
    Path(os.path.dirname(os.path.dirname(_THIS_DIR.parent))),
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.append(path_text)

from pcd_utils import (  # noqa: E402
    DEFAULT_CALIBRATION,
    StereoRgbCalibration,
    compute_disparity_sgbm,
    cv2,
    disparity_to_points_rectified,
    points_left_to_rgb_depth,
    read_image,
    rectified_left_to_original_left,
    rectify_stereo_pair,
    rgb8,
    rgb_depth_to_points_rgb,
    save_point_cloud,
    stereo_rgb_to_colored_point_cloud,
    transform_points,
)

try:  # noqa: E402
    from pcd_dnn_utils import FastFoundationStereoDisparity, stereo_rgb_to_colored_point_cloud_dnn
    _DNN_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # noqa: E402
    FastFoundationStereoDisparity = Any  # type: ignore[misc,assignment]
    stereo_rgb_to_colored_point_cloud_dnn = None  # type: ignore[assignment]
    _DNN_IMPORT_ERROR = exc

Resolution = tuple[int, int]
ColorOrder = Literal["RGB", "BGR"]
DepthBackend = Literal["sgbm", "dnn"]
OutputFrame = Literal["left", "left_rectified"]
SegmentOutputFrame = Literal["rgb", "left"]
TranslationUnit = Literal["m", "cm", "mm"]
YoloOverlapPolicy = Literal["highest_confidence", "first", "none"]
Matrix3x3 = tuple[tuple[float, float, float], ...]
Matrix4x4 = tuple[tuple[float, float, float, float], ...]
DistortionCoefficients = tuple[float, ...]

_DEFAULT_CALIBRATION = DEFAULT_CALIBRATION
_DNN_CACHE_KEY_FIELDS = ("repo_dir", "model_path", "model_dir", "device", "valid_iters", "max_disp", "hiera")
_YOLO_CACHE_KEY_FIELDS = ("model_path",)


def _as_resolution(value: Any) -> Resolution:
    return int(value[0]), int(value[1])


def _as_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _as_float_tuple(value: Any) -> DistortionCoefficients:
    return tuple(float(item) for item in value)


def _default_calibration_field(key: str, converter: Any = _as_matrix) -> Any:
    return Field(default=converter(_DEFAULT_CALIBRATION[key]))


def _model_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)

    for method_name in ("model_dump", "dict"):
        if hasattr(value, method_name):
            return getattr(value, method_name)()

    raise TypeError(f"Expected a dict/RpcModel-compatible object, got {type(value)!r}")


def _model_field_names(model_type: type[BaseModel]) -> tuple[str, ...]:
    fields = getattr(model_type, "model_fields", None) or getattr(model_type, "__fields__", {})
    return tuple(fields.keys())


def _depth_statistics(points_m: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if not points_m.size:
        return None, None, None

    depth_values = np.asarray(points_m, dtype=np.float64)[:, 2]
    depth_values = depth_values[np.isfinite(depth_values)]
    if not depth_values.size:
        return None, None, None

    return tuple(float(value) for value in (depth_values.min(), depth_values.max(), depth_values.mean()))


def _read_image_or_npy(path: str | Path, *, color: bool) -> np.ndarray:
    image_path = Path(path).expanduser()

    if image_path.suffix.lower() == ".npy":
        array = np.load(image_path, allow_pickle=False)

        if color:
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError(
                    f"Expected color .npy image with shape HxWx3 or HxWx4, "
                    f"got {array.shape} from {image_path}"
                )
            array = array[:, :, :3]
        else:
            if array.ndim == 2:
                pass
            elif array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            else:
                raise ValueError(
                    f"Expected grayscale .npy image with shape HxW or HxWx1, "
                    f"got {array.shape} from {image_path}"
                )

        return np.ascontiguousarray(array)

    return read_image(image_path, color=color)


def _save_cloud_npz(path: str | Path, cloud: Any) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "points_m": np.asarray(cloud.points_m),
        "colors_rgb": np.asarray(cloud.colors_rgb),
    }

    if cloud.disparity is not None:
        arrays["disparity"] = np.asarray(cloud.disparity)

    np.savez(output_path, **arrays)
    return output_path


def _torch_to_numpy(x: Any) -> np.ndarray:
    """Accept torch tensors, numpy arrays, or list-like values."""
    if x is None:
        return np.asarray([])
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _safe_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _safe_filename_text(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _resize_yolo_mask_to_rgb(mask: np.ndarray, rgb_hw: tuple[int, int], threshold: float) -> np.ndarray:
    """Resize one YOLO mask to the RGB image geometry using nearest neighbor."""
    rgb_h, rgb_w = rgb_hw
    m = np.asarray(mask, np.float32)

    if m.shape != (rgb_h, rgb_w):
        c = cv2()
        m = c.resize(m, (rgb_w, rgb_h), interpolation=c.INTER_NEAREST)

    return m > float(threshold)


def _segment_meta_dict(seg: "DetectSegment3D") -> dict[str, Any]:
    return {
        "instance_id": int(seg.instance_id),
        "class_id": int(seg.class_id),
        "class_name": seg.class_name,
        "confidence": float(seg.confidence),
        "bbox_xyxy_rgb": [float(v) for v in seg.bbox_xyxy_rgb],
        "mask_area_px": int(seg.mask_area_px),
        "point_count": int(len(seg.points_m)),
        "centroid_m": seg.centroid_m.astype(float).tolist(),
        "aabb_min_m": seg.aabb_min_m.astype(float).tolist(),
        "aabb_max_m": seg.aabb_max_m.astype(float).tolist(),
        "output_frame": seg.output_frame,
        "pcd_path": seg.pcd_path,
        "pixels_path": seg.pixels_path,
        "meta_path": seg.meta_path,
    }


def _save_segments_npz(path: str | Path, result: "DetectSegments3D") -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "points_m": np.asarray(result.points_m),
        "colors_rgb": np.asarray(result.colors_rgb),
        "pixels_rgb": np.asarray(result.pixels_rgb),
        "instance_ids": np.asarray(result.instance_ids),
        "class_ids": np.asarray(result.class_ids),
        "confidences": np.asarray(result.confidences),
        "instance_map": np.asarray(result.instance_map),
    }
    if result.depth_rgb_m is not None:
        arrays["depth_rgb_m"] = np.asarray(result.depth_rgb_m)
    if result.disparity is not None:
        arrays["disparity"] = np.asarray(result.disparity)

    np.savez(output_path, **arrays)
    return output_path


def _result_segments_to_summary(segments: list["DetectSegment3D"]) -> list["YoloSegmentSummary"]:
    summaries = []
    for seg in segments:
        summaries.append(
            YoloSegmentSummary(
                instance_id=int(seg.instance_id),
                class_id=int(seg.class_id),
                class_name=seg.class_name,
                confidence=float(seg.confidence),
                bbox_xyxy_rgb=tuple(float(v) for v in seg.bbox_xyxy_rgb),
                mask_area_px=int(seg.mask_area_px),
                point_count=int(len(seg.points_m)),
                depth_min_m=float(np.nanmin(seg.points_m[:, 2])) if len(seg.points_m) else None,
                depth_max_m=float(np.nanmax(seg.points_m[:, 2])) if len(seg.points_m) else None,
                depth_mean_m=float(np.nanmean(seg.points_m[:, 2])) if len(seg.points_m) else None,
                centroid_m=tuple(float(v) for v in seg.centroid_m),
                aabb_min_m=tuple(float(v) for v in seg.aabb_min_m),
                aabb_max_m=tuple(float(v) for v in seg.aabb_max_m),
                output_frame=seg.output_frame,
                pcd_path=seg.pcd_path,
                pixels_path=seg.pixels_path,
                meta_path=seg.meta_path,
            )
        )
    return summaries


@dataclass(frozen=True)
class DetectSegment3D:
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_rgb: tuple[float, float, float, float]
    points_m: np.ndarray
    colors_rgb: np.ndarray
    pixels_rgb: np.ndarray
    mask_area_px: int
    centroid_m: np.ndarray
    aabb_min_m: np.ndarray
    aabb_max_m: np.ndarray
    output_frame: SegmentOutputFrame
    pcd_path: str | None = None
    pixels_path: str | None = None
    meta_path: str | None = None


@dataclass(frozen=True)
class DetectSegments3D:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    pixels_rgb: np.ndarray
    instance_ids: np.ndarray
    class_ids: np.ndarray
    confidences: np.ndarray
    instance_map: np.ndarray
    segments: list[DetectSegment3D]
    depth_rgb_m: np.ndarray | None
    disparity: np.ndarray | None
    rectification: Any | None
    output_frame: SegmentOutputFrame


class DepthBaseModel(RpcModel):
    service: Literal["jrpc"] = "jrpc"


class DepthCalibrationParams(DepthBaseModel):
    source_translation_unit: TranslationUnit = "cm"
    rgb_resolution: Resolution # = _default_calibration_field("rgb_resolution", _as_resolution)
    left_resolution: Resolution # = _default_calibration_field("left_resolution", _as_resolution)
    right_resolution: Resolution # = _default_calibration_field("right_resolution", _as_resolution)
    rgb_intrinsics: Matrix3x3 # = _default_calibration_field("rgb_intrinsics")
    left_intrinsics: Matrix3x3 # = _default_calibration_field("left_intrinsics")
    right_intrinsics: Matrix3x3 # = _default_calibration_field("right_intrinsics")
    left_to_right_extrinsics: Matrix4x4 # = _default_calibration_field("left_to_right_extrinsics")
    left_to_rgb_extrinsics: Matrix4x4 # = _default_calibration_field("left_to_rgb_extrinsics")
    rgb_distortion: DistortionCoefficients # = _default_calibration_field("rgb_distortion", _as_float_tuple)
    left_distortion: DistortionCoefficients # = _default_calibration_field("left_distortion", _as_float_tuple)
    right_distortion: DistortionCoefficients # = _default_calibration_field("right_distortion", _as_float_tuple)


class SetDepthCalibrationResult(DepthBaseModel):
    configured: bool
    source_translation_unit: TranslationUnit
    rgb_resolution: Resolution
    left_resolution: Resolution
    right_resolution: Resolution
    stereo_baseline_m: float
    stereo_baseline_cm: float


class BackendOverrides(BaseModel):
    backend: DepthBackend | None = None
    repo_dir: str | None = None
    model_path: str | None = None
    model_dir: str | None = None
    device: str | None = None
    valid_iters: int | None = None
    max_disp: int | None = None
    hiera: bool | None = None
    model_scale: float | None = None
    stereo_input_color_order: ColorOrder | None = None
    remove_invisible: bool | None = None


BACKEND_KEYS = _model_field_names(BackendOverrides)


class BackendParams(BackendOverrides):
    backend: DepthBackend = "sgbm"
    device: str = "cuda"
    valid_iters: int = 8
    max_disp: int = 192
    hiera: bool = False
    model_scale: float = 1.0
    stereo_input_color_order: ColorOrder = "RGB"
    remove_invisible: bool = True


class BackendStatusResult(BackendParams):
    configured: bool = True
    predictor_loaded: bool = False
    dnn_available: bool = True
    dnn_error: str | None = None


class ToPcdParams(BackendOverrides):
    db_record: CustomRecord = CustomRecord.empty()
    left_path: str = ""
    right_path: str = ""
    rgb_path: str = ""
    output_pcd_path: str = "colored_cloud.pcd"
    calibration: DepthCalibrationParams | None = None
    input_color_order: ColorOrder = "BGR"
    rgb_image_is_undistorted: bool = False
    alpha: float = 0.0
    max_depth_m: float | None = 5.0
    stride: int = Field(default=1, ge=1)
    output_frame: OutputFrame = "left"
    save_binary_pcd: bool = True
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5

    def model_post_init(self, context):
        if not self.db_record.is_empty() and self.db_record.is_stereo:
            left_path = self.db_record.listup_left_image_paths
            right_path = self.db_record.listup_right_image_paths
            if len(left_path)!=1 or len(right_path)!=1:
                raise ValueError("Invalid left_path:{left_path}, right_path:{right_path}")
            self.left_path = str(left_path[0])
            self.right_path = str(right_path[0])
            self.rgb_path = str(left_path[0]).replace("left", "rgb")
            self.output_pcd_path = str(self.db_record.expect_pcd_path)            
            with open(self.db_record.expect_stereo_calib) as f:
                calib = json.load(f)
                if "calibration" in calib:
                    calib = calib["calibration"]
                self.calibration = DepthCalibrationParams.model_validate(calib,extra="ignore")
        return super().model_post_init(context)
    

class ToPcdResult(DepthBaseModel):
    backend: DepthBackend
    output_path: str
    point_count: int
    color_count: int
    size_bytes: int
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None
    calibration: str | None = None


class Ros2PublishParams(DepthBaseModel):
    ros2_node_name: str = "depth_segment_publisher"
    ros2_topic_prefix: str = "/perception/segments"
    ros2_frame_id: str = "camera_rgb_optical_frame"
    ros2_include_depth: bool = True
    ros2_include_markers: bool = True
    ros2_pretty_json: bool = False


class Ros2PublishResult(DepthBaseModel):
    published: bool
    point_count: int = 0
    segment_count: int = 0
    frame_id: str | None = None
    topics: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class YoloSegmentSummary(BaseModel):
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_rgb: tuple[float, float, float, float]
    mask_area_px: int
    point_count: int
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    centroid_m: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    output_frame: SegmentOutputFrame
    pcd_path: str | None = None
    pixels_path: str | None = None
    meta_path: str | None = None


class ToYoloSegmentsParams(BackendOverrides):
    left_path: str
    right_path: str
    rgb_path: str
    output_dir: str = "detect_segments_out"
    frame_name: str = "frame"
    calibration: DepthCalibrationParams | None = None
    input_color_order: ColorOrder = "BGR"
    rgb_image_is_undistorted: bool = False

    # Depth parameters
    alpha: float = 0.0
    max_depth_m: float | None = 10.0
    output_frame: SegmentOutputFrame = "rgb"
    save_binary_pcd: bool = True
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    splat_px: int = Field(default=1, ge=0)


class ToDetectSegmentsResult(DepthBaseModel):
    backend: DepthBackend
    output_dir: str
    frame_name: str
    output_frame: SegmentOutputFrame
    point_count: int
    segment_count: int
    instance_map_path: str | None = None
    depth_rgb_path: str | None = None
    combined_npz_path: str | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    ros2: Ros2PublishResult | None = None
    segments: list[YoloSegmentSummary] = Field(default_factory=list)


@dataclass
class DepthController:
    service_name: str = "jrpc"
    controller_name: str = "pcd"
    calibration_params: DepthCalibrationParams | None = field(default=None, init=False, repr=False)
    backend_params: BackendParams = field(default_factory=BackendParams)
    _dnn_predictor: FastFoundationStereoDisparity | None = field(default=None, init=False, repr=False)
    _dnn_predictor_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _yolo_model: Any | None = field(default=None, init=False, repr=False)
    _yolo_model_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _last_yolo_segments: DetectSegments3D | None = field(default=None, init=False, repr=False)
    _ros2_node: Any | None = field(default=None, init=False, repr=False)
    _ros2_publishers: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _ros2_publishers_key: tuple[str, str] | None = field(default=None, init=False, repr=False)

    def _build_calibration(self, params: DepthCalibrationParams | None = None) -> StereoRgbCalibration:
        calibration_data = _model_to_dict(params or self.calibration_params)
        calibration_data.pop("service", None)
        translation_unit = calibration_data.pop("source_translation_unit", "cm")
        return StereoRgbCalibration.from_dict(calibration_data, source_translation_unit=translation_unit)

    def _calibration_result(self, calibration: StereoRgbCalibration) -> SetDepthCalibrationResult:
        result_fields = (
            "source_translation_unit",
            "rgb_resolution",
            "left_resolution",
            "right_resolution",
            "stereo_baseline_m",
            "stereo_baseline_cm",
        )
        return SetDepthCalibrationResult(
            configured=True,
            **{field_name: getattr(calibration, field_name) for field_name in result_fields},
        )

    def _backend_result(self) -> BackendStatusResult:
        return BackendStatusResult(
            configured=True,
            predictor_loaded=self._dnn_predictor is not None,
            dnn_available=_DNN_IMPORT_ERROR is None,
            dnn_error=None if _DNN_IMPORT_ERROR is None else str(_DNN_IMPORT_ERROR),
            **{key: getattr(self.backend_params, key) for key in BACKEND_KEYS},
        )

    def _dnn_cache_key(self, backend: BackendParams) -> tuple[Any, ...]:
        values = [getattr(backend, key) for key in _DNN_CACHE_KEY_FIELDS]
        values[4] = int(values[4])
        values[5] = int(values[5])
        values[6] = bool(values[6])
        return tuple(values)

    def _get_dnn_predictor(self, backend: BackendParams):
        if _DNN_IMPORT_ERROR is not None:
            raise ImportError("DNN depth backend is unavailable; could not import pcd_dnn_utils") from _DNN_IMPORT_ERROR

        cache_key = self._dnn_cache_key(backend)
        if self._dnn_predictor is None or self._dnn_predictor_key != cache_key:
            self._dnn_predictor = FastFoundationStereoDisparity(
                repo_dir=backend.repo_dir,
                model_path=backend.model_path,
                model_dir=backend.model_dir,
                device=backend.device,
                valid_iters=int(backend.valid_iters),
                max_disp=int(backend.max_disp),
                hiera=bool(backend.hiera),
            )
            self._dnn_predictor_key = cache_key

        return self._dnn_predictor

    def _effective_backend(self, params: Any) -> BackendParams:
        backend_data = {key: getattr(self.backend_params, key) for key in BACKEND_KEYS}
        override_data = _model_to_dict(params)
        backend_data.update({key: override_data[key] for key in BACKEND_KEYS if override_data.get(key) is not None})
        return BackendParams(**backend_data)

    def _compute_rgb_aligned_depth(
        self,
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        rgb_image: np.ndarray,
        calibration: StereoRgbCalibration,
        backend: BackendParams,
        input_color_order: ColorOrder,
        rgb_image_is_undistorted: bool,
        alpha: float,
        min_disparity: int,
        num_disparities: int,
        block_size: int,
        max_depth_m: float | None,
        splat_px: int,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        if backend.backend == "sgbm":
            left_rect, right_rect, rect = rectify_stereo_pair(
                left_image,
                right_image,
                calibration,
                alpha=alpha,
            )
            disparity = compute_disparity_sgbm(
                left_rect,
                right_rect,
                min_disparity=min_disparity,
                num_disparities=num_disparities,
                block_size=block_size,
            )
        elif backend.backend == "dnn":
            if stereo_rgb_to_colored_point_cloud_dnn is None:
                raise ImportError("DNN depth backend is unavailable; could not import pcd_dnn_utils") from _DNN_IMPORT_ERROR
            cloud = stereo_rgb_to_colored_point_cloud_dnn(
                left_image=left_image,
                right_image=right_image,
                rgb_image=rgb_image,
                calibration=calibration,
                output_path=None,
                min_disparity=float(min_disparity),
                max_depth_m=max_depth_m,
                stride=1,
                output_frame="left",
                save_binary_pcd=True,
                input_color_order=input_color_order,
                alpha=alpha,
                rgb_image_is_undistorted=rgb_image_is_undistorted,
                disparity_predictor=self._get_dnn_predictor(backend),
                model_scale=float(backend.model_scale),
                stereo_input_color_order=backend.stereo_input_color_order,
                remove_invisible=backend.remove_invisible,
            )
            if cloud.disparity is None or cloud.rectification is None:
                raise RuntimeError("DNN depth result did not include disparity and rectification")
            disparity = cloud.disparity
            rect = cloud.rectification
        else:
            raise ValueError(f"Unsupported backend: {backend.backend}")

        rgb_h, rgb_w = np.asarray(rgb_image).shape[:2]
        points_rect, _xy_left_rect = disparity_to_points_rectified(
            disparity,
            rect,
            min_disparity=max(0.5, float(min_disparity)),
            max_depth_m=max_depth_m,
            stride=1,
        )

        if len(points_rect) == 0:
            return np.full((rgb_h, rgb_w), np.nan, np.float64), disparity, rect

        points_left = rectified_left_to_original_left(points_rect, rect)
        depth_rgb_m, _valid_depth_rgb = points_left_to_rgb_depth(
            points_left,
            rgb_image,
            calibration,
            rgb_image_is_undistorted=rgb_image_is_undistorted,
            splat_px=splat_px,
        )
        return depth_rgb_m, disparity, rect

    def _detect_result_to_segments(
        self,
        *,
        yolo_result: Any,
        depth_rgb_m: np.ndarray,
        disparity: np.ndarray | None,
        rectification: Any | None,
        rgb_image: np.ndarray,
        calibration: StereoRgbCalibration,
        output_dir: Path | None,
        frame_name: str,
        input_color_order: ColorOrder,
        output_frame: SegmentOutputFrame,
        rgb_image_is_undistorted: bool,
        mask_threshold: float,
        overlap_policy: YoloOverlapPolicy,
        min_points: int,
        save_pcd: bool,
        save_pixels: bool,
        save_meta: bool,
        save_binary_pcd: bool,
    ) -> DetectSegments3D:
        rgb_arr = np.asarray(rgb_image)
        rgb_h, rgb_w = rgb_arr.shape[:2]

        masks_obj = getattr(yolo_result, "masks", None)
        boxes_obj = getattr(yolo_result, "boxes", None)

        empty = DetectSegments3D(
            points_m=np.empty((0, 3), np.float64),
            colors_rgb=np.empty((0, 3), np.uint8),
            pixels_rgb=np.empty((0, 2), np.int32),
            instance_ids=np.empty((0,), np.int32),
            class_ids=np.empty((0,), np.int32),
            confidences=np.empty((0,), np.float32),
            instance_map=np.zeros((rgb_h, rgb_w), np.uint32),
            segments=[],
            depth_rgb_m=depth_rgb_m,
            disparity=disparity,
            rectification=rectification,
            output_frame=output_frame,
        )

        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return empty

        masks = _torch_to_numpy(masks_obj.data)
        if masks.ndim == 2:
            masks = masks[None, :, :]

        n = int(masks.shape[0]) if masks.ndim == 3 else 0
        if n == 0 or not np.isfinite(depth_rgb_m).any():
            return empty

        if boxes_obj is not None:
            class_ids_yolo = _torch_to_numpy(getattr(boxes_obj, "cls", None)).astype(np.int32).reshape(-1)
            confs_yolo = _torch_to_numpy(getattr(boxes_obj, "conf", None)).astype(np.float32).reshape(-1)
            bboxes_yolo = _torch_to_numpy(getattr(boxes_obj, "xyxy", None)).astype(np.float32).reshape(-1, 4)
        else:
            class_ids_yolo = np.zeros((n,), np.int32)
            confs_yolo = np.ones((n,), np.float32)
            bboxes_yolo = np.full((n, 4), np.nan, np.float32)

        class_ids_yolo = np.resize(class_ids_yolo, n)
        confs_yolo = np.resize(confs_yolo, n)
        if bboxes_yolo.shape[0] != n:
            bboxes_yolo = np.resize(bboxes_yolo, (n, 4))

        names = getattr(yolo_result, "names", None)

        points_rgb, xy_rgb = rgb_depth_to_points_rgb(
            depth_rgb_m,
            calibration,
            rgb_image_is_undistorted=rgb_image_is_undistorted,
        )
        if len(points_rgb) == 0:
            return empty

        x = xy_rgb[:, 0]
        y = xy_rgb[:, 1]
        colors_all = rgb8(rgb_arr[y, x, :3], input_color_order)

        if output_frame == "rgb":
            points_all = points_rgb
        elif output_frame == "left":
            points_all = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))
        else:
            raise ValueError("output_frame must be 'rgb' or 'left'")

        bool_masks = [
            _resize_yolo_mask_to_rgb(masks[i], (rgb_h, rgb_w), mask_threshold)
            for i in range(n)
        ]

        instance_map = np.zeros((rgb_h, rgb_w), np.uint32)
        if overlap_policy == "highest_confidence":
            score_map = np.full((rgb_h, rgb_w), -np.inf, np.float32)
            for i, m in enumerate(bool_masks):
                inst_id = i + 1
                conf = float(confs_yolo[i])
                update = m & (conf >= score_map)
                instance_map[update] = inst_id
                score_map[update] = conf
        elif overlap_policy == "first":
            for i, m in enumerate(bool_masks):
                inst_id = i + 1
                update = m & (instance_map == 0)
                instance_map[update] = inst_id
        elif overlap_policy == "none":
            for i, m in enumerate(bool_masks):
                inst_id = i + 1
                update = m & (instance_map == 0)
                instance_map[update] = inst_id
        else:
            raise ValueError("overlap_policy must be 'highest_confidence', 'first', or 'none'")

        seg_dir = None
        if output_dir is not None:
            seg_dir = output_dir / "segments"
            seg_dir.mkdir(parents=True, exist_ok=True)

        segments: list[DetectSegment3D] = []
        combined_points: list[np.ndarray] = []
        combined_colors: list[np.ndarray] = []
        combined_pixels: list[np.ndarray] = []
        combined_instance_ids: list[np.ndarray] = []
        combined_class_ids: list[np.ndarray] = []
        combined_confidences: list[np.ndarray] = []

        for i, m in enumerate(bool_masks):
            inst_id = i + 1
            class_id = int(class_ids_yolo[i])
            class_name = _safe_class_name(names, class_id)
            conf = float(confs_yolo[i])
            bbox = tuple(float(v) for v in bboxes_yolo[i].tolist())

            if overlap_policy == "none":
                keep = m[y, x]
            else:
                keep = instance_map[y, x] == inst_id

            seg_points = points_all[keep]
            seg_colors = colors_all[keep]
            seg_pixels = xy_rgb[keep]

            if len(seg_points) < int(min_points):
                continue

            centroid = np.nanmean(seg_points, axis=0)
            aabb_min = np.nanmin(seg_points, axis=0)
            aabb_max = np.nanmax(seg_points, axis=0)

            pcd_path = None
            pixels_path = None
            meta_path = None
            safe_name = _safe_filename_text(class_name)
            stem = f"{frame_name}_obj_{i:03d}_id{inst_id:03d}_{safe_name}_conf{conf:.2f}"

            if seg_dir is not None:
                if save_pcd:
                    pcd_file = seg_dir / f"{stem}.pcd"
                    save_point_cloud(pcd_file, seg_points, seg_colors, binary_pcd=save_binary_pcd)
                    pcd_path = str(pcd_file)

                if save_pixels:
                    pixels_file = seg_dir / f"{stem}_pixels.npy"
                    np.save(pixels_file, seg_pixels.astype(np.int32))
                    pixels_path = str(pixels_file)

            seg = DetectSegment3D(
                instance_id=inst_id,
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy_rgb=bbox,  # type: ignore[arg-type]
                points_m=seg_points.astype(np.float64),
                colors_rgb=seg_colors.astype(np.uint8),
                pixels_rgb=seg_pixels.astype(np.int32),
                mask_area_px=int(m.sum()),
                centroid_m=centroid.astype(np.float64),
                aabb_min_m=aabb_min.astype(np.float64),
                aabb_max_m=aabb_max.astype(np.float64),
                output_frame=output_frame,
                pcd_path=pcd_path,
                pixels_path=pixels_path,
                meta_path=None,
            )

            if seg_dir is not None and save_meta:
                meta_file = seg_dir / f"{stem}_meta.json"
                meta = _segment_meta_dict(seg)
                meta["meta_path"] = str(meta_file)
                meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                meta_path = str(meta_file)
                seg = replace(seg, meta_path=meta_path)

            segments.append(seg)
            combined_points.append(seg.points_m)
            combined_colors.append(seg.colors_rgb)
            combined_pixels.append(seg.pixels_rgb)
            combined_instance_ids.append(np.full((len(seg.points_m),), inst_id, np.int32))
            combined_class_ids.append(np.full((len(seg.points_m),), class_id, np.int32))
            combined_confidences.append(np.full((len(seg.points_m),), conf, np.float32))

        if combined_points:
            points_out = np.concatenate(combined_points, axis=0)
            colors_out = np.concatenate(combined_colors, axis=0)
            pixels_out = np.concatenate(combined_pixels, axis=0)
            instance_ids_out = np.concatenate(combined_instance_ids, axis=0)
            class_ids_out = np.concatenate(combined_class_ids, axis=0)
            confidences_out = np.concatenate(combined_confidences, axis=0)
        else:
            points_out = np.empty((0, 3), np.float64)
            colors_out = np.empty((0, 3), np.uint8)
            pixels_out = np.empty((0, 2), np.int32)
            instance_ids_out = np.empty((0,), np.int32)
            class_ids_out = np.empty((0,), np.int32)
            confidences_out = np.empty((0,), np.float32)

        return DetectSegments3D(
            points_m=points_out,
            colors_rgb=colors_out,
            pixels_rgb=pixels_out,
            instance_ids=instance_ids_out,
            class_ids=class_ids_out,
            confidences=confidences_out,
            instance_map=instance_map,
            segments=segments,
            depth_rgb_m=depth_rgb_m,
            disparity=disparity,
            rectification=rectification,
            output_frame=output_frame,
        )

    def _ensure_ros2_publishers(self, params: Ros2PublishParams) -> dict[str, Any]:
        try:
            import rclpy
            from sensor_msgs.msg import Image, PointCloud2
            from std_msgs.msg import String
            from visualization_msgs.msg import MarkerArray
        except ImportError as exc:
            raise ImportError("ROS 2 publishing requires rclpy and ROS 2 message packages") from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        key = (params.ros2_node_name, params.ros2_topic_prefix.rstrip("/"))
        if self._ros2_node is None or self._ros2_publishers_key != key:
            if self._ros2_node is not None:
                try:
                    self._ros2_node.destroy_node()
                except Exception:
                    pass

            self._ros2_node = rclpy.create_node(params.ros2_node_name)
            prefix = params.ros2_topic_prefix.rstrip("/")
            self._ros2_publishers = {
                "cloud": self._ros2_node.create_publisher(PointCloud2, f"{prefix}/cloud", 10),
                "instance_map": self._ros2_node.create_publisher(Image, f"{prefix}/instance_map", 10),
                "depth_rgb": self._ros2_node.create_publisher(Image, f"{prefix}/depth_rgb", 10),
                "info_json": self._ros2_node.create_publisher(String, f"{prefix}/info_json", 10),
                "markers": self._ros2_node.create_publisher(MarkerArray, f"{prefix}/markers", 10),
            }
            self._ros2_publishers_key = key

        return self._ros2_publishers

    def _publish_segments_ros2(self, result: DetectSegments3D, params: Ros2PublishParams) -> Ros2PublishResult:
        try:
            import rclpy
            from ros2_utils import build_segment_ros_messages

            publishers = self._ensure_ros2_publishers(params)
            stamp = self._ros2_node.get_clock().now() if self._ros2_node is not None else None
            msgs = build_segment_ros_messages(
                result,
                frame_id=params.ros2_frame_id,
                stamp=stamp,
                include_depth=params.ros2_include_depth,
                include_markers=params.ros2_include_markers,
                pretty_json=params.ros2_pretty_json,
            )

            publishers["cloud"].publish(msgs.cloud)
            publishers["instance_map"].publish(msgs.instance_map)
            publishers["info_json"].publish(msgs.info_json)
            if msgs.depth_rgb is not None:
                publishers["depth_rgb"].publish(msgs.depth_rgb)
            if msgs.markers is not None:
                publishers["markers"].publish(msgs.markers)

            if self._ros2_node is not None:
                rclpy.spin_once(self._ros2_node, timeout_sec=0.0)

            prefix = params.ros2_topic_prefix.rstrip("/")
            topics = {
                "cloud": f"{prefix}/cloud",
                "instance_map": f"{prefix}/instance_map",
                "info_json": f"{prefix}/info_json",
            }
            if params.ros2_include_depth:
                topics["depth_rgb"] = f"{prefix}/depth_rgb"
            if params.ros2_include_markers:
                topics["markers"] = f"{prefix}/markers"

            return Ros2PublishResult(
                published=True,
                point_count=int(len(result.points_m)),
                segment_count=int(len(result.segments)),
                frame_id=params.ros2_frame_id,
                topics=topics,
                error=None,
            )
        except Exception as exc:
            return Ros2PublishResult(
                published=False,
                point_count=int(len(result.points_m)),
                segment_count=int(len(result.segments)),
                frame_id=params.ros2_frame_id,
                topics={},
                error=str(exc),
            )

    def set_calibration(self, params: DepthCalibrationParams) -> SetDepthCalibrationResult:
        self.calibration_params = params
        return self._calibration_result(self._build_calibration(params))

    def calibration(self, params: EmptyParams) -> SetDepthCalibrationResult:
        del params
        return self._calibration_result(self._build_calibration())

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        old_cache_key = self._dnn_cache_key(self.backend_params)
        self.backend_params = params
        if old_cache_key != self._dnn_cache_key(params):
            self._dnn_predictor = None
            self._dnn_predictor_key = None
        return self._backend_result()

    def backend(self, params: EmptyParams) -> BackendStatusResult:
        del params
        return self._backend_result()

    def to_pcd(self, params: ToPcdParams) -> ToPcdResult:

        calibration=self._build_calibration(params.calibration)
        output_path = Path(params.output_pcd_path).expanduser()
        output_suffix = output_path.suffix.lower()

        if output_suffix not in {".pcd", ".npz"}:
            raise ValueError(f"output_path must end with .pcd or .npz, got: {output_path}")

        backend = self._effective_backend(params)
        conversion_args = dict(
            left_image=_read_image_or_npy(params.left_path, color=False),
            right_image=_read_image_or_npy(params.right_path, color=False),
            rgb_image=_read_image_or_npy(params.rgb_path, color=True),
            calibration=calibration,
            output_path=output_path if output_suffix == ".pcd" else None,
            min_disparity=float(params.min_disparity),
            max_depth_m=params.max_depth_m,
            stride=params.stride,
            output_frame=params.output_frame,
            save_binary_pcd=params.save_binary_pcd,
            input_color_order=params.input_color_order,
            alpha=params.alpha,
            rgb_image_is_undistorted=params.rgb_image_is_undistorted,
        )

        if backend.backend == "sgbm":
            cloud = stereo_rgb_to_colored_point_cloud(
                **conversion_args,
                num_disparities=params.num_disparities,
                block_size=params.block_size,
            )
        elif backend.backend == "dnn":
            if stereo_rgb_to_colored_point_cloud_dnn is None:
                raise ImportError("DNN depth backend is unavailable; could not import pcd_dnn_utils") from _DNN_IMPORT_ERROR
            cloud = stereo_rgb_to_colored_point_cloud_dnn(
                **conversion_args,
                disparity_predictor=self._get_dnn_predictor(backend),
                model_scale=float(backend.model_scale),
                stereo_input_color_order=backend.stereo_input_color_order,
                remove_invisible=backend.remove_invisible,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.backend}")

        if output_suffix == ".npz":
            _save_cloud_npz(output_path, cloud)

        depth_min_m, depth_max_m, depth_mean_m = _depth_statistics(cloud.points_m)
        disparity_height, disparity_width = (None, None) if cloud.disparity is None else cloud.disparity.shape[:2]

        return ToPcdResult(
            backend=backend.backend,
            output_path=str(output_path),
            point_count=int(cloud.points_m.shape[0]),
            color_count=int(cloud.colors_rgb.shape[0]),
            size_bytes=int(output_path.stat().st_size),
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            depth_mean_m=depth_mean_m,
            disparity_width=disparity_width,
            disparity_height=disparity_height,
            calibration=str(calibration),
        )

    def detect_segments_to_pcd(self, params: ToYoloSegmentsParams) -> ToDetectSegmentsResult:
        output_dir = Path(params.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = self._effective_backend(params)
        calibration = self._build_calibration(params.calibration)

        left_image = _read_image_or_npy(params.left_path, color=False)
        right_image = _read_image_or_npy(params.right_path, color=False)
        rgb_image = _read_image_or_npy(params.rgb_path, color=True)

        yolo_result:YoloDetectResult = _read_detect_result(params.yolo_result_path)
        depth_rgb_m, disparity, rect = self._compute_rgb_aligned_depth(
            left_image=left_image,
            right_image=right_image,
            rgb_image=rgb_image,
            calibration=calibration,
            backend=backend,
            input_color_order=params.input_color_order,
            rgb_image_is_undistorted=params.rgb_image_is_undistorted,
            alpha=params.alpha,
            min_disparity=params.min_disparity,
            num_disparities=params.num_disparities,
            block_size=params.block_size,
            max_depth_m=params.max_depth_m,
            splat_px=params.splat_px,
        )

        result = self._detect_result_to_segments(
            yolo_result=yolo_result,
            depth_rgb_m=depth_rgb_m,
            disparity=disparity,
            rectification=rect,
            rgb_image=rgb_image,
            calibration=calibration,
            output_dir=output_dir,
            frame_name=params.frame_name,
            input_color_order=params.input_color_order,
            output_frame=params.output_frame,
            rgb_image_is_undistorted=params.rgb_image_is_undistorted,
            # mask_threshold=params.mask_threshold,
            # overlap_policy=params.overlap_policy,
            # min_points=params.min_points,
            # save_pcd=params.save_pcd,
            # save_pixels=params.save_pixels,
            # save_meta=params.save_meta,
            save_binary_pcd=params.save_binary_pcd,
        )
        self._last_yolo_segments = result

        # instance_map_path = None
        # depth_rgb_path = None
        # combined_npz_path = None
        # if params.save_instance_map:
        #     path = output_dir / f"{params.frame_name}_instance_map.npy"
        #     np.save(path, result.instance_map.astype(np.uint32))
        #     instance_map_path = str(path)
        # if params.save_depth_rgb and result.depth_rgb_m is not None:
        #     path = output_dir / f"{params.frame_name}_depth_rgb_m.npy"
        #     np.save(path, result.depth_rgb_m.astype(np.float32))
        #     depth_rgb_path = str(path)
        # if params.save_combined_npz:
        #     path = output_dir / f"{params.frame_name}_segments_combined.npz"
        #     _save_segments_npz(path, result)
        #     combined_npz_path = str(path)

        ros2_result = None
        if params.ros2_publish:
            ros2_result = self._publish_segments_ros2(
                result,
                Ros2PublishParams(
                    ros2_node_name=params.ros2_node_name,
                    ros2_topic_prefix=params.ros2_topic_prefix,
                    ros2_frame_id=params.ros2_frame_id,
                    ros2_include_depth=params.ros2_include_depth,
                    ros2_include_markers=params.ros2_include_markers,
                    ros2_pretty_json=params.ros2_pretty_json,
                ),
            )

        depth_min_m, depth_max_m, depth_mean_m = _depth_statistics(result.points_m)
        disparity_height, disparity_width = (None, None) if result.disparity is None else result.disparity.shape[:2]

        return ToDetectSegmentsResult(
            backend=backend.backend,
            yolo_model_path=yolo_result.yolo_config.model_name,
            output_dir=str(output_dir),
            frame_name=params.frame_name,
            output_frame=result.output_frame,
            point_count=int(len(result.points_m)),
            segment_count=int(len(result.segments)),
            # instance_map_path=instance_map_path,
            # depth_rgb_path=depth_rgb_path,
            # combined_npz_path=combined_npz_path,
            disparity_width=disparity_width,
            disparity_height=disparity_height,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            depth_mean_m=depth_mean_m,
            ros2=ros2_result,
            segments=_result_segments_to_summary(result.segments),
        )

    def publish_last_yolo_segments(self, params: Ros2PublishParams) -> Ros2PublishResult:
        if self._last_yolo_segments is None:
            raise ValueError("No YOLO segment result is cached. Call yolo_segments_to_pcd first.")
        return self._publish_segments_ros2(self._last_yolo_segments, params)

    def ros2_status(self, params: EmptyParams) -> Ros2PublishResult:
        del params
        prefix = None if self._ros2_publishers_key is None else self._ros2_publishers_key[1]
        topics: dict[str, str] = {}
        if prefix is not None:
            topics = {
                "cloud": f"{prefix}/cloud",
                "instance_map": f"{prefix}/instance_map",
                "depth_rgb": f"{prefix}/depth_rgb",
                "info_json": f"{prefix}/info_json",
                "markers": f"{prefix}/markers",
            }
        return Ros2PublishResult(
            published=self._ros2_node is not None,
            point_count=0 if self._last_yolo_segments is None else int(len(self._last_yolo_segments.points_m)),
            segment_count=0 if self._last_yolo_segments is None else int(len(self._last_yolo_segments.segments)),
            frame_id=None,
            topics=topics,
            error=None,
        )


def run_server(controller_name: str = "pcd") -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    Iox2JsonRpcServer(DepthController(controller_name=controller_name)).run_forever()
