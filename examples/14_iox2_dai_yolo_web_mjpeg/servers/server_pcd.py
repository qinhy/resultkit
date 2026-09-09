from __future__ import annotations

from functools import lru_cache
import importlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, RLock, Thread
import time
from typing import Any, Dict, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
import torch
import cupy as cp

from common import *
from common import HookDispatcher
from iox2_jsonrpc import EmptyParams, RpcModel
from store.custom_record_store import CustomRecord
from resultkit.logger import logger

LOG_SERVICE = "jrpc"
LOG_CONTROLLER = "pcd"

_THIS_DIR = Path(__file__).absolute().parent
for path in (
    _THIS_DIR,
    _THIS_DIR.parent,
    Path(os.path.dirname(os.path.dirname(_THIS_DIR.parent))),
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.append(path_text)
        
from pcd_utils import StereoRgbCalibration as StereoRgbCalibrationCpu

PCD_BACKEND_MODULES = {
    "cpu": "pcd_backend_cpu",
    "cuda": "pcd_backend_cuda",
    "vpi": "pcd_backend_vpi",
}

@lru_cache(maxsize=None)
def load_pcd_backend(name):
    try:
        module_name = PCD_BACKEND_MODULES[name]
    except KeyError:
        raise ValueError(f"Unsupported backend: {name}")

    return importlib.import_module(module_name)

try:  # noqa: E402
    from pcd_dnn_utils import FastFoundationStereoDisparity
    _DNN_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # noqa: E402
    FastFoundationStereoDisparity = Any  # type: ignore[misc,assignment]  
    _DNN_IMPORT_ERROR = exc

logger(
    f"[{LOG_SERVICE}:{LOG_CONTROLLER}:init] module loaded",
    extra={
        "module_dir": str(_THIS_DIR),
        "dnn_available": _DNN_IMPORT_ERROR is None,
        "dnn_error": None if _DNN_IMPORT_ERROR is None else str(_DNN_IMPORT_ERROR),
    },
)

Resolution = tuple[int, int]
ColorOrder = Literal["RGB", "BGR"]
DepthBackend = Literal["sgbm", "dnn", "vpi"]
OutputFrame = Literal["left", "left_rectified"]
SegmentOutputFrame = Literal["rgb", "left"]
TranslationUnit = Literal["m", "cm", "mm"]
YoloOverlapPolicy = Literal["highest_confidence", "first", "none"]
Matrix3x3 = tuple[tuple[float, float, float], ...]
Matrix4x4 = tuple[tuple[float, float, float, float], ...]
DistortionCoefficients = tuple[float, ...]

_DNN_CACHE_KEY_FIELDS = ("repo_dir", "model_path", "model_dir", "device", "valid_iters", "max_disp", "hiera")

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
    
    pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND","cpu"))
    return pcd_backend.read_image(image_path, color=color)


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
    logger(
        f"[{LOG_SERVICE}:{LOG_CONTROLLER}:save_cloud_npz] point cloud archive saved",
        extra={
            "output_path": str(output_path),
            "point_count": int(arrays["points_m"].shape[0]),
            "has_disparity": "disparity" in arrays,
        },
    )
    return output_path


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
    device_id:Optional[str] = None


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
    repo_dir: str | None = "./examples/14_iox2_dai_yolo_web_mjpeg/fast-foundationstereo"
    model_path: str | None = "weights/23-36-37/model_best_bp2_serialize.pth"
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
    backend: DepthBackend = "dnn" #"sgbm"
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
    db_record: CustomRecord | None = None
    left_path: str = ""
    right_path: str = ""
    rgb_path: str = ""
    output_pcd_path: str = "colored_cloud.pcd"
    calibration: DepthCalibrationParams | None = None
    input_color_order: ColorOrder = "BGR"
    rgb_image_is_undistorted: bool = False
    alpha: float = 0.0
    max_depth_m: float | None = 2.0
    stride: int = Field(default=1, ge=1)
    output_frame: OutputFrame = "left"
    save_binary_pcd: bool = True
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    hook_urls: list[list[str]] = Field(default_factory=list)

    @staticmethod
    def from_db_record(db_record: CustomRecord) -> list[ToPcdParams]:
        if isinstance(db_record,dict):
            db_record = CustomRecord.model_validate(db_record)
        if db_record.is_empty():return []
        img_ext = db_record.listup_left_image_paths[0].suffix

        left_parent_paths = db_record.listup_left_image_parent_paths
        left_path = [p/f"left{img_ext}" for p in left_parent_paths]
        right_path = [p/f"right{img_ext}" for p in left_parent_paths]
        rgb_path = [p/f"rgb{img_ext}" for p in left_parent_paths]
        pcd_path = db_record.expected_pcd_path
        calib_path = db_record.expected_calib_path

        res = []
        for rgb, l, r, lp in zip(rgb_path, left_path, right_path, left_parent_paths):
            if rgb.exists() and l.exists() and r.exists():
                cam_name = lp.name

                param = ToPcdParams(rgb_path=str(rgb), left_path=str(l), right_path=str(r))
                param.output_pcd_path = str(pcd_path/f"{cam_name}.pcd")
                
                with open(calib_path / f"{cam_name}.json" ) as f:
                    calib = json.load(f)
                    allowed_fields = set(_model_field_names(DepthCalibrationParams))
                    calibration_data = {key: value for key, value in calib.items() if key in allowed_fields}
                    param.calibration = DepthCalibrationParams.model_validate(calibration_data)

                res.append(param)
        return res
    
    def get_output_path(self):return self.output_pcd_path

class ToPcdResult(DepthBaseModel):
    backend: DepthBackend
    output_path: str = ""
    point_count: int
    color_count: int
    size_bytes: int
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None
    calibration: str | None = None
    error: str | None = None


class PcdAsyncResult(DepthBaseModel):
    running: bool = False
    queued: bool = False
    queue_size: int = 0
    requested_output_path: str | None = None
    current_output_path: str | None = None
    last_result: ToPcdResult | None = None
    error: str | None = None


class PcdStatusResult(DepthBaseModel):
    msg: str


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
    db_record: CustomRecord | None = None
    left_path: str = ""
    right_path: str = ""
    rgb_path: str = ""
    output_dir: str = "detect_segments_out"
    frame_name: str = "frame"
    calibration: DepthCalibrationParams | None = None
    input_color_order: ColorOrder = "BGR"
    rgb_image_is_undistorted: bool = False

    # Depth parameters
    alpha: float = 0.0
    max_depth_m: float | None = 2.0
    output_frame: SegmentOutputFrame = "rgb"
    save_binary_pcd: bool = True
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    splat_px: int = Field(default=1, ge=0)
    hook_urls: list[list[str]] = Field(default_factory=list)

    @staticmethod
    def from_db_record(db_record: CustomRecord) -> list[ToYoloSegmentsParams]:
        if isinstance(db_record,dict):
            db_record = CustomRecord.model_validate(db_record)
        if db_record.is_empty():return []
        
        img_ext = db_record.listup_left_image_paths[0].suffix

        left_parent_paths = db_record.listup_left_image_parent_paths
        left_path = [p/f"left{img_ext}" for p in left_parent_paths]
        right_path = [p/f"right{img_ext}" for p in left_parent_paths]
        rgb_path = [p/f"rgb{img_ext}" for p in left_parent_paths]
        pcd_path = db_record.expected_pcd_path
        calib_path = db_record.expected_calib_path

        res = []
        for rgb, l, r, lp in zip(rgb_path, left_path, right_path, left_parent_paths):
            if rgb.exists() and l.exists() and r.exists():
                cam_name = lp.name

                param = ToPcdParams(rgb_path=str(rgb), left_path=str(l), right_path=str(r))
                param.output_pcd_path = str(pcd_path/f"{cam_name}.pcd")
                output_dir = str(Path(db_record.expected_pcd_path)/Path(param.get_output_path()).stem)
                
                seg_params:ToYoloSegmentsParams = ToYoloSegmentsParams.model_validate(param.model_dump())
                seg_params.output_dir = output_dir
                
                with open(calib_path / f"{cam_name}.json" ) as f:
                    calib:Dict = json.load(f)
                    allowed_fields = set(_model_field_names(DepthCalibrationParams))
                    calibration_data = {key: value for key, value in calib.items() if key in allowed_fields}
                    seg_params.calibration = DepthCalibrationParams.model_validate(calibration_data)

                res.append(seg_params)
        return res
    
    def get_output_path(self):return self.output_dir


class ToDetectSegmentsResult(DepthBaseModel):
    backend: DepthBackend
    output_dir: str
    frame_name: str = "left"
    output_frame: SegmentOutputFrame = "left"
    point_count: int = -1
    segment_count: int = -1
    instance_map_path: str | None = None
    depth_rgb_path: str | None = None
    combined_npz_path: str | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    segments: list[YoloSegmentSummary] = Field(default_factory=list)


@dataclass
class PcdRunner:
    """Owns all point-cloud processing and its single background worker.

    The runner is the processing layer. It owns calibration/backend settings,
    cached DNN resources, conversion functions, queue lifecycle, result state,
    completion-hook dispatch. It intentionally
    processes one heavy job at a time to avoid concurrent GPU/OpenCV pressure.
    """

    input_queue: Queue[ToPcdParams | ToYoloSegmentsParams] = field(
        default_factory=lambda: Queue(maxsize=128),
        repr=False,
    )
    calibration_params: DepthCalibrationParams | None = None
    backend_params: BackendParams = field(default_factory=BackendParams)
    hook_dispatcher: HookDispatcher = field(default_factory=HookDispatcher, repr=False)

    _dnn_predictor: Any | None = field(default=None, init=False, repr=False)
    _dnn_predictor_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _yolo_model: Any | None = field(default=None, init=False, repr=False)
    _yolo_model_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)
    _lifecycle_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _state_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _config_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _process_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    _exception: Exception | None = field(default=None, init=False, repr=False)
    _last_result: ToPcdResult | None = field(default=None, init=False, repr=False)
    _current_output_path: str | None = field(default=None, init=False, repr=False)
    
    _calibration_cpus: Dict[str,StereoRgbCalibrationCpu] = field(default_factory=dict)
    _calibrations: Dict[str,Any] = field(default_factory=dict)
    _rectifiers: Dict[str,Any] = field(default_factory=dict)


    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="PcdRunner", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting queued work after the current conversion returns."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def snapshot(self) -> tuple[str | None, ToPcdResult | None, Exception | None]:
        with self._state_lock:
            return self._current_output_path, self._last_result, self._exception

    def _run(self) -> None:

        while not self._stop_event.is_set():
            try:
                params: ToPcdParams | ToYoloSegmentsParams = self.input_queue.get(timeout=0.2)
            except Empty:
                continue
            output_path = params.get_output_path()

            with self._state_lock:
                self._current_output_path = output_path


            try:
                with self._process_lock:
                    if isinstance(params, ToPcdParams):
                        result = self._convert_to_pcd(params)
                    elif isinstance(params, ToYoloSegmentsParams):
                        result = self._detect_segments_to_pcd(params)

                with self._state_lock:
                    self._last_result = result
                    self._exception = None


                if params.hook_urls:
                    try:
                        self.hook_dispatcher.dispatch(
                            db_record=params.db_record,
                            hook_chains=params.hook_urls,
                        )
                    except Exception:
                        pass

            except Exception as exc:
                with self._state_lock:
                    self._exception = exc
            finally:
                with self._state_lock:
                    self._current_output_path = None
                self.input_queue.task_done()

    def _build_calibration(self, params: DepthCalibrationParams | None = None):
        if params is None:
            with self._config_lock:
                params = self.calibration_params
        
        key = params.device_id
        if key is None or key not in self._calibrations: 
            if key is None: key = "latest"
            calibration_data = _model_to_dict(params)
            calibration_data.pop("service", None)
            translation_unit = calibration_data.pop("source_translation_unit", "cm")
            calibration = StereoRgbCalibrationCpu.from_dict(
                calibration_data,
                source_translation_unit=translation_unit,
            )
            pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))
            self._calibration_cpus[key] = calibration
            self._calibrations[key] = pcd_backend.StereoRgbCalibration.from_cpu(self._calibration_cpus)
            self._rectifiers[key] = calibration.get_rectifier(alpha=0.0)
            self._rectifiers[key].calibration = self._calibration_cpus[key]
        return self._calibration_cpus[key],key

    def _calibration_result(self, calibration: StereoRgbCalibrationCpu) -> SetDepthCalibrationResult:
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
        with self._config_lock:
            backend_data = {key: getattr(self.backend_params, key) for key in BACKEND_KEYS}
            predictor_loaded = self._dnn_predictor is not None
        return BackendStatusResult(
            configured=True,
            predictor_loaded=predictor_loaded,
            dnn_available=_DNN_IMPORT_ERROR is None,
            dnn_error=None if _DNN_IMPORT_ERROR is None else str(_DNN_IMPORT_ERROR),
            **backend_data,
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
            logger(f"[PcdRunner:get_dnn_predictor:info] loading dnn model")
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
        with self._config_lock:
            backend_data = {key: getattr(self.backend_params, key) for key in BACKEND_KEYS}
        override_data = _model_to_dict(params)
        backend_data.update({key: override_data[key] for key in BACKEND_KEYS if override_data.get(key) is not None})
        return BackendParams(**backend_data)

    def set_calibration(self, params: DepthCalibrationParams) -> SetDepthCalibrationResult:
        with self._config_lock:
            self.calibration_params = params
        result = self._calibration_result(self._build_calibration(params)[0])
        return result

    def calibration(self, params: EmptyParams) -> SetDepthCalibrationResult:
        del params
        with self._config_lock:
            configured = self.calibration_params
        result = self._calibration_result(self._build_calibration(configured)[0])
        return result

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        # pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))
        # Literal["sgbm", "dnn", "vpi"]
        os.environ["PCD_BACKEND"] = {"sgbm":"cpu","dnn":"cuda",
                                     "vpi":"vpi",}[params.backend]
        
        pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))
        logger(f"[PcdRunner:set_backend:info] set as {(params.backend,os.environ["PCD_BACKEND"])}",
            extra={"pcd_backend":str(pcd_backend)})

        with self._process_lock:
            with self._config_lock:
                old_cache_key = self._dnn_cache_key(self.backend_params)
                self.backend_params = params
                cache_reset = old_cache_key != self._dnn_cache_key(params)
                if cache_reset:
                    self._dnn_predictor = None
                    self._dnn_predictor_key = None
        result = self._backend_result()
        return result

    def backend(self, params: EmptyParams) -> BackendStatusResult:
        del params
        result = self._backend_result()
        return result

    def _compute_rgb_aligned_depth(
        self,
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        rgb_image: np.ndarray,
        calibration: StereoRgbCalibrationCpu,
        backend: BackendParams,
        input_color_order: ColorOrder,
        rgb_image_is_undistorted: bool,
        alpha: float,
        min_disparity: int,
        num_disparities: int,
        block_size: int,
        max_depth_m: float | None,
        device_id: str | None,
    ):
        pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))
        height, width = left_image.shape[:2]
        
        # calibration_cpu = calibration
        # calibration = pcd_backend.StereoRgbCalibration.from_cpu(calibration_cpu)
        # rectifier = calibration.get_rectifier(alpha=alpha)
        # rectifier.calibration = calibration_cpu
        calibration = self._calibration_cpus[device_id]
        # calibration = self._calibrations[device_id]
        rectifier = self._rectifiers[device_id]
        left_rect, right_rect, rect = rectifier.rectify(left_image, right_image)

        confidence_u16 = None
        if backend.backend == "sgbm" or backend.backend == "vpi":
            predictor = pcd_backend.SGBMDisparityPredictor( width=width,height=height,
                                                num_disparities=num_disparities,
                                                min_disparity=min_disparity,
                                                block_size=block_size,
                                            )
            disparity = predictor.predict(left_rect, right_rect)
            if isinstance(disparity, tuple):
                disparity,confidence_u16 = disparity
        elif backend.backend == "dnn":
            predictor = self._get_dnn_predictor(backend) # FastFoundationStereoDisparity
            disparity = predictor.predict(left_rect, right_rect, input_color_order=input_color_order)

        else:
            raise ValueError(f"Unsupported backend: {backend.backend}")
        
        logger(f"[PcdRunner:compute_rgb_aligned_depth:info] got disparity",
            extra={"predictor": str(predictor), "pcd_backend":str(pcd_backend)})
        
        arg_com = dict(
            disparity=disparity,
            min_disparity=max(0.5, float(min_disparity)),
            max_depth_m=max_depth_m,
            stride=1
        )
        if confidence_u16 is not None:
            arg_com["confidence_u16"] = confidence_u16
        points_rect, _xy = rect.disparity_to_points_rectified(**arg_com)

        if len(points_rect) == 0: return None,None
            
        rgb_image = pcd_backend.image_gpu(rgb_image)        
        points_left = pcd_backend.rectified_left_to_original_left(points_rect, rect)
        uv, _ = pcd_backend.project_points_to_rgb_pixels(
            points_left,
            rgb_image,
            calibration,
            rgb_image_is_undistorted=rgb_image_is_undistorted,
        )
        return points_left,uv
    
    def _convert_to_pcd(self, params: ToPcdParams) -> ToPcdResult:
        if Path(params.get_output_path()).exists():
            return ToPcdResult(
                backend=self.backend_params.backend,
                output_path=str(params.get_output_path()),
                point_count=-1,
                color_count=-1,
                size_bytes=-1,
                error=f"output path already exists: {params.get_output_path()}",
            )

        calibration,device_id = self._build_calibration(params.calibration)
        output_path = Path(params.get_output_path()).expanduser()
        output_suffix = output_path.suffix.lower()
        if output_suffix not in {".pcd", ".npz"}:
            raise ValueError(f"output_path must end with .pcd or .npz, got: {output_path}")

        backend = self._effective_backend(params)
        rgb_image=_read_image_or_npy(params.rgb_path, color=True)
        points_left,rgb_uv = self._compute_rgb_aligned_depth(
            left_image=_read_image_or_npy(params.left_path, color=False),
            right_image=_read_image_or_npy(params.right_path, color=False),
            rgb_image=rgb_image,
            calibration=calibration,
            backend=backend,
            input_color_order=params.input_color_order,
            rgb_image_is_undistorted=params.rgb_image_is_undistorted,
            alpha=params.alpha,
            min_disparity=float(params.min_disparity),
            num_disparities=params.num_disparities,
            block_size=params.block_size,
            max_depth_m=params.max_depth_m,
            device_id=device_id,
        )
        pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))

        uv_finite = np.isfinite(rgb_uv).all(axis=1)
        safe_uv = np.where(np.isfinite(rgb_uv), rgb_uv, 0)
        u = np.rint(safe_uv[:, 0]).astype(np.int64)
        v = np.rint(safe_uv[:, 1]).astype(np.int64)
        rgb_h,rgb_w = rgb_image.shape[:2]
        inside = (
            uv_finite
            & (u >= 0)
            & (u < rgb_w)
            & (v >= 0)
            & (v < rgb_h)
        )
        points_left = points_left[inside]
        u = u[inside]
        v = v[inside]
        if len(points_left) == 0:
            raise RuntimeError("No 3D points project inside the RGB image")
        
        sampled_colors = rgb_image[v, u, :3]
        colors_rgb = pcd_backend.rgb8(sampled_colors)        
        pcd_backend.save_point_cloud(
            output_path,
            points_left,
            colors_rgb,
            binary_pcd=True,
        )

        result = ToPcdResult(
            backend=backend.backend,
            output_path=str(output_path),
            point_count=-1,#int(cloud.points_m.shape[0]),
            color_count=-1,#int(cloud.colors_rgb.shape[0]),
            size_bytes=int(output_path.stat().st_size),
            # depth_min_m=depth_min_m,
            # depth_max_m=depth_max_m,
            # depth_mean_m=depth_mean_m,
            # disparity_width=disparity_width,
            # disparity_height=disparity_height,
            calibration=str(calibration),
        )

        return result

    def _pcd_async_result(
        self,
        params: ToPcdParams | None = None,
        *,
        queued: bool = False,
        error: str | None = None,
    ) -> PcdAsyncResult:
        current_output, last_result, exception = self.snapshot()
        runner_error = None
        if exception is not None:
            runner_error = f"{exception.__class__.__name__}: {exception}"

        return PcdAsyncResult(
            running=self.is_running,
            queued=queued,
            queue_size=self.input_queue.qsize(),
            requested_output_path=params.get_output_path() if params is not None else None,
            current_output_path=current_output,
            last_result=last_result,
            error=error or runner_error,
        )

    def submit_to_pcd(self, params: ToPcdParams | ToYoloSegmentsParams) -> PcdAsyncResult:
        """Queue a conversion and return immediately."""
        if not self.is_running: self.start()


        try:
            self.input_queue.put_nowait(params)
        except Full:
            return self._pcd_async_result(
                params,
                error=f"PCD queue is full (capacity={self.input_queue.maxsize})",
            )

        return self._pcd_async_result(params, queued=True)

    def to_pcd_status(self, params: EmptyParams) -> PcdAsyncResult:
        del params
        result = self._pcd_async_result()
        return result

    def to_pcd_stop(self, params: EmptyParams) -> PcdAsyncResult:
        del params
        self.stop()
        return self._pcd_async_result()

    def convert_to_pcd_sync(self, params: ToPcdParams) -> ToPcdResult:
        """Run through the same complete processor without the queue."""
        with self._process_lock:
            return self._convert_to_pcd(params)

    def _detect_segments_to_pcd(self, params: ToYoloSegmentsParams, architecture="yolo"):

        output_dir = Path(params.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = self._effective_backend(params)
        calibration,device_id = self._build_calibration(params.calibration)

        left_image = _read_image_or_npy(params.left_path, color=False)
        right_image = _read_image_or_npy(params.right_path, color=False)
        rgb_image = _read_image_or_npy(params.rgb_path, color=True)

        detections = json.loads(Path(
                        params.rgb_path.replace("imgs",architecture
                                       ).replace("rgb.jpg","rgb.json")
                                       ).read_text(encoding="utf-8"))
        
        points_left,rgb_uv = self._compute_rgb_aligned_depth(
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
            device_id=device_id,
        )

        pcd_backend = load_pcd_backend(os.getenv("PCD_BACKEND", "cpu"))
        pcd_backend.split_cloud_uv(
            points_left,
            rgb_uv,
            rgb_image,
            detections,
            output_dir,
            min_points=1,
            erode_pixels=0,
            exclusive=False,
            save_background=False,
            save_full_cloud=True,
            binary_pcd=True,
        )
        
        return ToPcdResult(
            backend=backend.backend,
            output_path=str(output_dir),
            point_count=len(points_left),
            color_count=len(points_left),
            size_bytes=-1,
            calibration=str(calibration),
        )

    def detect_segments_to_pcd(self, params: ToYoloSegmentsParams) -> ToDetectSegmentsResult:
        with self._process_lock:
            return self._detect_segments_to_pcd(params)


@dataclass
class DepthController:
    """Thin JSON-RPC facade; all processing lives in ``PcdRunner``."""

    service_name: str = "jrpc"
    controller_name: str = "pcd"
    runner: PcdRunner = field(default_factory=PcdRunner, repr=False)

    def __post_init__(self) -> None:
        logger(
            f"[{self.service_name}:{self.controller_name}:init] controller initialized",
            extra={
                "service_name": self.service_name,
                "controller_name": self.controller_name,
                "backend": self.runner.backend_params.backend,
                "queue_capacity": self.runner.input_queue.maxsize,
            },
        )

    def status(self, params: EmptyParams) -> PcdStatusResult:
        return PcdStatusResult(msg=f"{self.runner.snapshot()}")

    def set_calibration(self, params: DepthCalibrationParams) -> SetDepthCalibrationResult:
        return self.runner.set_calibration(params)

    def calibration(self, params: EmptyParams) -> SetDepthCalibrationResult:
        return self.runner.calibration(params)

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        return self.runner.set_backend(params)

    def backend(self, params: EmptyParams) -> BackendStatusResult:
        return self.runner.backend(params)

    def to_pcd(self, params: ToPcdParams) -> PcdAsyncResult:
        has_db_record = bool(params.db_record)
        logger(
            f"[{self.service_name}:{self.controller_name}:to_pcd] conversion requested",
            extra={
                "output_path": params.get_output_path(),
                "left_path": params.left_path,
                "right_path": params.right_path,
                "rgb_path": params.rgb_path,
                "has_db_record": has_db_record,
            },
        )
        try:
            if has_db_record:
                db_record = params.db_record
            else:
                db_record = CustomRecord.empty()
                
            if db_record.is_empty():
                return self.runner.submit_to_pcd(params)

            derived_params = ToPcdParams.from_db_record(db_record)

            logger(
                f"[{self.service_name}:{self.controller_name}:to_pcd] derived jobs({len(derived_params)})",
                extra={
                    "conversion_count": len(derived_params),
                    "output_paths": [
                        item.get_output_path() for item in derived_params
                    ],
                },
            )
            result = PcdAsyncResult()
            for conversion_params in derived_params:
                result = self.runner.submit_to_pcd(conversion_params)
            return result
        except Exception as e:
            logger(
                f"[{self.service_name}:{self.controller_name}:to_pcd:error] conversion request failed, {e}",level="error",
                extra={"output_path": params.get_output_path()},
            )
            raise

    def to_pcd_status(self, params: EmptyParams) -> PcdAsyncResult:
        return self.runner.to_pcd_status(params)

    def to_pcd_stop(self, params: EmptyParams) -> PcdAsyncResult:
        return self.runner.to_pcd_stop(params)

    def to_pcd_sync(self, params: ToPcdParams) -> ToPcdResult:
        return self.runner.convert_to_pcd_sync(params)

    def detect_segments_to_pcd(self, params: ToYoloSegmentsParams) -> ToDetectSegmentsResult:        
        has_db_record = bool(params.db_record)
        logger(f"[{self.service_name}:{self.controller_name}:detect_segments_to_pcd] conversion requested")

        try:
            if has_db_record:
                db_record = params.db_record
            else:
                db_record = CustomRecord.empty()                
            if db_record.is_empty():
                return self.runner.submit_to_pcd(params)

            derived_params = ToYoloSegmentsParams.from_db_record(db_record)

            logger(
                f"[{self.service_name}:{self.controller_name}:detect_segments_to_pcd] derived jobs({len(derived_params)})",
                extra={
                    "conversion_count": len(derived_params),
                    "output_paths": [
                        item.output_dir for item in derived_params
                    ],
                },
            )
            result = PcdAsyncResult()
            for conversion_params in derived_params:
                result = self.runner.submit_to_pcd(conversion_params)
            return result
        except Exception as e:
            logger(
                f"[{self.service_name}:{self.controller_name}:detect_segments_to_pcd:error] conversion request failed, {e}",level="error",
                extra={"output_path": params.output_dir},
            )
            raise

def run_server(controller_name: str = "pcd") -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    logger(
        f"[{LOG_SERVICE}:{controller_name}:run_server] starting RPC server",
        extra={"service_name": LOG_SERVICE, "controller_name": controller_name},
    )
    try:
        Iox2JsonRpcServer(
            DepthController(
                service_name=LOG_SERVICE,
                controller_name=controller_name,
            )
        ).run_forever()
    except KeyboardInterrupt:
        logger(
            f"[{LOG_SERVICE}:{controller_name}:run_server] server interrupted",
            level="warning",
        )
        raise
    except Exception:
        logger(
            f"[{LOG_SERVICE}:{controller_name}:run_server:error] server stopped with an error",level="error",
            extra={"service_name": LOG_SERVICE, "controller_name": controller_name},
        )
        raise
    finally:
        logger(f"[{LOG_SERVICE}:{controller_name}:run_server] server stopped")


if __name__ == "__main__":
    run_server()
