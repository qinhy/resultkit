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
from matops import MatDevice, MatLib, TorchMatOps, MatOps
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
        
from disparity_predictors import (DisparityPredictor,
                                SGBMDisparityPredictor,
                                FastFoundationStereoDisparity,
                                SGBMDisparityPredictorCuda,
                                VPIStereoDisparityGPU)

from pcd_calculation import (StereoRectifier, StereoRgbCalibration, project_points_to_rgb_pixels, read_image, rectified_left_to_original_left, rgb8, save_pcd, split_cloud_uv,
                             )

PCD_BACKEND_MODULES = {
    "cpu": "pcd_backend_cpu",
    "cuda": "pcd_backend_cuda",
    "vpi": "pcd_backend_vpi",
}
ALL_PCD_BACKENDS = {
        "cpu": (MatLib.TORCH, MatDevice.CPU,  TorchMatOps(),SGBMDisparityPredictor()),
        "dnn": (MatLib.TORCH, MatDevice.CUDA, TorchMatOps(device=MatDevice.CUDA),FastFoundationStereoDisparity(
                    repo_dir="./fast-foundationstereo",
                    model_path="weights/23-36-37/model_best_bp2_serialize.pth",
                )),
        "cuda": (MatLib.TORCH, MatDevice.CUDA,TorchMatOps(device=MatDevice.CUDA),SGBMDisparityPredictorCuda(
                    width=1280,height=800,
                )),
        # "vpi": (MatLib.TORCH, MatDevice.CUDA, TorchMatOps(device=MatDevice.CUDA),VPIStereoDisparityGPU()),
    }


@lru_cache(maxsize=None)
def load_pcd_backend(name):
    try:
        module_name = PCD_BACKEND_MODULES[name]
    except KeyError:
        raise ValueError(f"Unsupported backend: {name}")

    return importlib.import_module(module_name)


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


def _model_field_names(model_type: type[BaseModel]) -> tuple[str, ...]:
    fields = getattr(model_type, "model_fields", None) or getattr(model_type, "__fields__", {})
    return tuple(fields.keys())


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
    calib_path: str = ""
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
                param.calib_path = calib_path / f"{cam_name}.json"
                
                # with open(calib_path / f"{cam_name}.json" ) as f:
                #     calib = json.load(f)
                #     allowed_fields = set(_model_field_names(DepthCalibrationParams))
                #     calibration_data = {key: value for key, value in calib.items() if key in allowed_fields}
                #     param.calibration = DepthCalibrationParams.model_validate(calibration_data)

                res.append(param)
        return res
    
    def get_output_path(self):return self.output_pcd_path


class ToYoloSegmentsParams(ToPcdParams):
    output_dir: str = "detect_segments_out"
    detection_path: RecordPath | None = None

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
        yolo_path = db_record.expected_yolo_path

        res = []
        for rgb, l, r, lp in zip(rgb_path, left_path, right_path, left_parent_paths):
            if rgb.exists() and l.exists() and r.exists():
                cam_name = lp.name

                param = ToPcdParams(rgb_path=str(rgb), left_path=str(l), right_path=str(r))
                param.output_pcd_path = str(pcd_path/f"{cam_name}.pcd")
                output_dir = str(Path(db_record.expected_pcd_path)/Path(param.get_output_path()).stem)                
                seg_params:ToYoloSegmentsParams = ToYoloSegmentsParams.model_validate(param.model_dump())
                seg_params.output_dir = output_dir
                seg_params.calib_path = calib_path / f"{cam_name}.json"
                seg_params.detection_path = list((yolo_path / cam_name).glob("*.json"))[0]
                
                # with open(calib_path / f"{cam_name}.json" ) as f:
                #     calib:Dict = json.load(f)
                #     allowed_fields = set(_model_field_names(DepthCalibrationParams))
                #     calibration_data = {key: value for key, value in calib.items() if key in allowed_fields}
                #     seg_params.calibration = DepthCalibrationParams.model_validate(calibration_data)

                res.append(seg_params)
        return res
    
    def get_output_path(self):return self.output_dir


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
    """Runs point-cloud processing on a single background worker."""

    input_queue: Queue[ToPcdParams | ToYoloSegmentsParams] = field(
        default_factory=lambda: Queue(maxsize=128), repr=False
    )
    hook_dispatcher: HookDispatcher = field(default_factory=HookDispatcher, repr=False)
    backend_name: str = field(default="cpu")

    _dnn_predictor: Any | None = field(default=None, init=False, repr=False)
    _dnn_predictor_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    _lifecycle_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _state_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _config_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _process_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    _exception: Exception | None = field(default=None, init=False, repr=False)
    _last_result: ToPcdResult | None = field(default=None, init=False, repr=False)
    _current_output_path: str | None = field(default=None, init=False, repr=False)

    _backends = ALL_PCD_BACKENDS
    _backend_params: tuple[str,str,MatOps,DisparityPredictor] | None = field(default=None, init=False, repr=False)
    
    _rectifiers: Dict[str,Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:        
        if self._backend_params is None:
            self._backend_params = self._backends[self.backend_name]

        with self._lifecycle_lock:
            if self.is_running: return
            self._stop_event.clear()
            self._thread = Thread(target=self._run,
                name="PcdRunner",daemon=True,)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop after the current conversion returns."""
        self._stop_event.set()
        if self.is_running: self._thread.join(timeout=timeout)

    def snapshot(self) -> tuple[str | None, ToPcdResult | None, Exception | None]:
        with self._state_lock:
            return self._current_output_path, self._last_result, self._exception

    def set_backend(self, name:Literal["cpu","dnn","cuda","vpi"]="cpu"):
        if name not in self._backends:
            raise ValueError(f"only supports {self._backends.keys()}, got {name}")
        self.backend_name = name
        self._backend_params = self._backends[name]

    def _set_state(self,*,
        output_path: str | None = None,
        result: ToPcdResult | None = None,
        exception: Exception | None = None,
    ) -> None:
        with self._state_lock:
            self._current_output_path = output_path
            if result is not None:
                self._last_result = result
            self._exception = exception

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                params = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            self._set_state(output_path=params.get_output_path())

            try:
                with self._process_lock:
                    result = self._convert_to_pcd(params,segments=hasattr(params, "detection_path"))
                    
                self._set_state(
                    output_path=params.get_output_path(),
                    result=result,
                )

                if params.hook_urls:
                    try:
                        self.hook_dispatcher.dispatch(
                            db_record=params.db_record,
                            hook_chains=params.hook_urls,
                        )
                    except Exception:
                        pass

            except Exception as exc:
                self._set_state(
                    output_path=params.get_output_path(),
                    exception=exc,
                )
            finally:
                self._set_state(output_path=None)
                self.input_queue.task_done()

    def _convert_to_pcd(self,
        params: ToPcdParams | ToYoloSegmentsParams,
        segments: bool = False,
    ):
        start_time = time.perf_counter()
        min_disparity, max_depth_m, stride = 0.0, 5.0, 1
        mathlib, calc_dev, op, disp_predictor = self._backend_params
        effective_min_disparity = max(0.5, float(min_disparity))

        def safe(fn, default="<unknown>"):
            try:
                return fn()
            except Exception:
                return default

        def shape(x):
            return safe(lambda: tuple(op.shape(x)), safe(lambda: tuple(x.shape)))

        def length(x):
            return 0 if x is None else safe(lambda: len(x))

        def meta(x, device=True):
            parts = [f"shape={shape(x)}", f"dtype={safe(lambda: str(x.dtype))}"]
            if device:
                parts.append(f"device={safe(lambda: str(x.device))}")
            return ", ".join(parts)

        def elapsed():
            return f"{time.perf_counter() - start_time:.3f}s"

        def log(stage, message):
            logger(f"[PcdRunner:to_pcd:{stage}] {message}")

        def load_image(name, path, color):
            image = read_image(path, ops=op, color=color)
            log("LOAD", f"{name} loaded {meta(image)}")
            return image

        log(
            "START",
            f"segments={segments}, backend={self.backend_name}, mathlib={mathlib}, "
            f"calc_dev={calc_dev}, predictor={type(disp_predictor).__name__}, params={params}",
        )
        log(
            "CONFIG",
            f"min_disparity={min_disparity}, effective_min_disparity={effective_min_disparity}, "
            f"max_depth_m={max_depth_m}, stride={stride}",
        )
        log(
            "PATHS",
            f"left={params.left_path}, right={params.right_path}, rgb={params.rgb_path}, "
            f"calib={params.calib_path}",
        )
        log(
            "PATHS",
            f"detection={params.detection_path}, output_dir={params.output_dir}"
            if segments
            else f"output_pcd={params.output_pcd_path}",
        )

        log("LOAD", f"loading images... elapsed={elapsed()}")
        left_img = load_image("left", params.left_path, "gray")
        right_img = load_image("right", params.right_path, "gray")
        rgb_img = load_image("rgb", params.rgb_path, "RGB")
        log("LOAD", f"images ready elapsed={elapsed()}")

        log("CALIB", f"loading calibration from {params.calib_path}")
        calib_json = json.loads(Path(params.calib_path).read_text())
        raw_device_id = calib_json.get("device_id")
        cache_key = str(raw_device_id) if raw_device_id is not None else None
        log("CALIB", f"device_id={raw_device_id!r}, cache_key={cache_key!r}")

        # Calibration is still needed even when the rectifier is cached.
        calib = StereoRgbCalibration.from_dict(calib_json, ops=op)
        log(
            "CALIB",
            f"calibration created left_resolution={getattr(calib, 'left_resolution', '<unknown>')}, "
            f"elapsed={elapsed()}",
        )

        device_cache = self._rectifiers.get(cache_key) if cache_key is not None else None
        cached = device_cache.get(self.backend_name) if device_cache else None

        if cache_key is None:
            log("CACHE", "device_id is missing; rectifier will not be cached")
        elif not device_cache:
            log("CACHE", f"MISS device={cache_key!r}: device has no rectifier cache")
        else:
            log("CACHE", f"device cache found for {cache_key!r}; cached_backends={list(device_cache)}")
            log("CACHE", f"{'HIT' if cached else 'MISS'} device={cache_key!r}, backend={self.backend_name!r}")

        if cached:
            rectifier, rect = cached
            log("RECTIFIER", f"using cached rectifier={type(rectifier).__name__}")
        else:
            log("RECTIFIER", "creating StereoRectifier")
            rectifier = StereoRectifier(calib, ops=op)
            rect = rectifier.make(calib.left_resolution)
            log("RECTIFIER", f"rectification model created type={type(rect).__name__}")
            if cache_key is not None:
                self._rectifiers.setdefault(cache_key, {})[self.backend_name] = (rectifier, rect)
                log("CACHE", f"stored rectifier device={cache_key!r}, backend={self.backend_name!r}")

        log("RECTIFY", f"starting stereo rectification left_shape={shape(left_img)}, right_shape={shape(right_img)}")
        left_rect, right_rect, rect = rectifier.rectify(left_img, right_img)
        log(
            "RECTIFY",
            f"complete left_rect={shape(left_rect)}, right_rect={shape(right_rect)}, "
            f"left_dtype={safe(lambda: str(left_rect.dtype))}, right_dtype={safe(lambda: str(right_rect.dtype))}, "
            f"elapsed={elapsed()}",
        )

        log("DISPARITY", f"starting predictor={type(disp_predictor).__name__}")
        disparity = disp_predictor.predict(left_rect, right_rect)
        log("DISPARITY", f"prediction complete {meta(disparity)}, elapsed={elapsed()}")
        try:
            log("DISPARITY", f"finite_mask_shape={shape(op.isfinite(disparity))}")
        except Exception as exc:
            log("DISPARITY:DEBUG", f"could not calculate finite disparity mask: {type(exc).__name__}: {exc}")

        log("REPROJECT", "converting disparity to rectified 3D points")
        points_rect, _ = rect.disparity_to_points_rectified(
            disparity,
            min_disparity=effective_min_disparity,
            max_depth_m=max_depth_m,
            stride=stride,
        )
        log("REPROJECT", f"generated points_rect count={length(points_rect)}, {meta(points_rect)}, elapsed={elapsed()}")
        if len(points_rect) == 0:
            log(
                "ERROR",
                f"no valid 3D points reconstructed. disparity_shape={shape(disparity)}, "
                f"min_disparity={effective_min_disparity}, max_depth_m={max_depth_m}",
            )
            raise RuntimeError("No valid 3D points reconstructed from disparity")

        log("TRANSFORM", f"transforming {len(points_rect)} points from rectified-left to original-left coordinates")
        points_left = rectified_left_to_original_left(points_rect, rect)
        log("TRANSFORM", f"points_left count={length(points_left)}, {meta(points_left)}, elapsed={elapsed()}")

        log("RGB_PROJECT", f"projecting {length(points_left)} 3D points into RGB image shape={shape(rgb_img)}")
        rgb_uv, _ = project_points_to_rgb_pixels(
            points_left,
            rgb_img,
            calib,
            rgb_image_is_undistorted=False,
        )
        log("RGB_PROJECT", f"projection complete rgb_uv_{meta(rgb_uv)}, elapsed={elapsed()}")

        if rgb_uv is None:
            log("ERROR", "RGB projection returned rgb_uv=None")
            raise RuntimeError("RGB projection returned no UV coordinates")
        if length(rgb_uv) != length(points_left):
            log("WARNING", f"point/UV count mismatch: points_left={length(points_left)}, rgb_uv={length(rgb_uv)}")

        if segments:
            log("SEGMENTS", f"loading detections from {params.detection_path}")
            with params.detection_path.open("r", encoding="utf-8") as handle:
                detections = json.load(handle)
            log("SEGMENTS", f"detections loaded count={length(detections)}")
            log(
                "SEGMENTS",
                f"splitting point cloud points={length(points_left)}, rgb_uv={length(rgb_uv)}, output_dir={params.output_dir}",
            )
            try:
                split_cloud_uv(
                    op.to_numpy(points_left),
                    op.to_numpy(rgb_uv),
                    op.to_numpy(rgb_img),
                    detections,
                    params.output_dir,
                    min_points=1,
                    erode_pixels=0,
                    exclusive=False,
                    save_background=False,
                    save_full_cloud=True,
                    binary_pcd=True,
                )
            except Exception as exc:
                log("error", str(exc))
            log("DONE", f"segmented point-cloud export completed elapsed={elapsed()}")
            return

        log("UV_FILTER", "checking projected RGB coordinates")
        finite_uv_mask = op.isfinite(rgb_uv)
        uv_finite = op.all(finite_uv_mask)  # Preserves the original reduction behavior.
        safe_uv = op.where(finite_uv_mask, rgb_uv, 0)
        u = op.astype_int64(op.round(safe_uv[:, 0]))
        v = op.astype_int64(op.round(safe_uv[:, 1]))
        rgb_h, rgb_w = op.shape(rgb_img)[:2]
        inside = uv_finite & (u >= 0) & (v >= 0) & (u < rgb_w) & (v < rgb_h)
        log("UV_FILTER", f"finite_mask_shape={shape(uv_finite)}, RGB bounds width={rgb_w}, height={rgb_h}")
        log("UV_FILTER", f"inside_mask_shape={shape(inside)}")

        before_filter_count = length(points_left)
        points_left, u, v = points_left[inside], u[inside], v[inside]
        after_filter_count = length(points_left)
        rejected_count = safe(lambda: before_filter_count - after_filter_count)
        log(
            "UV_FILTER",
            f"before={before_filter_count}, inside={after_filter_count}, rejected={rejected_count}",
        )

        if len(points_left) == 0:
            log("ERROR", "all reconstructed 3D points were rejected during RGB-image bounds filtering")
            raise RuntimeError("No 3D points project inside the RGB image")

        log("COLOR", f"sampling RGB colors for {len(points_left)} points")
        sampled_colors = rgb_img[v, u, :3]
        log("COLOR", f"sampled_colors {meta(sampled_colors, device=False)}")
        colors_rgb8 = rgb8(sampled_colors, ops=op)
        log("COLOR", f"colors_rgb8 {meta(colors_rgb8, device=False)}")

        log("SAVE", f"saving PCD path={params.output_pcd_path}, points={len(points_left)}, binary=True")
        save_pcd(params.output_pcd_path, points_left, colors_rgb8, ops=op, binary=True)
        log("DONE", f"PCD saved successfully path={params.output_pcd_path}, points={len(points_left)}, elapsed={elapsed()}")

    def submit_to_pcd(self, params: ToPcdParams | ToYoloSegmentsParams) -> PcdAsyncResult:
        """Queue a conversion and return immediately."""
        if not self.is_running: self.start()

        try:
            self.input_queue.put_nowait(params)
        except Full:
            return PcdAsyncResult()
            # return self._pcd_async_result(
            #     params,
            #     error=f"PCD queue is full (capacity={self.input_queue.maxsize})",
            # )
        return PcdAsyncResult()
        # return self._pcd_async_result(params, queued=True)
                
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
                "backend": self.runner.backend_name,
                "queue_capacity": self.runner.input_queue.maxsize,
            },
        )

    def status(self, params: EmptyParams) -> PcdStatusResult:
        return PcdStatusResult(msg=f"{self.runner.snapshot()}")

    # def set_calibration(self, params: DepthCalibrationParams) -> SetDepthCalibrationResult:
    #     return self.runner.set_calibration(params)

    # def calibration(self, params: EmptyParams) -> SetDepthCalibrationResult:
    #     return self.runner.calibration(params)

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        return self.runner.set_backend(params)

    # def backend(self, params: EmptyParams) -> BackendStatusResult:
    #     return self.runner.backend(params)

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

    # def to_pcd_status(self, params: EmptyParams) -> PcdAsyncResult:
    #     return self.runner.to_pcd_status(params)

    # def to_pcd_stop(self, params: EmptyParams) -> PcdAsyncResult:
    #     return self.runner.to_pcd_stop(params)

    # def to_pcd_sync(self, params: ToPcdParams) -> ToPcdResult:
    #     return self.runner.convert_to_pcd_sync(params)

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
