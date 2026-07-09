#!/usr/bin/env python3
"""Single-frame record-store RPC server for dual_rgb and rgb_stereo MJPEG streams.

Flow implemented by this server:

    watch(mode="dual_rgb" | "rgb_stereo")
        -> subscribe to the matching EncodedImageMatPubSub topic

    capture(mode="dual_rgb", field_id="...", gis={...})
        -> verify/ensure subscription
        -> capture one fresh synchronized MJPEG bundle
        -> save one CustomStore record, including GIS
        -> call the optional YOLO hook
        -> finish

    capture(mode="rgb_stereo", field_id="...", gis=None)
        -> verify/ensure subscription
        -> capture one fresh synchronized RGB+left+right MJPEG bundle
        -> save one CustomStore record
        -> call optional YOLO hook
        -> call optional PCD hook
        -> optionally publish produced PCDs through a configured command hook
        -> finish

The camera servers intentionally publish encoded JPEG bytes. This server writes
those MJPEG/JPEG frame bytes directly as the CustomStore's canonical .jpg camera
images. The disparity image remains PNG because it is depth data, not a camera
MJPEG frame.

Downstream server_yolo/server_pcd/ROS2 APIs were not part of the uploaded files,
so this server exposes small generic hooks:
    - HTTP JSON-RPC hook: --yolo-url / --pcd-url
    - Shell hook: --yolo-command / --pcd-command / --ros2-publish-command

Every hook receives the record path, mode, field_id, record_id, image paths, and
GIS payload. Hook commands may use Python format placeholders such as
{record_path}, {mode}, {field_id}, {record_id}, {pcd_path}, and {topic}.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import requests

from common import EmptyParams, RpcModel, openapi_doc, CustomRecord, CustomStore, RecordMode
from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


parser = argparse.ArgumentParser(description="Run the capture record-store RPC server.")
parser.add_argument("--service-name", default="jrpc", help="iceoryx2 RPC service name")
parser.add_argument("--controller-name", default="store", help="record-store controller name")
parser.add_argument("--store-root", default="recording", help="root directory for CustomStore records")
parser.add_argument("--camera-service-name", default=None, help="service prefix used by camera topics; default is --service-name")
parser.add_argument("--dual-rgb-controller", default="cameraDualRGB", help="dual-RGB camera controller name")
parser.add_argument("--rgb-stereo-controller", default="cameraRgbd", help="RGB-stereo camera controller name")
parser.add_argument("--dual-rgb-topic", default=None, help="explicit dual_rgb MJPEG topic override")
parser.add_argument("--rgb-stereo-topic", default=None, help="explicit rgb_stereo MJPEG topic override")
parser.add_argument("--subscriber-capacity-bytes", type=int, default=64 * 1024 * 1024)
parser.add_argument("--capture-timeout-s", type=float, default=3.0)
parser.add_argument("--auto-subscribe-on-capture", action="store_true", default=True)
parser.add_argument("--no-auto-subscribe-on-capture", dest="auto_subscribe_on_capture", action="store_false")
parser.add_argument("--save-original-mjpeg", action="store_true", default=True, help="deprecated no-op; MJPEG/JPEG frames are always saved as canonical .jpg images")
parser.add_argument("--no-save-original-mjpeg", dest="save_original_mjpeg", action="store_false", help="deprecated no-op; canonical camera images are still saved as .jpg")
parser.add_argument("--require-png", action="store_true", default=False, help="deprecated no-op; camera frames are not decoded to PNG")
args = parser.parse_args()


NS_PER_SECOND = 1_000_000_000
PUBSUB_HEADER_BYTES = 8
DEFAULT_CAMERA_SERVICE_NAME = args.camera_service_name or args.service_name

from server_dual_rgb import BUNDLE_MAGIC as DUAL_RGB_MAGIC
from server_dual_rgb import BUNDLE_VERSION as DUAL_RGB_VERSION
from server_dual_rgb import BUNDLE_FORMAT as DUAL_RGB_FORMAT
from server_dual_rgb import BUNDLE_HEADER as DUAL_RGB_HEADER
from server_dual_rgb import BUNDLE_PREFIX as DUAL_RGB_PREFIX
from server_dual_rgb import BUNDLE_TYPE as DUAL_RGB_TYPE
from server_dual_rgb import CAM_MODE as DUAL_RGB_CAM_MODE

from server_rgb_stereo import BUNDLE_MAGIC as RGB_STEREO_MAGIC
from server_rgb_stereo import BUNDLE_VERSION as RGB_STEREO_VERSION
from server_rgb_stereo import BUNDLE_FORMAT as RGB_STEREO_FORMAT
from server_rgb_stereo import BUNDLE_HEADER as RGB_STEREO_HEADER
from server_dual_rgb import BUNDLE_PREFIX as RGB_STEREO_PREFIX
from server_dual_rgb import BUNDLE_TYPE as RGB_STEREO_TYPE
from server_rgb_stereo import CAM_MODE as RGB_STEREO_CAM_MODE

from store.custom_record_store import VALID_MODES
from store.custom_record_store import DUAL_RGB_CAM_NAME_0, DUAL_RGB_CAM_NAME_1, RGB_STEREO_CAM_NAME


def default_dual_rgb_topic() -> str:
    return f"{DEFAULT_CAMERA_SERVICE_NAME}:{args.dual_rgb_controller}:{DUAL_RGB_PREFIX}_{DUAL_RGB_TYPE}"


def default_rgb_stereo_topic() -> str:
    return f"{DEFAULT_CAMERA_SERVICE_NAME}:{args.rgb_stereo_controller}:{RGB_STEREO_PREFIX}_{RGB_STEREO_TYPE}"


class StoreBaseModel(RpcModel):
    service: str = args.service_name


class StoreConfig(StoreBaseModel):
    """Record-store settings and downstream hook endpoints."""

    root_path: str = args.store_root
    dual_rgb_topic: str = args.dual_rgb_topic or default_dual_rgb_topic()
    rgb_stereo_topic: str = args.rgb_stereo_topic or default_rgb_stereo_topic()
    subscriber_capacity_bytes: int = args.subscriber_capacity_bytes
    capture_timeout_s: float = args.capture_timeout_s
    auto_subscribe_on_capture: bool = args.auto_subscribe_on_capture
    # Deprecated compatibility knobs. Camera frames are always saved as canonical .jpg files.
    save_original_mjpeg: bool = args.save_original_mjpeg
    require_png: bool = args.require_png


class WatchParams(StoreBaseModel):
    mode: RecordMode


class CaptureParams(StoreBaseModel):
    mode: RecordMode
    field_id: str = "field_01"
    gis: Any | None = None
    capture_timeout_s: float | None = None
    fresh_frame: bool = True
    
    call_yolo: bool = False
    call_pcd: bool = False
    publish_ros2: bool = False

    hook_urls:list[str] = ["http://localhost:8000/controllers/yolo/start"]

    validate: bool = False


class StoreStatusResult(StoreBaseModel):
    opened: bool
    root_path: str
    watched_modes: list[str]
    subscribed_modes: list[str]
    dual_rgb_topic: str
    rgb_stereo_topic: str
    last_error: str | None = None
    last_capture: dict[str, Any] | None = None


class WatchResult(StoreBaseModel):
    ok: bool
    mode: RecordMode
    topic: str
    subscribed_modes: list[str]
    error: str | None = None


class CaptureResult(StoreBaseModel):
    ok: bool
    mode: RecordMode
    field_id: str
    record_id: str | None = None
    date_utc: str | None = None
    record_path: str | None = None
    topic: str | None = None
    frame_index: int | None = None
    timestamp_ns_utc: int | None = None
    images: list[str] = []
    validation_issues: list[str] = []
    yolo: dict[str, Any] | None = None
    pcd: dict[str, Any] | None = None
    ros2: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class DualRGBMjpegBundle:
    frame_index: int
    pts_ns: int
    rgb_width: int
    rgb_height: int
    camera0: bytes
    camera1: bytes


@dataclass(frozen=True)
class RGBStereoMjpegBundle:
    frame_index: int
    pts_ns: int
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    rgb: bytes
    left: bytes
    right: bytes


@dataclass(frozen=True)
class CapturedFrame:
    mode: RecordMode
    topic: str
    frame_index: int
    pts_ns: int
    bundle: DualRGBMjpegBundle | RGBStereoMjpegBundle
    received_ns_utc: int


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def model_to_dict(model: Any) -> dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, Mapping):
        return dict(model)
    return dict(getattr(model, "__dict__", {}))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def packet_nbytes(pkt: Any) -> int:
    if hasattr(pkt, "nbytes") and callable(pkt.nbytes):
        return int(pkt.nbytes())
    return int(getattr(pkt, "valid_nbytes", 0) or 0)


def packet_payload_bytes(pkt: Any) -> bytes:
    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        if isinstance(payload, np.ndarray):
            return payload.tobytes()
        return bytes(payload)

    arr = np.asarray(getattr(pkt, "data", pkt), dtype=np.uint8).reshape(-1)
    n = max(0, min(packet_nbytes(pkt), int(arr.size)))
    return arr[:n].tobytes()



def safe_rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except Exception:
        return path.as_posix()


def utc_now_ns() -> int:
    return time.time_ns()


# ---------------------------------------------------------------------------
# Bundle unpackers. These mirror the camera/debug files without importing them,
# because those modules parse CLI args at import time.
# ---------------------------------------------------------------------------


def unpack_dual_rgb_mjpeg_bundle(payload_or_packet: Any) -> DualRGBMjpegBundle:
    payload = packet_payload_bytes(payload_or_packet)
    if len(payload) < DUAL_RGB_HEADER.size:
        raise ValueError(f"Payload too small for {DUAL_RGB_FORMAT}: {len(payload)} bytes")

    (
        magic,
        version,
        header_nbytes,
        frame_index,
        pts_ns,
        rgb_width,
        rgb_height,
        camera0_nbytes,
        camera1_nbytes,
    ) = DUAL_RGB_HEADER.unpack(payload[: DUAL_RGB_HEADER.size])

    if magic != DUAL_RGB_MAGIC:
        raise ValueError(f"Invalid dual RGB MJPEG bundle magic: {magic!r}")
    if int(version) != DUAL_RGB_VERSION:
        raise ValueError(f"Unsupported dual RGB MJPEG bundle version: {version}")
    if int(header_nbytes) < DUAL_RGB_HEADER.size:
        raise ValueError(f"Invalid dual RGB MJPEG bundle header size: {header_nbytes}")

    camera0_start = int(header_nbytes)
    camera1_start = camera0_start + int(camera0_nbytes)
    payload_end = camera1_start + int(camera1_nbytes)
    if payload_end > len(payload):
        raise ValueError(f"Truncated dual RGB MJPEG bundle: need {payload_end}, got {len(payload)}")

    return DualRGBMjpegBundle(
        frame_index=int(frame_index),
        pts_ns=int(pts_ns),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        camera0=payload[camera0_start:camera1_start],
        camera1=payload[camera1_start:payload_end],
    )


def unpack_rgb_stereo_mjpeg_bundle(payload_or_packet: Any) -> RGBStereoMjpegBundle:
    payload = packet_payload_bytes(payload_or_packet)
    if len(payload) < RGB_STEREO_HEADER.size:
        raise ValueError(f"Payload too small for {RGB_STEREO_FORMAT}: {len(payload)} bytes")

    (
        magic,
        version,
        header_nbytes,
        frame_index,
        pts_ns,
        rgb_width,
        rgb_height,
        stereo_width,
        stereo_height,
        rgb_nbytes,
        left_nbytes,
        right_nbytes,
    ) = RGB_STEREO_HEADER.unpack(payload[: RGB_STEREO_HEADER.size])

    if magic != RGB_STEREO_MAGIC:
        raise ValueError(f"Invalid RGB+stereo MJPEG bundle magic: {magic!r}")
    if int(version) != RGB_STEREO_VERSION:
        raise ValueError(f"Unsupported RGB+stereo MJPEG bundle version: {version}")
    if int(header_nbytes) < RGB_STEREO_HEADER.size:
        raise ValueError(f"Invalid RGB+stereo MJPEG bundle header size: {header_nbytes}")

    rgb_start = int(header_nbytes)
    left_start = rgb_start + int(rgb_nbytes)
    right_start = left_start + int(left_nbytes)
    payload_end = right_start + int(right_nbytes)
    if payload_end > len(payload):
        raise ValueError(f"Truncated RGB+stereo MJPEG bundle: need {payload_end}, got {len(payload)}")

    return RGBStereoMjpegBundle(
        frame_index=int(frame_index),
        pts_ns=int(pts_ns),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        stereo_width=int(stereo_width),
        stereo_height=int(stereo_height),
        rgb=payload[rgb_start:left_start],
        left=payload[left_start:right_start],
        right=payload[right_start:payload_end],
    )


# ---------------------------------------------------------------------------
# Subscriber handling
# ---------------------------------------------------------------------------


def make_subscriber(topic: str, capacity_bytes: int) -> "Model4Mat.EncodedImageMatPubSub":
    sub = Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.MJPEG,
        color_format=ColorFormat.BGR,
        frame_index=0,
        pts_ns=0,
        dts_ns=0,
        is_keyframe=True,
        width=0,
        height=0,
        valid_nbytes=0,
        data=np.zeros((int(capacity_bytes),), dtype=np.uint8),
    )
    sub.set_id(topic).init()
    sub.is_pub = False
    sub.valid_nbytes = 0
    return sub


@dataclass
class ModeSubscriber:
    mode: RecordMode
    topic: str
    capacity_bytes: int
    subscriber: Any = field(default=None, init=False, repr=False)
    last_returned_frame_index: int | None = None
    last_error: str | None = None

    def open(self) -> None:
        if self.subscriber is None:
            self.subscriber = make_subscriber(self.topic, self.capacity_bytes)

    def read_one(self, timeout_s: float, *, fresh: bool = True) -> CapturedFrame:
        self.open()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        last_error: str | None = None

        while True:
            if time.monotonic() > deadline:
                suffix = f" Last error: {last_error}" if last_error else ""
                raise TimeoutError(f"Timed out waiting for {self.mode} frame on {self.topic!r}.{suffix}")

            pkt = self.subscriber.sub()
            if packet_nbytes(pkt) <= 0:
                time.sleep(0.001)
                continue

            try:
                if self.mode == RGB_STEREO_CAM_MODE:
                    bundle = unpack_dual_rgb_mjpeg_bundle(pkt)
                else:
                    bundle = unpack_rgb_stereo_mjpeg_bundle(pkt)
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.001)
                continue

            frame_index = int(bundle.frame_index)
            if fresh and self.last_returned_frame_index is not None and frame_index == self.last_returned_frame_index:
                time.sleep(0.001)
                continue

            self.last_returned_frame_index = frame_index
            return CapturedFrame(
                mode=self.mode,
                topic=self.topic,
                frame_index=frame_index,
                pts_ns=int(bundle.pts_ns),
                bundle=bundle,
                received_ns_utc=utc_now_ns(),
            )


# ---------------------------------------------------------------------------
# Record writing
# ---------------------------------------------------------------------------


def add_default_calibration(record: CustomRecord, frame: CapturedFrame) -> None:
    if frame.mode == RGB_STEREO_CAM_MODE:
        bundle = frame.bundle
        assert isinstance(bundle, DualRGBMjpegBundle)
        calib = {
            "source": "server_store_default",
            "note": "Replace with real camera calibration when available.",
            "width": bundle.rgb_width,
            "height": bundle.rgb_height,
        }
        record.add_calibration(DUAL_RGB_CAM_NAME_0, {**calib, "camera": DUAL_RGB_CAM_NAME_0})
        record.add_calibration(DUAL_RGB_CAM_NAME_1, {**calib, "camera": DUAL_RGB_CAM_NAME_1})
        record.add_extrinsics(
            {
                "source": "server_store_default",
                "note": "Identity placeholder; replace with measured cam_from to cam_to extrinsics.",
                "from": DUAL_RGB_CAM_NAME_0,
                "to": DUAL_RGB_CAM_NAME_1,
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
    else:
        bundle = frame.bundle
        assert isinstance(bundle, RGBStereoMjpegBundle)
        record.add_calibration(
            RGB_STEREO_CAM_NAME,
            {
                "source": "server_store_default",
                "note": "Replace with real RGB/stereo calibration when available.",
                "camera": RGB_STEREO_CAM_NAME,
                "rgb_width": bundle.rgb_width,
                "rgb_height": bundle.rgb_height,
                "stereo_width": bundle.stereo_width,
                "stereo_height": bundle.stereo_height,
            },
        )


def add_gis_payload(record: CustomRecord, gis: Any | None) -> None:
    """Write canonical GIS files, even if the request does not provide GIS."""
    # default_coordinate_system = {
    #     "source": "server_store_default",
    #     "name": "unknown",
    #     "note": "No coordinate system was provided in the capture request.",
    # }
    # default_location = {
    #     "source": "server_store_default",
    #     "status": "not_provided",
    # }
    # default_pose = {
    #     "source": "server_store_default",
    #     "status": "not_provided",
    #     "frame": "unknown",
    # }

    if gis is None:
        # record.add_gis("location", default_location)
        # record.add_gis("pose", default_pose)
        # record.add_gis("coordinate_system", default_coordinate_system)
        return

    if isinstance(gis, Mapping):
        if "location" in gis or "pose" in gis or "coordinate_system" in gis:
            if "location" in gis:
                record.add_gis("location", gis["location"])
            if "pose" in gis:
                record.add_gis("pose", gis["pose"])
            if "coordinate_system" in gis:
                record.add_gis("coordinate_system", gis["coordinate_system"])

            for optional_kind in ("geofences", "map_notes"):
                if optional_kind in gis:
                    record.add_gis(optional_kind, gis[optional_kind])  # type: ignore[arg-type]
        # else:
        #     # A flat GIS payload, e.g. {lat, lon, alt}. Treat it as the location.
        #     record.add_gis("location", dict(gis))
        #     record.add_gis("pose", default_pose)
        #     record.add_gis("coordinate_system", gis.get("coordinate_system", default_coordinate_system))
        write_json(record.path / "gis" / "request.json", gis)
        return

    # Keep unusual request payloads rather than losing them.
    # record.add_gis("location", {"source": "request", "value": gis})
    # record.add_gis("pose", default_pose)
    # record.add_gis("coordinate_system", default_coordinate_system)
    # write_json(record.path / "gis" / "request.json", {"value": gis})


def save_dual_rgb_images(record: CustomRecord, bundle: DualRGBMjpegBundle) -> list[str]:
    """Save synchronized dual-RGB MJPEG frames as canonical .jpg camera images."""
    saved: list[str] = []
    for camera_id, jpeg in ((DUAL_RGB_CAM_NAME_0, bundle.camera0), (DUAL_RGB_CAM_NAME_1, bundle.camera1)):
        record.add_image(camera_id, "rgb", jpeg)
        saved.append(f"imgs/{camera_id}/rgb.jpg")
    return saved


def save_rgb_stereo_images(record: CustomRecord, bundle: RGBStereoMjpegBundle) -> list[str]:
    """Save synchronized RGB/stereo MJPEG frames as canonical .jpg camera images."""
    saved: list[str] = []
    for stream, jpeg in (("rgb", bundle.rgb), ("left", bundle.left), ("right", bundle.right)):
        record.add_image(RGB_STEREO_CAM_NAME, stream, jpeg)
        saved.append(f"imgs/cam_c/{stream}.jpg")
    return saved


def save_frame_to_store(
    store: CustomStore,
    frame: CapturedFrame,
    *,
    field_id: str,
    gis: Any | None,
) -> tuple[CustomRecord, list[str]]:
    # Use wall-clock UTC for the record directory. Camera pts_ns is a frame PTS, not UTC.
    timestamp_ns_utc = utc_now_ns()
    last_error: Exception | None = None
    for sequence in range(1, 1000):
        try:
            record = store.add_record(
                mode=frame.mode,
                timestamp_ns_utc=timestamp_ns_utc,
                field_id=field_id,
                sequence=sequence,
                exist_ok=False,
            )
            break
        except FileExistsError as exc:
            last_error = exc
            continue
    else:
        raise FileExistsError(f"Could not allocate unique record_id: {last_error}")

    add_default_calibration(record, frame)
    add_gis_payload(record, gis)

    if frame.mode == RGB_STEREO_CAM_MODE:
        assert isinstance(frame.bundle, DualRGBMjpegBundle)
        images = save_dual_rgb_images(record, frame.bundle)
    else:
        assert isinstance(frame.bundle, RGBStereoMjpegBundle)
        images = save_rgb_stereo_images(record, frame.bundle)

    write_json(
        record.path / "logs" / "capture_frame.json",
        {
            "mode": frame.mode,
            "topic": frame.topic,
            "frame_index": frame.frame_index,
            "camera_pts_ns": frame.pts_ns,
            "received_ns_utc": frame.received_ns_utc,
            "saved_ns_utc": timestamp_ns_utc,
        },
    )
    return record, images


# ---------------------------------------------------------------------------
# Downstream hooks
# ---------------------------------------------------------------------------


def call_http_jsonrpc(url: str, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    payload = json.dumps({"jsonrpc": "2.0", "id": int(time.time_ns() & 0x7FFFFFFF),
                          "method": method, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP JSON-RPC call failed for {url!r}: {exc}") from exc

    parsed = json.loads(body) if body else {}
    if isinstance(parsed, Mapping) and parsed.get("error") is not None:
        raise RuntimeError(f"JSON-RPC error from {url!r}: {parsed['error']}")
    return dict(parsed) if isinstance(parsed, Mapping) else {"result": parsed}


def store_raw_hook_result(record: CustomRecord, name: str, result: dict[str, Any]) -> None:
    write_json(record.path / "logs" / f"hook_{name}_result.json", result)


def extract_result_payload(result: dict[str, Any]) -> Any:
    """Unwrap common JSON-RPC result nesting."""
    value: Any = result
    for key in ("http_jsonrpc", "result"):
        if isinstance(value, Mapping) and key in value:
            value = value[key]
    if isinstance(value, Mapping) and "parsed_json" in value:
        return value["parsed_json"]
    return value


def decode_base64_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


# ---------------------------------------------------------------------------
# RPC controller
# ---------------------------------------------------------------------------


@dataclass
class StoreController:
    service_name: str = "jrpc"
    controller_name: str = "store"
    config: StoreConfig = field(default_factory=StoreConfig)
    opened: bool = True

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _subscribers: dict[str, ModeSubscriber] = field(default_factory=dict, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_capture: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @staticmethod
    def openapi_examples() -> dict[str, Any]:
        return {
            **openapi_doc("store_status", id=1, params={}),
            **openapi_doc("store_configure", id=2, params=model_to_dict(StoreConfig())),
            **openapi_doc("store_watch", id=3, params={"mode": DUAL_RGB_CAM_MODE}),
            **openapi_doc(
                "store_capture",
                id=4,
                params={"mode": DUAL_RGB_CAM_MODE, "field_id": "field_01", "gis": None},
            ),
        }

    def _topic_for_mode(self, mode: RecordMode) -> str:
        if mode == RGB_STEREO_CAM_MODE:
            return self.config.dual_rgb_topic
        if mode == DUAL_RGB_CAM_MODE:
            return self.config.rgb_stereo_topic
        raise ValueError(f"Unsupported mode: {mode!r}")

    def _subscriber_for_mode_unlocked(self, mode: RecordMode) -> ModeSubscriber | None:
        return self._subscribers.get(mode)

    def _ensure_subscriber_unlocked(self, mode: RecordMode) -> ModeSubscriber:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
        current = self._subscribers.get(mode)
        topic = self._topic_for_mode(mode)
        if current is not None and current.topic == topic:
            current.open()
            return current
        sub = ModeSubscriber(mode=mode, topic=topic, capacity_bytes=int(self.config.subscriber_capacity_bytes))
        sub.open()
        self._subscribers[mode] = sub
        return sub

    def _status_unlocked(self) -> StoreStatusResult:
        subscribed_modes = sorted(self._subscribers.keys())
        return StoreStatusResult(
            opened=self.opened,
            root_path=str(self.config.root_path),
            watched_modes=subscribed_modes,
            subscribed_modes=subscribed_modes,
            dual_rgb_topic=self.config.dual_rgb_topic,
            rgb_stereo_topic=self.config.rgb_stereo_topic,
            last_error=self._last_error,
            last_capture=self._last_capture,
        )

    def configure(self, params: StoreConfig) -> StoreStatusResult:
        with self._lock:
            try:
                self.config = params or self.config
                # Topic/capacity changes require fresh subscribers.
                self._subscribers.clear()
                self._last_error = None
            except Exception:
                self._last_error = traceback.format_exc()
            return self._status_unlocked()

    def watch(self, params: WatchParams) -> WatchResult:
        with self._lock:
            mode = params.mode
            try:
                sub = self._ensure_subscriber_unlocked(mode)
                self._last_error = None
                return WatchResult(ok=True, mode=mode, topic=sub.topic, subscribed_modes=sorted(self._subscribers.keys()))
            except Exception:
                self._last_error = traceback.format_exc()
                return WatchResult(
                    ok=False,
                    mode=mode,
                    topic=self._topic_for_mode(mode) if mode in VALID_MODES else "",
                    subscribed_modes=sorted(self._subscribers.keys()),
                    error=self._last_error,
                )

    def unwatch(self, params: WatchParams) -> WatchResult:
        with self._lock:
            mode = params.mode
            self._subscribers.pop(mode, None)
            return WatchResult(ok=True, mode=mode, topic=self._topic_for_mode(mode), subscribed_modes=sorted(self._subscribers.keys()))

    def status(self, params: EmptyParams) -> StoreStatusResult:
        with self._lock:
            return self._status_unlocked()

    def capture(self, params: CaptureParams) -> CaptureResult:
        with self._lock:
            mode = params.mode
            field_id = str(params.field_id or "field_01")
            try:
                sub = self._subscriber_for_mode_unlocked(mode)
                if sub is None:
                    if not self.config.auto_subscribe_on_capture:
                        raise RuntimeError(f"Mode {mode!r} is not subscribed. Call watch(mode={mode!r}) first.")
                    sub = self._ensure_subscriber_unlocked(mode)

                timeout_s = float(params.capture_timeout_s or self.config.capture_timeout_s)
                frame = sub.read_one(timeout_s, fresh=bool(params.fresh_frame))

                store = CustomStore(self.config.root_path)
                record, images = save_frame_to_store(
                    store,
                    frame,
                    field_id=field_id,
                    gis=params.gis,
                )

                for hook_url in params.hook_urls:
                    try:
                        # hook_url = "http://localhost:8000/controllers/yolo/start"
                        # db_record = {
                        #     "root_path": ".",
                        #     "mode": "dual_rgb",
                        #     "field_id": "null",
                        #     "record_id": "000000.000000000JST",
                        #     "timestamp_ns_utc": 0,
                        #     "date_utc": "1970-01-01",
                        #     "path": r"dual_rgb\1970-01-01\null\000000.000000000JST",
                        # }
                        params = {"db_record": json.dumps(record)}
                        def push_request(hook_url=hook_url, params=params):
                            try:
                                requests.get(hook_url, params=params, timeout=0.5)
                            except requests.RequestException:
                                pass  # ignore timeout / connection errors
                        executor = ThreadPoolExecutor(max_workers=8)
                        executor.submit(push_request)

                    except Exception:
                        pass


                result_dict = {
                    "ok": True,
                    "mode": mode,
                    "field_id": field_id,
                    "record_id": record.record_id,
                    "date_utc": record.date_utc,
                    "record_path": record.path.as_posix(),
                    "topic": frame.topic,
                    "frame_index": frame.frame_index,
                    "timestamp_ns_utc": record.timestamp_ns_utc,
                    "images": images,
                }
                self._last_capture = result_dict
                self._last_error = None
                return CaptureResult(**result_dict)
            except Exception:
                self._last_error = traceback.format_exc()
                result = CaptureResult(ok=False, mode=mode, field_id=field_id, error=self._last_error)
                self._last_capture = model_to_dict(result)
                return result


def run_server(service_name: str = "jrpc", controller_name: str = "store") -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    Iox2JsonRpcServer(StoreController(service_name=service_name, controller_name=controller_name)).run_forever()


if __name__ == "__main__":
    run_server(service_name=args.service_name, controller_name=args.controller_name)
