#!/usr/bin/env python3
"""Tiny rgb_stereo multi-stream record-store RPC server v5.

Public RPC API uses stream lists only:
    watch({"stream_ids": ["left", "right"]})
    capture({"stream_ids": ["left", "right"], "field_id": "field_all"})

For one camera, still pass a list:
    capture({"stream_ids": ["left"]})
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import requests

from common import EmptyParams, HookDispatcher, RpcModel, openapi_doc, CustomStore, RecordPath, CustomRecord
from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat
from server_rgb_stereo import BUNDLE_MAGIC, BUNDLE_VERSION, BUNDLE_FORMAT, BUNDLE_HEADER, BUNDLE_PREFIX, BUNDLE_TYPE
# from store.custom_record_store import RGB_STEREO_CAM_NAME
from resultkit.logger import logger

VERSION = "v5"
RECORD_MODE = "rgb_stereo"
DEFAULT_CONTROLLER = "cameraRgbd"
REST_URL_BASE="http://localhost:8000"

parser = argparse.ArgumentParser(description="Run the rgb_stereo multi-stream record-store RPC server.")
parser.add_argument("--service-name", default="jrpc")
parser.add_argument("--controller-name", default="store")
parser.add_argument("--store-root", default="recording")
parser.add_argument("--record-mode", default="rgb_stereo")
parser.add_argument("--camera-service-name", default=None)
parser.add_argument("--rgbd-controller", type=str, default=DEFAULT_CONTROLLER, help="comma-separated controllers, e.g. rgbd_left,rgbd_right")
parser.add_argument("--rgbd-stream", type=str, default=None, help="comma-separated stream IDs aligned with controllers")
parser.add_argument("--subscriber-capacity-bytes", type=int, default=64 * 1024 * 1024)
parser.add_argument("--capture-timeout-s", type=float, default=3.0)
args = parser.parse_args()


def csv(text: str | None) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]

RECORD_MODE = args.record_mode
CONTROLLERS = csv(args.rgbd_controller) or [DEFAULT_CONTROLLER]
STREAM_IDS = csv(args.rgbd_stream) or CONTROLLERS
CAMERA_SERVICE = args.camera_service_name or args.service_name
STREAMS = {
    (STREAM_IDS[i] if i < len(STREAM_IDS) else controller): f"{CAMERA_SERVICE}:{controller}:{BUNDLE_PREFIX}_{BUNDLE_TYPE}"
    for i, controller in enumerate(CONTROLLERS)
}
ThreadExecutor = ThreadPoolExecutor(max_workers=8)

logger(f"[{args.service_name}:{args.controller_name}:init] module configuration loaded",
    extra={
        "version": VERSION,
        "service_name": args.service_name,
        "controller_name": args.controller_name,
        "store_root": args.store_root,
        "streams": STREAMS,
    },
)


def default_rgbd_rest(stream_id: str,func="status") -> str:
    return f"{REST_URL_BASE}/controllers/{stream_id}/{func}"

class StoreModel(RpcModel):
    service: str = args.service_name


class StoreConfig(StoreModel):
    version: str = VERSION
    root_path: str = args.store_root
    streams: dict[str, str] = STREAMS
    subscriber_capacity_bytes: int = args.subscriber_capacity_bytes
    capture_timeout_s: float = args.capture_timeout_s


class StreamsParams(StoreModel):
    stream_ids: list[str] = STREAM_IDS


class CaptureParams(StreamsParams):
    field_id: str = "field_all"
    meta: dict = field(default_factory=dict)
    capture_timeout_s: float | None = None
    fresh_frame: bool = True
    # hook_urls:list[list[str]] = [[]]
    hook_urls:list[list[str]] = [
            ["http://localhost:8000/controllers/yolo/start",
             "http://localhost:8000/controllers/pcd/to_pcd"
            ],
    ]
    how_to_use_meta:str="""{{ "key1":{{data1 ...}} }}, 
        {{ "gnss":{{ ... }},
        "arm":{{ "run_id":{{...}}, "data":{{...}} }}, }}"""


class StatusResult(StoreModel):
    opened: bool
    root_path: str
    streams: dict[str, str]
    subscribed_streams: list[str]
    last_error: str | None = None
    last_capture: CaptureResult | None = None


class WatchResult(StoreModel):
    ok: bool
    stream_ids: list[str]
    subscribed_streams: list[str]
    error: str | None = None


class StreamCapture(StoreModel):
    ok: bool
    stream_id: str
    field_id: str
    record_id: str | None = None
    date_utc: str | None = None
    record_path: str | None = None
    topic: str | None = None
    frame_index: int | None = None
    timestamp_ns_utc: int | None = None
    images: list[str] = []
    db_record: dict | None = None
    error: str | None = None


class CaptureResult(StoreModel):
    ok: bool
    stream_ids: list[str]
    field_id: str
    captures: list[StreamCapture]
    error: str | None = None


@dataclass(frozen=True)
class Bundle:
    frame_index: int
    pts_ns: int
    rgb: bytes
    left: bytes
    right: bytes


@dataclass(frozen=True)
class Frame:
    stream_id: str
    topic: str
    frame_index: int
    pts_ns: int
    bundle: Bundle
    received_ns_utc: int


@dataclass
class Subscriber:
    stream_id: str
    topic: str
    capacity_bytes: int
    sub: Any = field(default=None, init=False, repr=False)
    last_frame_index: int | None = None

    def open(self) -> None:
        if self.sub is None:
            logger(f"[{args.service_name}:{args.controller_name}:Subscriber:open] opening subscriber",
                extra={
                    "stream_id": self.stream_id,
                    "topic": self.topic,
                    "capacity_bytes": self.capacity_bytes,
                },
            )
            try:
                self.sub = Model4Mat.EncodedImageMatPubSub(
                    codec=CodecFormat.MJPEG,
                    color_format=ColorFormat.BGR,
                    frame_index=0,
                    pts_ns=0,
                    dts_ns=0,
                    is_keyframe=True,
                    width=0,
                    height=0,
                    valid_nbytes=0,
                    data=np.zeros((self.capacity_bytes,), dtype=np.uint8),
                )
                self.sub.set_id(self.topic).init()
                self.sub.is_pub = False
                self.sub.valid_nbytes = 0
                logger(f"[{args.service_name}:{args.controller_name}:Subscriber:open] subscriber opened",
                    extra={"stream_id": self.stream_id, "topic": self.topic},
                )
            except Exception:
                self.sub = None
                logger(f"[{args.service_name}:{args.controller_name}:Subscriber:open:error] failed to open subscriber",level="error",
                    extra={"stream_id": self.stream_id, "topic": self.topic},
                )
                raise

    def read(self, timeout_s: float, fresh: bool) -> Frame:
        self.open()
        logger(f"[{args.service_name}:{args.controller_name}:Subscriber:read] waiting for frame",
            extra={
                "stream_id": self.stream_id,
                "topic": self.topic,
                "timeout_s": timeout_s,
                "fresh": fresh,
                "last_frame_index": self.last_frame_index,
            },
        )
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        last_error = None
        invalid_packet_count = 0
        while time.monotonic() <= deadline:
            pkt = self.sub.sub()
            if packet_size(pkt) <= 0:
                time.sleep(0.001)
                continue
            try:
                bundle = unpack(pkt)
            except Exception as exc:
                last_error = str(exc)
                invalid_packet_count += 1
                time.sleep(0.001)
                continue
            if fresh and bundle.frame_index == self.last_frame_index:
                time.sleep(0.001)
                continue
            self.last_frame_index = bundle.frame_index
            logger(f"[{args.service_name}:{args.controller_name}:Subscriber:read] frame received",
                extra={
                    "stream_id": self.stream_id,
                    "topic": self.topic,
                    "frame_index": bundle.frame_index,
                    "pts_ns": bundle.pts_ns,
                    "invalid_packet_count": invalid_packet_count,
                },
            )
            return Frame(self.stream_id, self.topic, bundle.frame_index, bundle.pts_ns, bundle, time.time_ns())
        suffix = f" Last error: {last_error}" if last_error else ""
        logger(f"[{args.service_name}:{args.controller_name}:Subscriber:read] timed out waiting for frame",
            level="warning",
            extra={
                "stream_id": self.stream_id,
                "topic": self.topic,
                "timeout_s": timeout_s,
                "fresh": fresh,
                "invalid_packet_count": invalid_packet_count,
                "last_error": last_error,
            },
        )
        raise TimeoutError(f"Timed out waiting for {self.stream_id!r} on {self.topic!r}.{suffix}")


def packet_size(pkt: Any) -> int:
    return int(pkt.nbytes()) if hasattr(pkt, "nbytes") and callable(pkt.nbytes) else int(getattr(pkt, "valid_nbytes", 0) or 0)


def packet_bytes(pkt: Any) -> bytes:
    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        return payload.tobytes() if isinstance(payload, np.ndarray) else bytes(payload)
    arr = np.asarray(getattr(pkt, "data", pkt), dtype=np.uint8).reshape(-1)
    return arr[: max(0, min(packet_size(pkt), int(arr.size)))].tobytes()


def unpack(pkt: Any) -> Bundle:
    payload = packet_bytes(pkt)
    if len(payload) < BUNDLE_HEADER.size:
        raise ValueError(f"Payload too small for {BUNDLE_FORMAT}: {len(payload)} bytes")
    magic, version, header_nbytes, frame_index, pts_ns, _rw, _rh, _sw, _sh, rgb_n, left_n, right_n = BUNDLE_HEADER.unpack(payload[: BUNDLE_HEADER.size])
    if magic != BUNDLE_MAGIC or int(version) != BUNDLE_VERSION or int(header_nbytes) < BUNDLE_HEADER.size:
        raise ValueError("Invalid rgb_stereo MJPEG bundle header")
    rgb_start = int(header_nbytes)
    left_start = rgb_start + int(rgb_n)
    right_start = left_start + int(left_n)
    end = right_start + int(right_n)
    if end > len(payload):
        raise ValueError(f"Truncated {BUNDLE_FORMAT}: need {end}, got {len(payload)}")
    return Bundle(int(frame_index), int(pts_ns), payload[rgb_start:left_start], payload[left_start:right_start], payload[right_start:end])


def write_json(path: RecordPath, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger(f"[{args.service_name}:{args.controller_name}:write_json] wrote JSON file", extra={"path": path.as_posix()})


def save_frame(cam_name:str, store: CustomStore, frame: Frame, field_id: str,
               ts, meta: dict, calib: Any | None) -> StreamCapture:    
    # ts = time.time_ns()
    logger(f"[{args.service_name}:{args.controller_name}:save_frame] saving frame",
        extra={
            "stream_id": frame.stream_id,
            "camera_name": cam_name,
            "field_id": field_id,
            "frame_index": frame.frame_index,
            "timestamp_ns": ts,
        },
    )
    for sequence in range(1, 1000):
        try:
            record = store.add_record(RECORD_MODE, ts, field_id, sequence, exist_ok=True)
            break
        except FileExistsError:
            pass
    else:
        raise FileExistsError("Could not allocate unique record_id")

    logger(f"[{args.service_name}:{args.controller_name}:save_frame] record allocated",
        extra={
            "stream_id": frame.stream_id,
            "record_id": record.record_id,
            "record_path": record.path.as_posix(),
            "sequence": sequence,
        },
    )

    if "gnss" in meta:
        gnss:dict = meta["gnss"]
        if gnss is None or len(gnss)==0: return
        logger(f"[{args.service_name}:{args.controller_name}:add_gnss] saving GIS data",
            extra={"record_id": getattr(record, "record_id", None), "mapping": isinstance(gnss, Mapping)},
        )
        record.add_gnss(gnss, kind="baselink")
        
    if "arm" in meta:
        arm:dict = meta["arm"]
        if arm is None or len(arm)==0: return
        logger(f"[{args.service_name}:{args.controller_name}:add_arm] saving GIS data",
            extra={"record_id": getattr(record, "record_id", None), "mapping": isinstance(arm, Mapping)},
        )
        if arm.get("run_id",None) is None:
            logger(f"[{args.service_name}:{args.controller_name}:add_arm:error] arm data has no run_id")
        else:
            data = arm.get("data",arm)
            record.get_arm().add_result(arm["run_id"],data)

    images = []
    for name, data in (("rgb", frame.bundle.rgb), ("left", frame.bundle.left), ("right", frame.bundle.right)):
        record.add_image(cam_name, name, data)
        images.append(f"imgs/{cam_name}/{name}.jpg")

    logger(f"[{args.service_name}:{args.controller_name}:save_frame] images saved",
        extra={
            "stream_id": frame.stream_id,
            "record_id": record.record_id,
            "images": images,
        },
    )

    if calib:
        record.add_calibration(cam_name,calib)
        logger(f"[{args.service_name}:{args.controller_name}:save_frame] calibration saved",
            extra={"stream_id": frame.stream_id, "record_id": record.record_id},
        )

    write_json(record.path / "logs" / "capture_frame.json", {
        "stream_id": frame.stream_id,
        "topic": frame.topic,
        "frame_index": frame.frame_index,
        "camera_pts_ns": frame.pts_ns,
        "received_ns_utc": frame.received_ns_utc,
        "saved_ns_utc": ts,
    })
    result = StreamCapture(
        ok=True,
        stream_id=frame.stream_id,
        field_id=field_id,
        record_id=record.record_id,
        date_utc=record.date_utc,
        record_path=record.path.as_posix(),
        topic=frame.topic,
        frame_index=frame.frame_index,
        timestamp_ns_utc=record.timestamp_ns_utc,
        images=images,
        db_record=json.loads(record.model_dump_json()),
    )
    logger(f"[{args.service_name}:{args.controller_name}:save_frame] frame saved",
        extra={
            "stream_id": frame.stream_id,
            "record_id": record.record_id,
            "record_path": record.path.as_posix(),
            "frame_index": frame.frame_index,
        },
    )
    return result


def dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(getattr(model, "__dict__", {}))


@dataclass
class StoreController:
    service_name: str = args.service_name
    controller_name: str = args.controller_name
    config: StoreConfig = field(default_factory=StoreConfig)
    opened: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _subs: dict[str, Subscriber] = field(default_factory=dict, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_capture: CaptureResult | None = field(default=None, init=False, repr=False)
    _calibration_dicts: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _hooks: HookDispatcher = field(default_factory=HookDispatcher,init=False,repr=False)

    def __post_init__(self) -> None:
        logger(f"[{args.service_name}:{args.controller_name}:init] controller initialized",
            extra={
                "service_name": self.service_name,
                "controller_name": self.controller_name,
                "root_path": self.config.root_path,
                "streams": self.config.streams,
            },
        )

    @staticmethod
    def openapi_examples() -> dict[str, Any]:
        return {
            **openapi_doc("store_status", id=1, params={}),
            **openapi_doc("store_watch", id=2, params={"stream_ids": STREAM_IDS}),
            **openapi_doc("store_capture", id=3, params={"stream_ids": STREAM_IDS, "field_id": "field_all", "gnss": None}),
        }

    def _check(self, stream_ids: list[str]) -> list[str]:
        if not isinstance(stream_ids, list) or not stream_ids or not all(isinstance(x, str) and x for x in stream_ids):
            logger(f"[{args.service_name}:{args.controller_name}:check] invalid stream_ids",
                level="warning",
                extra={"stream_ids": stream_ids},
            )
            raise ValueError("stream_ids must be a non-empty list of strings, e.g. ['left'] or ['left', 'right']")
        unknown = [x for x in stream_ids if x not in self.config.streams]
        if unknown:
            logger(f"[{args.service_name}:{args.controller_name}:check] unknown stream_ids",
                level="warning",
                extra={"unknown": unknown, "known": sorted(self.config.streams)},
            )
            raise ValueError(f"Unknown stream_ids {unknown!r}; known={sorted(self.config.streams)}")
        return stream_ids

    def _sub(self, stream_id: str) -> Subscriber:
        topic = self.config.streams[stream_id]
        sub = self._subs.get(stream_id)
        if sub is None or sub.topic != topic:
            logger(f"[{args.service_name}:{args.controller_name}:sub] creating subscriber",
                extra={"stream_id": stream_id, "topic": topic},
            )
            sub = Subscriber(stream_id, topic, int(self.config.subscriber_capacity_bytes))
            sub.open()
            self._subs[stream_id] = sub
        else:
            logger(f"[{args.service_name}:{args.controller_name}:sub] reusing subscriber",
                extra={"stream_id": stream_id, "topic": topic},
            )
        return sub

    def _status(self) -> StatusResult:
        return StatusResult(
            opened=self.opened,
            root_path=str(self.config.root_path),
            streams=dict(self.config.streams),
            subscribed_streams=sorted(self._subs),
            last_error=self._last_error,
            last_capture=self._last_capture,
        )

    def configure(self, params: StoreConfig) -> StatusResult:
        with self._lock:
            logger(f"[{args.service_name}:{args.controller_name}:configure] applying configuration",
                extra={"config": dump_model(params) if params is not None else None},
            )
            self.config = params or self.config
            self._subs.clear()
            self._last_error = None
            result = self._status()
            logger(f"[{args.service_name}:{args.controller_name}:configure] configuration applied",
                extra={"root_path": result.root_path, "streams": result.streams},
            )
            return result

    def status(self, params: EmptyParams) -> StatusResult:
        with self._lock:
            result = self._status()
            logger(f"[{args.service_name}:{args.controller_name}:status] status requested",
                extra={
                    "opened": result.opened,
                    "subscribed_streams": result.subscribed_streams,
                    "has_last_error": result.last_error is not None,
                },
            )
            return result

    def _set_calib(self, stream_id):
        if stream_id not in self._calibration_dicts:
            url = default_rgbd_rest(stream_id,"calibration")
            logger(f"[{args.service_name}:{args.controller_name}:calibration] requesting calibration",
                extra={"stream_id": stream_id, "url": url},
            )
            calib = requests.get(url)
            self._calibration_dicts[stream_id] = calib.json()
            if "calibration" in self._calibration_dicts[stream_id]:
                self._calibration_dicts[stream_id] = self._calibration_dicts[stream_id]["calibration"]
            logger(f"[{args.service_name}:{args.controller_name}:calibration] calibration cached",
                extra={
                    "stream_id": stream_id,
                    "status_code": calib.status_code,
                    "keys": list(self._calibration_dicts[stream_id]),
                },
            )
        else:
            logger(f"[{args.service_name}:{args.controller_name}:calibration] using cached calibration",
                extra={"stream_id": stream_id},
            )

    def watch(self, params: StreamsParams) -> WatchResult:
        with self._lock:
            logger(f"[{args.service_name}:{args.controller_name}:watch] watch requested",
                extra={"stream_ids": getattr(params, "stream_ids", [])},
            )
            try:
                stream_ids = self._check(params.stream_ids)
                for stream_id in stream_ids:
                    self._sub(stream_id)
                    self._set_calib(stream_id)
                self._last_error = None
                result = WatchResult(ok=True, stream_ids=stream_ids, subscribed_streams=sorted(self._subs))
                logger(f"[{args.service_name}:{args.controller_name}:watch] streams watched",
                    extra={"stream_ids": stream_ids, "subscribed_streams": result.subscribed_streams},
                )
                return result
            except Exception:
                self._last_error = traceback.format_exc()
                logger(f"[{args.service_name}:{args.controller_name}:watch:error] watch failed",level="error",
                    extra={"stream_ids": getattr(params, "stream_ids", [])},
                )
                return WatchResult(ok=False, stream_ids=getattr(params, "stream_ids", []), subscribed_streams=sorted(self._subs), error=self._last_error)

    def unwatch(self, params: StreamsParams) -> WatchResult:
        with self._lock:
            logger(f"[{args.service_name}:{args.controller_name}:unwatch] unwatch requested",
                extra={"stream_ids": getattr(params, "stream_ids", [])},
            )
            try:
                stream_ids = self._check(params.stream_ids)
                for stream_id in stream_ids:
                    self._subs.pop(stream_id, None)
                self._last_error = None
                result = WatchResult(ok=True, stream_ids=stream_ids, subscribed_streams=sorted(self._subs))
                logger(f"[{args.service_name}:{args.controller_name}:unwatch] streams unwatched",
                    extra={"stream_ids": stream_ids, "subscribed_streams": result.subscribed_streams},
                )
                return result
            except Exception:
                self._last_error = traceback.format_exc()
                logger(f"[{args.service_name}:{args.controller_name}:unwatch:error] unwatch failed",level="error",
                    extra={"stream_ids": getattr(params, "stream_ids", [])},
                )
                return WatchResult(ok=False, stream_ids=getattr(params, "stream_ids", []), subscribed_streams=sorted(self._subs), error=self._last_error)

    def capture(self, params: CaptureParams) -> CaptureResult:
        with self._lock:
            field_id = str(params.field_id or "field_all")
            logger(f"[{args.service_name}:{args.controller_name}:capture] capture requested",
                extra={
                    "stream_ids": getattr(params, "stream_ids", []),
                    "field_id": field_id,
                    "fresh_frame": getattr(params, "fresh_frame", True),
                    "capture_timeout_s": getattr(params, "capture_timeout_s", None),
                },
            )
            try:
                stream_ids = self._check(params.stream_ids)
                subs = {}
                for stream_id in stream_ids:
                    subs[stream_id] = self._sub(stream_id)
                    self._set_calib(stream_id)
                timeout_s = float(params.capture_timeout_s or self.config.capture_timeout_s)
                captures: list[StreamCapture] = []
                store = CustomStore(self.config.root_path)
                logger(f"[{args.service_name}:{args.controller_name}:capture] subscribers ready",
                    extra={
                        "stream_ids": stream_ids,
                        "timeout_s": timeout_s,
                        "root_path": self.config.root_path,
                    },
                )

                with ThreadPoolExecutor(max_workers=len(subs)) as pool:
                    futures = {pool.submit(sub.read, timeout_s, bool(params.fresh_frame)): stream_id for stream_id, sub in subs.items()}
                    frames: dict[str, Frame] = {}
                    for future in as_completed(futures):
                        stream_id = futures[future]
                        try:
                            frames[stream_id] = future.result()
                            logger(f"[{args.service_name}:{args.controller_name}:capture] frame acquired",
                                extra={
                                    "stream_id": stream_id,
                                    "frame_index": frames[stream_id].frame_index,
                                    "topic": frames[stream_id].topic,
                                },
                            )
                        except Exception:
                            error = traceback.format_exc()
                            logger(f"[{args.service_name}:{args.controller_name}:capture:error] frame acquisition failed",level="error",
                                extra={"stream_id": stream_id, "field_id": field_id},
                            )
                            captures.append(StreamCapture(ok=False, stream_id=stream_id, field_id=field_id,
                                    topic=self.config.streams[stream_id], error=error))

                ts = time.time_ns()
                for stream_id in stream_ids:
                    if stream_id not in frames:continue

                    try:
                        calib = self._calibration_dicts.get(stream_id)
                        captures.append(save_frame(stream_id, store, frames[stream_id],
                                                    field_id, ts, params.meta,
                                                    calib))
                        logger(f"[{args.service_name}:{args.controller_name}:capture] stream capture saved ({stream_id})",
                            extra={
                                "stream_id": stream_id,
                                "field_id": field_id,
                                "record_id": captures[-1].record_id,
                            },
                        )
                    except Exception:
                        error = traceback.format_exc()
                        logger(f"[{args.service_name}:{args.controller_name}:capture:error] failed to save stream capture",level="error",
                            extra={"stream_id": stream_id, "field_id": field_id},
                        )
                        captures.append(StreamCapture(ok=False, stream_id=stream_id, field_id=field_id,
                                topic=self.config.streams[stream_id], error=error))

                captures.sort(key=lambda x: stream_ids.index(x.stream_id))
                ok = all(x.ok for x in captures)
                error = None if ok else "\n".join(f"{x.stream_id}: {x.error}" for x in captures if not x.ok)
                result = CaptureResult(ok=ok, stream_ids=stream_ids, field_id=field_id, captures=captures, error=error)
                self._last_capture = result
                self._last_error = error
                logger(f"[{args.service_name}:{args.controller_name}:capture] capture completed",
                    level="info" if ok else "warning",
                    extra={
                        "ok": ok,
                        "stream_ids": stream_ids,
                        "field_id": field_id,
                        "success_count": sum(1 for capture in captures if capture.ok),
                        "failure_count": sum(1 for capture in captures if not capture.ok),
                    },
                )

                db_records = {x.db_record["record_id"]:x for x in captures if x.db_record is not None}
                for capture in db_records.values():
                    db_record = capture.db_record
                    if db_record is not None:
                        logger(f"[{args.service_name}:{args.controller_name}:capture] dispatching hooks ({capture.record_id})",
                            extra={
                                "stream_id": stream_id,
                                "record_id": capture.record_id,
                                "hook_chains": params.hook_urls,
                            },
                        )
                        self._hooks.dispatch(
                            db_record=db_record,
                            hook_chains=params.hook_urls,
                        )
                        logger(f"[{args.service_name}:{args.controller_name}:capture] hooks dispatched",
                            extra={"stream_id": stream_id, "record_id": capture.record_id},
                        )

                return result
            except Exception:
                self._last_error = traceback.format_exc()
                logger(f"[{args.service_name}:{args.controller_name}:capture:error] capture failed",level="error",
                    extra={
                        "stream_ids": getattr(params, "stream_ids", []),
                        "field_id": field_id,
                    },
                )
                result = CaptureResult(ok=False, stream_ids=getattr(params, "stream_ids", []), field_id=field_id, captures=[], error=self._last_error)
                self._last_capture = dump_model(result)
                return result


def run_server(service_name: str = args.service_name, controller_name: str = args.controller_name) -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer
    logger(f"[{args.service_name}:{controller_name}:run_server] starting RPC server",
        extra={"service_name": service_name, "controller_name": controller_name},
    )
    try:
        Iox2JsonRpcServer(StoreController(service_name=service_name, controller_name=controller_name)).run_forever()
    except KeyboardInterrupt:
        logger(f"[{args.service_name}:{controller_name}:run_server] server interrupted", level="warning")
        raise
    except Exception:
        logger(f"[{args.service_name}:{args.controller_name}:run_server:error] server stopped with an error",level="error",
            extra={"service_name": service_name, "controller_name": controller_name},
        )
        raise
    finally:
        logger(f"[{args.service_name}:{controller_name}:run_server] server stopped")


if __name__ == "__main__":
    run_server()
