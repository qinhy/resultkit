#!/usr/bin/env python3
"""
Tiny debug viewer for the dual-RGB MJPEG bundle published by server_dual_rgb.py.

Run:
    python debug_vis_server_dual_rgb.py

Typical custom topic:
    python debug_vis_server_dual_rgb.py --topic OkadCamA:camera:dual_rgb_mjpeg

Controls:
    q or ESC  quit

This viewer is intentionally the only place that uses cv2. The camera server stays
encoded-byte-only.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Keep this like the resultkit examples: allow running from a tests/examples folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


DEFAULT_TOPIC = "OkadCamA:camera:dual_rgb_mjpeg"
DEFAULT_CAPACITY_BYTES = 64 * 1024 * 1024

BUNDLE_MAGIC = b"DRGB"
BUNDLE_VERSION = 1
BUNDLE_HEADER = struct.Struct("<4sHHQQIIII")


@dataclass(frozen=True)
class Bundle:
    frame_index: int
    pts_ns: int
    rgb_width: int
    rgb_height: int
    camera0: bytes
    camera1: bytes


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


def unpack_bundle(pkt: Any) -> Bundle:
    payload = packet_payload_bytes(pkt)
    if len(payload) < BUNDLE_HEADER.size:
        raise ValueError(f"bundle too small: {len(payload)} bytes")

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
    ) = BUNDLE_HEADER.unpack(payload[: BUNDLE_HEADER.size])

    if magic != BUNDLE_MAGIC:
        raise ValueError(f"bad bundle magic: {magic!r}")
    if int(version) != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version: {version}")
    if int(header_nbytes) < BUNDLE_HEADER.size:
        raise ValueError(f"bad bundle header size: {header_nbytes}")

    camera0_start = int(header_nbytes)
    camera1_start = camera0_start + int(camera0_nbytes)
    end = camera1_start + int(camera1_nbytes)
    if end > len(payload):
        raise ValueError(f"truncated bundle: need {end} bytes, got {len(payload)}")

    return Bundle(
        frame_index=int(frame_index),
        pts_ns=int(pts_ns),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        camera0=payload[camera0_start:camera1_start],
        camera1=payload[camera1_start:end],
    )


def decode_jpeg(payload: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


def maybe_resize(img: np.ndarray, scale: float) -> np.ndarray:
    scale = float(scale)
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = max(1, int(w * (height / max(h, 1))))
    return cv2.resize(img, (width, height))


def put_text(img: np.ndarray, text: str) -> np.ndarray:
    out = img
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug imshow viewer for one dual-RGB MJPEG bundle topic.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--capacity-bytes", type=int, default=DEFAULT_CAPACITY_BYTES)
    parser.add_argument("--rgb-scale", type=float, default=0.20)
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--stats-every", type=int, default=30)
    parser.add_argument("--separate-windows", action="store_true", help="Show camera0 and camera1 in separate OpenCV windows")
    args = parser.parse_args()

    sub = make_subscriber(args.topic, args.capacity_bytes)
    print(f"Subscribing: {args.topic!r}")
    print("Press q or ESC to quit.")

    frames = 0
    t0 = time.perf_counter()

    while True:
        pkt = sub.sub()
        if packet_nbytes(pkt) <= 0:
            time.sleep(0.001)
            continue

        bundle = unpack_bundle(pkt)
        camera0 = decode_jpeg(bundle.camera0)
        camera1 = decode_jpeg(bundle.camera1)

        frames += 1
        label = f"frame={bundle.frame_index} cam0={len(bundle.camera0)} cam1={len(bundle.camera1)}"

        camera0_view = put_text(maybe_resize(camera0, args.rgb_scale), f"camera0 | {label}")
        camera1_view = put_text(maybe_resize(camera1, args.rgb_scale), f"camera1 | {label}")

        if args.separate_windows:
            cv2.imshow("camera0", camera0_view)
            cv2.imshow("camera1", camera1_view)
        else:
            h = min(camera0_view.shape[0], camera1_view.shape[0])
            camera0_side = resize_to_height(camera0_view, h)
            camera1_side = resize_to_height(camera1_view, h)
            cv2.imshow("dual rgb", np.hstack((camera0_side, camera1_side)))

        if frames % max(1, args.stats_every) == 0:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            print(f"frames={frames}, fps={frames / elapsed:.2f}, {label}")

        key = cv2.waitKey(max(1, int(args.wait_ms))) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
