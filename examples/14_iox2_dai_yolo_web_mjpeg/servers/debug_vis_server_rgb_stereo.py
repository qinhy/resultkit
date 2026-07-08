#!/usr/bin/env python3
"""
Tiny debug viewer for the RGB+stereo MJPEG bundle published by server_dai_clean_mjpeg_bundle.py.

Run:
    python debug_vis_server_rgb_stereo.py

Typical custom topic:
    python debug_vis_server_rgb_stereo.py --topic OkadCamA:camera:rgb_stereo_mjpeg

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


DEFAULT_TOPIC = "OkadCamA:camera:rgb_stereo_mjpeg"
DEFAULT_CAPACITY_BYTES = 64 * 1024 * 1024

BUNDLE_MAGIC = b"RSMJ"
BUNDLE_VERSION = 1
BUNDLE_HEADER = struct.Struct("<4sHHQQIIIIIII")


@dataclass(frozen=True)
class Bundle:
    frame_index: int
    pts_ns: int
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    rgb: bytes
    left: bytes
    right: bytes


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
        stereo_width,
        stereo_height,
        rgb_nbytes,
        left_nbytes,
        right_nbytes,
    ) = BUNDLE_HEADER.unpack(payload[: BUNDLE_HEADER.size])

    if magic != BUNDLE_MAGIC:
        raise ValueError(f"bad bundle magic: {magic!r}")
    if int(version) != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version: {version}")
    if int(header_nbytes) < BUNDLE_HEADER.size:
        raise ValueError(f"bad bundle header size: {header_nbytes}")

    rgb_start = int(header_nbytes)
    left_start = rgb_start + int(rgb_nbytes)
    right_start = left_start + int(left_nbytes)
    end = right_start + int(right_nbytes)
    if end > len(payload):
        raise ValueError(f"truncated bundle: need {end} bytes, got {len(payload)}")

    return Bundle(
        frame_index=int(frame_index),
        pts_ns=int(pts_ns),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        stereo_width=int(stereo_width),
        stereo_height=int(stereo_height),
        rgb=payload[rgb_start:left_start],
        left=payload[left_start:right_start],
        right=payload[right_start:end],
    )


def decode_jpeg(payload: bytes, flags: int) -> np.ndarray:
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


def put_text(img: np.ndarray, text: str) -> np.ndarray:
    out = img
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug imshow viewer for one RGB+stereo MJPEG bundle topic.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--capacity-bytes", type=int, default=DEFAULT_CAPACITY_BYTES)
    parser.add_argument("--rgb-scale", type=float, default=0.25)
    parser.add_argument("--stereo-scale", type=float, default=0.5)
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--stats-every", type=int, default=30)
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
        rgb = decode_jpeg(bundle.rgb, cv2.IMREAD_COLOR)
        left = decode_jpeg(bundle.left, cv2.IMREAD_GRAYSCALE)
        right = decode_jpeg(bundle.right, cv2.IMREAD_GRAYSCALE)

        frames += 1
        label = f"frame={bundle.frame_index} rgb={len(bundle.rgb)} L={len(bundle.left)} R={len(bundle.right)}"

        rgb_view = put_text(maybe_resize(rgb, args.rgb_scale), label)
        left_view = maybe_resize(left, args.stereo_scale)
        right_view = maybe_resize(right, args.stereo_scale)
        stereo_view = put_text(np.hstack((left_view, right_view)), "left | right")

        cv2.imshow("rgb", rgb_view)
        cv2.imshow("stereo", stereo_view)

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
