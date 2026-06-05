#!/usr/bin/env python3
"""
Minimal H264 file -> NVDEC -> CUDA IPC -> YOLO -> FastAPI monitor demo.

Runtime pipeline:

    .h264 Annex-B file
        -> access-unit splitter, one encoded frame at a time
        -> PyNvVideoCodec / NVDEC, decoded in CUDA memory
        -> resultkit Model4Mat.ImageMatCUDAPubSub CUDA IPC image
        -> YOLO CUDA worker
        -> FastAPI MJPEG monitor

Run in three terminals:

    # Terminal 1: read .h264, NVDEC decode, publish CUDA IPC image
    python cil.py decode-pub --input demo.h264 --width 1280 --height 720 --fps 30 --loop

    # Terminal 2: subscribe decoded frames, publish YOLO overlay frames
    python cil.py torch --width 1280 --height 720 --fps 30

    # Terminal 3: serve the YOLO monitor at http://127.0.0.1:8000
    python cil.py web --width 1280 --height 720 --fps 30 --image-topic ImageMatCUDAPubSub:yolo

Notes:
    - No FFmpeg is used at runtime.
    - The input should be raw H264 Annex-B, not MP4.
    - Best input is encoded with AUD NALs, repeat SPS/PPS headers, and no B frames:
        -x264-params repeat-headers=1:aud=1 -bf 0

PS:
ffmpeg -i input.mp4 \
  -an \
  -vf "fps=30,format=yuv420p" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -profile:v baseline \
  -bf 0 \
  -g 30 \
  -keyint_min 30 \
  -sc_threshold 0 \
  -x264-params "repeat-headers=1:aud=1" \
  -f h264 \
  demo_iponly_30fps.h264
"""

from __future__ import annotations

import argparse
from typing import Iterable

from cuda_ipc_runtime import (
    DEFAULT_IMAGE_TOPIC,
    Config,
    decode_pub_loop,
)
from torch_runtime import torch_loop
from web_runtime import web_loop

def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(
        description="Demo: raw .h264 Annex-B file -> NVDEC -> CUDA IPC -> YOLO -> FastAPI monitor."
    )
    p.add_argument("role", choices=("decode-pub", "web", "pub", "torch"))
    p.add_argument("--input", dest="input_path", help="raw H264 Annex-B file, required for decode-pub")
    p.add_argument("--width", type=int, default=1280, help="decoded/source frame width")
    p.add_argument("--height", type=int, default=720, help="decoded/source frame height")
    p.add_argument("--fps", type=int, default=30, help="playback/publish pacing FPS")
    p.add_argument("--device", type=int, default=0, help="CUDA device id")
    p.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    p.add_argument("--num-slots", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stats-every", type=int, default=100)
    p.add_argument("--loop", action="store_true", help="loop the .h264 file in decode-pub mode")
    p.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--host", default="127.0.0.1", help="FastAPI bind host for web mode")
    p.add_argument("--port", type=int, default=8000, help="FastAPI bind port for web mode")
    p.add_argument("--monitor-width", type=int, default=640, help="web monitor image width")
    p.add_argument("--jpeg-quality", type=int, default=80, help="web monitor JPEG quality, 1..95")
    p.add_argument(
        "--require-aud",
        action="store_true",
        help="fail if the .h264 stream has no AUD NALs instead of using the simple fallback splitter",
    )
    return p.parse_args(argv)


def config_from_args(args) -> Config:
    return Config(
        input_path=args.input_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        device=args.device,
        image_topic=args.image_topic,
        num_slots=args.num_slots,
        max_frames=args.max_frames,
        stats_every=args.stats_every,
        loop=args.loop,
        flip_y=args.flip_y,
        require_aud=args.require_aud,
        host=args.host,
        port=args.port,
        monitor_width=args.monitor_width,
        jpeg_quality=args.jpeg_quality,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    role = {"pub": "decode-pub"}.get(args.role, args.role)

    try:
        if role == "decode-pub":
            decode_pub_loop(cfg)
        elif role == "web":
            web_loop(cfg)
        elif role == "torch":
            torch_loop(cfg)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
