#!/usr/bin/env python3
from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from cuda_ipc_runtime import Config, FpsMeter, close_quietly, make_cuda_image_endpoint
from torch_runtime import YOLO_TOPIC, pushed_cuda_context


@dataclass
class LatestJpegFrame:
    data: bytes | None = None
    sequence: int = -1
    updated_at: float = 0.0


class FrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest = LatestJpegFrame()

    def publish(self, data: bytes, sequence: int) -> None:
        with self._condition:
            self._latest = LatestJpegFrame(
                data=data,
                sequence=int(sequence),
                updated_at=time.time(),
            )
            self._condition.notify_all()

    def snapshot(self) -> LatestJpegFrame:
        with self._condition:
            return self._latest

    def wait_for_next(self, last_sequence: int, timeout: float = 2.0) -> LatestJpegFrame:
        deadline = time.time() + timeout
        with self._condition:
            while self._latest.sequence <= last_sequence:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._latest


def _resize_to_monitor(frame, cfg: Config):
    import torch
    import torch.nn.functional as F

    if bool(cfg.flip_y):
        frame = torch.flip(frame, dims=(0,))

    monitor_width = max(1, min(int(cfg.monitor_width), int(frame.shape[1])))
    if monitor_width == int(frame.shape[1]):
        return frame.contiguous()

    scale = monitor_width / float(frame.shape[1])
    monitor_height = max(1, int(round(float(frame.shape[0]) * scale)))

    nchw = frame.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    resized = F.interpolate(
        nchw,
        size=(monitor_height, monitor_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0).clamp_(0, 255).to(dtype=torch.uint8).contiguous()


def _jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    quality = max(1, min(95, int(quality)))
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def _subscriber_loop(cfg: Config, store: FrameStore, stop: threading.Event) -> None:
    import pycuda.driver as cuda
    import torch

    cuda.init()
    torch.cuda.set_device(cfg.device)
    torch.empty(1, device=f"cuda:{cfg.device}")

    ctx = cuda.Device(int(cfg.device)).retain_primary_context()
    image_sub = None
    meter = FpsMeter("web-monitor", cfg.stats_every)
    last_sequence = -1

    try:
        with pushed_cuda_context(ctx):
            image_sub = make_cuda_image_endpoint(cfg, is_pub=False)

        print(
            f"web: subscribing {cfg.image_topic!r}, serving {cfg.monitor_width}px JPEG monitor",
            flush=True,
        )

        while not stop.is_set():
            with pushed_cuda_context(ctx):
                image_sub.sub(copy=False, sync=True)

                if getattr(image_sub, "_remote_mem", None) is None:
                    frame = None
                    sequence = None
                else:
                    sequence = int(getattr(image_sub, "sequence", -1))
                    if sequence == last_sequence:
                        frame = None
                    else:
                        last_sequence = sequence
                        remote_t = image_sub.get_data_torch(copy=False, sync=False)
                        frame = remote_t.clone(memory_format=torch.contiguous_format)
                        torch.cuda.synchronize(cfg.device)
                        del remote_t

            if frame is None:
                time.sleep(0.001)
                continue

            monitor = _resize_to_monitor(frame, cfg)
            torch.cuda.synchronize(cfg.device)
            rgb = monitor.detach().cpu().numpy()
            store.publish(_jpeg_bytes(rgb, cfg.jpeg_quality), int(sequence))
            meter.tick()

            del frame
            del monitor

    finally:
        try:
            with pushed_cuda_context(ctx):
                if image_sub is not None:
                    close_quietly(image_sub)
        finally:
            ctx.detach()


def _preview_scale_text(cfg: Config) -> str:
    source_width = max(1, int(cfg.width))
    monitor_width = max(1, min(int(cfg.monitor_width), source_width))
    divisor = math.gcd(monitor_width, source_width)
    return f"preview scale at {monitor_width // divisor}/{source_width // divisor}"


def _html(cfg: Config) -> str:
    scale_text = _preview_scale_text(cfg)
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Arial, Helvetica, sans-serif;
      background: #111316;
      color: #e8eaed;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid #2c3036;
      background: #191c20;
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 0;
    }
    .meta {
      color: #aab0b7;
      font-size: 13px;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    main {
      display: grid;
      place-items: center;
      padding: 16px;
    }
    img {
      width: min(100%, 960px);
      height: auto;
      background: #050607;
      border: 1px solid #2c3036;
    }
    a {
      color: #9fc7ff;
      font-size: 13px;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <header>
    <h1>YOLO Monitor</h1>
    <div class="actions">
      <span class="meta">{scale_text}</span>
      <a href="/snapshot.jpg">snapshot</a>
    </div>
  </header>
  <main>
    <img src="/stream.mjpg" alt="YOLO monitor stream">
  </main>
</body>
</html>
""".replace("{scale_text}", scale_text)


def create_app(store: FrameStore, cfg: Config):
    try:
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI web mode requires: pip install fastapi uvicorn") from exc

    app = FastAPI(title="YOLO Monitor")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _html(cfg)

    @app.get("/snapshot.jpg")
    def snapshot():
        latest = store.snapshot()
        if latest.data is None:
            raise HTTPException(status_code=503, detail="No frame has been received yet")
        return Response(content=latest.data, media_type="image/jpeg")

    @app.get("/healthz")
    def healthz():
        latest = store.snapshot()
        age = None if latest.updated_at <= 0 else time.time() - latest.updated_at
        return {"ok": latest.data is not None, "sequence": latest.sequence, "age_seconds": age}

    @app.get("/stream.mjpg")
    def stream():
        boundary = "frame"

        def gen():
            last_sequence = -1
            while True:
                latest = store.wait_for_next(last_sequence)
                if latest.data is None or latest.sequence <= last_sequence:
                    continue
                last_sequence = latest.sequence
                yield (
                    b"--" + boundary.encode("ascii") + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n"
                    + latest.data
                    + b"\r\n"
                )

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    return app


def web_loop(cfg: Config) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI web mode requires: pip install fastapi uvicorn") from exc

    if cfg.image_topic == "ImageMatCUDAPubSub:h264FileDemo":
        cfg = replace(cfg, image_topic=YOLO_TOPIC)

    store = FrameStore()
    stop = threading.Event()
    thread = threading.Thread(target=_subscriber_loop, args=(cfg, store, stop), daemon=True)
    thread.start()

    print(f"web: http://{cfg.host}:{cfg.port}", flush=True)
    server_config = uvicorn.Config(create_app(store, cfg), host=cfg.host, port=int(cfg.port), log_level="info")
    server = uvicorn.Server(server_config)
    try:
        server.run()
    finally:
        stop.set()
        thread.join(timeout=5.0)
