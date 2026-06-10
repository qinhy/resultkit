from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import io
import json
import logging
import threading
from typing import Any

from PIL import Image
import cv2
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from webapi import create_auto_discover_fastapi_app

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))
from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
from resultkit.mat import DataType

def print_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, indent=2)


def create_fastapi_app():
    """Create the zero-registration FastAPI gateway.

    The API process does not mount CameraController directly. Start this API,
    then start any RPC services separately; the gateway discovers them and
    refreshes /controllers/** when GET /refresh or POST /refresh is called.
    """

    return create_auto_discover_fastapi_app(
        title="Auto-discovered RPC API",
        description="Discovers iox2 JSON-RPC services and exposes them through simple /controllers/** routes.",
        refresh_on_startup=True,
        refresh_interval_s=None,
        install_dynamic_openapi=True,
    )


def make_cuda_image_endpoint(image_topic,height,width, is_pub: bool=False, device="cuda:0"):
    import pycuda.gpuarray as gpuarray
    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError("Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build")
    try:
        data = gpuarray.empty((int(height), int(width), 3), dtype=np.uint8)
    except Exception as e:
        print(e)
        raise ValueError(
            "PyCUDA context is trying to allocate GPU memory, but probably no current one. "
            "There are maybe muti-context exits."
        )

    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.RGB,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=device,
        data=data,
    )
    img.set_id(image_topic).init()

    # Some resultkit versions use is_pub, some infer it from pub/sub calls.
    try:
        img.is_pub = bool(is_pub)
    except Exception:
        pass

    return img


_CUDA_LOCK = threading.Lock()
_CUDA_CTX = None
def get_cuda_context(device: int = 0):
    global _CUDA_CTX
    import pycuda.driver as cuda

    cuda.init()

    if _CUDA_CTX is None:
        # Use CUDA primary context instead of making/detaching a new user context
        # for every HTTP request.
        _CUDA_CTX = cuda.Device(int(device)).retain_primary_context()

    return _CUDA_CTX


@contextmanager
def pycuda_context(device: int):
    import pycuda.driver as cuda

    ctx = get_cuda_context(device)
    ctx.push()
    try:
        yield
        cuda.Context.synchronize()
    finally:
        cuda.Context.pop()

def numpy_image_to_png_bytes(arr) -> bytes:
    # Remove batch dimension if present
    if arr.ndim == 4:
        arr = arr[0]

    # Convert CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # Convert float image to uint8
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Grayscale with trailing channel
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    image = Image.fromarray(arr)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()



async def imgstream(
    stream_type: str,
    stream_name: str,
    width: int = Query(default=1280, gt=0),
    height: int = Query(default=720, gt=0),
    step: int = Query(default=10, gt=0),
):
    img = None
    topic = f"{stream_type}:{stream_name}"

    try:
        with pycuda_context(0):
            img = make_cuda_image_endpoint(topic, height, width)

            while True:
                with _CUDA_LOCK:
                    img.sub()

                    # rgb
                    tensor = img.get_data_torch(copy=False)

                    # Downsample on GPU before moving to CPU.
                    # Assumes image layout is H x W x C or H x W.
                    if step > 1:
                        tensor = tensor[::step, ::step, ...].contiguous()
                    else:
                        tensor = tensor.contiguous()

                    # Move data fully to CPU while CUDA context and img are still alive.
                    arr = tensor.detach().cpu().numpy().copy()[:,:,::-1] # to bgr

                # Encode outside the lock.
                ok, encoded = cv2.imencode(".jpg", arr)
                if not ok:
                    continue

                frame = encoded.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n" + frame + b"\r\n"
                )

                await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        # Client disconnected. Let FastAPI/Starlette handle cancellation.
        raise

    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass

async def stream_res(
    stream_type: str,
    stream_name: str,
    width: int = Query(default=1280, gt=0),
    height: int = Query(default=720, gt=0),
    step: int = Query(default=10, gt=0),
):
    return StreamingResponse(
        imgstream(stream_type, stream_name, width, height, step),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
        },
    )


def run_api(host: str, port: int, reload: bool = False) -> None:
    """Run the auto-discovery FastAPI gateway."""

    # Import here so existing non-HTTP modes work without uvicorn installed.
    import uvicorn
    app = create_fastapi_app()
    app.add_api_route(
        "/imgstream/{stream_type}/{stream_name}",
        stream_res,
        methods=["GET"],
        name="imgstream",
        tags=["image"],
    )

    uvicorn.run(
        app=app,
        factory=reload,
        host=host,
        port=port,
        reload=reload,
    )


def run_client() -> None:
    """Discover an iceoryx2 JSON-RPC service and call the camera methods."""

    # Import here so `python camera_all_in_one.py local` works without iceoryx2.
    from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

    registry = Iox2RpcRegistry.discover_all()

    logging.basicConfig(
        filename="client.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Program started")
    logging.warning("Something might be wrong")
    logging.error("Something failed")

    logging.info("\n=== Discovered JSON-RPC catalog ===")
    logging.info(json.dumps(registry.catalog(), indent=2))

    logging.info("\n=== camera.status ===")
    logging.info(registry.call_unique("camera.status"))

    logging.info("\n=== camera.capture exposure_ms=25 ===")
    logging.info(registry.call_unique("camera.capture", {"exposure_ms": 25}))

    logging.info("\n=== camera.capture default exposure ===")
    logging.info(registry.call_unique("camera.capture"))

    logging.info("\n=== camera.close ===")
    logging.info(registry.call_unique("camera.close"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one camera example for the iox2_jsonrpc library."
    )
    parser.add_argument(
        "mode",
        choices=["client", "api"],
        help=(
            "local = no iceoryx2 needed; "
            "server/client = real iceoryx2 request-response transport; "
            "api = auto-discovery FastAPI RPC gateway"
        ),
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for FastAPI modes. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI modes. Default: 8000",
    )

    args = parser.parse_args()

    if args.mode == "client":
        run_client()
    elif args.mode == "api":
        run_api(host=args.host, port=args.port)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
