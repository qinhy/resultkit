from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import io
import json
import logging
import threading
from typing import Any, Dict, List

from PIL import Image
import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel
import requests

from webapi import create_auto_discover_fastapi_app

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))

from store.custom_record_store import CustomRecord,YoloRecord

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


def call_localhost(controller: str, method: str, params: dict = None) -> dict | None:
    if params is None:
        params = {}
    url = f"http://localhost:8000/controllers/{controller}/{method}"
    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response body: {e.response.text}")
        return None
    

def last_capture_record(store_name:str="store_dual")->CustomRecord:
    store_status = call_localhost(store_name,"status")
    res = store_status["last_capture"]["captures"][-1]["db_record"]
    return CustomRecord(**res)

def last_capture_record_rgb(store_name:str="store_dual"):
    store_status = call_localhost(store_name,"status")
    res = store_status["last_capture"]["captures"][-1]["db_record"]
    res = CustomRecord(**res)
    file_path = str(res.listup_rgb_image_paths[-1])
    if not Path(file_path).is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}",
        )
    return FileResponse(file_path)

def last_ai_records()->List[Dict]:
    store_status = call_localhost("store","status")
    res = store_status["last_capture"]["captures"][-1]["db_record"]
    res = CustomRecord(**res)
    res = [r.model_dump() for r in res.get_yolo_list()]
    return res


def get_debug_file(path: str):
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}",
        )
    return FileResponse(file_path)


def yolo_debug():
    return FileResponse(
        "yolo_debug.html",
        media_type="text/html",
    )


def run_api(host: str, port: int, reload: bool = False) -> None:
    """Run the auto-discovery FastAPI gateway."""

    # Import here so existing non-HTTP modes work without uvicorn installed.
    import uvicorn
    app = create_fastapi_app()

    app.add_api_route("/debug/last_capture_record",
        last_capture_record,methods=["GET"], name="debug", tags=["debug"],)
    
    app.add_api_route("/debug/last_capture_record_rgb",
        last_capture_record_rgb,methods=["GET"], name="debug", tags=["debug"],)
    
    app.add_api_route("/debug/last_ai_records",
        last_ai_records,methods=["GET"], name="debug", tags=["debug"],)

    app.add_api_route("/debug/get_debug_file",
        get_debug_file, methods=["GET"], name="debug", tags=["debug"])
    
    app.add_api_route("/debug/yolo",
        yolo_debug,methods=["GET"],tags=["debug"],)
    
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
