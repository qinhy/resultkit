from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from iox2_jsonrpc import (
    ControllerRpcEndpoint,
    EmptyParams,
    JsonRpcRequest,
    RpcModel,
    describe_controller,
)

from webapi import create_auto_discover_fastapi_app

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


def run_api(host: str, port: int, reload: bool = False) -> None:
    """Run the auto-discovery FastAPI gateway."""

    # Import here so existing non-HTTP modes work without uvicorn installed.
    import uvicorn

    uvicorn.run(
        create_fastapi_app(),
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
