from __future__ import annotations

import argparse

import uvicorn

import common  # noqa: F401
from webapi import create_auto_discover_fastapi_app


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI gateway for auto-discovered iceoryx2 JSON-RPC services")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--refresh-interval-s",
        type=float,
        default=0.0,
        help="Periodic refresh interval. Use 0 to refresh only on startup and manual GET /refresh.",
    )
    args = parser.parse_args()
    refresh_interval_s = args.refresh_interval_s if args.refresh_interval_s > 0 else None

    app = create_auto_discover_fastapi_app(
        title="Auto-discovered YOLO RPC API",
        description="Discovers iox2 JSON-RPC services and exposes them through /controllers/** routes.",
        refresh_on_startup=True,
        refresh_interval_s=refresh_interval_s,
        install_dynamic_openapi=True,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
