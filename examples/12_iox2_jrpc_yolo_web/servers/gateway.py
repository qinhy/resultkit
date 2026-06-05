from __future__ import annotations

import argparse
from typing import Annotated

from fastapi import Body, Response
import uvicorn

from common import *
from iox2_jsonrpc.gateway import FastApiJsonRpcGateway, JsonRpcHttpRequest
from iox2_jsonrpc.services import JsonRpcServiceRegistry

from server_decodepub import DecodePubController
from server_glshow import GlShowController

JsonRpcHttpBody = Annotated[
    JsonRpcHttpRequest | list[JsonRpcHttpRequest],
    Body(
        openapi_examples={
            **(DecodePubController.openapi_examples()),
            **(GlShowController.openapi_examples()),
        }
        
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI gateway for JSON-RPC over iceoryx2 services")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    registry = JsonRpcServiceRegistry.from_list([
        DecodePubController.JsonRpcServiceDescriptor(),
        GlShowController.JsonRpcServiceDescriptor(),
    ])
    gw = FastApiJsonRpcGateway(registry)
    app = gw.create_app()
    
    @app.post("/{service_name}/rpc", summary="Forward JSON-RPC to one iceoryx2 service")
    async def rpc_by_prefix(service_name: str, body: JsonRpcHttpBody) -> Response:
        return await gw._forward_body(service_name, body)

    @app.post("/rpc/{service_name}", summary="Forward JSON-RPC to one iceoryx2 service")
    async def rpc_by_namespace(service_name: str, body: JsonRpcHttpBody) -> Response:
        return await gw._forward_body(service_name, body)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
