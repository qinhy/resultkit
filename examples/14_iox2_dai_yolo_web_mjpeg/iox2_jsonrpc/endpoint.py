from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .controller import iter_rpc_methods
from .models import JsonRpcRequest, JsonRpcResponse

PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ControllerRpcEndpoint:
    """In-process JSON-RPC endpoint for a typed controller."""

    def __init__(self, controller: Any):
        self.controller = controller
        self.controller_name = controller.controller_name

    def handle_bytes(self, raw: bytes) -> bytes:
        try:
            request = JsonRpcRequest.model_validate_json(raw)
        except Exception as exc:
            response = JsonRpcResponse.fail(
                id=None,
                code=PARSE_ERROR,
                message="Parse error",
                data=str(exc),
            )
            return response.model_dump_json().encode()

        response = self.handle(request)
        return response.model_dump_json().encode()

    def handle(self, request: JsonRpcRequest) -> JsonRpcResponse:
        expected_prefix = f"{self.controller_name}."

        if not request.method.startswith(expected_prefix):
            return JsonRpcResponse.fail(
                id=request.id,
                code=METHOD_NOT_FOUND,
                message=f"Unknown controller for method: {request.method}",
            )

        local_method_name = request.method.removeprefix(expected_prefix)

        for binding in iter_rpc_methods(self.controller):
            if binding.name != local_method_name:
                continue

            try:
                params = binding.params_model.model_validate(request.params or {})
            except ValidationError as exc:
                return JsonRpcResponse.fail(
                    id=request.id,
                    code=INVALID_PARAMS,
                    message="Invalid params",
                    data=exc.errors(),
                )

            try:
                result = binding.function(params)
            except Exception as exc:  # pragma: no cover - controller dependent
                return JsonRpcResponse.fail(
                    id=request.id,
                    code=INTERNAL_ERROR,
                    message="Internal error",
                    data=str(exc),
                )

            return JsonRpcResponse.ok(id=request.id, result=result)

        return JsonRpcResponse.fail(
            id=request.id,
            code=METHOD_NOT_FOUND,
            message=f"Method not found: {request.method}",
        )
