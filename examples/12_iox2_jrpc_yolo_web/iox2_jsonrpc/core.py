from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class MethodDefinition:
    name: str
    handler: Callable[..., Any]
    params_model: type[BaseModel] | None = None
    result_model: type[BaseModel] | None = None


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodDefinition] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        params_model: type[BaseModel] | None = None,
        result_model: type[BaseModel] | None = None,
    ) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("JSON-RPC method name must be a non-empty string")
        self._methods[name] = MethodDefinition(
            name=name,
            handler=handler,
            params_model=params_model,
            result_model=result_model,
        )

    def get(self, name: str) -> MethodDefinition:
        try:
            return self._methods[name]
        except KeyError as exc:
            raise JsonRpcError(METHOD_NOT_FOUND, "Method not found") from exc

    def names(self) -> list[str]:
        return sorted(self._methods)


class JsonRpcProcessor:
    def __init__(self, registry: MethodRegistry) -> None:
        self._registry = registry

    def handle_bytes(self, payload: bytes) -> bytes | None:
        response = self.handle_text(payload.decode("utf-8"))
        if response is None:
            return None
        return json.dumps(response, separators=(",", ":")).encode("utf-8")

    def handle_text(self, text: str) -> Any | None:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return self._error_response(None, PARSE_ERROR, "Parse error")
        return self.handle_value(value)

    def handle_value(self, value: Any) -> Any | None:
        if isinstance(value, list):
            if not value:
                return self._error_response(None, INVALID_REQUEST, "Invalid Request")
            responses = []
            for item in value:
                response = self._handle_single(item)
                if response is not None:
                    responses.append(response)
            return responses or None

        return self._handle_single(value)

    def _handle_single(self, request: Any) -> dict[str, Any] | None:
        request_id = None
        is_notification = False

        if not isinstance(request, dict):
            return self._error_response(None, INVALID_REQUEST, "Invalid Request")

        request_id = request.get("id")
        is_notification = "id" not in request

        try:
            method_name = self._validate_request_shape(request)
            result = self._call_method(method_name, request.get("params"))
        except JsonRpcError as exc:
            if is_notification:
                return None
            return self._error_response(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001 - JSON-RPC converts internal exceptions.
            if is_notification:
                return None
            return self._error_response(
                request_id,
                INTERNAL_ERROR,
                "Internal error",
                {"exception": exc.__class__.__name__},
            )

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": self._jsonable(result)}

    def _validate_request_shape(self, request: dict[str, Any]) -> str:
        if request.get("jsonrpc") != "2.0":
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
        method_name = request.get("method")
        if not isinstance(method_name, str) or not method_name:
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
        if "params" in request and not isinstance(request["params"], dict):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", {"reason": "params must be an object"})
        return method_name

    def _call_method(self, method_name: str, raw_params: Any) -> Any:
        method = self._registry.get(method_name)
        params_obj: BaseModel | dict[str, Any]

        if method.params_model is None:
            params_obj = raw_params or {}
        else:
            try:
                params_obj = method.params_model.model_validate(raw_params or {})
            except ValidationError as exc:
                raise JsonRpcError(INVALID_PARAMS, "Invalid params", exc.errors()) from exc

        result = method.handler(params_obj)
        if inspect.isawaitable(result):
            raise JsonRpcError(INTERNAL_ERROR, "Internal error", {"reason": "async handlers are not supported in this tiny sync worker"})

        if method.result_model is None:
            return result

        try:
            return method.result_model.model_validate(result)
        except ValidationError as exc:
            raise JsonRpcError(INTERNAL_ERROR, "Internal error", {"result_validation": exc.errors()}) from exc

    def _error_response(self, request_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = self._jsonable(data)
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        return value
