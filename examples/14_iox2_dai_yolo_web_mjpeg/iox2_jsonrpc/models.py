from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

JsonRpcId = str | int | None


class RpcModel(BaseModel):
    """Base model used by the library's JSON-RPC data structures."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class EmptyParams(RpcModel):
    """Convenience params model for RPC methods that accept no input."""


class JsonRpcRequest(RpcModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    method: str
    params: dict[str, Any] | None = None


class JsonRpcErrorObject(RpcModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(RpcModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    result: Any | None = None
    error: JsonRpcErrorObject | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> JsonRpcResponse:
        if self.result is not None and self.error is not None:
            raise ValueError("JSON-RPC response cannot contain both result and error")
        return self

    @staticmethod
    def ok(id: JsonRpcId, result: Any) -> JsonRpcResponse:
        if isinstance(result, BaseModel):
            result = result.model_dump()

        return JsonRpcResponse(id=id, result=result)

    @staticmethod
    def fail(
        id: JsonRpcId,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> JsonRpcResponse:
        return JsonRpcResponse(
            id=id,
            error=JsonRpcErrorObject(
                code=code,
                message=message,
                data=data,
            ),
        )


class RpcMethodDescriptor(RpcModel):
    name: str
    jsonrpc_method: str
    params_model: str
    result_model: str
    params_schema: dict[str, Any]
    result_schema: dict[str, Any]


class RpcServiceDescriptor(RpcModel):
    protocol: Literal["jsonrpc-2.0"] = "jsonrpc-2.0"
    service: str
    controller: str
    rpc_endpoint: str
    schema_endpoint: str
    methods: list[RpcMethodDescriptor]
