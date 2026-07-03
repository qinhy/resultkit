from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, get_type_hints, runtime_checkable

from pydantic import BaseModel

from .models import RpcMethodDescriptor, RpcServiceDescriptor


@runtime_checkable
class RpcController(Protocol):
    """Structural protocol for a controller object."""

    service_name: str
    controller_name: str


@dataclass(frozen=True)
class RpcMethodBinding:
    name: str
    function: Callable[[BaseModel], BaseModel]
    params_model: type[BaseModel]
    result_model: type[BaseModel]

    @property
    def jsonrpc_method(self) -> str:
        return self.name


def _validate_controller(controller: Any) -> None:
    if not hasattr(controller, "service_name"):
        raise TypeError("RPC controller must define service_name")
    if not hasattr(controller, "controller_name"):
        raise TypeError("RPC controller must define controller_name")


def iter_rpc_methods(controller: Any) -> Iterator[RpcMethodBinding]:
    """Yield public controller methods with typed Pydantic params/results."""

    _validate_controller(controller)

    for method_name in dir(controller):
        if method_name.startswith("_"):
            continue

        fn = getattr(controller, method_name)

        if not callable(fn):
            continue

        hints = get_type_hints(fn)
        params_model = hints.get("params")
        result_model = hints.get("return")

        if not isinstance(params_model, type):
            continue
        if not isinstance(result_model, type):
            continue
        if not issubclass(params_model, BaseModel):
            continue
        if not issubclass(result_model, BaseModel):
            continue

        yield RpcMethodBinding(
            name=method_name,
            function=fn,
            params_model=params_model,
            result_model=result_model,
        )


def rpc_endpoint_name(controller: RpcController) -> str:
    return f"{controller.service_name}/{controller.controller_name}/rpc"


def schema_endpoint_name(controller: RpcController) -> str:
    return f"{controller.service_name}/{controller.controller_name}/schema"


def describe_controller(
    controller: RpcController,
    rpc_endpoint: str | None = None,
    schema_endpoint: str | None = None,
) -> RpcServiceDescriptor:
    """Build a JSON-serializable service descriptor for a controller."""

    methods: list[RpcMethodDescriptor] = []

    for binding in iter_rpc_methods(controller):
        jsonrpc_method = f"{controller.controller_name}.{binding.name}"
        methods.append(
            RpcMethodDescriptor(
                name=binding.name,
                jsonrpc_method=jsonrpc_method,
                params_model=binding.params_model.__name__,
                result_model=binding.result_model.__name__,
                params_schema=binding.params_model.model_json_schema(),
                result_schema=binding.result_model.model_json_schema(),
            )
        )

    return RpcServiceDescriptor(
        service=controller.service_name,
        controller=controller.controller_name,
        rpc_endpoint=rpc_endpoint or rpc_endpoint_name(controller),
        schema_endpoint=schema_endpoint or schema_endpoint_name(controller),
        methods=methods,
    )
