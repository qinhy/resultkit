from __future__ import annotations

import argparse
import os
from pathlib import Path
import inspect
import sys
from typing import Literal, get_type_hints

from pydantic import BaseModel

from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute())))
from iox2_jsonrpc.core import JsonRpcProcessor, MethodRegistry


def _is_pydantic_model(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def register_controller(
    registry: MethodRegistry,
    *,
    prefix: str,
    controller: object,
) -> None:
    for method_name, method in inspect.getmembers(controller, predicate=inspect.ismethod):
        if method_name.startswith("_"):
            continue

        signature = inspect.signature(method)
        params = list(signature.parameters.values())

        # Only register methods like:
        # def start(self, params: SomeParams) -> SomeResult
        if len(params) != 1:
            continue

        param_name = params[0].name
        hints = get_type_hints(method)

        params_model = hints.get(param_name)
        result_model = hints.get("return")

        if not _is_pydantic_model(params_model):
            continue

        if not _is_pydantic_model(result_model):
            continue

        rpc_name = f"{prefix}.{method_name}"

        registry.register(
            rpc_name,
            method,
            params_model=params_model,
            result_model=result_model,
        )



class EmptyParams(BaseModel):
    pass

class RpcHealthResult(BaseModel):
    ok: bool = True
    methods: list[str]

def build_processor(controller,prefix:str) -> JsonRpcProcessor:
    registry = MethodRegistry()

    register_controller(
        registry,
        prefix=prefix,
        controller=controller,
    )

    def rpc_health(params: EmptyParams) -> RpcHealthResult:
        return RpcHealthResult(methods=registry.names())

    registry.register(
        "rpc.health",
        rpc_health,
        params_model=EmptyParams,
        result_model=RpcHealthResult,
    )

    return JsonRpcProcessor(registry)