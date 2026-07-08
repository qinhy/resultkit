from __future__ import annotations

import os
import sys

from pathlib import Path

EXAMPLE_DIR = os.path.dirname(os.path.dirname(Path(__file__).absolute()))

if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from iox2_jsonrpc import EmptyParams, RpcModel


def openapi_doc(key="yolo_status", id=1, params={}):
    """Return OpenAPI document for the given key."""
    service_name = key.split("_")[0]
    func_name = key.replace(service_name+"_", "")
    return {key: {
                "summary": f"{service_name}: {func_name}",
                "description": f"Use service with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": f"{service_name}.{func_name}",
                    "params": params,
                },
            }}


__all__ = ["EmptyParams", "RpcModel", "PreviewProcess",
           "AsyncPreviewProcess", "CaptureRequestProcess",
           "UltralyticsYoloProcess", "AsyncYoloProcess", "AsyncWorkerPipelineProcess",
           "WorkerFrame", "WorkerContext", "WorkerPipelineProcess", "StoppableLoop",
           "openapi_doc"]
