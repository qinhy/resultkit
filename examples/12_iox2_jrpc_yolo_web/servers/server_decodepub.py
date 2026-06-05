from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Literal
from pydantic import BaseModel, Field


from common import EmptyParams, build_processor
from iox2_jsonrpc.services import JsonRpcServiceDescriptor
from iox2_jsonrpc.iox2_transport import Iceoryx2JsonRpcServer

from cuda_ipc_runtime import Config, DecodePubLoop

class DecodePubBaseModel(BaseModel):
    service: Literal["decodepub"] = "decodepub"

    def model_post_init(self, context):
        print(f"[DecodePub {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class StartDecodePubParams(DecodePubBaseModel):
    input_path: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    device: int = 0
    image_topic: str = "ImageMatCUDAPubSub:h264FileDemo"
    num_slots: int = 3
    max_frames: int | None = None
    stats_every: int = 100
    loop: bool = False
    flip_y: bool = True
    require_aud: bool = False


class DecodePubResult(DecodePubBaseModel):
    running: bool
    profile: str = "decodepub"
    error: str | None = None


@dataclass
class DecodePubController:
    class JsonRpcServiceDescriptor(JsonRpcServiceDescriptor):
        name: str="decodepub"
        iceoryx2_service: str="jsonrpc/decodepub"
        timeout_seconds: float = Field(default=5.0, gt=0)    
    
    running: bool = False
    dec_pub: DecodePubLoop = None

    @staticmethod
    def openapi_examples():
        return {            
        "decodepub_status": {
            "summary": "decodepub: decodepub.status",
            "description": "Use service_name = decodepub with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "decodepub.status",
                "params": {}
            },
        },
        "decodepub_start": {
            "summary": "decodepub: decodepub.start",
            "description": "Use service_name = decodepub with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "decodepub.start",
                "params": {
                    "input_path": "./examples/demo.h264",
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "device": 0,
                    "loop": True
                }
            }
        },
        "decodepub_stop": {
            "summary": "decodepub: decodepub.stop",
            "description": "Use service_name = decodepub with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "decodepub.stop",
                "params": {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "decodepub.stop",
                    "params": {}
                }
            }
        },
    }
    
    def _result(self) -> DecodePubResult:
        err = None
        running = False

        if self.dec_pub is not None:
            running = self.dec_pub.is_running
            if self.dec_pub.exception is not None:
                err = f"{self.dec_pub.exception.__class__.__name__}: {self.dec_pub.exception}"

        self.running = running
        return DecodePubResult(running=running, error=err)

    def start(self, params: StartDecodePubParams) -> DecodePubResult:
        if self.dec_pub is not None:
            self.dec_pub.stop()

        cfg = Config(**params.model_dump(exclude={"service"}))
        self.dec_pub = DecodePubLoop(cfg)
        self.dec_pub.start(blocking=False)
        return self._result()

    def stop(self, params: EmptyParams) -> DecodePubResult:
        if self.dec_pub is None:
            self.running = False
            return DecodePubResult(running=False)

        self.dec_pub.stop()
        return self._result()

    def status(self, params: EmptyParams) -> DecodePubResult:
        return self._result()


def main() -> None:
    iox_service_name = DecodePubController.JsonRpcServiceDescriptor().iceoryx2_service
    processor=build_processor(DecodePubController(),prefix="decodepub")
    server = Iceoryx2JsonRpcServer(service_name=iox_service_name, processor=processor)
    server.serve_forever()


if __name__ == "__main__":
    main()
