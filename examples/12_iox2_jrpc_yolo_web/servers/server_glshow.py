from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Literal
from pydantic import BaseModel, Field


from common import EmptyParams, build_processor
from iox2_jsonrpc.services import JsonRpcServiceDescriptor
from iox2_jsonrpc.iox2_transport import Iceoryx2JsonRpcServer

from cuda_ipc_runtime import Config, GlShowLoop

class GlShowBaseModel(BaseModel):
    service: Literal["glshow"] = "glshow"

    def model_post_init(self, context):
        print(f"[GlShow {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class StartGlShowParams(GlShowBaseModel):
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


class GlShowResult(GlShowBaseModel):
    running: bool
    error: str | None = None


@dataclass
class GlShowController:
    class JsonRpcServiceDescriptor(JsonRpcServiceDescriptor):
        name: str="glshow"
        iceoryx2_service: str="jsonrpc/glshow"
        timeout_seconds: float = Field(default=5.0, gt=0)    
    
    running: bool = False
    glshow_sub: GlShowLoop = None

    @staticmethod
    def openapi_examples():
        return {            
        "glshow_status": {
            "summary": "glshow: glshow.status",
            "description": "Use service_name = glshow with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "glshow.status",
                "params": {}
            },
        },
        "glshow_start": {
            "summary": "glshow: glshow.start",
            "description": "Use service_name = glshow with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "glshow.start",
                "params": {
                    "image_topic":"ImageMatCUDAPubSub:h264FileDemo",
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "device": 0,
                    "loop": True
                }
            }
        },
        "glshow_stop": {
            "summary": "glshow: glshow.stop",
            "description": "Use service_name = glshow with /{service_name}/rpc.",
            "value": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "glshow.stop",
                "params": {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "glshow.stop",
                    "params": {}
                }
            }
        },
    }
    
    def _result(self) -> GlShowResult:
        err = None
        running = False

        if self.glshow_sub is not None:
            running = self.glshow_sub.is_running
            if self.glshow_sub.exception is not None:
                err = f"{self.glshow_sub.exception.__class__.__name__}: {self.glshow_sub.exception}"

        self.running = running
        return GlShowResult(running=running, error=err)

    def start(self, params: StartGlShowParams) -> GlShowResult:
        if self.glshow_sub is not None:
            self.glshow_sub.stop()

        cfg = Config(**params.model_dump(exclude={"service"}))
        self.glshow_sub = GlShowLoop(cfg)
        self.glshow_sub.start(blocking=False)
        return self._result()

    def stop(self, params: EmptyParams) -> GlShowResult:
        if self.glshow_sub is None:
            self.running = False
            return GlShowResult(running=False)

        self.glshow_sub.stop()
        return self._result()

    def status(self, params: EmptyParams) -> GlShowResult:
        return self._result()


def main() -> None:
    iox_service_name = GlShowController.JsonRpcServiceDescriptor().iceoryx2_service
    processor=build_processor(GlShowController(),prefix="glshow")
    server = Iceoryx2JsonRpcServer(service_name=iox_service_name, processor=processor)
    server.serve_forever()


if __name__ == "__main__":
    main()
