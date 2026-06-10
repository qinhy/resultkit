from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


from common import EmptyParams, RpcModel

from cuda_ipc_runtime import Config, GlShowLoop


class GlShowBaseModel(RpcModel):
    service: Literal["glshow"] = "glshow"

    def model_post_init(self, context):
        print(f"[GlShow {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class StartGlShowParams(GlShowBaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    device: int = 0
    image_topic: str = "ImageMatCUDAPubSub:h264FileDemo"
    num_slots: int = 3
    max_frames: int | None = None
    stats_every: int = 100
    loop: bool = True
    flip_y: bool = True
    require_aud: bool = False


class GlShowResult(GlShowBaseModel):
    running: bool
    error: str | None = None


@dataclass
class GlShowController:
    running: bool = False
    glshow_sub: GlShowLoop = None
    service_name: str = "jsonrpc"
    controller_name: str = "glshow"

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
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(GlShowController())
    server.run_forever()


if __name__ == "__main__":
    main()
