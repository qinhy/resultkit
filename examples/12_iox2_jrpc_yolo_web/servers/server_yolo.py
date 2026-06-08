from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from common import EmptyParams, build_processor
from iox2_jsonrpc.iox2_transport import Iceoryx2JsonRpcServer
from iox2_jsonrpc.services import JsonRpcServiceDescriptor

from cuda_ipc_runtime import Config
from torch_runtime import YOLO_TOPIC, YoloLoop


class YoloBaseModel(BaseModel):
    service: Literal["yolo"] = "yolo"

    def model_post_init(self, context):
        print(f"[Yolo {self.__class__.__name__}]")
        return super().model_post_init(context)


class StartYoloParams(YoloBaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    device: int = 0
    input_topic: str = "ImageMatCUDAPubSub:h264FileDemo"
    output_topic: str = YOLO_TOPIC
    num_slots: int = 3
    max_frames: int | None = None
    stats_every: int = 100


class YoloResult(YoloBaseModel):
    running: bool
    input_topic: str = "ImageMatCUDAPubSub:h264FileDemo"
    output_topic: str = YOLO_TOPIC
    error: str | None = None


@dataclass
class YoloController:
    class JsonRpcServiceDescriptor(JsonRpcServiceDescriptor):
        name: str = "yolo"
        iceoryx2_service: str = "jsonrpc/yolo"
        timeout_seconds: float = Field(default=5.0, gt=0)

    running: bool = False
    yolo_loop: YoloLoop = None

    @staticmethod
    def openapi_examples():
        return {
            "yolo_status": {
                "summary": "yolo: yolo.status",
                "description": "Use service_name = yolo with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "yolo.status",
                    "params": {},
                },
            },
            "yolo_start": {
                "summary": "yolo: yolo.start",
                "description": "Use service_name = yolo with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "yolo.start",
                    "params": {
                        "input_topic": "ImageMatCUDAPubSub:h264FileDemo",
                        "output_topic": "ImageMatCUDAPubSub:yolo",
                        "width": 1280,
                        "height": 720,
                        "fps": 30,
                        "device": 0,
                    },
                },
            },
            "yolo_stop": {
                "summary": "yolo: yolo.stop",
                "description": "Use service_name = yolo with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "yolo.stop",
                    "params": {},
                },
            },
        }

    def _result(self) -> YoloResult:
        err = None
        running = False
        input_topic = "ImageMatCUDAPubSub:h264FileDemo"
        output_topic = YOLO_TOPIC

        if self.yolo_loop is not None:
            running = self.yolo_loop.is_running
            input_topic = self.yolo_loop.cfg.image_topic
            output_topic = self.yolo_loop.output_topic
            if self.yolo_loop.exception is not None:
                err = f"{self.yolo_loop.exception.__class__.__name__}: {self.yolo_loop.exception}"

        self.running = running
        return YoloResult(running=running, input_topic=input_topic, output_topic=output_topic, error=err)

    def start(self, params: StartYoloParams) -> YoloResult:
        if self.yolo_loop is not None:
            self.yolo_loop.stop()

        values = params.model_dump(exclude={"service", "input_topic", "output_topic"})
        cfg = Config(**values, image_topic=params.input_topic)
        self.yolo_loop = YoloLoop(cfg, output_topic=params.output_topic)
        self.yolo_loop.start(blocking=False)
        return self._result()

    def stop(self, params: EmptyParams) -> YoloResult:
        if self.yolo_loop is None:
            self.running = False
            return YoloResult(running=False)

        self.yolo_loop.stop()
        return self._result()

    def status(self, params: EmptyParams) -> YoloResult:
        return self._result()


def main() -> None:
    iox_service_name = YoloController.JsonRpcServiceDescriptor().iceoryx2_service
    processor = build_processor(YoloController(), prefix="yolo")
    server = Iceoryx2JsonRpcServer(service_name=iox_service_name, processor=processor)
    server.serve_forever()


if __name__ == "__main__":
    main()
