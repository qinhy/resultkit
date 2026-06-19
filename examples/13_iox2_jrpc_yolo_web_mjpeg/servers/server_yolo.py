from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from common import EmptyParams, RpcModel, openapi_doc

from cuda_ipc_runtime import Config
from torch_runtime import YoloLoop, YoloSettings as YoloSettingsRuntime


class YoloBaseModel(RpcModel):
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
    output_topic: str = "ImageMatCUDAPubSub:yolo"
    num_slots: int = 3
    max_frames: int | None = None
    stats_every: int = 100


class YoloSettings(YoloBaseModel):
    """Runtime settings for model inference and detection serialization."""

    model_name: str = "yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    max_detections: int = 100
    stride: int = 32

    def as_runtime(self) -> YoloSettingsRuntime:
        return YoloSettingsRuntime(**self.model_dump(exclude={"service"}))


class YoloResult(StartYoloParams):
    model_config = {"extra": "ignore"}
    running: bool
    model: YoloSettings = None
    error: str | None = None


@dataclass
class YoloController:# (JsonRpcController)
    running: bool = False
    yolo_loop: YoloLoop = None
    service_name: str = "jsonrpc"
    controller_name: str = "yolo"

    @staticmethod
    def openapi_examples():
        return {
            **openapi_doc("yolo_status", id=1, params={}),
            **openapi_doc("yolo_start",  id=2, params=StartYoloParams().model_dump()),
            **openapi_doc("yolo_stop",   id=3, params={}),
            **openapi_doc("yolo_set_model", id=4, params=YoloSettings().model_dump()),
        }

    def _result(self, err = None) -> YoloResult:
        res = YoloResult(
            running = False,
            input_topic = "",
            output_topic = "",
            error = None,
        )

        if self.yolo_loop is not None:
            res.running = self.yolo_loop.is_running
            res.input_topic = self.yolo_loop.cfg.image_topic
            res.output_topic = self.yolo_loop.output_topic
            res.model = YoloSettings(**self.yolo_loop.detector.settings.__dict__)
            if self.yolo_loop.exception is not None:
                res.error = f"{self.yolo_loop.exception.__class__.__name__}: {self.yolo_loop.exception}"

        return res
    
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
    
    def set_model(self, params: YoloSettings) -> YoloResult:
        if self.yolo_loop is None:return self._result()
        try:
            self.yolo_loop.change_detector(settings=params.as_runtime())
        except Exception as e:
            return self._result(str(e))
        return self._result()


def main() -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer
    from utils import set_default_font_hex_path
    set_default_font_hex_path("./font/unifont_all.hex")

    server = Iox2JsonRpcServer(YoloController())
    server.run_forever()


if __name__ == "__main__":
    main()
