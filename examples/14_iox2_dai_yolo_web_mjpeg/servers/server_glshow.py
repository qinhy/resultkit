from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Literal


from common import EmptyParams, RpcModel, openapi_doc
from cuda_ipc_runtime import Config, StoppableLoop, make_cuda_image_endpoint


class GlShowLoop(StoppableLoop):
    """CUDA IPC image subscriber -> OpenGL viewer loop."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.viewer = None
        self.image_sub = None

    def stop(self, *, join: bool = True, timeout: float | None = None) -> None:
        super().stop(join=False)

        # ImageMatCudaGlViewer is expected to own the GL run loop. Different
        # resultkit builds expose different shutdown method names, so try the
        # common ones if the viewer is already initialized.
        viewer = self.viewer
        if viewer is not None:
            for method_name in ("stop", "close", "shutdown", "destroy", "quit"):
                method = getattr(viewer, method_name, None)
                if callable(method):
                    try:
                        method()
                        break
                    except Exception:
                        pass

        thread = self._thread
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        from resultkit.cudavis import ImageMatCudaGlViewer

        cfg = self.cfg

        # Follow resultkit's GL path: initialize the viewer first, then create
        # the CUDA IPC image endpoint in the CUDA/GL context owned by viewer.
        self.viewer = ImageMatCudaGlViewer(
            width=int(cfg.width),
            height=int(cfg.height),
            fps=float(cfg.fps),
            device=int(cfg.device),
            flip_y=bool(cfg.flip_y),
            max_frames=cfg.max_frames,
        )
        self.viewer.init()

        self.image_sub = make_cuda_image_endpoint(cfg, is_pub=False)

        print(
            f"show: GL viewer subscribing {cfg.image_topic!r} "
            f"({cfg.width}x{cfg.height} @ {cfg.fps} fps)",
            flush=True,
        )

        try:
            self.viewer.run(img=self.image_sub)
        finally:
            try:
                self.image_sub.close()
            except Exception:
                pass
            self.image_sub = None
            self.viewer = None


class GlShowBaseModel(RpcModel):
    service: Literal["glshow"] = "glshow"

    def model_post_init(self, context):
        print(f"[GlShow {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class StartGlShowParams(GlShowBaseModel):
    width: int = 4032
    height: int = 3040
    fps: int = 10
    device: int = 0
    image_topic: str = "ImageMatCUDAPubSub:daiRgb"
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
            **openapi_doc("glshow_status", id=1, params={}),
            **openapi_doc("glshow_start",  id=2, params=StartGlShowParams().model_dump()),
            **openapi_doc("glshow_stop",   id=3, params={}),
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
