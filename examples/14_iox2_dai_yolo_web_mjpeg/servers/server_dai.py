from __future__ import annotations

import argparse
import base64
import multiprocessing as mp
import os
import queue as py_queue
import sys
import threading
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from common import EmptyParams, RpcModel, openapi_doc


StreamName = Literal["rgb", "stereo", "left", "right"]
STREAMS: tuple[StreamName, ...] = ("rgb", "stereo", "left", "right")

parser = argparse.ArgumentParser(
    description="Run the camera controller JSON-RPC server."
)
parser.add_argument(
    "--service-name",
    default="OkadCamA",
    help="Name of the iceoryx2 service.",
)
parser.add_argument(
    "--controller-name",
    default="camera",
    help="Name of the camera controller.",
)
parser.add_argument(
    "--device",
    default="169.254.1.222",
    help="Name of the camera controller.",
)
args = parser.parse_args()


class CameraBaseModel(RpcModel):
    service: str = args.service_name


class CameraConfig(CameraBaseModel):
    """Runtime settings for the DepthAI PoE RGB+stereo RPC camera server.

    Keep all DepthAI/Torch/CUDA/OpenCV frame work in the child process. The RPC
    parent only owns queues, metadata, and JPEG base64 strings, so native decoder
    failures cannot directly crash the RPC server process.
    """

    uuid: str = f"{args.service_name}:{args.controller_name}"
    sources: list[str] = [args.device]

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800
    capture_fps: int = 15

    rgb_codec: str = "h265"
    stereo_codec: str = "h265"
    rgb_bitrate_kbps: int = 60000
    stereo_bitrate_kbps: int = 6000

    decoder_backend: str = "pynvvideocodec"
    gst_nvivafilter_so: str = "./libdepthai_cuda_preprocess.so"
    gst_nvivafilter_dtype: str = "fp16"
    gst_nvivafilter_channel_order: str = "rgba"
    decoder_output_color: str = "rgbp"
    stereo_decoder_output_color: str = "rgbp"

    rgb_camera_socket: str = "CAM_A"
    left_camera_socket: str = "CAM_B"
    right_camera_socket: str = "CAM_C"

    normalize_rgb: bool = True
    normalize_stereo: bool = True

    # CUDA image publisher settings. The publisher is created inside the
    # camera worker process so the JSON-RPC parent never touches CUDA state.
    pub_enabled: bool = True
    pub_stream: StreamName = "rgb"
    pub_device: int = 0
    image_topic: str = "ImageMatCUDAPubSub:daiRgb"
    num_slots: int = 3
    pub_width: int | None = None
    pub_height: int | None = None
    pub_required: bool = False

    preview: bool = False
    preview_rgb_downsample: int = 10
    preview_stereo_downsample: int = 4

    queue_max_size: int = 8
    capture_wait_s: float = 2.0
    close_join_timeout_s: float = 5.0
    close_terminate_timeout_s: float = 2.0
    worker_watchdog_interval_s: float = 0.5

    retry_forever: bool = True
    retry_delay_s: float = 1.0


class CaptureParams(CameraBaseModel):
    """Params for camera.capture.

    exposure_ms and iso are kept for wire compatibility. This generator path does
    not apply manual exposure unless the generator implementation adds that API.
    """

    stream: StreamName = "rgb"
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    exposure_ms: int | None = Field(default=None, ge=1, le=33)
    iso: int | None = Field(default=None, ge=100, le=1600)


class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int
    device_id: str
    worker_alive: bool
    width: int
    height: int
    rgb_width: int
    rgb_height: int
    stereo_width: int
    stereo_height: int
    fps: int
    last_frame_id: int | None = None
    last_frame_age_s: float | None = None
    last_error: str | None = None
    worker_state: str | None = None
    restart_count: int = 0
    process_restart_count: int = 0
    worker_exitcode: int | None = None
    publishing: bool = False
    image_topic: str | None = None
    pub_stream: StreamName | None = None
    last_published_frame_id: int | None = None
    last_publish_error: str | None = None


class CaptureResult(CameraStatusResult):
    frame_id: int
    frame_timestamp_s: float
    stream: StreamName
    tensor_shape: tuple[int, ...]
    jpeg_base64: str


# ---------------------------------------------------------------------------
# Shared helpers. Parent helpers must stay JSON/pydantic/queue-only.
# ---------------------------------------------------------------------------


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _put_drop_oldest(q: Any, item: Any) -> None:
    """Put without blocking forever; if full, drop one stale item."""

    try:
        q.put_nowait(item)
        return
    except py_queue.Full:
        pass

    try:
        q.get_nowait()
    except (py_queue.Empty, Exception):
        pass

    try:
        q.put_nowait(item)
    except py_queue.Full:
        pass


def _drain_queue(q: Any) -> None:
    while q is not None:
        try:
            q.get_nowait()
        except (py_queue.Empty, Exception):
            return


def _shape(x: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _int_or_keep(current: int | None, value: Any) -> int | None:
    try:
        return int(value) if value is not None else current
    except Exception:
        return current


def _float_or_keep(current: float | None, value: Any) -> float | None:
    try:
        return float(value) if value is not None else current
    except Exception:
        return current



def _to_cv_array(mat: Any, *, downsample: int = 1, rgb_to_bgr: bool = True) -> Any:
    """Convert a Torch tensor image to a CPU OpenCV array inside the worker."""

    import torch

    x = mat.detach()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = x.permute(1, 2, 0)

    step = max(1, int(downsample))
    if step > 1:
        x = x[::step, ::step]

    if x.dtype.is_floating_point:
        max_value = float(x.max()) if x.numel() else 1.0
        x = x * 255.0 if max_value <= 1.5 else x
        x = x.clamp(0, 255)
    x = x.to(dtype=torch.uint8)

    arr = x.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr


def _tensor_to_jpeg_base64_in_worker(tensor: Any, jpeg_quality: int, *, rgb_to_bgr: bool) -> str:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        _to_cv_array(tensor, downsample=1, rgb_to_bgr=rgb_to_bgr),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed while encoding camera frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _drain_capture_requests(capture_request_queue: Any) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    while True:
        try:
            req = capture_request_queue.get_nowait()
        except (py_queue.Empty, Exception):
            return requests
        if isinstance(req, dict):
            requests.append(req)


def _sleep_until_stopped(stop_event: Any, seconds: float) -> None:
    end_s = time.monotonic() + max(0.0, seconds)
    while not stop_event.is_set() and time.monotonic() < end_s:
        time.sleep(min(0.1, end_s - time.monotonic()))

# ---------------------------------------------------------------------------
# Worker-only helpers. Keep Torch/OpenCV/DepthAI/CUDA imports inside functions.
# ---------------------------------------------------------------------------

def _build_dai_gen_from_config(config: dict[str, Any]) -> Any:
    from resultkit.dai.rgb_stereo_generator import DepthAIPoeRGBStereoTorchGenerator
    config:CameraConfig = CameraConfig(**config)
    return DepthAIPoeRGBStereoTorchGenerator(
        uuid=config.uuid,
        sources=config.sources,
        rgb_width=config.rgb_width,
        rgb_height=config.rgb_height,
        stereo_width=config.stereo_width,
        stereo_height=config.stereo_height,
        capture_fps=config.capture_fps,
        rgb_codec=config.rgb_codec,
        stereo_codec=config.stereo_codec,
        rgb_bitrate_kbps=config.rgb_bitrate_kbps,
        stereo_bitrate_kbps=config.stereo_bitrate_kbps,
        decoder_backend=config.decoder_backend,
        gst_nvivafilter_so=config.gst_nvivafilter_so,
        gst_nvivafilter_dtype=config.gst_nvivafilter_dtype,
        gst_nvivafilter_channel_order=config.gst_nvivafilter_channel_order,
        decoder_output_color=config.decoder_output_color,
        stereo_decoder_output_color=config.stereo_decoder_output_color,
        rgb_camera_socket=config.rgb_camera_socket,
        left_camera_socket=config.left_camera_socket,
        right_camera_socket=config.right_camera_socket,
        normalize_rgb=config.normalize_rgb,
        normalize_stereo=config.normalize_stereo,
        color_types=[],
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )


@contextmanager
def _pycuda_context(device: int):
    """Use the CUDA primary context so PyCUDA, Torch, and CUDA IPC agree."""

    import pycuda.driver as cuda

    cuda.init()
    ctx = cuda.Device(int(device)).retain_primary_context()
    ctx.push()
    try:
        yield
    finally:
        ctx.pop()
        ctx.detach()


def _publisher_context(config: dict[str, Any]):
    if not bool(config.get("pub_enabled", True)):
        return nullcontext()
    return _pycuda_context(int(config.get("pub_device", 0)))


def _mat_device(device: int) -> Any:
    from resultkit.mat import MatDevice

    return getattr(MatDevice, f"CUDA{int(device)}", MatDevice.CUDA0)


def _stream_size_from_config(config: dict[str, Any], stream: str) -> tuple[int, int]:
    if config.get("pub_width") is not None and config.get("pub_height") is not None:
        return int(config["pub_width"]), int(config["pub_height"])
    if stream == "rgb":
        return int(config["rgb_width"]), int(config["rgb_height"])
    return int(config["stereo_width"]), int(config["stereo_height"])


def _stream_is_normalized(config: dict[str, Any], stream: str) -> bool:
    if stream == "rgb":
        return bool(config.get("normalize_rgb", True))
    return bool(config.get("normalize_stereo", True))


def _make_cuda_image_endpoint(config: dict[str, Any], *, stream: str, is_pub: bool = True) -> Any:
    import numpy as np
    import pycuda.gpuarray as gpuarray
    from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
    from resultkit.mat import DataType

    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError("Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build")

    width, height = _stream_size_from_config(config, stream)
    try:
        data = gpuarray.empty((height, width, 3), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(
            "PyCUDA could not allocate the publisher image buffer. "
            "A CUDA primary context must be current in the camera worker."
        ) from exc

    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.RGB,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=_mat_device(int(config.get("pub_device", 0))),
        data=data,
        num_slots=int(config.get("num_slots", 3)),
    )
    img.set_id(str(config.get("image_topic", "ImageMatCUDAPubSub:daiRgb"))).init()

    try:
        img.is_pub = bool(is_pub)
    except Exception:
        pass

    return img


def _as_cuda_hwc_rgb8(tensor: Any, *, width: int, height: int, normalized: bool) -> Any:
    """Normalize a DepthAI Torch image tensor to CUDA HWC/RGB/uint8."""

    import torch

    t = tensor.detach()
    if not getattr(t, "is_cuda", False):
        raise RuntimeError("publisher expected a CUDA tensor from DepthAI")

    # Drop a leading batch dimension. The generator commonly returns NCHW/NHWC.
    if t.ndim == 4:
        if int(t.shape[0]) < 1:
            raise RuntimeError(f"empty batch tensor cannot be published: {tuple(t.shape)}")
        t = t[0]

    if t.ndim == 2:
        # Mono/depth-like stream -> RGB by replication.
        t = t[:height, :width].unsqueeze(-1).expand(-1, -1, 3)
    elif t.ndim == 3 and int(t.shape[0]) in (1, 3, 4) and int(t.shape[-1]) not in (1, 3, 4):
        # CHW/RGBP/RGBA planar -> HWC/RGB.
        t = t[:, :height, :width]
        if int(t.shape[0]) == 1:
            t = t.expand(3, -1, -1)
        t = t[:3].permute(1, 2, 0)
    elif t.ndim == 3 and int(t.shape[-1]) in (1, 3, 4):
        # HWC/RGB/RGBA -> HWC/RGB.
        t = t[:height, :width, :]
        if int(t.shape[-1]) == 1:
            t = t.expand(-1, -1, 3)
        t = t[:, :, :3]
    else:
        raise RuntimeError(f"unsupported publisher tensor shape: {tuple(t.shape)}")

    if tuple(int(v) for v in t.shape) != (height, width, 3):
        raise RuntimeError(
            f"publisher tensor shape {tuple(t.shape)} does not match endpoint "
            f"shape {(height, width, 3)}. Check rgb/stereo dimensions or pub_width/pub_height."
        )

    if t.dtype != torch.uint8:
        if getattr(t.dtype, "is_floating_point", False) or t.dtype.is_floating_point:
            if normalized:
                t = t.clamp(0.0, 1.0) * 255.0
            else:
                t = t.clamp(0.0, 255.0)
        else:
            t = t.clamp(0, 255)
        t = t.to(dtype=torch.uint8)

    return t.contiguous()


def _publish_latest_frame(
    image_pub: Any,
    config: dict[str, Any],
    tensors: dict[StreamName, Any],
    *,
    frame_id: int,
) -> int:
    stream = str(config.get("pub_stream", "rgb"))
    if stream not in tensors:
        raise RuntimeError(f"Unsupported pub_stream: {stream!r}")

    width, height = _stream_size_from_config(config, stream)
    image_pub.pub(
        data=_as_cuda_hwc_rgb8(
            tensors[stream],  # type: ignore[index]
            width=width,
            height=height,
            normalized=_stream_is_normalized(config, stream),
        )
    )
    return frame_id


def _close_quietly(obj: Any) -> None:
    try:
        if obj is not None:
            obj.close()
    except Exception:
        pass


def _emit_worker_status(status_queue: Any, state: str, **fields: Any) -> None:
    fields.setdefault("timestamp_s", time.time())
    fields.setdefault("error", None)
    _put_drop_oldest(status_queue, {"state": state, **fields})


def _unpack_frame(gen: Any, mats: Any) -> dict[StreamName, Any]:
    rgb, stereo, left, right = gen.unpack_packed_tensor(mats[0].data)
    return {"rgb": rgb, "stereo": stereo, "left": left, "right": right}


def _emit_frame_status(
    status_queue: Any,
    tensors: dict[StreamName, Any],
    *,
    frame_id: int,
    frame_timestamp_s: float,
    restart_count: int,
    publishing: bool = False,
    last_published_frame_id: int | None = None,
    last_publish_error: str | None = None,
) -> None:
    shapes = {f"{stream}_shape": _shape(tensor) for stream, tensor in tensors.items()}
    _emit_worker_status(
        status_queue,
        "running",
        restart_count=restart_count,
        last_frame_id=frame_id,
        last_frame_timestamp_s=frame_timestamp_s,
        publishing=publishing,
        last_published_frame_id=last_published_frame_id,
        last_publish_error=last_publish_error,
        **shapes,
    )


def _handle_capture_requests(
    capture_request_queue: Any,
    capture_result_queue: Any,
    latest_tensors: dict[StreamName, Any] | None,
    *,
    frame_id: int,
    frame_timestamp_s: float,
) -> None:
    for req in _drain_capture_requests(capture_request_queue):
        request_id = int(req.get("request_id", -1))
        stream = str(req.get("stream", "rgb"))
        quality = int(req.get("jpeg_quality", 85))

        try:
            if latest_tensors is None:
                raise RuntimeError("No latest frame is available yet")
            if stream not in latest_tensors:
                raise RuntimeError(f"Unsupported stream: {stream!r}")

            tensor = latest_tensors[stream]  # type: ignore[index]
            _put_drop_oldest(
                capture_result_queue,
                {
                    "ok": True,
                    "request_id": request_id,
                    "frame_id": frame_id,
                    "frame_timestamp_s": frame_timestamp_s,
                    "stream": stream,
                    "tensor_shape": _shape(tensor),
                    "jpeg_base64": _tensor_to_jpeg_base64_in_worker(
                        tensor,
                        quality,
                        rgb_to_bgr=(stream == "rgb"),
                    ),
                },
            )
        except Exception:
            _put_drop_oldest(
                capture_result_queue,
                {"ok": False, "request_id": request_id, "error": traceback.format_exc()},
            )


def _show_preview(config: dict[str, Any], tensors: dict[StreamName, Any], stop_event: Any) -> None:
    import cv2

    rgb_step = int(config["preview_rgb_downsample"])
    stereo_step = int(config["preview_stereo_downsample"])
    cv2.imshow("rpc_camera_rgb", _to_cv_array(tensors["rgb"][0], downsample=rgb_step, rgb_to_bgr=True))
    cv2.imshow("rpc_camera_left", _to_cv_array(tensors["left"][0], downsample=stereo_step, rgb_to_bgr=False))
    cv2.imshow("rpc_camera_right", _to_cv_array(tensors["right"][0], downsample=stereo_step, rgb_to_bgr=False))

    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        stop_event.set()


def _release_worker_resources(gen: Any, image_pub: Any = None) -> str | None:
    error = None
    try:
        _close_quietly(image_pub)
    except Exception:
        error = traceback.format_exc()
    try:
        if gen is not None:
            gen.release()
    except Exception:
        error = traceback.format_exc()
    try:
        import cv2
        cv2.destroyWindow("rpc_camera_rgb")
        cv2.destroyWindow("rpc_camera_left")
        cv2.destroyWindow("rpc_camera_right")
    except Exception:
        pass
    return error


def _camera_worker(
    config: dict[str, Any],
    status_queue: Any,
    capture_request_queue: Any,
    capture_result_queue: Any,
    stop_event: Any,
) -> None:
    """Child process that owns DepthAI/Torch/CUDA and returns JSON-safe data."""

    print("DepthAI RPC camera worker starting.", flush=True)
    frame_id = restart_count = 0
    last_status_emit_s = 0.0
    retry_forever = bool(config.get("retry_forever", True))
    retry_delay_s = float(config.get("retry_delay_s", 1.0))

    while not stop_event.is_set():
        gen = latest_tensors = image_pub = None
        latest_timestamp_s = 0.0
        last_published_frame_id = None
        last_publish_error = None
        publish_disabled_after_error = False
        restart_count += 1
        _emit_worker_status(
            status_queue,
            "starting",
            restart_count=restart_count,
            last_frame_id=frame_id or None,
            publishing=False,
        )

        try:
            with _publisher_context(config):
                try:
                    gen = _build_dai_gen_from_config(config)
                    pub_stream = str(config.get("pub_stream", "rgb"))
                    if bool(config.get("pub_enabled", True)):
                        image_pub = _make_cuda_image_endpoint(config, stream=pub_stream, is_pub=True)
                        width, height = _stream_size_from_config(config, pub_stream)
                        print(
                            f"server_dai: DepthAI {pub_stream!r} -> CUDA pub "
                            f"{config.get('image_topic')!r} ({width}x{height})",
                            flush=True,
                        )

                    _emit_worker_status(
                        status_queue,
                        "running",
                        restart_count=restart_count,
                        last_frame_id=frame_id or None,
                        publishing=image_pub is not None,
                        last_published_frame_id=last_published_frame_id,
                    )

                    for mats in gen:
                        if stop_event.is_set():
                            break

                        frame_id += 1
                        latest_timestamp_s = time.time()
                        latest_tensors = _unpack_frame(gen, mats)

                        if image_pub is not None:
                            try:
                                last_published_frame_id = _publish_latest_frame(
                                    image_pub,
                                    config,
                                    latest_tensors,
                                    frame_id=frame_id,
                                )
                                last_publish_error = None
                            except Exception:
                                last_publish_error = traceback.format_exc()
                                print(last_publish_error, flush=True)
                                _emit_worker_status(
                                    status_queue,
                                    "publish_error",
                                    error=last_publish_error,
                                    restart_count=restart_count,
                                    last_frame_id=frame_id or None,
                                    last_frame_timestamp_s=latest_timestamp_s or None,
                                    publishing=False,
                                    last_published_frame_id=last_published_frame_id,
                                    last_publish_error=last_publish_error,
                                )
                                if bool(config.get("pub_required", False)):
                                    raise
                                _close_quietly(image_pub)
                                image_pub = None
                                publish_disabled_after_error = True

                        now = time.monotonic()
                        if now - last_status_emit_s >= 1.0:
                            last_status_emit_s = now
                            _emit_frame_status(
                                status_queue,
                                latest_tensors,
                                frame_id=frame_id,
                                frame_timestamp_s=latest_timestamp_s,
                                restart_count=restart_count,
                                publishing=image_pub is not None and not publish_disabled_after_error,
                                last_published_frame_id=last_published_frame_id,
                                last_publish_error=last_publish_error,
                            )

                        _handle_capture_requests(
                            capture_request_queue,
                            capture_result_queue,
                            latest_tensors,
                            frame_id=frame_id,
                            frame_timestamp_s=latest_timestamp_s,
                        )

                        if config.get("preview", False):
                            _show_preview(config, latest_tensors, stop_event)

                    if not stop_event.is_set():
                        _emit_worker_status(
                            status_queue,
                            "ended",
                            error="DepthAI generator ended; restarting camera session.",
                            restart_count=restart_count,
                            last_frame_id=frame_id or None,
                            last_frame_timestamp_s=latest_timestamp_s or None,
                            publishing=False,
                            last_published_frame_id=last_published_frame_id,
                            last_publish_error=last_publish_error,
                        )
                finally:
                    err = _release_worker_resources(gen, image_pub)
                    if err:
                        print(err, flush=True)
                        _emit_worker_status(
                            status_queue,
                            "release_error",
                            error=err,
                            restart_count=restart_count,
                            last_frame_id=frame_id or None,
                            publishing=False,
                        )

        except KeyboardInterrupt:
            stop_event.set()
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            _emit_worker_status(
                status_queue,
                "error",
                error=err,
                restart_count=restart_count,
                last_frame_id=frame_id or None,
                publishing=False,
                last_published_frame_id=last_published_frame_id,
                last_publish_error=last_publish_error,
            )

        if stop_event.is_set() or not retry_forever:
            break

        _emit_worker_status(
            status_queue,
            "retry_wait",
            restart_count=restart_count,
            last_frame_id=frame_id or None,
        )
        _sleep_until_stopped(stop_event, retry_delay_s)

    _emit_worker_status(
        status_queue,
        "stopped",
        restart_count=restart_count,
        last_frame_id=frame_id or None,
    )
    print("DepthAI RPC camera worker stopped.", flush=True)


# ---------------------------------------------------------------------------
# RPC controller. Parent process owns only queues, metadata, and JPEG text.
# ---------------------------------------------------------------------------


@dataclass
class CameraController:
    """JSON-RPC controller backed by an isolated DepthAI worker process."""

    service_name: str = "serverCam"
    controller_name: str = "camera"
    config: CameraConfig = field(default_factory=CameraConfig)
    opened: bool = False
    captures: int = 0

    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _ctx: Any = field(default=None, init=False, repr=False)
    _status_queue: Any = field(default=None, init=False, repr=False)
    _capture_request_queue: Any = field(default=None, init=False, repr=False)
    _capture_result_queue: Any = field(default=None, init=False, repr=False)
    _stop_event: Any = field(default=None, init=False, repr=False)
    _process: Any = field(default=None, init=False, repr=False)

    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_worker_state: str | None = field(default=None, init=False, repr=False)
    _restart_count: int = field(default=0, init=False, repr=False)
    _process_restart_count: int = field(default=0, init=False, repr=False)
    _last_frame_id: int | None = field(default=None, init=False, repr=False)
    _last_frame_timestamp_s: float | None = field(default=None, init=False, repr=False)
    _publishing: bool = field(default=False, init=False, repr=False)
    _last_published_frame_id: int | None = field(default=None, init=False, repr=False)
    _last_publish_error: str | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)

    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @staticmethod
    def openapi_examples():
        return {
            **openapi_doc("camera_status", id=1, params={}),
            **openapi_doc("camera_start",  id=2, params=CameraConfig().model_dump()),
            **openapi_doc("camera_stop",   id=3, params={}),
            **openapi_doc("camera_capture",id=4, params=CaptureParams().model_dump()),
        }
    
    def _is_worker_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _worker_exitcode(self) -> int | None:
        try:
            return None if self._process is None else self._process.exitcode
        except Exception:
            return None

    def _reset_runtime_state_unlocked(self) -> None:
        self._last_error = None
        self._last_worker_state = None
        self._last_frame_id = None
        self._last_frame_timestamp_s = None
        self._publishing = False
        self._last_published_frame_id = None
        self._last_publish_error = None
        self._restart_count = 0
        self._process_restart_count = 0
        self._request_id = 0

    def _apply_worker_status_unlocked(self, msg: dict[str, Any]) -> None:
        state = msg.get("state")
        if state:
            self._last_worker_state = str(state)
        self._restart_count = _int_or_keep(self._restart_count, msg.get("restart_count")) or 0
        self._last_frame_id = _int_or_keep(self._last_frame_id, msg.get("last_frame_id"))
        self._last_frame_timestamp_s = _float_or_keep(
            self._last_frame_timestamp_s,
            msg.get("last_frame_timestamp_s"),
        )
        if "publishing" in msg:
            self._publishing = bool(msg.get("publishing"))
        self._last_published_frame_id = _int_or_keep(
            self._last_published_frame_id,
            msg.get("last_published_frame_id"),
        )
        if msg.get("last_publish_error"):
            self._last_publish_error = str(msg["last_publish_error"])
        if msg.get("error"):
            self._last_error = str(msg["error"])

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                msg = self._status_queue.get_nowait()
            except py_queue.Empty:
                break
            except Exception as exc:
                self._last_error = f"Failed reading worker status queue: {exc!r}"
                break
            if isinstance(msg, dict):
                self._apply_worker_status_unlocked(msg)

    def _cleanup_queues_unlocked(self) -> None:
        for attr in ("_status_queue", "_capture_request_queue", "_capture_result_queue"):
            q = getattr(self, attr)
            if q is None:
                continue
            try:
                _drain_queue(q)
                q.close()
                q.join_thread()
            except Exception:
                pass
            setattr(self, attr, None)
        self._stop_event = None

    def _ensure_watchdog_unlocked(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="camera-rpc-worker-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            try:
                interval_s = max(0.1, float(self.config.worker_watchdog_interval_s))
            except Exception:
                interval_s = 0.5

            if self._watchdog_stop.wait(interval_s):
                break

            try:
                with self._state_lock:
                    if self.opened and self._process is not None and not self._is_worker_alive():
                        self._restart_dead_worker_unlocked()
                    else:
                        self._drain_status_queue()
            except Exception:
                self._last_error = traceback.format_exc()

    def _start_worker_unlocked(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._status_queue = self._ctx.Queue(maxsize=32)
        self._capture_request_queue = self._ctx.Queue(maxsize=int(self.config.queue_max_size))
        self._capture_result_queue = self._ctx.Queue(maxsize=int(self.config.queue_max_size))
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_camera_worker,
            args=(
                _model_to_dict(self.config),
                self._status_queue,
                self._capture_request_queue,
                self._capture_result_queue,
                self._stop_event,
            ),
            daemon=False,
        )
        self._process.start()
        self.opened = True
        self._last_worker_state = "process_started"
        self._ensure_watchdog_unlocked()

    def _restart_dead_worker_unlocked(self) -> None:
        """Recover if the worker process died from a native crash."""

        if not self.opened or self._process is None or self._is_worker_alive():
            return

        self._drain_status_queue()
        exitcode = self._worker_exitcode()
        self._process_restart_count += 1
        self._last_error = f"Camera worker process exited with code {exitcode}; restarting."

        try:
            self._process.join(timeout=0.0)
        except Exception:
            pass

        self._process = None
        self._cleanup_queues_unlocked()
        self._last_worker_state = "process_restarting"

        try:
            self._start_worker_unlocked()
        except Exception:
            self.opened = False
            self._last_worker_state = "process_restart_failed"
            self._last_error = traceback.format_exc()

    def _stop_worker_unlocked(self) -> None:
        process, stop_event = self._process, self._stop_event

        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass

        if process is not None:
            try:
                process.join(timeout=float(self.config.close_join_timeout_s))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=float(self.config.close_terminate_timeout_s))
            except Exception:
                pass

        self._process = None
        self.opened = False
        self._drain_status_queue()
        self._cleanup_queues_unlocked()
        self._last_worker_state = "closed"

    def _status_unlocked(self) -> CameraStatusResult:
        if self.opened and self._process is not None and not self._is_worker_alive():
            self._restart_dead_worker_unlocked()
        self._drain_status_queue()

        age_s = None
        if self._last_frame_timestamp_s is not None:
            age_s = max(0.0, time.time() - self._last_frame_timestamp_s)

        return CameraStatusResult(**{
            "opened": bool(self.opened),
            "captures": self.captures,
            "device_id": ",".join(self.config.sources),
            "worker_alive": self._is_worker_alive(),
            "width": self.config.rgb_width,
            "height": self.config.rgb_height,
            "rgb_width": self.config.rgb_width,
            "rgb_height": self.config.rgb_height,
            "stereo_width": self.config.stereo_width,
            "stereo_height": self.config.stereo_height,
            "fps": self.config.capture_fps,
            "last_frame_id": self._last_frame_id,
            "last_frame_age_s": age_s,
            "last_error": self._last_error,
            "worker_state": self._last_worker_state,
            "restart_count": self._restart_count,
            "process_restart_count": self._process_restart_count,
            "worker_exitcode": self._worker_exitcode(),
            "publishing": bool(self._publishing),
            "image_topic": self.config.image_topic,
            "pub_stream": self.config.pub_stream,
            "last_published_frame_id": self._last_published_frame_id,
            "last_publish_error": self._last_publish_error,
        })

    def open(self, params: CameraConfig) -> CameraStatusResult:
        """Start the isolated camera worker process."""

        config = params or self.config
        with self._state_lock:
            try:
                if self._is_worker_alive() and config == self.config:
                    self.opened = True
                    return self._status_unlocked()

                if self._process is not None:
                    self._stop_worker_unlocked()

                self.config = config
                self._reset_runtime_state_unlocked()
                self._start_worker_unlocked()
            except Exception:
                self.opened = False
                self._last_worker_state = "open_failed"
                self._last_error = traceback.format_exc()
            return self._status_unlocked()

    def close(self, params: EmptyParams) -> CameraStatusResult:
        """Stop the worker process and release the camera."""

        with self._state_lock:
            self._stop_worker_unlocked()
            return self._status_unlocked()

    def status(self, params: EmptyParams) -> CameraStatusResult:
        with self._state_lock:
            return self._status_unlocked()

    def _put_capture_request_unlocked(self, request: dict[str, Any]) -> None:
        if self._capture_request_queue is None:
            raise RuntimeError("Camera worker request queue is not available")
        try:
            self._capture_request_queue.put_nowait(request)
        except py_queue.Full:
            _drain_queue(self._capture_request_queue)
            self._capture_request_queue.put_nowait(request)

    def _wait_capture_result_unlocked(self, request_id: int, wait_s: float) -> dict[str, Any] | None:
        if self._capture_result_queue is None:
            return None

        deadline = time.monotonic() + max(0.0, wait_s)
        while time.monotonic() < deadline:
            try:
                result = self._capture_result_queue.get(
                    timeout=min(0.1, max(0.0, deadline - time.monotonic()))
                )
            except py_queue.Empty:
                if self.opened and self._process is not None and not self._is_worker_alive():
                    return None
                continue
            except Exception:
                return None

            if isinstance(result, dict) and int(result.get("request_id", -999999)) == request_id:
                return result
        return None

    def _capture_once_unlocked(self, request: dict[str, Any]) -> dict[str, Any] | None:
        self._request_id += 1
        request = {**request, "request_id": self._request_id, "timestamp_s": time.time()}
        self._put_capture_request_unlocked(request)
        return self._wait_capture_result_unlocked(
            self._request_id,
            wait_s=float(self.config.capture_wait_s),
        )

    def _capture_size(self, stream: str) -> tuple[int, int]:
        if stream == "rgb":
            return self.config.rgb_width, self.config.rgb_height
        return self.config.stereo_width, self.config.stereo_height

    def _build_capture_result(
        self,
        result: dict[str, Any],
        status: CameraStatusResult,
        stream: str,
    ) -> CaptureResult:
        width, height = self._capture_size(stream)
        payload = _model_to_dict(status)
        payload.update({
            "captures": self.captures,
            "width": width,
            "height": height,
            "frame_id": int(result["frame_id"]),
            "frame_timestamp_s": float(result["frame_timestamp_s"]),
            "stream": stream,
            "tensor_shape": tuple(int(x) for x in result["tensor_shape"]),
            "jpeg_base64": str(result["jpeg_base64"]),
        })
        return CaptureResult(**payload)

    def capture(self, params: CaptureParams) -> CaptureResult:
        """Ask the worker to JPEG-encode the latest stream and return it."""

        with self._state_lock:
            if self._process is not None and not self._is_worker_alive() and self.opened:
                self._restart_dead_worker_unlocked()
            elif not self._is_worker_alive():
                self.open(self.config)

            if not self.opened or not self._is_worker_alive():
                status = self._status_unlocked()
                raise RuntimeError(
                    "Camera worker is not running. "
                    f"worker_state={status.worker_state!r}, last_error={status.last_error!r}"
                )

            _drain_queue(self._capture_result_queue)
            request = {"stream": params.stream, "jpeg_quality": int(params.jpeg_quality)}
            result = self._capture_once_unlocked(request)

            # If a native crash killed the worker while capture was waiting,
            # restart once and give the new worker one more capture interval.
            if result is None and self.opened and self._process is not None and not self._is_worker_alive():
                self._restart_dead_worker_unlocked()
                if self.opened and self._is_worker_alive():
                    _drain_queue(self._capture_result_queue)
                    result = self._capture_once_unlocked(request)

            if result is None:
                status = self._status_unlocked()
                raise RuntimeError(
                    f"No JPEG capture received within {self.config.capture_wait_s}s. "
                    f"worker_alive={status.worker_alive}, "
                    f"worker_state={status.worker_state!r}, "
                    f"last_error={status.last_error!r}"
                )

            if not result.get("ok", False):
                self._last_error = str(result.get("error", "capture failed"))
                raise RuntimeError(self._last_error)

            self.captures += 1
            self._last_frame_id = int(result["frame_id"])
            self._last_frame_timestamp_s = float(result["frame_timestamp_s"])
            status = self._status_unlocked()
            stream = str(result.get("stream", params.stream))
            return self._build_capture_result(result, status, stream)


def run_server(service_name: str = "serverCam", controller_name: str = "camera") -> None:
    """Run the real camera controller as an iceoryx2 JSON-RPC service."""

    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    mp.freeze_support()
    Iox2JsonRpcServer(CameraController(service_name=service_name,
                                       controller_name=controller_name)).run_forever()



if __name__ == "__main__":
    run_server(
        service_name=args.service_name,
        controller_name=args.controller_name,
    )