#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import struct
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import torch

# Keep this if the demo lives in an examples/tests folder next to resultkit.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from cuda_ipc_runtime import Config, FpsMeter, FramePacer, StoppableLoop, make_cuda_image_endpoint, mat_device


YOLO_TOPIC = "ImageMatCUDAPubSub:yolo"

# Environment overrides:
#   set YOLO_MODEL=yolov8n.pt
#   set YOLO_CONF=0.25
#   set YOLO_IOU=0.45
#   set YOLO_MAX_DET=100
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "100"))


@contextmanager
def pushed_cuda_context(ctx):
    """
    Temporarily make the CUDA primary context current for PyCUDA/resultkit calls.

    Important:
        Do not keep this pushed while running YOLO / torchvision / PyTorch.
        Push only around resultkit CUDA IPC operations.
    """
    ctx.push()
    try:
        yield
    finally:
        ctx.pop()


def make_cuda_yolo_endpoint(cfg: Config, *, is_pub: bool, output_topic: str = YOLO_TOPIC):
    import pycuda.gpuarray as gpuarray
    from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
    from resultkit.mat import DataType

    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise RuntimeError("Model4Mat.ImageMatCUDAPubSub is not available in this resultkit build")

    data = gpuarray.empty((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)

    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.RGB,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=mat_device(cfg.device),
        data=data,
        num_slots=int(cfg.num_slots),
    )

    img.set_id(output_topic).init()

    img.init()

    try:
        img.is_pub = bool(is_pub)
    except Exception:
        pass

    return img


@lru_cache(maxsize=8)
def _get_yolo_model(device_index: int):
    """
    Load the YOLO model once per CUDA device.

    Requires:
        pip install ultralytics opencv-python
    """
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL)
    model.to(f"cuda:{int(device_index)}")
    return model


def _uint16_le(value: int) -> bytes:
    return struct.pack("<H", max(0, min(65535, int(value))))


def _encode_detections_into_pixels(
    out_img: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None,
    conf: torch.Tensor | None,
    cls: torch.Tensor | None,
) -> torch.Tensor:
    """
    Encode YOLO result into the first bytes of the output RGB image.

    Layout:
        bytes 0..7    magic: b"YOLORES1"
        bytes 8..9    detection count uint16 little-endian
        bytes 10..15   reserved

    Per detection, 12 bytes:
        x1 uint16
        y1 uint16
        x2 uint16
        y2 uint16
        conf uint16, confidence * 10000
        cls uint16

    The output is still a normal HWC RGB uint8 image, so it can be published
    through ImageMatCUDAPubSub.
    """
    if out_img.dtype != torch.uint8:
        raise ValueError("out_img must be torch.uint8")

    if out_img.ndim != 3 or int(out_img.shape[-1]) != 3:
        raise ValueError(f"out_img must be HWC RGB, got {tuple(out_img.shape)}")

    if not out_img.is_contiguous():
        out_img = out_img.contiguous()

    device = out_img.device
    flat = out_img.reshape(-1)

    if boxes_xyxy is None or conf is None or cls is None or boxes_xyxy.numel() == 0:
        payload = b"YOLORES1" + _uint16_le(0) + bytes(6)
    else:
        boxes_cpu = boxes_xyxy[:YOLO_MAX_DET].detach().round().to(torch.int32).cpu()
        conf_cpu = conf[:YOLO_MAX_DET].detach().cpu()
        cls_cpu = cls[:YOLO_MAX_DET].detach().round().to(torch.int32).cpu()

        count = min(int(boxes_cpu.shape[0]), YOLO_MAX_DET)

        payload = bytearray()
        payload += b"YOLORES1"
        payload += _uint16_le(count)
        payload += bytes(6)

        for i in range(count):
            x1, y1, x2, y2 = [int(v) for v in boxes_cpu[i].tolist()]
            score = int(float(conf_cpu[i]) * 10000.0)
            class_id = int(cls_cpu[i])

            payload += _uint16_le(x1)
            payload += _uint16_le(y1)
            payload += _uint16_le(x2)
            payload += _uint16_le(y2)
            payload += _uint16_le(score)
            payload += _uint16_le(class_id)

        payload = bytes(payload)

    if len(payload) > flat.numel():
        raise RuntimeError(
            f"YOLO encoded payload needs {len(payload)} bytes, "
            f"but output image only has {flat.numel()} bytes"
        )

    payload_np = np.frombuffer(payload, dtype=np.uint8).copy()
    encoded = torch.from_numpy(payload_np).to(device=device, dtype=torch.uint8)
    flat[: encoded.numel()] = encoded

    return out_img

def yolo_step(img_uint8: torch.Tensor) -> torch.Tensor:
    """
    Input:
        img_uint8:
            CUDA HWC RGB uint8 tensor, shape [H, W, 3].

    Output:
        CUDA HWC RGB uint8 tensor, same original shape.

    Behavior:
        - Crops edges so H and W are divisible by 32.
        - For 720x1280 input, YOLO sees 704x1280.
        - Boxes are shifted back to full-frame coordinates.
        - Output image stays 720x1280.
    """
    if not img_uint8.is_cuda:
        raise RuntimeError("yolo_step expects a CUDA tensor")

    if img_uint8.dtype != torch.uint8:
        raise RuntimeError(f"yolo_step expects torch.uint8, got {img_uint8.dtype}")

    if img_uint8.ndim != 3 or int(img_uint8.shape[-1]) != 3:
        raise RuntimeError(f"yolo_step expects HWC RGB image, got shape {tuple(img_uint8.shape)}")

    img_uint8 = img_uint8.contiguous()

    device_index = int(img_uint8.device.index or 0)
    model = _get_yolo_model(device_index)

    h = int(img_uint8.shape[0])
    w = int(img_uint8.shape[1])

    crop_h = h - (h % 32)
    crop_w = w - (w % 32)

    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(f"image too small after stride crop: original shape={tuple(img_uint8.shape)}")

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    bottom = top + crop_h
    right = left + crop_w

    img_crop = img_uint8[top:bottom, left:right, :].contiguous()

    # Ultralytics tensor input must be BCHW and H/W divisible by stride 32.
    # Input is RGB float in range 0..1.
    inp = (
        img_crop
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )

    with torch.inference_mode():
        results = model.predict(
            source=inp,
            device=f"cuda:{device_index}",
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
            verbose=False,
        )

    result = results[0]
    boxes = getattr(result, "boxes", None)

    boxes_xyxy = None
    conf = None
    cls = None

    if boxes is not None and len(boxes) > 0:
        boxes_xyxy = boxes.xyxy.clone()
        conf = boxes.conf
        cls = boxes.cls

        # Convert crop-local coordinates back to original full-frame coordinates.
        boxes_xyxy[:, [0, 2]] += left
        boxes_xyxy[:, [1, 3]] += top

    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", {})

    out = draw_boxes_gpu_with_bitmap_labels(
            img_uint8,
            boxes_xyxy=boxes_xyxy,
            conf=conf,
            cls=cls,
            names=names,
            font_scale=2,
            color_rgb=DEFAULT_COLOR_PALETTE_RGB,
    )
    # out = draw_boxes_cpu(
    #     img_uint8,
    #     boxes_xyxy=boxes_xyxy,
    #     conf=conf,
    #     cls=cls,
    #     names=names,
    # )

    out = _encode_detections_into_pixels(
        out,
        boxes_xyxy=boxes_xyxy,
        conf=conf,
        cls=cls,
    )

    return out.contiguous()


class YoloLoop(StoppableLoop):
    """CUDA IPC image subscriber -> YOLO CUDA image publisher loop."""

    def __init__(self, cfg: Config, *, output_topic: str = YOLO_TOPIC):
        super().__init__(cfg)
        self.output_topic = output_topic

    def _run(self) -> None:
        torch_loop(self.cfg, output_topic=self.output_topic, stop_event=self._stop_event)


def torch_loop(cfg: Config, output_topic: str = YOLO_TOPIC, stop_event=None) -> None:
    import pycuda.driver as cuda

    cuda.init()

    # Initialize PyTorch CUDA first.
    torch.cuda.set_device(cfg.device)
    torch.empty(1, device=f"cuda:{cfg.device}")

    # Use CUDA primary context, same context family PyTorch uses.
    ctx = cuda.Device(int(cfg.device)).retain_primary_context()

    image_sub = None
    yolo_pub = None
    
    pacer = FramePacer(cfg.fps)
    meter = FpsMeter("yolo-pub", cfg.stats_every)

    try:
        # Create resultkit endpoints while PyCUDA primary context is current.
        with pushed_cuda_context(ctx):
            image_sub = make_cuda_image_endpoint(cfg, is_pub=False)
            yolo_pub = make_cuda_yolo_endpoint(cfg, is_pub=True, output_topic=output_topic)

        last_sequence = -1

        print(
            f"torch-yolo: subscribing {cfg.image_topic!r}, publishing {output_topic!r}",
            flush=True,
        )

        published = 0
        while stop_event is None or not stop_event.is_set():
            # Resultkit/PyCUDA IPC receive must happen with the PyCUDA context current.
            # CRITICAL: do not let YOLO run directly on the remote IPC tensor.
            # Copy it into local PyTorch CUDA memory immediately, then release the
            # remote tensor before leaving the PyCUDA/resultkit section.
            with pushed_cuda_context(ctx):
                image_sub.sub(copy=False, sync=True)

                if getattr(image_sub, "_remote_mem", None) is None:
                    frame = None
                    sequence = None
                    # remote_ptr = 0
                else:
                    sequence = int(getattr(image_sub, "sequence", -1))

                    if sequence == last_sequence:
                        frame = None
                        # remote_ptr = 0
                    else:
                        last_sequence = sequence

                        remote_t = image_sub.get_data_torch(copy=False, sync=False)
                        # remote_ptr = int(remote_t.data_ptr())

                        # This clone breaks the lifetime dependency on the IPC memory
                        # handle. YOLO/torchvision will only see normal PyTorch CUDA
                        # memory, not resultkit's remote IPC slot.
                        frame = remote_t.clone(memory_format=torch.contiguous_format)

                        # Make the clone complete before resultkit is allowed to open,
                        # close, or switch IPC handles on the next iteration.
                        torch.cuda.synchronize(cfg.device)

                        del remote_t

            if frame is None:
                time.sleep(0.001)
                continue

            # print(
            #     f"seq={sequence} shape={tuple(frame.shape)} "
            #     f"dtype={frame.dtype} device={frame.device} "
            #     f"remote_ptr={hex(remote_ptr)} local_ptr={hex(frame.data_ptr())}",
            #     flush=True,
            # )

            # Run YOLO outside the pushed PyCUDA context, using only the local clone.
            yolo_res = yolo_step(frame)

            # Make sure PyTorch/YOLO kernels and copies are complete before publishing.
            torch.cuda.synchronize(cfg.device)

            # Publish result with PyCUDA/resultkit context current.            
            with pushed_cuda_context(ctx):
                yolo_pub.pub(data=yolo_res)

            # Drop references aggressively so Python does not hold old IPC-backed
            # tensors across resultkit slot switches.            
            published += 1
            meter.tick()
            pacer.sleep()
            del frame
            del yolo_res

            if cfg.max_frames is not None and published >= cfg.max_frames:
                return

    finally:
        # Close resultkit endpoints with PyCUDA context current.
        try:
            with pushed_cuda_context(ctx):
                if yolo_pub is not None:
                    try:
                        yolo_pub.close()
                    except Exception:
                        pass

                if image_sub is not None:
                    try:
                        image_sub.close()
                    except Exception:
                        pass
        finally:
            ctx.detach()
