import contextlib
import threading
import time
import traceback
from typing import List, Literal, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import torch

from .generator import ImageMatGenerator
from ..MatModel import ColorFormat

ColorType = ColorFormat
from ..logger import logger


class _MjpegStreamRuntime:
    """Host-side state for one MJPEG DepthAI stream."""

    def __init__(self, name: str):
        self.name = name
        self.depthai_q = None
        self.decode_thread = None
        self.packet_count = 0
        self.byte_count = 0
        self.decoded_frame_count = 0
        self.latest_tensor = None
        self.latest_tensor_normalized = False
        self.latest_frame_index = 0
        self.latest_at = 0.0


class _DepthAIPoeRGBStereoMjpegBottomTorchTensorCapture:
    """
    RGB + stereo capture using MJPEG on all three streams.

    Device side:
        CAM_A RGB   -> NV12/YUV420 -> VideoEncoder MJPEG
        CAM_B left  -> NV12/YUV400p/GRAY8 -> VideoEncoder MJPEG
        CAM_C right -> NV12/YUV400p/GRAY8 -> VideoEncoder MJPEG

    Host side:
        RGB, left and right MJPEG frames are decoded with OpenCV. The latest
        decoded left/right grayscale frames are packed into the RGB tensor.

    Returned tensor shape:
        [1, 3, rgb_height, rgb_width]

    Packing layout:
        left.flatten()  is written into the top payload rows.
        right.flatten() is written into the bottom payload rows.

    The top and bottom payload rows are overwritten by stereo data.
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.stop_event = threading.Event()

        self.rgb = _MjpegStreamRuntime("rgb")
        self.left = _MjpegStreamRuntime("left")
        self.right = _MjpegStreamRuntime("right")

        self.combined_frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_combined_count = 0

        self._latest_stereo_lock = threading.Lock()
        self._stereo_ready = threading.Event()

        self._released = False
        self._exit_stack = contextlib.ExitStack()

        # Lazy-loaded only when owner.mjpeg_decode_backend == "torchvision-cuda".
        # This keeps the OpenCV path dependency-compatible.
        self._torchvision_decode_jpeg = None
        self._torchvision_image_read_mode = None
        self._torchvision_cuda_warning_printed = False
        self._force_opencv_decode = False

        self._start()

    def _open_device(self):
        src = str(self.source).strip()
        if src.startswith("depthai://"):
            src = src.replace("depthai://", "", 1).strip()
        if src in ("", "auto", "default", "none", "None"):
            return dai.Device()
        try:
            return dai.Device(dai.DeviceInfo(src))
        except Exception:
            self._log("error", f"could not open DepthAI device: {src}")
            raise

    @staticmethod
    def _camera_socket(socket_name: str):
        if hasattr(dai.CameraBoardSocket, socket_name):
            return getattr(dai.CameraBoardSocket, socket_name)
        aliases = {"RGB": "CAM_A", "LEFT": "CAM_B", "RIGHT": "CAM_C"}
        alias = aliases.get(socket_name)
        if alias and hasattr(dai.CameraBoardSocket, alias):
            return getattr(dai.CameraBoardSocket, alias)
        raise ValueError(f"Unsupported camera socket: {socket_name}")

    @staticmethod
    def _depthai_mjpeg_profile():
        profile = dai.VideoEncoderProperties.Profile
        for name in ("MJPEG", "Mjpeg", "JPEG", "Jpeg"):
            if hasattr(profile, name):
                return getattr(profile, name)
        raise RuntimeError("This DepthAI build does not expose VideoEncoderProperties.Profile.MJPEG")

    @staticmethod
    def _img_frame_type(type_name: str):
        if hasattr(dai.ImgFrame.Type, type_name):
            return getattr(dai.ImgFrame.Type, type_name)

        if type_name == "GRAY8":
            for fallback in ("YUV400p", "YUV400P", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        if type_name == "YUV400p":
            for fallback in ("YUV400P", "GRAY8", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        raise ValueError(f"Unsupported DepthAI ImgFrame.Type: {type_name}")

    @staticmethod
    def _resize_mode(mode_name: str):
        if hasattr(dai.ImgResizeMode, mode_name):
            return getattr(dai.ImgResizeMode, mode_name)
        raise ValueError(f"Unsupported DepthAI ImgResizeMode: {mode_name}")

    def _mjpeg_decode_backend(self) -> str:
        if self._force_opencv_decode:
            return "opencv"
        return str(getattr(self.owner, "mjpeg_decode_backend", "opencv")).strip().lower()

    def _use_torchvision_cuda_decode(self) -> bool:
        return self._mjpeg_decode_backend() in ("torchvision-cuda", "torchvision_cuda", "cuda", "nvjpeg")

    def _ensure_torchvision_cuda_decode(self):
        """
        Lazy import/validation for TorchVision CUDA JPEG decode.

        torchvision.io.decode_jpeg(input, device="cuda") keeps the encoded JPEG
        bytes on CPU, decodes with nvJPEG, and returns a CUDA uint8 tensor.
        """
        if self._torchvision_decode_jpeg is not None and self._torchvision_image_read_mode is not None:
            return

        device = self._torch_device()
        if device.type != "cuda":
            raise RuntimeError(
                "mjpeg_decode_backend='torchvision-cuda' requires a CUDA torch device. "
                f"Current _torch_device() is {device}. Set torch_device='cuda:0' or gpu_id>=0."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("mjpeg_decode_backend='torchvision-cuda' requires torch.cuda.is_available().")

        try:
            from torchvision.io import ImageReadMode, decode_jpeg
        except Exception as e:
            raise RuntimeError(
                "mjpeg_decode_backend='torchvision-cuda' requires torchvision with image I/O support. "
                "Install a torchvision build matching your torch/CUDA version."
            ) from e

        self._torchvision_decode_jpeg = decode_jpeg
        self._torchvision_image_read_mode = ImageReadMode

    def _maybe_fallback_to_opencv_after_cuda_decode_error(self, err: Exception) -> bool:
        if not bool(getattr(self.owner, "mjpeg_decode_fallback_to_opencv", True)):
            return False

        if not self._torchvision_cuda_warning_printed:
            self._log(
                "warning",
                "TorchVision CUDA MJPEG decode failed; falling back to OpenCV CPU decode. "
                f"Error was: {type(err).__name__}: {err}"
            )
            self._torchvision_cuda_warning_printed = True

        self._force_opencv_decode = True
        try:
            self.owner.mjpeg_decode_backend = "opencv"
        except Exception:
            pass
        return True

    def _encoded_numpy_to_cpu_tensor(self, encoded: np.ndarray) -> torch.Tensor:
        # torchvision.io.decode_jpeg requires the compressed JPEG byte tensor to
        # be CPU uint8. The decoded output goes to device='cuda:...'.
        if encoded.dtype != np.uint8:
            encoded = encoded.astype(np.uint8, copy=False)
        if not encoded.flags.c_contiguous:
            encoded = np.ascontiguousarray(encoded)
        return torch.from_numpy(encoded.reshape(-1))

    def _resize_chw_tensor(self, tensor: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        original_dtype = tensor.dtype
        x = tensor.unsqueeze(0).to(dtype=torch.float32)
        x = torch.nn.functional.interpolate(
            x,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[0]
        if original_dtype == torch.uint8:
            x = x.round().clamp_(0, 255).to(dtype=torch.uint8)
        return x.contiguous()

    def _resize_hw_tensor(self, tensor: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        original_dtype = tensor.dtype
        x = tensor.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        x = torch.nn.functional.interpolate(
            x,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        if original_dtype == torch.uint8:
            x = x.round().clamp_(0, 255).to(dtype=torch.uint8)
        return x.contiguous()

    @staticmethod
    def _packet_to_uint8_array(packet) -> np.ndarray:
        data = packet.getData()
        if isinstance(data, bytes):
            return np.frombuffer(data, dtype=np.uint8)
        if isinstance(data, bytearray):
            return np.frombuffer(bytes(data), dtype=np.uint8)
        return np.asarray(data, dtype=np.uint8).reshape(-1)

    def _create_mjpeg_encoder(self, pipeline, source_output, *, frame_rate: float, quality: int, name: str):
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            source_output,
            frameRate=frame_rate,
            profile=self._depthai_mjpeg_profile(),
        )

        quality = int(max(1, min(100, quality)))
        for setter_name in ("setQuality", "setLossless"):
            if hasattr(encoder, setter_name):
                try:
                    if setter_name == "setQuality":
                        getattr(encoder, setter_name)(quality)
                    else:
                        getattr(encoder, setter_name)(quality >= 100)
                    break
                except Exception as e:
                    self._log("warning", f"could not set OAK {name} MJPEG quality with {setter_name}: {e}")

        return encoder

    def _create_depthai_pipeline(self):
        owner = self.owner

        raw_pipeline = dai.Pipeline(self.device)
        if hasattr(raw_pipeline, "__enter__"):
            pipeline = self._exit_stack.enter_context(raw_pipeline)
        else:
            pipeline = raw_pipeline

        rgb_socket = self._camera_socket(owner.rgb_camera_socket)
        left_socket = self._camera_socket(owner.left_camera_socket)
        right_socket = self._camera_socket(owner.right_camera_socket)

        rgb_cam = pipeline.create(dai.node.Camera).build(rgb_socket)
        left_cam = pipeline.create(dai.node.Camera).build(left_socket)
        right_cam = pipeline.create(dai.node.Camera).build(right_socket)

        rgb_input_type = self._img_frame_type(owner.rgb_mjpeg_input_type)
        rgb_frame = rgb_cam.requestOutput(
            (owner.rgb_width, owner.rgb_height),
            rgb_input_type,
            self._resize_mode(owner.rgb_resize_mode),
            owner.capture_fps,
        )

        stereo_input_type = self._img_frame_type(owner.stereo_mjpeg_input_type)
        stereo_resize_mode = self._resize_mode(owner.stereo_resize_mode)
        left_frame = left_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            stereo_input_type,
            stereo_resize_mode,
            owner.capture_fps,
        )
        right_frame = right_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            stereo_input_type,
            stereo_resize_mode,
            owner.capture_fps,
        )

        rgb_encoder = self._create_mjpeg_encoder(
            pipeline,
            rgb_frame,
            frame_rate=owner.capture_fps,
            quality=owner.rgb_mjpeg_quality,
            name="RGB",
        )
        left_encoder = self._create_mjpeg_encoder(
            pipeline,
            left_frame,
            frame_rate=owner.capture_fps,
            quality=owner.stereo_mjpeg_quality,
            name="left",
        )
        right_encoder = self._create_mjpeg_encoder(
            pipeline,
            right_frame,
            frame_rate=owner.capture_fps,
            quality=owner.stereo_mjpeg_quality,
            name="right",
        )

        self.rgb.depthai_q = rgb_encoder.out.createOutputQueue(
            maxSize=owner.rgb_depthai_queue_size,
            blocking=True,
        )
        self.left.depthai_q = left_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )
        self.right.depthai_q = right_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )

        pipeline.start()
        return pipeline

    def _log(self, path: str, msg: str):
        level = "info"
        if path in ["error", "warning", "debug", "info"]:
            level = path
        logger(f"[{self.owner.uuid}:{path}] {msg}", level=level)

    def _payload_rows_per_stereo(self) -> int:
        stereo_values = int(self.owner.stereo_width) * int(self.owner.stereo_height)
        values_per_payload_row = 3 * int(self.owner.rgb_width)
        return (stereo_values + values_per_payload_row - 1) // values_per_payload_row

    def _start(self):
        owner = self.owner

        payload_rows_per_stereo = self._payload_rows_per_stereo()
        total_payload_rows = 2 * payload_rows_per_stereo
        bottom_payload_start = int(owner.rgb_height) - payload_rows_per_stereo
        if bottom_payload_start < payload_rows_per_stereo:
            raise ValueError(
                "Stereo payload does not fit inside RGB tensor rows: "
                f"payload_rows_per_stereo={payload_rows_per_stereo}, "
                f"total_payload_rows={total_payload_rows}, rgb_height={owner.rgb_height}."
            )

        self._log("info", "Opening DepthAI device...")
        self.device = self._open_device()

        self._log("info", "Connected DepthAI device:")
        self._log("status", f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        self._log("status", f"  Cameras: {self.device.getConnectedCameras()}")

        self._log("info", "Starting DepthAI RGB + MJPEG stereo tensor pipeline:")
        self._log("status", f"  Source: {self.source}")
        self._log("status", f"  RGB socket: {owner.rgb_camera_socket}")
        self._log("status", f"  Left socket: {owner.left_camera_socket}")
        self._log("status", f"  Right socket: {owner.right_camera_socket}")
        self._log("status", f"  RGB size: {owner.rgb_width}x{owner.rgb_height}")
        self._log("status", f"  Stereo size: {owner.stereo_width}x{owner.stereo_height}")
        self._log("status", f"  Capture FPS: {owner.capture_fps}")
        self._log("status", f"  OAK encoder: MJPEG")
        self._log("status", f"  RGB MJPEG quality: {owner.rgb_mjpeg_quality}")
        self._log("status", f"  Stereo MJPEG quality: {owner.stereo_mjpeg_quality}")
        self._log("status", f"  RGB MJPEG input type: {owner.rgb_mjpeg_input_type}")
        self._log("status", f"  Stereo MJPEG input type: {owner.stereo_mjpeg_input_type}")
        self._log("status", f"  MJPEG decode backend: {getattr(owner, 'mjpeg_decode_backend', 'opencv')}")
        self._log("status", f"  Output: [1, 3, {owner.rgb_height}, {owner.rgb_width}]")
        self._log("status", f"  Left payload rows at top: {payload_rows_per_stereo}")
        self._log("status", f"  Right payload rows start row: {bottom_payload_start}")
        self._log("info", "  Note: top and bottom RGB rows are overwritten by stereo payload")
        self._log("status", f"  normalize_rgb: {owner.normalize_rgb}")
        self._log("status", f"  normalize_stereo: {owner.normalize_stereo}")
        self._log("status", f"  torch_device: {self._torch_device()}")

        if self._use_torchvision_cuda_decode():
            self._ensure_torchvision_cuda_decode()

        self.pipeline = self._create_depthai_pipeline()

        for stream in (self.left, self.right):
            stream.decode_thread = threading.Thread(
                target=self._stereo_decode_loop,
                args=(stream,),
                daemon=True,
            )
            stream.decode_thread.start()

        self._log("info", "DepthAI RGB + MJPEG stereo tensor pipeline ready.")

    def _read_packet(self, stream: _MjpegStreamRuntime):
        while not self.stop_event.is_set():
            try:
                if hasattr(stream.depthai_q, "tryGet"):
                    packet = stream.depthai_q.tryGet()
                    if packet is None:
                        time.sleep(0.001)
                        continue
                    return packet
                return stream.depthai_q.get()
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise
        raise StopIteration

    def _decode_mjpeg_rgb_packet(self, packet, stream: _MjpegStreamRuntime) -> torch.Tensor:
        encoded = self._packet_to_uint8_array(packet)
        stream.packet_count += 1
        stream.byte_count += int(encoded.nbytes)

        if self._use_torchvision_cuda_decode():
            try:
                return self._decode_mjpeg_rgb_packet_torchvision_cuda(encoded, stream)
            except Exception as e:
                if not self._maybe_fallback_to_opencv_after_cuda_decode_error(e):
                    raise

        return self._decode_mjpeg_rgb_packet_opencv(encoded, stream)

    def _decode_mjpeg_rgb_packet_opencv(self, encoded: np.ndarray, stream: _MjpegStreamRuntime) -> torch.Tensor:
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV failed to decode {stream.name} MJPEG packet")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if tuple(rgb.shape[:2]) != (int(self.owner.rgb_height), int(self.owner.rgb_width)):
            if self.owner.strict_rgb_shape:
                raise ValueError(
                    f"Decoded RGB frame shape {tuple(rgb.shape[:2])} does not match "
                    f"({self.owner.rgb_height}, {self.owner.rgb_width})."
                )
            rgb = cv2.resize(rgb, (int(self.owner.rgb_width), int(self.owner.rgb_height)), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
        stream.decoded_frame_count += 1
        stream.latest_tensor_normalized = False

        self.owner.on_rgb_tensor(tensor, stream.decoded_frame_count)

        if self.owner.show_rgb_preview:
            self._show_small_rgb_preview(tensor)

        tensor = tensor.unsqueeze(0).to(device=self._torch_device(), non_blocking=self.owner.non_blocking_gpu_copy)
        if self.owner.normalize_rgb:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        return tensor

    def _decode_mjpeg_rgb_packet_torchvision_cuda(self, encoded: np.ndarray, stream: _MjpegStreamRuntime) -> torch.Tensor:
        self._ensure_torchvision_cuda_decode()
        encoded_tensor = self._encoded_numpy_to_cpu_tensor(encoded)
        device = self._torch_device()

        tensor = self._torchvision_decode_jpeg(
            encoded_tensor,
            mode=self._torchvision_image_read_mode.RGB,
            device=device,
        )
        # tensor: [3, H, W], uint8, CUDA

        if tuple(tensor.shape[-2:]) != (int(self.owner.rgb_height), int(self.owner.rgb_width)):
            if self.owner.strict_rgb_shape:
                raise ValueError(
                    f"Decoded RGB frame shape {tuple(tensor.shape[-2:])} does not match "
                    f"({self.owner.rgb_height}, {self.owner.rgb_width})."
                )
            tensor = self._resize_chw_tensor(
                tensor,
                height=int(self.owner.rgb_height),
                width=int(self.owner.rgb_width),
            )

        stream.decoded_frame_count += 1
        stream.latest_tensor_normalized = False

        self.owner.on_rgb_tensor(tensor, stream.decoded_frame_count)

        if self.owner.show_rgb_preview:
            self._show_small_rgb_preview(tensor)

        tensor = tensor.unsqueeze(0)
        if self.owner.normalize_rgb:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        return tensor.contiguous()

    def _decode_mjpeg_gray_packet(self, packet, stream: _MjpegStreamRuntime) -> torch.Tensor:
        encoded = self._packet_to_uint8_array(packet)
        stream.packet_count += 1
        stream.byte_count += int(encoded.nbytes)

        if self._use_torchvision_cuda_decode():
            try:
                return self._decode_mjpeg_gray_packet_torchvision_cuda(encoded, stream)
            except Exception as e:
                if not self._maybe_fallback_to_opencv_after_cuda_decode_error(e):
                    raise

        return self._decode_mjpeg_gray_packet_opencv(encoded, stream)

    def _decode_mjpeg_gray_packet_opencv(self, encoded: np.ndarray, stream: _MjpegStreamRuntime) -> torch.Tensor:
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"OpenCV failed to decode {stream.name} MJPEG packet")

        stereo_h = int(self.owner.stereo_height)
        stereo_w = int(self.owner.stereo_width)
        if tuple(gray.shape[:2]) != (stereo_h, stereo_w):
            if self.owner.strict_stereo_shape:
                raise ValueError(
                    f"Decoded {stream.name} stereo frame shape {tuple(gray.shape[:2])} does not match "
                    f"({self.owner.stereo_height}, {self.owner.stereo_width})."
                )
            gray = cv2.resize(gray, (stereo_w, stereo_h), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(np.ascontiguousarray(gray)).contiguous()
        tensor = tensor.to(device=self._torch_device(), non_blocking=self.owner.non_blocking_gpu_copy)
        stream.decoded_frame_count += 1
        stream.latest_tensor_normalized = False
        return tensor

    def _decode_mjpeg_gray_packet_torchvision_cuda(self, encoded: np.ndarray, stream: _MjpegStreamRuntime) -> torch.Tensor:
        self._ensure_torchvision_cuda_decode()
        encoded_tensor = self._encoded_numpy_to_cpu_tensor(encoded)
        device = self._torch_device()

        tensor = self._torchvision_decode_jpeg(
            encoded_tensor,
            mode=self._torchvision_image_read_mode.GRAY,
            device=device,
        )
        # tensor: [1, H, W], uint8, CUDA
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            gray = tensor[0]
        elif tensor.ndim == 2:
            gray = tensor
        else:
            raise ValueError(f"TorchVision returned unexpected gray JPEG tensor shape {tuple(tensor.shape)}")

        stereo_h = int(self.owner.stereo_height)
        stereo_w = int(self.owner.stereo_width)
        if tuple(gray.shape[-2:]) != (stereo_h, stereo_w):
            if self.owner.strict_stereo_shape:
                raise ValueError(
                    f"Decoded {stream.name} stereo frame shape {tuple(gray.shape[-2:])} does not match "
                    f"({self.owner.stereo_height}, {self.owner.stereo_width})."
                )
            gray = self._resize_hw_tensor(gray, height=stereo_h, width=stereo_w)

        stream.decoded_frame_count += 1
        stream.latest_tensor_normalized = False
        return gray.contiguous()

    def _stereo_decode_loop(self, stream: _MjpegStreamRuntime):
        try:
            while not self.stop_event.is_set():
                packet = self._read_packet(stream)
                gray = self._decode_mjpeg_gray_packet(packet, stream)

                with self._latest_stereo_lock:
                    stream.latest_tensor = gray
                    stream.latest_frame_index = stream.decoded_frame_count
                    stream.latest_at = time.monotonic()
                    if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                        self._stereo_ready.set()

        except StopIteration:
            pass
        except Exception:
            if not self.stop_event.is_set() and not self._released:
                self._log("error", f"DepthAI {stream.name} MJPEG stereo decode thread failed:")
                traceback.print_exc()
        finally:
            self._stereo_ready.set()

    def _get_latest_stereo_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
            return self.left.latest_tensor, self.right.latest_tensor

        timeout = float(getattr(self.owner, "stereo_startup_timeout_sec", 2.0))
        if timeout > 0:
            self._stereo_ready.wait(timeout=timeout)

        with self._latest_stereo_lock:
            if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                return self.left.latest_tensor, self.right.latest_tensor

        if getattr(self.owner, "allow_missing_stereo", False):
            device = self._torch_device()
            zero = torch.zeros(
                (int(self.owner.stereo_height), int(self.owner.stereo_width)),
                dtype=torch.uint8,
                device=device,
            )
            return zero, zero

        while not self.stop_event.is_set() and not self._released:
            self._stereo_ready.wait(timeout=0.01)
            with self._latest_stereo_lock:
                if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                    return self.left.latest_tensor, self.right.latest_tensor

        raise StopIteration

    def _torch_device(self):
        owner = self.owner
        if owner.torch_device:
            return torch.device(owner.torch_device)
        if torch.cuda.is_available() and owner.gpu_id is not None and owner.gpu_id >= 0:
            return torch.device(f"cuda:{owner.gpu_id}")
        return torch.device("cpu")

    def _show_small_rgb_preview(self, tensor: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        if tensor.ndim != 3:
            self._log("error", f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        if tensor.shape[0] == 3:
            small_hwc = tensor[:, ::stride, ::stride].permute(1, 2, 0).contiguous()
        elif tensor.shape[-1] == 3:
            small_hwc = tensor[::stride, ::stride, :].contiguous()
        else:
            self._log("error", f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        small = small_hwc.detach()
        if small.dtype.is_floating_point and float(small.max()) <= 1.5:
            small = small.mul(255.0)
        small_rgb = small.cpu().numpy()
        if small_rgb.dtype != np.uint8:
            small_rgb = np.clip(small_rgb, 0, 255).astype(np.uint8)
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(owner.rgb_window_name, small_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _show_small_stereo_preview(self, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        left = left_gray[::stride, ::stride]
        right = right_gray[::stride, ::stride]
        preview = torch.cat((left, right), dim=1).detach()
        if owner.normalize_stereo and preview.dtype.is_floating_point:
            preview = preview.mul(255.0)
        if preview.dtype != torch.uint8:
            preview = preview.clamp(0, 255).to(dtype=torch.uint8)
        cv2.imshow(owner.stereo_window_name, preview.cpu().numpy())
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _rgb_tensor_to_bchw3(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        if rgb_tensor.ndim != 4:
            raise ValueError(f"Expected RGB tensor with 4 dims, got {tuple(rgb_tensor.shape)}")
        if rgb_tensor.shape[1] == 3:
            return rgb_tensor.contiguous()
        if rgb_tensor.shape[-1] == 3:
            return rgb_tensor.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "Packed RGB+stereo output needs 3 RGB channels. "
            f"Got RGB tensor shape {tuple(rgb_tensor.shape)}."
        )

    def _prepare_stereo_flat_for_rgb(self, gray: torch.Tensor, rgb: torch.Tensor, *, source_normalized: bool = False) -> torch.Tensor:
        flat = gray.reshape(-1)
        if flat.device != rgb.device or flat.dtype != rgb.dtype:
            flat = flat.to(device=rgb.device, dtype=rgb.dtype, non_blocking=self.owner.non_blocking_gpu_copy)
        if self.owner.normalize_stereo and flat.dtype.is_floating_point and not source_normalized:
            flat = flat / 255.0
        return flat

    def _copy_flat_into_payload_channels(self, payload: torch.Tensor, flat: torch.Tensor, start_offset: int = 0):
        """
        Copy a 1-D stereo payload into BCHW rows without payload.reshape(...).

        Storage order used by pack/unpack:
            channel 0 rows, channel 1 rows, then channel 2 rows.
        """

        if payload.ndim != 4 or payload.shape[0] != 1 or payload.shape[1] != 3:
            raise ValueError(f"Expected payload [1, 3, rows, width], got {tuple(payload.shape)}")

        remaining = int(flat.numel())
        src_pos = 0
        dst_pos = int(start_offset)
        per_channel = int(payload.shape[2] * payload.shape[3])

        for channel in range(3):
            if remaining <= 0:
                break
            if dst_pos >= per_channel:
                dst_pos -= per_channel
                continue

            dst_view = payload[0, channel].reshape(-1)
            n = min(remaining, per_channel - dst_pos)
            dst_view[dst_pos:dst_pos + n].copy_(flat[src_pos:src_pos + n], non_blocking=True)
            src_pos += n
            remaining -= n
            dst_pos = 0

        if remaining != 0:
            raise ValueError("Stereo payload did not fit into payload rows.")

    def _fill_payload_tail_channels(self, payload: torch.Tensor, start_offset: int, value: float):
        per_channel = int(payload.shape[2] * payload.shape[3])
        total = 3 * per_channel
        pos = int(start_offset)
        if pos >= total:
            return
        for channel in range(3):
            if pos >= per_channel:
                pos -= per_channel
                continue
            dst_view = payload[0, channel].reshape(-1)
            dst_view[pos:].fill_(value)
            pos = 0

    def _pack_mjpeg_stereo(self, rgb_tensor: torch.Tensor, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        rgb = self._rgb_tensor_to_bchw3(rgb_tensor)
        b, _, rgb_h, rgb_w = rgb.shape

        if b != 1:
            raise ValueError(f"Expected batch size 1, got {b}.")

        stereo_h = int(owner.stereo_height)
        stereo_w = int(owner.stereo_width)

        if tuple(left_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(
                f"Left stereo shape {tuple(left_gray.shape)} does not match {(stereo_h, stereo_w)}"
            )

        if tuple(right_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(
                f"Right stereo shape {tuple(right_gray.shape)} does not match {(stereo_h, stereo_w)}"
            )

        stereo_one_values = stereo_h * stereo_w
        values_per_payload_row = 3 * int(rgb_w)
        payload_rows_per_stereo = (stereo_one_values + values_per_payload_row - 1) // values_per_payload_row
        total_payload_rows = 2 * payload_rows_per_stereo

        if total_payload_rows > int(rgb_h):
            raise ValueError(
                "Stereo payload does not fit in top + bottom RGB tensor rows: "
                f"payload_rows_per_stereo={payload_rows_per_stereo}, "
                f"total_payload_rows={total_payload_rows}, rgb_height={rgb_h}."
            )

        top_payload = rgb[:, :, :payload_rows_per_stereo, :]
        bottom_payload = rgb[:, :, rgb_h - payload_rows_per_stereo:, :]

        left_flat = self._prepare_stereo_flat_for_rgb(
            left_gray,
            rgb,
            source_normalized=bool(getattr(self.left, "latest_tensor_normalized", False)),
        )[:stereo_one_values]

        right_flat = self._prepare_stereo_flat_for_rgb(
            right_gray,
            rgb,
            source_normalized=bool(getattr(self.right, "latest_tensor_normalized", False)),
        )[:stereo_one_values]

        self._copy_flat_into_payload_channels(top_payload, left_flat, start_offset=0)
        self._copy_flat_into_payload_channels(bottom_payload, right_flat, start_offset=0)

        if getattr(owner, "clear_unused_payload_tail", False):
            pad_value = float(owner.packed_stereo_pad_value)
            self._fill_payload_tail_channels(top_payload, start_offset=stereo_one_values, value=pad_value)
            self._fill_payload_tail_channels(bottom_payload, start_offset=stereo_one_values, value=pad_value)

        return rgb

    def next_frame(self):
        if self._released:
            raise StopIteration

        try:
            rgb_packet = self._read_packet(self.rgb)
            rgb_tensor = self._decode_mjpeg_rgb_packet(rgb_packet, self.rgb)
            left_gray, right_gray = self._get_latest_stereo_tensors()

            if self.owner.show_stereo_preview:
                self._show_small_stereo_preview(left_gray, right_gray)

            self.combined_frame_count += 1
            packed_tensor = self._pack_mjpeg_stereo(rgb_tensor, left_gray, right_gray)
            self.owner.on_rgb_stereo_tensor(packed_tensor, self.combined_frame_count)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                combined_fps = (self.combined_frame_count - self.last_combined_count) / dt
                elapsed = max(now - self.started_at, 1e-6)
                rgb_mbps = self.rgb.byte_count * 8.0 / elapsed / 1_000_000
                left_mbps = self.left.byte_count * 8.0 / elapsed / 1_000_000
                right_mbps = self.right.byte_count * 8.0 / elapsed / 1_000_000
                self._log(
                    "info",
                    f"combined={self.combined_frame_count}, "
                    f"fps={combined_fps:.2f}, "
                    f"rgb_dec={self.rgb.decoded_frame_count}, "
                    f"left_dec={self.left.decoded_frame_count}, "
                    f"right_dec={self.right.decoded_frame_count}, "
                    f"mbps rgb/left/right={rgb_mbps:.1f}/{left_mbps:.1f}/{right_mbps:.1f}, "
                    f"payload_rows_per_side={self.owner.packed_stereo_rows_per_side}",
                )
                self.last_log_at = now
                self.last_combined_count = self.combined_frame_count

            return packed_tensor

        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event.is_set():
                raise StopIteration
            raise

    def release(self):
        self._log("debug", "release() called")

        if self._released:
            self._log("debug", "release() skipped: already released")
            return

        self._log("info", "Starting DepthAI MJPEG resource release")
        self._released = True
        self.stop_event.set()
        self._stereo_ready.set()

        streams = (self.rgb, self.left, self.right)

        for stream in streams:
            try:
                if stream.depthai_q is not None and hasattr(stream.depthai_q, "close"):
                    self._log("debug", f"Closing DepthAI queue for stream '{stream.name}'")
                    stream.depthai_q.close()
            except Exception as e:
                self._log("warning", f"Error closing DepthAI queue for stream '{stream.name}': {e}")

        for stream in (self.left, self.right):
            try:
                if stream.decode_thread is not None:
                    self._log("debug", f"Joining decode thread for stream '{stream.name}'")
                    stream.decode_thread.join(timeout=2.0)
                    if stream.decode_thread.is_alive():
                        self._log("warning", f"Decode thread for stream '{stream.name}' did not exit within timeout")
            except Exception as e:
                self._log("error", f"Error joining decode thread for stream '{stream.name}': {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self._log("info", "Stopping DepthAI pipeline")
                    self.pipeline.stop()
                    self._log("info", "DepthAI pipeline stopped")
        except Exception as e:
            self._log("warning", f"DepthAI pipeline stop failed during release: {e}")

        try:
            self._exit_stack.close()
        except Exception as e:
            self._log("warning", f"Error closing exit stack during release: {e}")

        for stream in streams:
            stream.latest_tensor = None
            stream.latest_tensor_normalized = False
            stream.depthai_q = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self._log("info", "Closing DepthAI device")
                self.device.close()
                self._log("info", "DepthAI device closed")
        except Exception as e:
            self._log("warning", f"Error closing DepthAI device during release: {e}")

        self.pipeline = None
        self.device = None

        for window_name in (self.owner.rgb_window_name, self.owner.stereo_window_name):
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        self._log("info", "DepthAI MJPEG resource release completed")


class DepthAIPoeRGBStereoMjpegTorchGenerator(ImageMatGenerator):
    """
    ImageMatGenerator-style DepthAI PoE RGB + stereo generator using MJPEG.

    This version encodes RGB, left mono, and right mono on the OAK device using
    MJPEG and decodes the JPEG frames on the host with OpenCV. It intentionally
    removes the H264/H265 decoder backends and does not use PyNvVideoCodec,
    GStreamer, NVDEC, or nvivafilter.

    Output per source:
        one torch.Tensor with shape [1, 3, rgb_height, rgb_width]

    Layout:
        left stereo is packed into the top payload rows.
        right stereo is packed into the bottom payload rows.
        rows between those payload regions remain RGB.
    """

    color_types: List['ColorType'] = []

    capture_fps: float = 15.0

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800

    rgb_camera_socket: Literal["CAM_A", "RGB"] = "CAM_A"
    left_camera_socket: Literal["CAM_B", "LEFT"] = "CAM_B"
    right_camera_socket: Literal["CAM_C", "RIGHT"] = "CAM_C"

    # Compatibility-only fields. This refactor always uses MJPEG regardless of
    # old h264/h265 kwargs that callers may still pass.
    codec: Literal["mjpeg"] = "mjpeg"
    rgb_codec: Literal["mjpeg"] = "mjpeg"
    stereo_codec: Literal["mjpeg"] = "mjpeg"
    bitrate_kbps: int = 0
    rgb_bitrate_kbps: int = 0
    stereo_bitrate_kbps: int = 0

    rgb_mjpeg_quality: int = 90
    stereo_mjpeg_quality: int = 85
    mjpeg_quality: int = 90

    # DepthAI VideoEncoder MJPEG usually accepts NV12. Stereo defaults to NV12
    # for compatibility; OpenCV decodes it as grayscale on the host.
    rgb_mjpeg_input_type: Literal["NV12", "BGR888p", "RGB888p", "YUV420p"] = "NV12"
    stereo_mjpeg_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"

    # Kept as aliases for old configs; __init__ maps stereo_encoder_input_type
    # into stereo_mjpeg_input_type when provided.
    stereo_encoder_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"

    rgb_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"
    stereo_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"

    gpu_id: int = 0
    torch_device: Optional[str] = None
    non_blocking_gpu_copy: bool = True

    # "opencv" keeps the current stable CPU decode path.
    # "torchvision-cuda" uses torchvision.io.decode_jpeg(..., device="cuda") / nvJPEG.
    mjpeg_decode_backend: Literal["opencv", "torchvision-cuda"] = "opencv"
    mjpeg_decode_fallback_to_opencv: bool = True

    normalize_rgb: bool = True
    normalize_stereo: bool = True
    strict_rgb_shape: bool = True
    strict_stereo_shape: bool = True

    packed_stereo_pad_value: float = 0.0
    clear_unused_payload_tail: bool = False

    rgb_depthai_queue_size: int = 8
    stereo_depthai_queue_size: int = 8

    stereo_startup_timeout_sec: float = 2.0
    allow_missing_stereo: bool = False

    log_fps: bool = True

    show_rgb_preview: bool = False
    show_stereo_preview: bool = False
    preview_stride: int = 10
    rgb_window_name: str = "DepthAI RGB MJPEG small preview"
    stereo_window_name: str = "DepthAI MJPEG stereo small preview - left | right"

    # Let the camera/encoder control FPS.
    fps: int = 0

    def __init__(self, *args, **kwargs):
        # Keep old configuration code working while forcing this class to MJPEG.
        for key in ("codec", "rgb_codec", "stereo_codec"):
            if key in kwargs and str(kwargs[key]).lower() not in ("mjpeg", "jpeg", "jpg"):
                logger(f"[DepthAIPoeRGBStereoMjpegTorchGenerator:warning] Ignoring {key}={kwargs[key]!r}; MJPEG is always used.", level="warning")
            kwargs[key] = "mjpeg"

        if "mjpeg_quality" in kwargs:
            kwargs.setdefault("rgb_mjpeg_quality", kwargs["mjpeg_quality"])
            kwargs.setdefault("stereo_mjpeg_quality", kwargs["mjpeg_quality"])
        if "stereo_encoder_input_type" in kwargs and "stereo_mjpeg_input_type" not in kwargs:
            kwargs["stereo_mjpeg_input_type"] = kwargs["stereo_encoder_input_type"]

        super().__init__(*args, **kwargs)

    @property
    def packed_stereo_rows_per_side(self) -> int:
        stereo_values = int(self.stereo_height) * int(self.stereo_width)
        values_per_row = 3 * int(self.rgb_width)
        return (stereo_values + values_per_row - 1) // values_per_row

    @property
    def packed_stereo_rows(self) -> int:
        return 2 * int(self.packed_stereo_rows_per_side)

    @property
    def stereo_payload_start_row(self) -> int:
        """Start row of the bottom/right stereo payload."""
        return int(self.rgb_height) - int(self.packed_stereo_rows_per_side)

    @property
    def rgb_valid_start_row(self) -> int:
        """First RGB row after the top/left stereo payload."""
        return int(self.packed_stereo_rows_per_side)

    @property
    def rgb_valid_height(self) -> int:
        return int(self.rgb_height) - int(self.packed_stereo_rows)

    @property
    def packed_height(self) -> int:
        return int(self.rgb_height)

    def unpack_packed_tensor(self, packed: torch.Tensor):
        """
        Returns (rgb_with_payload, stereo, left, right).

        Top rows contain the left stereo payload.
        Bottom rows contain the right stereo payload.

        Payload order inside each region is:
            channel 0 rows, channel 1 rows, channel 2 rows
        """

        if packed.ndim != 4 or packed.shape[1] != 3:
            raise ValueError(f"Expected tensor [B, 3, H, W], got {tuple(packed.shape)}.")

        b, _, rgb_h, rgb_w = packed.shape

        stereo_h = int(self.stereo_height)
        stereo_w = int(self.stereo_width)

        stereo_one_values = stereo_h * stereo_w
        values_per_payload_row = 3 * int(rgb_w)
        payload_rows_per_stereo = (stereo_one_values + values_per_payload_row - 1) // values_per_payload_row
        total_payload_rows = 2 * payload_rows_per_stereo

        if total_payload_rows > int(rgb_h):
            raise ValueError(
                "Packed stereo payload cannot fit in top + bottom rows: "
                f"payload_rows_per_stereo={payload_rows_per_stereo}, "
                f"total_payload_rows={total_payload_rows}, rgb_height={rgb_h}."
            )

        top_payload = packed[:, :, :payload_rows_per_stereo, :]
        bottom_payload = packed[:, :, rgb_h - payload_rows_per_stereo:, :]

        def payload_region_to_flat(region: torch.Tensor) -> torch.Tensor:
            return torch.cat([region[:, c, :, :].reshape(b, -1) for c in range(3)], dim=1)

        left_flat = payload_region_to_flat(top_payload)[:, :stereo_one_values]
        right_flat = payload_region_to_flat(bottom_payload)[:, :stereo_one_values]

        left = left_flat.reshape(b, stereo_h, stereo_w)
        right = right_flat.reshape(b, stereo_h, stereo_w)
        stereo = torch.stack([left, right], dim=1)

        return packed, stereo, left, right

    def rgb_without_stereo_payload(self, packed: torch.Tensor) -> torch.Tensor:
        """Return only the contiguous RGB rows between the top and bottom stereo payload regions."""
        return packed[:, :, self.rgb_valid_start_row:self.stereo_payload_start_row, :]

    def _tensor_color_type(self):
        for name in ("RGBP", "RGB_CHW", "RGB", "BGR"):
            if hasattr(ColorType, name):
                return getattr(ColorType, name)
        return ColorType.BGR

    def on_rgb_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def on_rgb_stereo_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def create_frame_generator(self, idx, source):
        tensor_color_type = self._tensor_color_type()
        if idx >= len(self.color_types):
            self.color_types.append(tensor_color_type)
        else:
            self.color_types[idx] = tensor_color_type

        capture = self.register_resource(
            _DepthAIPoeRGBStereoMjpegBottomTorchTensorCapture(
                owner=self,
                source=source,
                idx=idx,
            )
        )

        def gen(capture=capture):
            while True:
                try:
                    yield capture.next_frame()
                except StopIteration:
                    return
                except Exception:
                    if capture.stop_event.is_set() or capture._released:
                        return
                    logger(f"[{self.uuid}:warning] DepthAI RGB + MJPEG stereo generator failed:", level="warning")
                    traceback.print_exc()
                    raise

        return gen()


# Backward-friendly alias for code that expects the old generic class name after
# importing this new file explicitly.
DepthAIPoeRGBStereoTorchGenerator = DepthAIPoeRGBStereoMjpegTorchGenerator


def to_small_cv(mat, s=10, rgb_to_bgr=True):
    """
    Accepts torch tensor in:
      CHW RGB: [3, H, W]
      HWC RGB: [H, W, 3]
      Gray:    [H, W]

    Returns uint8 CPU numpy image for cv2.imshow().
    """
    x = mat.detach()

    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    x = x[::s, ::s]

    if x.dtype.is_floating_point:
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = x.clamp(0, 255).to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    arr = x.cpu().numpy()

    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr


def _get_gen():
    return DepthAIPoeRGBStereoMjpegTorchGenerator(
        uuid="OkadCam:CamA",
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_mjpeg_quality=90,
        stereo_mjpeg_quality=85,
        rgb_mjpeg_input_type="NV12",
        stereo_mjpeg_input_type="NV12",
        # Use "torchvision-cuda" to decode MJPEG with nvJPEG into CUDA tensors.
        mjpeg_decode_backend="torchvision-cuda",#"opencv",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )


def test_rgb_stereo():
    """Small shape/unpack smoke test. This prints every tenth frame and is not a benchmark."""

    gen = _get_gen()

    try:
        for i, mats in enumerate(gen):
            packed = mats[0].data
            rgb, stereo, left, right = gen.unpack_packed_tensor(packed)

            rgb_valid = gen.rgb_without_stereo_payload(rgb)

            small_rgb = to_small_cv(rgb_valid[0], s=10, rgb_to_bgr=True)
            small_left = to_small_cv(left[0], s=4, rgb_to_bgr=False)
            small_right = to_small_cv(right[0], s=4, rgb_to_bgr=False)

            cv2.imshow("small_rgb", small_rgb)
            cv2.imshow("small_left", small_left)
            cv2.imshow("small_right", small_right)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if i % 10 == 0:
                print(
                    "packed", tuple(packed.shape), packed.device, packed.dtype,
                    "rgb", tuple(rgb.shape),
                    "rgb_valid", tuple(rgb_valid.shape),
                    "stereo", tuple(stereo.shape),
                    "left", tuple(left.shape),
                    "right", tuple(right.shape),
                )

    finally:
        gen.release()
        cv2.destroyAllWindows()


def benchmark_rgb_stereo(duration_sec: float = 10.0, warmup_sec: float = 2.0):
    """No per-frame printing/unpacking benchmark."""

    gen = _get_gen()
    total = 0
    measured = 0
    first_shape = None
    first_dtype = None
    first_device = None
    t0 = time.monotonic()
    measure_t0 = None

    try:
        for mats in gen:
            packed = mats[0].data
            total += 1
            if first_shape is None:
                first_shape = tuple(packed.shape)
                first_dtype = packed.dtype
                first_device = packed.device
            now = time.monotonic()
            if measure_t0 is None and now - t0 >= warmup_sec:
                measure_t0 = now
                measured = 0
            if measure_t0 is not None:
                measured += 1
                if now - measure_t0 >= duration_sec:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = max(time.monotonic() - (measure_t0 or t0), 1e-6)
        fps = measured / elapsed if measure_t0 is not None else total / max(time.monotonic() - t0, 1e-6)
        print(
            f"benchmark frames={measured if measure_t0 is not None else total}, "
            f"fps={fps:.2f}, shape={first_shape}, dtype={first_dtype}, device={first_device}"
        )
    finally:
        gen.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    benchmark_rgb_stereo()
