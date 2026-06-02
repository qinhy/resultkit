
import os
import sys
from typing import Optional, Union

import cv2
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat, MatLib, DataType, MatStore


def _opencv_ext(codec) -> str:
    codec = CodecFormat(codec)

    if codec in {
        CodecFormat.JPEG,
        CodecFormat.MJPEG,
    }:
        return ".jpg"

    if codec == CodecFormat.PNG:
        return ".png"

    raise NotImplementedError(
        f"cv2.imencode/imdecode cannot handle standalone {codec.value} video packets. "
        "Use a stateful decoder backend for H264/H265/HEVC/AV1."
    )

def from_opencv_frame(
    frame: np.ndarray,
    codec: Union[str, CodecFormat] = CodecFormat.JPEG,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    frame_index: int = -1,
    pts_ns: Optional[int] = None,
    encode_params: Optional[list[int]] = None,
):
    """
    Encode one decoded frame into one EncodedImageMat payload.

    This is correct for JPEG/PNG/MJPEG-style streaming frames.
    It is not H264/H265 inter-frame video compression.
    """
    codec = CodecFormat(codec)
    color_format = ColorFormat(color_format)

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        height, width = arr.shape
        encode_arr = arr

    elif arr.ndim == 3 and arr.shape[-1] == 3:
        height, width = arr.shape[:2]

        if color_format == ColorFormat.RGB:
            encode_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif color_format == ColorFormat.BGR:
            encode_arr = arr
        else:
            raise ValueError(f"Unsupported color_format for 3-channel frame: {color_format}")

    else:
        raise ValueError(f"Expected HW or HWC-3 frame, got {arr.shape}")

    ext = _opencv_ext(codec)
    ok, buf = cv2.imencode(ext, encode_arr, encode_params or [])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for codec={codec.value}")

    return Model4Mat.EncodedImageMat(
        codec=codec,
        color_format=color_format,
        frame_index=frame_index,
        pts_ns=pts_ns,
        is_keyframe=True,
        width=int(width),
        height=int(height),
        valid_nbytes=int(buf.size),
        data=np.asarray(buf, dtype=np.uint8).reshape(-1),
    )

def decode_opencv_frame(
    img,
    color_format: Optional[Union[str, ColorFormat]] = None,
) -> "Model4Mat.ImageMat":
    """
    Decode one encoded frame payload with OpenCV.

    Works for JPEG/PNG/MJPEG payloads.
    Does not work for standalone H264/H265/HEVC/AV1 packets.
    """
    _opencv_ext(img.codec)

    target_color = (
        ColorFormat(color_format)
        if color_format is not None
        else ColorFormat(img.color_format)
    )

    decoded_bgr = cv2.imdecode(img.payload(), cv2.IMREAD_UNCHANGED)
    if decoded_bgr is None:
        raise RuntimeError(f"cv2.imdecode failed for codec={img.codec.value}")

    if decoded_bgr.ndim == 2:
        arr = decoded_bgr
        out_color = ColorFormat.GRAY

    elif target_color == ColorFormat.BGR:
        arr = decoded_bgr
        out_color = ColorFormat.BGR

    elif target_color == ColorFormat.RGB:
        arr = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        out_color = ColorFormat.RGB

    elif target_color in {
        ColorFormat.GRAY,
        ColorFormat.BAYER,
    }:
        arr = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2GRAY)
        out_color = target_color

    else:
        raise ValueError(f"Unsupported target color format: {target_color}")

    return Model4Mat.ImageMat(
        color_format=out_color,
        data=arr,
    )


def make_frame(i: int, h=64, w=96) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frame[:, :, 1] = int(i * 20) % 255
    frame[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    return frame


def test_streaming_encoded_frames_opencv_mjpeg():
    packets = []

    for i in range(10):
        frame = make_frame(i)

        pkt = from_opencv_frame(
            frame,
            codec=CodecFormat.MJPEG,
            color_format=ColorFormat.BGR,
            frame_index=i,
            pts_ns=i * 33_333_333,
        )

        packets.append(pkt)

    decoded = []

    for pkt in packets:
        img = decode_opencv_frame(
            pkt,
            color_format=ColorFormat.BGR,
        )
        decoded.append(img)

    assert len(decoded) == 10
    assert all(pkt.nbytes() > 0 for pkt in packets)
    assert decoded[0].data.shape == make_frame(0).shape
    assert packets[3].frame_index == 3
    assert packets[3].pts_ns == 3 * 33_333_333


test_streaming_encoded_frames_opencv_mjpeg()