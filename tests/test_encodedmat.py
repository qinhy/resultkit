"""
Test one H264 access-unit packet per Model4Mat.EncodedImageMat using FFmpeg.

Concept:
    one decoded frame
    -> one H264 Annex B access unit bytes
    -> one EncodedImageMat
    -> one decoded ImageMat

This is NOT an MP4-container test.
This is a streaming-packet test.

Run directly:
    python test_ffmpeg_h264_encoded_image_packet.py
    python test_ffmpeg_h264_encoded_image_packet.py --ffmpeg-bin /path/to/ffmpeg

You can also set:
    FFMPEG_BIN=/path/to/ffmpeg python test_ffmpeg_h264_encoded_image_packet.py

Run with pytest:
    pytest -q test_ffmpeg_h264_encoded_image_packet.py

Requirements:
    - ffmpeg on PATH
    - numpy
    - opencv-python
    - your resultkit package importable
"""

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional, Union

import cv2
import numpy as np


# Adjust this exactly like your current test file.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def set_ffmpeg_bin(ffmpeg_bin: str):
    """
    Set the FFmpeg executable used by this test module.

    Examples:
        set_ffmpeg_bin("ffmpeg")
        set_ffmpeg_bin("/usr/local/bin/ffmpeg")
        set_ffmpeg_bin(r"C:\\ffmpeg\\bin\\ffmpeg.exe")
    """
    global FFMPEG_BIN
    if not ffmpeg_bin:
        raise ValueError("ffmpeg_bin must be a non-empty string")
    FFMPEG_BIN = ffmpeg_bin


def _pytest_skip_or_raise(message: str):
    """
    Skip when running under pytest; raise otherwise.
    """
    try:
        import pytest  # type: ignore

        pytest.skip(message)
    except ImportError:
        raise RuntimeError(message)


def require_ffmpeg():
    resolved = shutil.which(FFMPEG_BIN)
    if resolved is None:
        _pytest_skip_or_raise(
            f"ffmpeg executable was not found: {FFMPEG_BIN!r}. "
            "Pass --ffmpeg-bin /path/to/ffmpeg or set FFMPEG_BIN."
        )
    return resolved


def run_ffmpeg(cmd: list[str], input_bytes: bytes) -> bytes:
    """
    Run FFmpeg with stdin bytes and return stdout bytes.
    """
    ffmpeg_bin = require_ffmpeg()
    cmd = [ffmpeg_bin, *cmd[1:]]

    proc = subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n\n"
            f"CMD:\n{' '.join(cmd)}\n\n"
            f"STDERR:\n{proc.stderr.decode(errors='replace')}"
        )

    return proc.stdout


def make_frame(i: int, h: int = 64, w: int = 96) -> np.ndarray:
    """
    Make one deterministic BGR test frame with shape HWC.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frame[:, :, 1] = int(i * 20) % 255
    frame[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    return frame


def _as_bgr24_frame(
    frame: np.ndarray,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> np.ndarray:
    """
    Convert input frame to contiguous uint8 BGR HWC for FFmpeg rawvideo.
    """
    color_format = ColorFormat(color_format)

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC-3 frame, got shape {arr.shape}")

    if color_format == ColorFormat.BGR:
        return np.ascontiguousarray(arr)

    if color_format == ColorFormat.RGB:
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    raise ValueError(f"H264 test expects RGB or BGR input, got {color_format}")


def encode_one_h264_access_unit_ffmpeg(
    frame: np.ndarray,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    fps: int = 30,
    crf: int = 18,
) -> bytes:
    """
    Encode one decoded frame into one H264 Annex B access unit.

    Returned bytes are suitable for one Model4Mat.EncodedImageMat.

    Important:
    - This packet may contain multiple NAL units.
    - SPS/PPS headers are repeated so this packet is independently decodable.
    - Every frame is forced to be a keyframe for simple packet-level testing.
    """
    require_ffmpeg()

    bgr = _as_bgr24_frame(frame, color_format=color_format)
    h, w = bgr.shape[:2]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",

        # Input raw BGR frame from stdin.
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",

        # Encode exactly one frame.
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-crf",
        str(crf),

        # keyint=1 makes every encoded frame an IDR/keyframe.
        # repeat-headers=1 puts SPS/PPS headers into the output packet.
        "-x264-params",
        "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",

        # Broad H264 compatibility.
        "-pix_fmt",
        "yuv420p",

        # Raw H264 Annex B bytestream, not MP4.
        "-f",
        "h264",
        "pipe:1",
    ]

    payload = run_ffmpeg(cmd, bgr.tobytes())

    if not payload:
        raise RuntimeError("FFmpeg returned an empty H264 payload")

    return payload


def encoded_image_from_ffmpeg_h264_frame(
    frame: np.ndarray,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    frame_index: int = -1,
    pts_ns: Optional[int] = None,
    fps: int = 30,
    crf: int = 18,
) -> "Model4Mat.EncodedImageMat":
    """
    Build one EncodedImageMat from one decoded frame.

    EncodedImageMat.data contains one H264 Annex B access unit.
    """
    color_format = ColorFormat(color_format)

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC-3 frame, got shape {arr.shape}")

    h, w = arr.shape[:2]

    payload = encode_one_h264_access_unit_ffmpeg(
        arr,
        color_format=color_format,
        fps=fps,
        crf=crf,
    )

    return Model4Mat.EncodedImageMat(
        codec=CodecFormat.H264,
        color_format=color_format,
        frame_index=frame_index,
        pts_ns=pts_ns,
        dts_ns=pts_ns,
        is_keyframe=True,
        width=int(w),
        height=int(h),
        valid_nbytes=len(payload),
        data=np.frombuffer(payload, dtype=np.uint8).copy(),
    )


def _packet_payload_bytes(pkt: "Model4Mat.EncodedImageMat") -> bytes:
    """
    Read the valid encoded payload from EncodedImageMat.

    Supports either:
        pkt.payload()
    or:
        pkt.data[:pkt.valid_nbytes]
    """
    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        if isinstance(payload, np.ndarray):
            return payload.tobytes()
        return bytes(payload)

    valid_nbytes = int(getattr(pkt, "valid_nbytes", len(pkt.data)))
    return np.asarray(pkt.data).reshape(-1)[:valid_nbytes].tobytes()


def _packet_nbytes(pkt: "Model4Mat.EncodedImageMat") -> int:
    if hasattr(pkt, "nbytes") and callable(pkt.nbytes):
        return int(pkt.nbytes())

    return int(getattr(pkt, "valid_nbytes", len(pkt.data)))


def decode_one_h264_access_unit_ffmpeg(
    packet_bytes: bytes,
    width: int,
    height: int,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> np.ndarray:
    """
    Decode one H264 Annex B access unit into one frame.

    Returns:
        HWC uint8 frame in requested RGB/BGR format.
    """
    require_ffmpeg()

    color_format = ColorFormat(color_format)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",

        # Input is raw H264 Annex B access unit bytes.
        "-f",
        "h264",
        "-i",
        "pipe:0",

        # Decode exactly one frame to raw BGR.
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    raw = run_ffmpeg(cmd, packet_bytes)

    expected = int(width) * int(height) * 3
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} decoded bytes, got {len(raw)}")

    bgr = np.frombuffer(raw, dtype=np.uint8).reshape(int(height), int(width), 3).copy()

    if color_format == ColorFormat.BGR:
        return bgr

    if color_format == ColorFormat.RGB:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unsupported output color_format: {color_format}")


def decode_ffmpeg_h264_packet_to_image_mat(
    pkt: "Model4Mat.EncodedImageMat",
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> "Model4Mat.ImageMat":
    """
    Decode one EncodedImageMat H264 packet into one ImageMat.
    """
    if CodecFormat(pkt.codec) != CodecFormat.H264:
        raise ValueError(f"Expected H264 packet, got {pkt.codec}")

    color_format = ColorFormat(color_format)

    frame = decode_one_h264_access_unit_ffmpeg(
        _packet_payload_bytes(pkt),
        width=int(pkt.width),
        height=int(pkt.height),
        color_format=color_format,
    )

    return Model4Mat.ImageMat(
        color_format=color_format,
        data=frame,
    )


def test_streaming_encoded_frames_ffmpeg_h264_packets():
    """
    Main test:
        one source frame -> one H264 EncodedImageMat packet -> one decoded ImageMat
    """
    packets = []
    decoded_frames = []
    src_frames = []

    for i in range(10):
        frame = make_frame(i)
        src_frames.append(frame)

        pkt = encoded_image_from_ffmpeg_h264_frame(
            frame,
            color_format=ColorFormat.BGR,
            frame_index=i,
            pts_ns=i * 33_333_333,
            fps=30,
            crf=18,
        )

        packets.append(pkt)

    for pkt in packets:
        img = decode_ffmpeg_h264_packet_to_image_mat(
            pkt,
            color_format=ColorFormat.BGR,
        )
        decoded_frames.append(img.data)

    src_frames = np.stack(src_frames, axis=0)
    decoded_frames = np.stack(decoded_frames, axis=0)

    assert len(packets) == 10
    assert all(_packet_nbytes(pkt) > 0 for pkt in packets)

    assert packets[3].frame_index == 3
    assert packets[3].pts_ns == 3 * 33_333_333
    assert CodecFormat(packets[3].codec) == CodecFormat.H264
    assert packets[3].is_keyframe is True

    assert decoded_frames.shape == src_frames.shape
    assert decoded_frames.dtype == np.uint8

    # H264 yuv420p is lossy. Do not require exact equality.
    mae = np.mean(np.abs(decoded_frames.astype(np.int16) - src_frames.astype(np.int16)))
    assert mae < 15.0, f"MAE too high: {mae}"
    print("MAE:",np.mean(np.abs(decoded_frames.astype(np.int16) - src_frames.astype(np.int16))))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test one H264 Annex B access-unit packet per "
            "Model4Mat.EncodedImageMat using FFmpeg."
        )
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=FFMPEG_BIN,
        help=(
            "Path/name of the ffmpeg executable. Defaults to the FFMPEG_BIN "
            "environment variable, or 'ffmpeg' on PATH."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    set_ffmpeg_bin(args.ffmpeg_bin)

    test_streaming_encoded_frames_ffmpeg_h264_packets()
    print(f"OK: one H264 access unit per EncodedImageMat packet using {require_ffmpeg()}")


if __name__ == "__main__":
    main()