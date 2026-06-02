"""
Clean H264 FFmpeg pub/sub demo for resultkit.Model4Mat.EncodedImageMatPubSub.

Streaming model:
    one decoded frame
    -> one H264 Annex B access-unit payload
    -> one EncodedImageMatPubSub packet
    -> pub/sub transport
    -> one decoded ImageMat

Run publisher:
    python gen_codec_vis_basic.py pub

Run subscriber:
    python gen_codec_vis_basic.py sub

Legacy compatible subscriber flag:
    python gen_codec_vis_basic.py --sub

Custom FFmpeg binary:
    python gen_codec_vis_basic.py pub --ffmpeg-bin /path/to/ffmpeg
    FFMPEG_BIN=/path/to/ffmpeg python gen_codec_vis_basic.py sub
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


# Adjust this exactly like your current test file if needed.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


DEFAULT_TOPIC = "EncodedImageMatPubSub:test"
DEFAULT_FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
NS_PER_SECOND = 1_000_000_000


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is missing or returns a non-zero exit code."""


@dataclass(frozen=True)
class StreamConfig:
    topic: str = DEFAULT_TOPIC
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    width: int = 96
    height: int = 64
    fps: int = 30
    crf: int = 18
    color_format: ColorFormat = ColorFormat.BGR
    stats_every: int = 100
    max_frames: Optional[int] = None
    capacity_scale: float = 2.0
    min_capacity_bytes: int = 64 * 1024
    display: bool = True
    window_name: str = "resultkit h264 pub/sub"
    wait_ms: Optional[int] = None
    poll_sleep_s: float = 0.001

    @property
    def frame_period_ns(self) -> int:
        return int(NS_PER_SECOND / max(self.fps, 1))

    @property
    def cv_wait_ms(self) -> int:
        if self.wait_ms is not None:
            return max(1, int(self.wait_ms))
        return max(1, int(1000 / max(self.fps, 1)))


class FFmpegRunner:
    def __init__(self, ffmpeg_bin: str = DEFAULT_FFMPEG_BIN):
        if not ffmpeg_bin:
            raise ValueError("ffmpeg_bin must be a non-empty string")
        self.ffmpeg_bin = ffmpeg_bin

    def resolve(self) -> str:
        resolved = shutil.which(self.ffmpeg_bin)
        if resolved is None:
            raise FFmpegError(
                f"ffmpeg executable was not found: {self.ffmpeg_bin!r}. "
                "Pass --ffmpeg-bin /path/to/ffmpeg or set FFMPEG_BIN."
            )
        return resolved

    def run(self, args: list[str], input_bytes: bytes) -> bytes:
        cmd = [self.resolve(), *args]
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")
            raise FFmpegError(
                "ffmpeg failed\n\n"
                f"CMD:\n{' '.join(cmd)}\n\n"
                f"STDERR:\n{stderr}"
            )

        return proc.stdout


class H264AnnexBCodec:
    """Stateless FFmpeg helper for one-keyframe-per-packet H264 testing."""

    def __init__(self, runner: FFmpegRunner, fps: int = 30, crf: int = 18):
        self.runner = runner
        self.fps = int(fps)
        self.crf = int(crf)

    @staticmethod
    def as_bgr24_frame(
        frame: np.ndarray,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> np.ndarray:
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

        raise ValueError(f"H264 pub/sub demo expects RGB or BGR input, got {color_format}")

    def encode_access_unit(
        self,
        frame: np.ndarray,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> bytes:
        """
        Encode one decoded frame into one H264 Annex B access unit.

        The returned payload may contain multiple NAL units. For this demo,
        every payload is independently decodable because libx264 is configured
        to emit an IDR/keyframe and repeat SPS/PPS headers for every frame.
        """
        bgr = self.as_bgr24_frame(frame, color_format=color_format)
        h, w = bgr.shape[:2]

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
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
            str(self.crf),
            "-x264-params",
            "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
            "pipe:1",
        ]

        payload = self.runner.run(args, bgr.tobytes())
        if not payload:
            raise FFmpegError("FFmpeg returned an empty H264 payload")
        return payload

    def decode_access_unit(
        self,
        packet_bytes: bytes,
        width: int,
        height: int,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> np.ndarray:
        """Decode one H264 Annex B access unit into one HWC uint8 frame."""
        color_format = ColorFormat(color_format)

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        raw = self.runner.run(args, packet_bytes)
        expected = int(width) * int(height) * 3
        if len(raw) != expected:
            raise FFmpegError(f"Expected {expected} decoded bytes, got {len(raw)}")

        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(int(height), int(width), 3).copy()

        if color_format == ColorFormat.BGR:
            return bgr
        if color_format == ColorFormat.RGB:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise ValueError(f"Unsupported output color_format: {color_format}")


class FpsMeter:
    def __init__(self, label: str, stats_every: int = 100):
        self.label = label
        self.stats_every = max(1, int(stats_every))
        self.start_t = time.perf_counter()
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += int(n)
        if self.count % self.stats_every != 0:
            return

        elapsed = time.perf_counter() - self.start_t
        if elapsed <= 0:
            return

        print(f"{self.label} FPS: {self.count / elapsed:.2f}")


def make_frame(i: int, h: int = 64, w: int = 96) -> np.ndarray:
    """Make one deterministic BGR test frame with shape HWC."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frame[:, :, 1] = int(i * 20) % 255
    frame[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    return frame


def payload_as_array(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.uint8).copy()


def packet_payload_bytes(pkt: "Model4Mat.EncodedImageMat") -> bytes:
    """Read only the valid encoded payload bytes from a packet model."""
    if hasattr(pkt, "payload") and callable(pkt.payload):
        payload = pkt.payload()
        if isinstance(payload, np.ndarray):
            return payload.tobytes()
        return bytes(payload)

    valid_nbytes = int(getattr(pkt, "valid_nbytes", len(pkt.data)))
    return np.asarray(pkt.data).reshape(-1)[:valid_nbytes].tobytes()


def packet_nbytes(pkt: "Model4Mat.EncodedImageMat") -> int:
    if hasattr(pkt, "nbytes") and callable(pkt.nbytes):
        return int(pkt.nbytes())
    return int(getattr(pkt, "valid_nbytes", len(pkt.data)))


def initial_pubsub_buffer(payload: bytes, config: StreamConfig) -> np.ndarray:
    """
    Create the initial buffer used to size the pub/sub service.

    Your working version used payload * 2. This keeps that idea, but makes the
    capacity explicit and easier to tune with --capacity-scale and
    --min-capacity-bytes.
    """
    payload_arr = payload_as_array(payload)
    capacity = max(
        int(np.ceil(payload_arr.size * max(config.capacity_scale, 1.0))),
        int(config.min_capacity_bytes),
    )
    data = np.zeros((capacity,), dtype=np.uint8)
    data[: payload_arr.size] = payload_arr
    return data


def update_packet_metadata(
    pkt: "Model4Mat.EncodedImageMatPubSub",
    *,
    frame_index: int,
    pts_ns: int,
    width: int,
    height: int,
    valid_nbytes: int,
) -> None:
    pkt.codec = CodecFormat.H264
    pkt.frame_index = int(frame_index)
    pkt.pts_ns = int(pts_ns)
    pkt.dts_ns = int(pts_ns)
    pkt.is_keyframe = True
    pkt.width = int(width)
    pkt.height = int(height)
    pkt.valid_nbytes = int(valid_nbytes)


def make_seed_packet(
    codec: H264AnnexBCodec,
    config: StreamConfig,
) -> "Model4Mat.EncodedImageMatPubSub":
    """Create an initialized packet model with enough buffer capacity."""
    frame = make_frame(0, h=config.height, w=config.width)
    payload = codec.encode_access_unit(frame, color_format=config.color_format)
    data = initial_pubsub_buffer(payload, config)

    return Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.H264,
        color_format=config.color_format,
        frame_index=0,
        pts_ns=0,
        dts_ns=0,
        is_keyframe=True,
        width=int(config.width),
        height=int(config.height),
        valid_nbytes=len(payload),
        data=data,
    )


def make_endpoint(
    codec: H264AnnexBCodec,
    config: StreamConfig,
    *,
    is_pub: bool,
) -> "Model4Mat.EncodedImageMatPubSub":
    endpoint = make_seed_packet(codec, config)
    endpoint.set_id(config.topic).init()
    endpoint.is_pub = bool(is_pub)
    return endpoint


def publish_frame(
    publisher: "Model4Mat.EncodedImageMatPubSub",
    codec: H264AnnexBCodec,
    config: StreamConfig,
    frame_index: int,
) -> int:
    frame = make_frame(frame_index, h=config.height, w=config.width)
    payload = codec.encode_access_unit(frame, color_format=config.color_format)
    pts_ns = frame_index * config.frame_period_ns

    update_packet_metadata(
        publisher,
        frame_index=frame_index,
        pts_ns=pts_ns,
        width=config.width,
        height=config.height,
        valid_nbytes=len(payload),
    )

    # Keep the actual published unit as one encoded access-unit payload.
    # This matches your working code and your streaming model.
    publisher.pub(data=payload_as_array(payload))
    return len(payload)


def decode_packet_to_image_mat(
    pkt: "Model4Mat.EncodedImageMat",
    codec: H264AnnexBCodec,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> "Model4Mat.ImageMat":
    if CodecFormat(pkt.codec) != CodecFormat.H264:
        raise ValueError(f"Expected H264 packet, got {pkt.codec}")

    frame = codec.decode_access_unit(
        packet_payload_bytes(pkt),
        width=int(pkt.width),
        height=int(pkt.height),
        color_format=color_format,
    )

    return Model4Mat.ImageMat(
        color_format=ColorFormat(color_format),
        data=frame,
    )


def publisher_loop(config: StreamConfig) -> None:
    runner = FFmpegRunner(config.ffmpeg_bin)
    codec = H264AnnexBCodec(runner, fps=config.fps, crf=config.crf)
    publisher = make_endpoint(codec, config, is_pub=True)
    meter = FpsMeter("Pub", stats_every=config.stats_every)

    print(
        f"Publishing H264 packets to topic={config.topic!r}, "
        f"size={config.width}x{config.height}, fps={config.fps}, crf={config.crf}"
    )

    frame_index = 0
    while config.max_frames is None or frame_index < config.max_frames:
        frame_index += 1
        publish_frame(publisher, codec, config, frame_index)
        meter.tick()


def subscriber_loop(config: StreamConfig) -> None:
    runner = FFmpegRunner(config.ffmpeg_bin)
    codec = H264AnnexBCodec(runner, fps=config.fps, crf=config.crf)
    subscriber = make_endpoint(codec, config, is_pub=False)
    meter = FpsMeter("Sub", stats_every=config.stats_every)

    print(f"Subscribing H264 packets from topic={config.topic!r}")

    frame_count = 0
    try:
        while config.max_frames is None or frame_count < config.max_frames:
            pkt = subscriber.sub()

            if packet_nbytes(pkt) <= 0:
                time.sleep(config.poll_sleep_s)
                continue

            img = decode_packet_to_image_mat(pkt, codec, color_format=config.color_format)
            frame_count += 1
            meter.tick()

            if config.display:
                cv2.imshow(config.window_name, img.get_data())
                if cv2.waitKey(config.cv_wait_ms) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(config.poll_sleep_s)
    finally:
        if config.display:
            cv2.destroyAllWindows()


def parse_color_format(value: str) -> ColorFormat:
    try:
        return ColorFormat(value)
    except ValueError:
        allowed = ", ".join(v.value for v in ColorFormat)
        raise argparse.ArgumentTypeError(f"invalid color format {value!r}; expected one of: {allowed}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or subscribe one H264 Annex B access-unit per EncodedImageMatPubSub."
    )

    parser.add_argument(
        "role",
        nargs="?",
        choices=("pub", "sub"),
        help="Run as publisher or subscriber. Default is pub unless --sub is used.",
    )
    parser.add_argument("--sub", action="store_true", help="Legacy alias for role=sub.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ffmpeg-bin", default=DEFAULT_FFMPEG_BIN)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--color-format", type=parse_color_format, default=ColorFormat.BGR)
    parser.add_argument("--stats-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--capacity-scale", type=float, default=2.0)
    parser.add_argument("--min-capacity-bytes", type=int, default=64 * 1024)
    parser.add_argument("--wait-ms", type=int, default=None)
    parser.add_argument("--poll-sleep-s", type=float, default=0.001)

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true", default=True)
    display_group.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--window-name", default="resultkit h264 pub/sub")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> tuple[str, StreamConfig]:
    role = "sub" if args.sub else (args.role or "pub")
    config = StreamConfig(
        topic=args.topic,
        ffmpeg_bin=args.ffmpeg_bin,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        color_format=args.color_format,
        stats_every=args.stats_every,
        max_frames=args.max_frames,
        capacity_scale=args.capacity_scale,
        min_capacity_bytes=args.min_capacity_bytes,
        display=args.display,
        window_name=args.window_name,
        wait_ms=args.wait_ms,
        poll_sleep_s=args.poll_sleep_s,
    )
    return role, config


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    role, config = config_from_args(args)

    try:
        if role == "sub":
            subscriber_loop(config)
        else:
            publisher_loop(config)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
