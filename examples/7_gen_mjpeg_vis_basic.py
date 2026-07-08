"""
Clean MJPEG pub/sub demo for resultkit.Model4Mat.EncodedImageMatPubSub.

Streaming model:
    one decoded BGR/RGB frame
    -> one JPEG-compressed MJPEG frame payload
    -> one EncodedImageMatPubSub packet
    -> pub/sub transport
    -> one decoded ImageMat

MJPEG note:
    In this demo, each MJPEG packet is represented as one standalone JPEG image.
    That matches resultkit.Model4Mat.EncodedImageMat's note that MJPEG frames can
    be decoded directly with OpenCV, just like JPEG bytes.

Run publisher:
    python 7_gen_mjpeg_vis_basic.py pub

Run subscriber:
    python 7_gen_mjpeg_vis_basic.py sub

Legacy compatible subscriber flag:
    python 7_gen_mjpeg_vis_basic.py --sub
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


# Adjust this exactly like your current test file if needed.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import CodecFormat, ColorFormat, Model4Mat


DEFAULT_TOPIC = "EncodedImageMatPubSub:mjpeg-test"
NS_PER_SECOND = 1_000_000_000
PUBSUB_HEADER_BYTES = 8  # EncodedImageMatPubSub stores valid_nbytes in first 64 bits.


@dataclass(frozen=True)
class StreamConfig:
    topic: str = DEFAULT_TOPIC
    width: int = 256
    height: int = 256
    fps: int = 30
    jpeg_quality: int = 90
    color_format: ColorFormat = ColorFormat.BGR
    stats_every: int = 100
    max_frames: Optional[int] = None
    capacity_scale: float = 2.0
    min_capacity_bytes: int = 64 * 1024
    display: bool = True
    window_name: str = "resultkit mjpeg pub/sub"
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


class MJPEGCodec:
    """OpenCV helper for one MJPEG frame represented as JPEG bytes."""

    def __init__(self, jpeg_quality: int = 90):
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))

    @staticmethod
    def as_bgr_frame(
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

        raise ValueError(f"MJPEG pub/sub demo expects RGB or BGR input, got {color_format}")

    def encode_frame(
        self,
        frame: np.ndarray,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> bytes:
        """Encode one decoded frame into one standalone JPEG/MJPEG payload."""
        bgr = self.as_bgr_frame(frame, color_format=color_format)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok or encoded.size == 0:
            raise RuntimeError("cv2.imencode returned an empty MJPEG/JPEG payload")
        return encoded.reshape(-1).tobytes()

    def decode_frame(
        self,
        packet_bytes: bytes,
        color_format: Union[str, ColorFormat] = ColorFormat.BGR,
    ) -> np.ndarray:
        """Decode one MJPEG/JPEG payload into one HWC uint8 frame."""
        color_format = ColorFormat(color_format)
        encoded = np.frombuffer(packet_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode failed to decode MJPEG/JPEG payload")

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


def make_frame(i: int, h: int = 256, w: int = 256) -> np.ndarray:
    """Make one deterministic BGR test frame with shape HWC."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frame[:, :, 1] = int(i * 7) % 255
    frame[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]

    # Add moving shapes/text so compression and display changes are visible.
    cx = int((i * 5) % max(w, 1))
    cy = int(h / 2 + np.sin(i * 0.12) * h * 0.25)
    radius = max(8, min(h, w) // 12)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
    cv2.putText(
        frame,
        f"MJPEG {i}",
        (10, max(24, h - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
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

    EncodedImageMatPubSub stores an 8-byte length header in the shared-memory
    slice before the encoded payload. Allocate enough capacity for both.
    """
    payload_arr = payload_as_array(payload)
    capacity = max(
        int(np.ceil((payload_arr.size + PUBSUB_HEADER_BYTES) * max(config.capacity_scale, 1.0))),
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
    pkt.codec = CodecFormat.MJPEG
    pkt.frame_index = int(frame_index)
    pkt.pts_ns = int(pts_ns)
    pkt.dts_ns = int(pts_ns)
    pkt.is_keyframe = True
    pkt.width = int(width)
    pkt.height = int(height)
    pkt.valid_nbytes = int(valid_nbytes)


def make_seed_packet(codec: MJPEGCodec, config: StreamConfig) -> "Model4Mat.EncodedImageMatPubSub":
    """Create an initialized packet model with enough buffer capacity."""
    frame = make_frame(0, h=config.height, w=config.width)
    payload = codec.encode_frame(frame, color_format=config.color_format)
    data = initial_pubsub_buffer(payload, config)

    return Model4Mat.EncodedImageMatPubSub(
        codec=CodecFormat.MJPEG,
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
    codec: MJPEGCodec,
    config: StreamConfig,
    *,
    is_pub: bool,
) -> "Model4Mat.EncodedImageMatPubSub":
    endpoint = make_seed_packet(codec, config)
    endpoint.set_id(config.topic).init()
    endpoint.is_pub = bool(is_pub)

    # Important for subscribers: until a real sample arrives, do not decode the
    # seed/capacity buffer as though it were a received packet.
    if not is_pub:
        endpoint.valid_nbytes = 0

    return endpoint


def publish_frame(
    publisher: "Model4Mat.EncodedImageMatPubSub",
    codec: MJPEGCodec,
    config: StreamConfig,
    frame_index: int,
) -> int:
    frame = make_frame(frame_index, h=config.height, w=config.width)
    payload = codec.encode_frame(frame, color_format=config.color_format)
    pts_ns = frame_index * config.frame_period_ns

    if len(payload) + PUBSUB_HEADER_BYTES > publisher.data.size:
        raise RuntimeError(
            f"Encoded MJPEG payload is too large for the pub/sub buffer: "
            f"payload={len(payload)} + header={PUBSUB_HEADER_BYTES}, "
            f"capacity={publisher.data.size}. Increase --capacity-scale or --min-capacity-bytes."
        )

    update_packet_metadata(
        publisher,
        frame_index=frame_index,
        pts_ns=pts_ns,
        width=config.width,
        height=config.height,
        valid_nbytes=len(payload),
    )

    publisher.pub(data=payload_as_array(payload))
    return len(payload)


def decode_packet_to_image_mat(
    pkt: "Model4Mat.EncodedImageMat",
    codec: MJPEGCodec,
    color_format: Union[str, ColorFormat] = ColorFormat.BGR,
) -> "Model4Mat.ImageMat":
    if CodecFormat(pkt.codec) != CodecFormat.MJPEG:
        raise ValueError(f"Expected MJPEG packet, got {pkt.codec}")

    frame = codec.decode_frame(packet_payload_bytes(pkt), color_format=color_format)

    return Model4Mat.ImageMat(
        color_format=ColorFormat(color_format),
        data=frame,
    )


def publisher_loop(config: StreamConfig) -> None:
    codec = MJPEGCodec(jpeg_quality=config.jpeg_quality)
    publisher = make_endpoint(codec, config, is_pub=True)
    meter = FpsMeter("Pub", stats_every=config.stats_every)

    print(
        f"Publishing MJPEG packets to topic={config.topic!r}, "
        f"size={config.width}x{config.height}, fps={config.fps}, "
        f"jpeg_quality={config.jpeg_quality}"
    )

    frame_index = 0
    next_t = time.perf_counter()
    while config.max_frames is None or frame_index < config.max_frames:
        frame_index += 1
        publish_frame(publisher, codec, config, frame_index)
        meter.tick()

        # Keep publisher near the requested frame rate.
        next_t += 1.0 / max(config.fps, 1)
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)


def subscriber_loop(config: StreamConfig) -> None:
    codec = MJPEGCodec(jpeg_quality=config.jpeg_quality)
    subscriber = make_endpoint(codec, config, is_pub=False)
    meter = FpsMeter("Sub", stats_every=config.stats_every)

    print(f"Subscribing MJPEG packets from topic={config.topic!r}")

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
    except ValueError as exc:
        allowed = ", ".join(v.value for v in ColorFormat)
        raise argparse.ArgumentTypeError(
            f"invalid color format {value!r}; expected one of: {allowed}"
        ) from exc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or subscribe one JPEG-compressed MJPEG frame per EncodedImageMatPubSub."
    )

    parser.add_argument(
        "role",
        nargs="?",
        choices=("pub", "sub"),
        help="Run as publisher or subscriber. Default is pub unless --sub is used.",
    )
    parser.add_argument("--sub", action="store_true", help="Legacy alias for role=sub.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
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
    parser.add_argument("--window-name", default="resultkit mjpeg pub/sub")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> tuple[str, StreamConfig]:
    role = "sub" if args.sub else (args.role or "pub")
    config = StreamConfig(
        topic=args.topic,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
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
