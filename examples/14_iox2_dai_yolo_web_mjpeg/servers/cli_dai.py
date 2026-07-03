from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Keep the same project-relative import style as your existing camera files.
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute())))



JSON = dict[str, Any]


def configure_logging(log_file: str | Path = "camera_rpc_cli.log") -> None:
    """Configure console + file logging without depending on project utils.py."""

    log_path = Path(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def _as_plain(value: Any) -> Any:
    """Convert pydantic/RPC results into JSON-printable Python objects."""

    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


def print_json_result(title: str, result: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(_as_plain(result), indent=2, default=str))


def save_capture_jpeg(result: Any, output_dir: str | Path = ".") -> dict[str, Any]:
    """Save jpeg_base64 from camera.capture and return a compact result."""

    data = _as_plain(result)
    if not isinstance(data, dict) or "jpeg_base64" not in data:
        return data

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stream = str(data.get("stream", "camera"))
    frame_id = data.get("frame_id", data.get("last_frame_id", "unknown"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = output_path / f"{stream}_frame_{frame_id}_{timestamp}.jpg"

    jpeg_bytes = base64.b64decode(data["jpeg_base64"])
    image_path.write_bytes(jpeg_bytes)

    compact = dict(data)
    compact["jpeg_path"] = str(image_path)
    compact["jpeg_base64_len"] = len(data["jpeg_base64"])
    compact.pop("jpeg_base64", None)
    return compact


class CameraRpcApi:
    """Thin JSON-RPC client wrapper for the real camera RPC methods."""

    def __init__(self, registry: Any, *, controller_name: str = "camera") -> None:
        self.registry = registry
        self.controller_name = controller_name

    @classmethod
    def discover(cls, *, controller_name: str = "camera", show_catalog: bool = False) -> "CameraRpcApi":
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

        registry = Iox2RpcRegistry.discover_all()
        if show_catalog:
            logging.info("\n=== Discovered JSON-RPC catalog ===")
            logging.info(json.dumps(registry.catalog(), indent=2, default=str))
        return cls(registry, controller_name=controller_name)

    def method(self, name: str) -> str:
        return f"{self.controller_name}.{name}"

    def call(self, name: str, params: JSON | None = None, *, timeout_s: float = 2.0) -> Any:
        method = self.method(name)
        logging.info("Calling %s", method)
        if params is None:
            return self.registry.call_unique(method, timeout_s=timeout_s)
        return self.registry.call_unique(method, params, timeout_s=timeout_s)

    def status(self, *, timeout_s: float = 2.0) -> Any:
        return self.call("status", timeout_s=timeout_s)

    def open(self, params: JSON | None = None, *, timeout_s: float = 10.0) -> Any:
        return self.call("open", params=params, timeout_s=timeout_s)

    def close(self, *, timeout_s: float = 5.0) -> Any:
        return self.call("close", timeout_s=timeout_s)

    def capture(self, params: JSON | None = None, *, timeout_s: float = 5.0) -> Any:
        return self.call("capture", params=params, timeout_s=timeout_s)


def _add_common_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server-name", default="camera", help="JSON-RPC controller name. Default: camera")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="RPC timeout in seconds.")
    parser.add_argument("--show-catalog", action="store_true", help="Log the discovered JSON-RPC catalog.")


def _add_camera_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uuid", default=None, help="DepthAI generator UUID, e.g. OkadCam:CamA.")
    parser.add_argument(
        "--source",
        "--sources",
        dest="sources",
        action="append",
        default=None,
        help="Camera source/IP. Can be repeated. Default from server: 169.254.1.222",
    )
    parser.add_argument("--rgb-width", type=int, default=None, help="RGB capture width.")
    parser.add_argument("--rgb-height", type=int, default=None, help="RGB capture height.")
    parser.add_argument("--stereo-width", type=int, default=None, help="Stereo capture width.")
    parser.add_argument("--stereo-height", type=int, default=None, help="Stereo capture height.")
    parser.add_argument("--capture-fps", "--fps", dest="capture_fps", type=int, default=None, help="Capture FPS.")
    parser.add_argument("--rgb-codec", default=None, help="RGB codec, e.g. h265.")
    parser.add_argument("--stereo-codec", default=None, help="Stereo codec, e.g. h265.")
    parser.add_argument("--rgb-bitrate-kbps", type=int, default=None, help="RGB bitrate in kbps.")
    parser.add_argument("--stereo-bitrate-kbps", type=int, default=None, help="Stereo bitrate in kbps.")
    parser.add_argument("--decoder-backend", default=None, help="Decoder backend, e.g. pynvvideocodec or gst-nvivafilter.")
    parser.add_argument("--gst-nvivafilter-so", default=None, help="Path to libdepthai_cuda_preprocess.so.")
    parser.add_argument("--gst-nvivafilter-dtype", default=None, help="gst-nvivafilter dtype, e.g. fp16.")
    parser.add_argument("--gst-nvivafilter-channel-order", default=None, help="gst-nvivafilter channel order, e.g. rgba.")
    parser.add_argument("--decoder-output-color", default=None, help="RGB decoder output color, e.g. rgbp.")
    parser.add_argument("--stereo-decoder-output-color", default=None, help="Stereo decoder output color, e.g. rgbp.")
    parser.add_argument("--rgb-camera-socket", default=None, help="RGB camera socket, e.g. CAM_A.")
    parser.add_argument("--left-camera-socket", default=None, help="Left camera socket, e.g. CAM_B.")
    parser.add_argument("--right-camera-socket", default=None, help="Right camera socket, e.g. CAM_C.")
    parser.add_argument("--no-normalize-rgb", action="store_true", help="Set normalize_rgb=false.")
    parser.add_argument("--no-normalize-stereo", action="store_true", help="Set normalize_stereo=false.")
    parser.add_argument("--preview", action="store_true", help="Enable server-side OpenCV preview windows in the worker process.")
    parser.add_argument("--preview-rgb-downsample", type=int, default=None, help="Preview RGB downsample stride.")
    parser.add_argument("--preview-stereo-downsample", type=int, default=None, help="Preview stereo downsample stride.")
    parser.add_argument("--queue-max-size", type=int, default=None, help="Multiprocessing latest-frame queue size.")
    parser.add_argument("--capture-wait-s", type=float, default=None, help="How long server capture waits for a frame.")
    parser.add_argument("--close-join-timeout-s", type=float, default=None, help="Worker graceful join timeout.")
    parser.add_argument("--close-terminate-timeout-s", type=float, default=None, help="Worker terminate join timeout.")
    parser.add_argument("--retry-delay-s", type=float, default=None, help="Delay before retrying the camera generator after a Python exception/end.")
    parser.add_argument("--worker-watchdog-interval-s", type=float, default=None, help="How often the RPC server checks for native-crashed worker processes.")
    parser.add_argument("--no-retry-forever", action="store_true", help="Disable infinite camera generator retry loop.")


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stream",
        choices=["rgb", "stereo", "left", "right"],
        default="rgb",
        help="Which latest tensor to capture as JPEG.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality, 1-100.")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"), help="Where capture JPEGs are saved.")


def camera_params_from_args(args: argparse.Namespace) -> JSON:
    """Build CameraConfig params from CLI flags, omitting defaults not explicitly set."""

    params: JSON = {"service": "serverCam"}

    direct_fields = [
        "uuid",
        "rgb_width",
        "rgb_height",
        "stereo_width",
        "stereo_height",
        "capture_fps",
        "rgb_codec",
        "stereo_codec",
        "rgb_bitrate_kbps",
        "stereo_bitrate_kbps",
        "decoder_backend",
        "gst_nvivafilter_so",
        "gst_nvivafilter_dtype",
        "gst_nvivafilter_channel_order",
        "decoder_output_color",
        "stereo_decoder_output_color",
        "rgb_camera_socket",
        "left_camera_socket",
        "right_camera_socket",
        "preview_rgb_downsample",
        "preview_stereo_downsample",
        "queue_max_size",
        "capture_wait_s",
        "close_join_timeout_s",
        "close_terminate_timeout_s",
        "retry_delay_s",
        "worker_watchdog_interval_s",
    ]
    for field in direct_fields:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                params[field] = value

    if getattr(args, "sources", None):
        params["sources"] = list(args.sources)

    if getattr(args, "no_normalize_rgb", False):
        params["normalize_rgb"] = False
    if getattr(args, "no_normalize_stereo", False):
        params["normalize_stereo"] = False
    if getattr(args, "preview", False):
        params["preview"] = True
    if getattr(args, "no_retry_forever", False):
        params["retry_forever"] = False

    # If the caller supplied no real config fields, let camera.open use server defaults.
    if set(params.keys()) == {"service"}:
        return {}
    return params


def capture_params_from_args(args: argparse.Namespace) -> JSON:
    return {
        "service": "serverCam",
        "stream": args.stream,
        "jpeg_quality": args.jpeg_quality,
    }


def make_api(args: argparse.Namespace) -> CameraRpcApi:
    configure_logging()
    return CameraRpcApi.discover(controller_name=args.server_name, show_catalog=args.show_catalog)


def command_server(args: argparse.Namespace) -> None:
    configure_logging()
    # Import lazily so client/help commands do not require the camera runtime.
    from server_dai import run_server

    run_server(controller_name=args.server_name)


def command_status(args: argparse.Namespace) -> None:
    api = make_api(args)
    result = api.status(timeout_s=args.timeout_s)
    print_json_result(api.method("status"), result)


def command_open(args: argparse.Namespace) -> None:
    api = make_api(args)
    params = camera_params_from_args(args)
    result = api.open(params=params or None, timeout_s=args.timeout_s)
    print_json_result(api.method("open"), result)


def command_close(args: argparse.Namespace) -> None:
    api = make_api(args)
    result = api.close(timeout_s=args.timeout_s)
    print_json_result(api.method("close"), result)


def command_capture(args: argparse.Namespace) -> None:
    api = make_api(args)
    if args.open_first:
        open_result = api.open(params=camera_params_from_args(args) or None, timeout_s=args.open_timeout_s)
        print_json_result(api.method("open"), open_result)

    result = api.capture(params=capture_params_from_args(args), timeout_s=args.timeout_s)
    compact = save_capture_jpeg(result, output_dir=args.output_dir)
    print_json_result(api.method("capture"), compact)

    if args.close_after:
        close_result = api.close(timeout_s=args.close_timeout_s)
        print_json_result(api.method("close"), close_result)


def command_client(args: argparse.Namespace) -> None:
    """Demo client: status -> open -> status -> capture one or more frames."""

    api = make_api(args)
    print_json_result(api.method("status"), api.status(timeout_s=args.timeout_s))
    print_json_result(
        api.method("open"),
        api.open(params=camera_params_from_args(args) or None, timeout_s=args.open_timeout_s),
    )
    print_json_result(api.method("status"), api.status(timeout_s=args.timeout_s))

    for i in range(args.count):
        result = api.capture(params=capture_params_from_args(args), timeout_s=args.timeout_s)
        compact = save_capture_jpeg(result, output_dir=args.output_dir)
        print_json_result(f"{api.method('capture')} #{i + 1}", compact)
        if i + 1 < args.count and args.interval_s > 0:
            time.sleep(args.interval_s)

    if not args.keep_open:
        print_json_result(api.method("close"), api.close(timeout_s=args.close_timeout_s))


def command_live(args: argparse.Namespace) -> None:
    """Open the camera and keep it running until Enter/Ctrl+C, then close."""

    api = make_api(args)
    params = camera_params_from_args(args)
    if args.preview:
        params["preview"] = True

    print_json_result(api.method("open"), api.open(params=params or None, timeout_s=args.open_timeout_s))

    print("\nCamera worker is running.")
    if params.get("preview"):
        print("Server-side OpenCV preview windows are enabled. Press q in a preview window or Enter here to close.")
    else:
        print("Preview is disabled. Press Enter or Ctrl+C here to close.")

    try:
        input("\nPress Enter to close camera... ")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Closing camera...")
    finally:
        print_json_result(api.method("close"), api.close(timeout_s=args.close_timeout_s))


def command_loop(args: argparse.Namespace) -> None:
    """Open once and capture repeatedly until count is reached or Ctrl+C."""

    api = make_api(args)
    print_json_result(
        api.method("open"),
        api.open(params=camera_params_from_args(args) or None, timeout_s=args.open_timeout_s),
    )

    captured = 0
    try:
        while args.count <= 0 or captured < args.count:
            captured += 1
            result = api.capture(params=capture_params_from_args(args), timeout_s=args.timeout_s)
            compact = save_capture_jpeg(result, output_dir=args.output_dir)
            print_json_result(f"{api.method('capture')} #{captured}", compact)
            if args.interval_s > 0:
                time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping capture loop...")
    finally:
        if not args.keep_open:
            print_json_result(api.method("close"), api.close(timeout_s=args.close_timeout_s))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real DepthAI RGB+stereo JSON-RPC camera CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", aliases=["serve"], help="Run the JSON-RPC camera server.")
    server.add_argument("--server-name", default="camera", help="JSON-RPC controller name. Default: camera")
    server.set_defaults(func=command_server)

    status = subparsers.add_parser("status", help="Call camera.status.")
    _add_common_client_args(status)
    status.set_defaults(func=command_status)

    open_cmd = subparsers.add_parser("open", help="Call camera.open and start the real camera worker.")
    _add_common_client_args(open_cmd)
    _add_camera_config_args(open_cmd)
    open_cmd.set_defaults(func=command_open)

    close = subparsers.add_parser("close", help="Call camera.close and stop the worker.")
    _add_common_client_args(close)
    close.set_defaults(func=command_close)

    capture = subparsers.add_parser("capture", help="Call camera.capture and save the JPEG.")
    _add_common_client_args(capture)
    _add_camera_config_args(capture)
    _add_capture_args(capture)
    capture.add_argument("--open-first", action="store_true", help="Call camera.open before capture.")
    capture.add_argument("--close-after", action="store_true", help="Call camera.close after capture.")
    capture.add_argument("--open-timeout-s", type=float, default=15.0, help="camera.open timeout.")
    capture.add_argument("--close-timeout-s", type=float, default=8.0, help="camera.close timeout.")
    capture.set_defaults(func=command_capture)

    client = subparsers.add_parser("client", help="Demo: status, open, status, capture N frames, close.")
    _add_common_client_args(client)
    _add_camera_config_args(client)
    _add_capture_args(client)
    client.add_argument("--count", type=int, default=2, help="Number of captures.")
    client.add_argument("--interval-s", type=float, default=0.0, help="Delay between captures.")
    client.add_argument("--keep-open", action="store_true", help="Do not close after demo captures.")
    client.add_argument("--open-timeout-s", type=float, default=15.0, help="camera.open timeout.")
    client.add_argument("--close-timeout-s", type=float, default=8.0, help="camera.close timeout.")
    client.set_defaults(func=command_client)

    live = subparsers.add_parser("live", aliases=["preview", "stream"], help="Open camera and keep it running until Enter/Ctrl+C.")
    _add_common_client_args(live)
    _add_camera_config_args(live)
    live.add_argument("--open-timeout-s", type=float, default=15.0, help="camera.open timeout.")
    live.add_argument("--close-timeout-s", type=float, default=8.0, help="camera.close timeout.")
    live.set_defaults(func=command_live)

    loop = subparsers.add_parser("loop", help="Open once and capture repeatedly.")
    _add_common_client_args(loop)
    _add_camera_config_args(loop)
    _add_capture_args(loop)
    loop.add_argument("--count", type=int, default=0, help="Number of captures. 0 means run until Ctrl+C.")
    loop.add_argument("--interval-s", type=float, default=1.0, help="Delay between captures.")
    loop.add_argument("--keep-open", action="store_true", help="Do not close after the loop.")
    loop.add_argument("--open-timeout-s", type=float, default=15.0, help="camera.open timeout.")
    loop.add_argument("--close-timeout-s", type=float, default=8.0, help="camera.close timeout.")
    loop.set_defaults(func=command_loop)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
