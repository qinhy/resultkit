from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any

try:
    from redis.exceptions import ResponseError
except Exception:  # pragma: no cover - keeps the wrapper importable if redis is absent.
    ResponseError = Exception  # type: ignore[misc,assignment]

from common import redis_for
from common import server_main

CONST_PREFIX = "const:"
DEFAULT_SERVICE = "/iox2redis/"
DUMP_FILE_FORMAT = "iox2redis-json-dump-v1"


class PersistenceError(RuntimeError):
    """Raised when startup persistence cannot be completed."""


def _status_code(result: Any) -> int:
    return 0 if result is None else int(result)


def _key_to_str(key: str | bytes) -> str:
    # This demo store treats keys as UTF-8 strings.
    return key.decode("utf-8") if isinstance(key, bytes) else str(key)


def _dump_payload_to_text(payload: bytes | str) -> str:
    if isinstance(payload, str):
        # DUMP payloads should be bytes. This fallback keeps compatibility with
        # custom clients that may already return a binary-safe string.
        payload = payload.encode("latin1")
    return base64.b64encode(payload).decode("ascii")


def dump_all_keys(
    r: Any,
    *,
    pattern: str = "*",
    include_const: bool = True,
) -> dict[str, str]:
    """Return {key: base64_dump_payload} for all matching live keys."""
    dumped: dict[str, str] = {}

    for raw_key in r.keys(pattern):
        key = _key_to_str(raw_key)
        if not include_const and key.startswith(CONST_PREFIX):
            continue

        payload = r.dump(key)
        if payload is None:
            # Key may have expired between KEYS and DUMP.
            continue

        dumped[key] = _dump_payload_to_text(payload)

    return dumped


def save_server_dump(
    r: Any,
    path: str | Path,
    *,
    pattern: str = "*",
    include_const: bool = True,
) -> int:
    """Atomically write a JSON dump file for all matching keys."""
    dump_path = Path(path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    keys = dump_all_keys(r, pattern=pattern, include_const=include_const)
    data = {
        "format": DUMP_FILE_FORMAT,
        "keys": keys,
    }

    tmp_path = dump_path.with_name(f".{dump_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(dump_path)
    return len(keys)


def load_server_dump(
    r: Any,
    path: str | Path,
    *,
    nx: bool = False,
    xx: bool = False,
    load_const: bool = True,
) -> dict[str, Any]:
    """Load a JSON dump file produced by save_server_dump()."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != DUMP_FILE_FORMAT:
        raise ValueError(f"unsupported dump file format: {data.get('format')!r}")

    results: dict[str, Any] = {}
    for key, encoded_payload in data["keys"].items():
        is_const = key.startswith(CONST_PREFIX)
        if is_const and not load_const:
            results[key] = "SKIPPED const key"
            continue

        payload = base64.b64decode(encoded_payload.encode("ascii"))

        try:
            if is_const:
                # const:* keys are write-once. Restore them as create-only so
                # const:server_info and already-present user constants are not overwritten.
                results[key] = r.load(key, payload, nx=True)
            else:
                results[key] = r.load(key, payload, nx=nx, xx=xx)
        except ResponseError as exc:
            message = str(exc)
            if is_const and "constant keys cannot be loaded" in message:
                results[key] = "SKIPPED const key: server does not support LOAD const:* yet"
            elif is_const and "already set" in message:
                results[key] = None
            else:
                raise

    return results


def _infer_service(server_args: list[str]) -> str:
    """Best-effort service inference from args passed to the wrapped server."""
    for index, arg in enumerate(server_args):
        if arg in {"--service", "--host"} and index + 1 < len(server_args):
            return server_args[index + 1]
        if arg.startswith("--service=") or arg.startswith("--host="):
            return arg.split("=", 1)[1]

    for arg in server_args:
        if not arg.startswith("-") and arg.startswith("/"):
            return arg

    return DEFAULT_SERVICE


def _persistence_client(service: str) -> Any:
    # Use binary responses so Redis DUMP payloads are not decoded as text.
    return redis_for(host=service, decode_responses=False)


def _wait_for_server(service: str, timeout_seconds: float) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        try:
            r = _persistence_client(service)
            if r.ping():
                return r
        except BaseException as exc:  # keep retrying while the child starts.
            last_error = exc
        time.sleep(0.1)

    detail = f": {last_error!r}" if last_error else ""
    raise PersistenceError(
        f"server did not become ready on service {service!r} within {timeout_seconds:.1f}s{detail}"
    )


def _run_wrapped_server(server_args: list[str]) -> None:
    # Ctrl+C is handled by the parent so it can dump the store before stopping the child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.argv = [sys.argv[0], *server_args]
    raise SystemExit(server_main())


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run iox2redis.server with optional JSON dump/load persistence. "
            "Unknown arguments are forwarded to iox2redis.server."
        )
    )
    parser.add_argument(
        "--store-file",
        type=Path,
        help="JSON file used for automatic LOAD on startup and DUMP on shutdown.",
    )
    parser.add_argument(
        "--persist-service",
        help=(
            "Service path used by the persistence client. If omitted, this wrapper "
            "tries to infer it from forwarded server args, then falls back to the demo default."
        ),
    )
    parser.add_argument(
        "--persist-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the wrapped server to become reachable; default: 10.",
    )
    parser.add_argument(
        "--persist-pattern",
        default="*",
        help="KEYS pattern to dump; default: '*'.",
    )
    parser.add_argument(
        "--persist-save-seconds",
        type=float,
        default=13.0,
        help="Optional periodic save interval. 0 disables periodic saves; default: 0.",
    )
    parser.add_argument(
        "--persist-no-load",
        action="store_true",
        help="Do not load the store file at startup.",
    )
    parser.add_argument(
        "--persist-no-dump",
        action="store_true",
        help="Do not dump the store file on shutdown.",
    )
    parser.add_argument(
        "--persist-skip-const",
        action="store_true",
        help="Do not dump const:* keys.",
    )
    parser.add_argument(
        "--persist-load-const",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load const:* keys when present; default: true.",
    )
    return parser.parse_known_args(argv)


def run_persistent_server(args: argparse.Namespace, server_args: list[str]) -> int:
    if args.store_file is None:
        # No persistence requested: preserve the original tiny entrypoint behavior.
        sys.argv = [sys.argv[0], *server_args]
        return _status_code(server_main())

    service = args.persist_service or _infer_service(server_args)
    store_file = Path(args.store_file)
    child = Process(target=_run_wrapped_server, args=(server_args,))

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    child.start()
    r: Any | None = None
    exit_code = 0

    try:
        r = _wait_for_server(service, args.persist_timeout)

        if not args.persist_no_load and store_file.exists():
            results = load_server_dump(r, store_file, load_const=args.persist_load_const)
            print(f"[persist] loaded {len(results)} keys from {store_file}")
        elif not args.persist_no_load:
            print(f"[persist] no existing store file at {store_file}; starting empty")

        last_save = time.monotonic()
        while child.is_alive() and not stop_requested:
            child.join(timeout=0.25)
            if (
                args.persist_save_seconds > 0
                and time.monotonic() - last_save >= args.persist_save_seconds
            ):
                count = save_server_dump(
                    r,
                    store_file,
                    pattern=args.persist_pattern,
                    include_const=not args.persist_skip_const,
                )
                last_save = time.monotonic()
                print(f"[persist] periodic dump wrote {count} keys to {store_file}")

        if child.exitcode not in (None, 0):
            exit_code = int(child.exitcode)

    except BaseException as exc:
        exit_code = 1
        print(f"[persist] error: {exc}", file=sys.stderr)
    finally:
        if r is not None and not args.persist_no_dump:
            try:
                count = save_server_dump(
                    r,
                    store_file,
                    pattern=args.persist_pattern,
                    include_const=not args.persist_skip_const,
                )
                print(f"[persist] shutdown dump wrote {count} keys to {store_file}")
            except BaseException as exc:
                exit_code = 1
                print(f"[persist] shutdown dump failed: {exc}", file=sys.stderr)

        if child.is_alive():
            child.terminate()
            child.join(timeout=2.0)
        if child.is_alive():
            child.kill()
            child.join(timeout=2.0)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv

    # Preserve exact original behavior unless the new persistence option is used.
    if "--store-file" not in raw_argv and not any(
        arg.startswith("--store-file=") for arg in raw_argv
    ):
        return _status_code(server_main())

    args, server_args = _parse_wrapper_args(raw_argv)
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    return run_persistent_server(args, server_args)


if __name__ == "__main__":
    raise SystemExit(main())

### Example usage:
# python server.py --store-file ./iox2redis-store.json

### With a custom service path forwarded to the underlying server:
# python server.py --store-file ./iox2redis-store.json -- --service /your/topic/to/iox2_server/

### Or explicitly tell the persistence wrapper which service to connect to:
# python server.py \ox2redis-store.json \
#   --persist-service /your/topic/to/iox2_server/ \
#   -- --service /your/topic/to/iox2_server/
# ```

### Periodic autosave every 30 seconds:
# python server.py --store-file ./iox2redis-store.json --persist-save-seconds 30
