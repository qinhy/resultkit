"""In-memory Redis-like JSON store and iceoryx2 server CLI."""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

from .codec import (
    CodecError,
    CommandFrame,
    ResponseFrame,
    decode_command,
    decode_json_value,
    encode_json_value,
    encode_response,
    key_to_str,
    value_to_json_text,
)

if TYPE_CHECKING:
    from .transport import Iox2RpcServer

SERVER_NAME: Final = "iox2redis-json"
DEFAULT_MAX_PAYLOAD_SIZE: Final = 64 * 1024
DEFAULT_POLL_NS: Final = 100_000
CONST_KEY_PREFIX: Final = "const:"
SERVER_INFO_KEY: Final = "const:server_info"
ROOT_PATHS: Final = frozenset({"$", "."})
DUMP_MAGIC: Final = b"IX2D"
DUMP_FORMAT_VERSION: Final = 1


def _error(message: str) -> ResponseFrame:
    return ResponseFrame("error", message=message)


def _wrong_args(command: str) -> ResponseFrame:
    return _error(f"ERR wrong number of arguments for {command}")


def _json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _handle_payload(payload: bytes, handler: Callable[[CommandFrame], ResponseFrame]) -> bytes:
    try:
        response = handler(decode_command(payload))
    except CodecError as exc:
        response = _error(f"ERR codec {exc}")
    except Exception as exc:  # noqa: BLE001
        response = _error(f"ERR {type(exc).__name__}: {exc}")
    return encode_response(response)


def _format_bytes(size: int) -> str:
    """Return a human-readable binary byte size."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{int(value)} {unit}" if value.is_integer() else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _is_const_key(key: str) -> bool:
    """Return whether a key belongs to the constant namespace."""
    return key.startswith(CONST_KEY_PREFIX)


def _coerce_dump_payload(value: Any) -> bytes:
    """Return bytes from a command argument used as a DUMP/LOAD payload."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise CodecError(f"dump payload must be bytes or str, got {type(value).__name__}")


def _encode_dump_payload(item: StoredValue, now: float) -> bytes:
    ttl_ms: int | None = None
    if item.expires_at is not None:
        ttl_ms = max(0, int(round((item.expires_at - now) * 1000)))

    frame = {
        "v": DUMP_FORMAT_VERSION,
        "is_json": item.is_json,
        "ttl_ms": ttl_ms,
        "value": encode_json_value(item.value),
    }
    data = json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return DUMP_MAGIC + data


def _decode_dump_payload(payload: Any, now: float) -> StoredValue:
    raw = _coerce_dump_payload(payload)
    if not raw.startswith(DUMP_MAGIC):
        raise CodecError("invalid dump payload magic")

    try:
        frame = json.loads(raw[len(DUMP_MAGIC) :].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CodecError(f"invalid dump payload: {exc}") from exc

    if frame.get("v") != DUMP_FORMAT_VERSION:
        raise CodecError(f"unsupported dump format version: {frame.get('v')!r}")

    is_json = frame.get("is_json")
    if not isinstance(is_json, bool):
        raise CodecError("dump payload is missing boolean is_json")

    ttl_ms = frame.get("ttl_ms")
    if ttl_ms is None:
        expires_at = None
    elif not isinstance(ttl_ms, bool) and isinstance(ttl_ms, (int, float)) and ttl_ms >= 0:
        expires_at = now + (float(ttl_ms) / 1000.0)
    else:
        raise CodecError("dump payload ttl_ms must be null or non-negative number")

    try:
        value = decode_json_value(frame["value"])
    except KeyError as exc:
        raise CodecError("dump payload is missing value") from exc
    return StoredValue(value=value, expires_at=expires_at, is_json=is_json)


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Immutable information describing the configured server."""

    name: str
    service_path: str
    max_payload_size: int
    max_payload_size_text: str
    poll_value: int
    poll_unit: str
    poll_is_default: bool
    const_key_prefix: str
    server_info_key: str

    @property
    def poll_text(self) -> str:
        suffix = " (default)" if self.poll_is_default else ""
        return f"{self.poll_value} {self.poll_unit}{suffix}"

    @property
    def ready_message(self) -> str:
        return f"IOX2REDIS_READY {self.service_path}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["poll_text"] = self.poll_text
        return data


@dataclass(slots=True)
class StoredValue:
    value: Any
    expires_at: float | None = None
    is_json: bool = False


def _compile_redis_glob(pattern: str) -> re.Pattern[str]:
    """Compile Redis-style glob syntax into a case-sensitive regex."""
    parts, idx = [r"\A"], 0
    while idx < len(pattern):
        char = pattern[idx]
        if char == "*":
            parts.append(".*")
            idx += 1
            continue
        if char == "?":
            parts.append(".")
            idx += 1
            continue
        if char == "\\":
            idx += 1
            if idx < len(pattern):
                parts.append(re.escape(pattern[idx]))
                idx += 1
            else:
                parts.append(re.escape("\\"))
            continue
        if char != "[":
            parts.append(re.escape(char))
            idx += 1
            continue

        end, negated = idx + 1, False
        if end < len(pattern) and pattern[end] in {"^", "!"}:
            negated = True
            end += 1

        class_parts: list[str] = []
        if end < len(pattern) and pattern[end] == "]":
            class_parts.append(r"\]")
            end += 1

        closed = False
        while end < len(pattern):
            class_char = pattern[end]
            if class_char == "]":
                closed = True
                break
            if class_char == "\\" and end + 1 < len(pattern):
                end += 1
                class_parts.append(re.escape(pattern[end]))
            elif class_char == "-":
                class_parts.append("-")
            else:
                class_parts.append(re.escape(class_char))
            end += 1

        if not closed or not class_parts:
            parts.append(r"\[")
            idx += 1
            continue

        parts.append("[" + ("^" if negated else "") + "".join(class_parts) + "]")
        idx = end + 1

    parts.append(r"\Z")
    return re.compile("".join(parts), re.DOTALL)


class JsonStore:
    """Small Redis-like in-memory store used by the server."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._items: dict[str, StoredValue] = {}
        self._clock = clock

    def handle_payload(self, payload: bytes) -> bytes:
        return _handle_payload(payload, self.handle)

    def handle(self, frame: CommandFrame) -> ResponseFrame:
        cmd, args = frame.command.upper(), frame.args
        if cmd == "PING":
            return self._ping(args)
        if cmd == "SET":
            return self._set(args)
        if cmd == "GET":
            return self._get(args)
        if cmd == "DEL":
            return self._del(args)
        if cmd == "EXISTS":
            return self._exists(args)
        if cmd == "MGET":
            return self._mget(args)
        if cmd == "KEYS":
            return self._keys(args)
        if cmd == "DUMP":
            return self._dump(args)
        if cmd == "LOAD":
            return self._load(args)
        if cmd == "JSON.SET":
            return self._json_set(args)
        if cmd == "JSON.GET":
            return self._json_get(args)
        return _error(f"ERR unsupported command {cmd}")

    def _purge_if_expired(self, key: str) -> None:
        item = self._items.get(key)
        if item is not None and item.expires_at is not None and item.expires_at <= self._clock():
            self._items.pop(key, None)

    def _get_item(self, key: str) -> StoredValue | None:
        self._purge_if_expired(key)
        return self._items.get(key)

    def key_exists(self, key: str) -> bool:
        """Return whether a live key exists."""
        return self._get_item(key) is not None

    def value_for_mget(self, key: str) -> Any | None:
        """Return a value using the same representation as MGET."""
        item = self._get_item(key)
        if item is None:
            return None
        return _json_text(item.value) if item.is_json else item.value

    def matching_keys(self, pattern: str) -> list[str]:
        """Return live keys matching a Redis-style glob."""
        matcher = _compile_redis_glob(pattern)
        for key in list(self._items):  # Purging mutates the dictionary.
            self._purge_if_expired(key)
        return [key for key in self._items if matcher.fullmatch(key)]

    def _ping(self, args: list[Any]) -> ResponseFrame:
        return ResponseFrame("pong") if not args else ResponseFrame("bulk", args[0])

    def _set(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 2:
            return _wrong_args("SET")

        key, value = key_to_str(args[0]), args[1]
        old = self._get_item(key)
        expires_at: float | None = None
        nx = xx = get_old = False
        idx = 2

        while idx < len(args):
            opt = key_to_str(args[idx]).upper()
            if opt == "EX" and idx + 1 < len(args):
                expires_at = self._clock() + float(key_to_str(args[idx + 1]))
                idx += 2
            elif opt == "PX" and idx + 1 < len(args):
                expires_at = self._clock() + float(key_to_str(args[idx + 1])) / 1000.0
                idx += 2
            elif opt == "NX":
                nx, idx = True, idx + 1
            elif opt == "XX":
                xx, idx = True, idx + 1
            elif opt == "GET":
                get_old, idx = True, idx + 1
            else:
                return _error(f"ERR unsupported SET option {opt}")

        if nx and xx:
            return _error("ERR NX and XX options are mutually exclusive")
        exists = old is not None
        if (nx and exists) or (xx and not exists):
            return ResponseFrame("nil")

        self._items[key] = StoredValue(value=value, expires_at=expires_at, is_json=False)
        if get_old:
            return ResponseFrame("nil") if old is None else ResponseFrame("bulk", old.value)
        return ResponseFrame("simple", "OK")

    def _get(self, args: list[Any]) -> ResponseFrame:
        if len(args) != 1:
            return _wrong_args("GET")
        item = self._get_item(key_to_str(args[0]))
        if item is None:
            return ResponseFrame("nil")
        return ResponseFrame("bulk", _json_text(item.value) if item.is_json else item.value)

    def _del(self, args: list[Any]) -> ResponseFrame:
        count = 0
        for raw_key in args:
            key = key_to_str(raw_key)
            self._purge_if_expired(key)
            if key in self._items:
                count += 1
                del self._items[key]
        return ResponseFrame("integer", count)

    def _exists(self, args: list[Any]) -> ResponseFrame:
        count = 0
        for raw_key in args:
            if self.key_exists(key_to_str(raw_key)):
                count += 1
        return ResponseFrame("integer", count)

    def _mget(self, args: list[Any]) -> ResponseFrame:
        return ResponseFrame(
            "array", [self.value_for_mget(key_to_str(raw_key)) for raw_key in args]
        )

    def _keys(self, args: list[Any]) -> ResponseFrame:
        if len(args) != 1:
            return _wrong_args("KEYS")
        return ResponseFrame("array", self.matching_keys(key_to_str(args[0])))

    def _dump(self, args: list[Any]) -> ResponseFrame:
        if len(args) != 1:
            return _wrong_args("DUMP")

        item = self._get_item(key_to_str(args[0]))
        if item is None:
            return ResponseFrame("nil")
        return ResponseFrame("bulk", _encode_dump_payload(item, self._clock()))

    def _load(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 2:
            return _wrong_args("LOAD")

        key = key_to_str(args[0])
        old = self._get_item(key)
        nx = xx = False
        idx = 2
        while idx < len(args):
            opt = key_to_str(args[idx]).upper()
            if opt == "NX":
                nx = True
            elif opt == "XX":
                xx = True
            else:
                return _error(f"ERR unsupported LOAD option {opt}")
            idx += 1

        if nx and xx:
            return _error("ERR NX and XX options are mutually exclusive")
        if (nx and old is not None) or (xx and old is None):
            return ResponseFrame("nil")

        try:
            item = _decode_dump_payload(args[1], self._clock())
        except CodecError as exc:
            return _error(f"ERR invalid dump payload: {exc}")

        self._items[key] = item
        return ResponseFrame("simple", "OK")

    def _json_set(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 3:
            return _wrong_args("JSON.SET")

        key, path = key_to_str(args[0]), key_to_str(args[1])
        if path not in ROOT_PATHS:
            return _error("ERR only root path '$' is supported")

        old = self._get_item(key)
        nx = xx = False
        idx = 3
        while idx < len(args):
            opt = key_to_str(args[idx]).upper()
            if opt == "NX":
                nx = True
            elif opt == "XX":
                xx = True
            else:
                return _error(f"ERR unsupported JSON.SET option {opt}")
            idx += 1

        if nx and xx:
            return _error("ERR NX and XX options are mutually exclusive")
        if (nx and old is not None) or (xx and old is None):
            return ResponseFrame("nil")

        try:
            value = json.loads(value_to_json_text(args[2]))
        except json.JSONDecodeError as exc:
            return _error(f"ERR invalid JSON: {exc.msg}")

        self._items[key] = StoredValue(value=value, expires_at=None, is_json=True)
        return ResponseFrame("simple", "OK")

    def _json_get(self, args: list[Any]) -> ResponseFrame:
        if not 1 <= len(args) <= 2:
            return _wrong_args("JSON.GET")

        key = key_to_str(args[0])
        if len(args) == 2 and key_to_str(args[1]) not in ROOT_PATHS:
            return _error("ERR only root path '$' is supported")

        item = self._get_item(key)
        if item is None:
            return ResponseFrame("nil")
        value = _json_text(item.value) if item.is_json else value_to_json_text(item.value)
        return ResponseFrame("bulk", value)


class ConstJsonStore(JsonStore):
    """Write-once store for keys in the reserved const: namespace."""

    def initialize_json(self, key: str, value: Any) -> None:
        """
        Add a server-owned JSON constant.

        This method is intended for initialization before requests are served.
        """
        self._validate_const_key(key)
        if self.key_exists(key):
            raise ValueError(f"constant key is already set: {key}")
        self._items[key] = StoredValue(value=value, expires_at=None, is_json=True)

    @staticmethod
    def _validate_const_key(key: str) -> None:
        if not _is_const_key(key):
            raise ValueError(f"constant keys must start with {CONST_KEY_PREFIX!r}")

    @staticmethod
    def _const_key_error(key: str) -> ResponseFrame:
        return _error(f"ERR constant key {key!r} is already set")

    @staticmethod
    def _const_prefix_error() -> ResponseFrame:
        return _error(f"ERR constant key must start with {CONST_KEY_PREFIX!r}")

    def _set(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 2:
            return _wrong_args("SET")

        key = key_to_str(args[0])
        if not _is_const_key(key):
            return self._const_prefix_error()
        if self.key_exists(key):
            return self._const_key_error(key)

        idx = 2
        while idx < len(args):
            if key_to_str(args[idx]).upper() in {"EX", "PX"}:
                return _error("ERR expiration is not allowed for constant keys")
            idx += 1
        return super()._set(args)

    def _json_set(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 3:
            return _wrong_args("JSON.SET")

        key = key_to_str(args[0])
        if not _is_const_key(key):
            return self._const_prefix_error()
        if self.key_exists(key):
            return self._const_key_error(key)
        return super()._json_set(args)

    def _del(self, args: list[Any]) -> ResponseFrame:
        return _error("ERR constant keys cannot be deleted")

    def _load(self, args: list[Any]) -> ResponseFrame:
        if len(args) < 2:
            return _wrong_args("LOAD")

        key = key_to_str(args[0])
        if not _is_const_key(key):
            return self._const_prefix_error()

        old = self._get_item(key)
        nx = xx = False
        idx = 2
        while idx < len(args):
            opt = key_to_str(args[idx]).upper()
            if opt == "NX":
                nx = True
            elif opt == "XX":
                xx = True
            else:
                return _error(f"ERR unsupported LOAD option {opt}")
            idx += 1

        if nx and xx:
            return _error("ERR NX and XX options are mutually exclusive")
        if old is not None:
            return ResponseFrame("nil") if nx else self._const_key_error(key)
        if xx:
            return ResponseFrame("nil")

        try:
            item = _decode_dump_payload(args[1], self._clock())
        except CodecError as exc:
            return _error(f"ERR invalid dump payload: {exc}")

        if item.expires_at is not None:
            return _error("ERR expiration is not allowed for constant keys")

        self._items[key] = item
        return ResponseFrame("simple", "OK")


class Iox2JsonServer:
    """iceoryx2 request-response server for routed JSON stores."""

    def __init__(
        self,
        service_name: str,
        *,
        max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
        poll_ns: int | None = None,
        poll_ms: int | None = None,
        store: JsonStore | None = None,
        const_store: ConstJsonStore | None = None,
    ) -> None:
        normalized_service_name = service_name.strip("/")
        if not normalized_service_name:
            raise ValueError("service name must not be empty")
        if max_payload_size <= 0:
            raise ValueError("max_payload_size must be greater than zero")
        if poll_ns is not None and poll_ns < 0:
            raise ValueError("poll_ns must not be negative")
        if poll_ms is not None and poll_ms < 0:
            raise ValueError("poll_ms must not be negative")

        self.service_name = normalized_service_name
        self.max_payload_size = max_payload_size
        self.poll_ns = poll_ns
        self.poll_ms = poll_ms

        if poll_ns is not None:
            poll_value, poll_unit, poll_is_default = poll_ns, "ns", False
        elif poll_ms is not None:
            poll_value, poll_unit, poll_is_default = poll_ms, "ms", False
        else:
            poll_value, poll_unit, poll_is_default = DEFAULT_POLL_NS, "ns", True

        self.info: Final[ServerInfo] = ServerInfo(
            name=SERVER_NAME,
            service_path=f"/{self.service_name}/",
            max_payload_size=max_payload_size,
            max_payload_size_text=_format_bytes(max_payload_size),
            poll_value=poll_value,
            poll_unit=poll_unit,
            poll_is_default=poll_is_default,
            const_key_prefix=CONST_KEY_PREFIX,
            server_info_key=SERVER_INFO_KEY,
        )
        self.store: Final[JsonStore] = store or JsonStore()
        self.const_store: Final[ConstJsonStore] = const_store or ConstJsonStore()

        if self.const_store.key_exists(SERVER_INFO_KEY):
            raise ValueError(f"{SERVER_INFO_KEY!r} is reserved by the server")
        self.const_store.initialize_json(SERVER_INFO_KEY, self.info.to_dict())

        self._transport: Iox2RpcServer | None = None
        self._closed = False

    def handle_payload(self, payload: bytes) -> bytes:
        """Decode, route and encode one request."""
        return _handle_payload(payload, self.handle)

    def handle(self, frame: CommandFrame) -> ResponseFrame:
        """Route a decoded command to the appropriate store."""
        cmd, args = frame.command.upper(), frame.args
        if cmd in {"GET", "SET", "JSON.GET", "JSON.SET", "DUMP", "LOAD"}:
            if not args:
                return self.store.handle(frame)
            target = self.const_store if _is_const_key(key_to_str(args[0])) else self.store
            return target.handle(frame)
        if cmd == "DEL":
            return self._routed_del(args)
        if cmd == "EXISTS":
            return self._routed_exists(args)
        if cmd == "MGET":
            return self._routed_mget(args)
        if cmd == "KEYS":
            return self._routed_keys(args)
        return self.store.handle(frame)

    def _store_for_key(self, key: str) -> JsonStore:
        return self.const_store if _is_const_key(key) else self.store

    def _routed_del(self, args: list[Any]) -> ResponseFrame:
        keys = [key_to_str(raw_key) for raw_key in args]
        if any(_is_const_key(key) for key in keys):
            return _error("ERR constant keys cannot be deleted")
        return self.store.handle(CommandFrame("DEL", args))

    def _routed_exists(self, args: list[Any]) -> ResponseFrame:
        count = 0
        for raw_key in args:
            key = key_to_str(raw_key)
            if self._store_for_key(key).key_exists(key):
                count += 1
        return ResponseFrame("integer", count)

    def _routed_mget(self, args: list[Any]) -> ResponseFrame:
        values: list[Any | None] = []
        for raw_key in args:
            key = key_to_str(raw_key)
            values.append(self._store_for_key(key).value_for_mget(key))
        return ResponseFrame("array", values)

    def _routed_keys(self, args: list[Any]) -> ResponseFrame:
        if len(args) != 1:
            return _wrong_args("KEYS")
        pattern = key_to_str(args[0])
        keys = list(
            dict.fromkeys(
                self.store.matching_keys(pattern) + self.const_store.matching_keys(pattern)
            )
        )
        return ResponseFrame("array", keys)

    def open(self) -> None:
        from .transport import Iox2RpcServer

        if self._closed or self._transport is not None:
            return
        transport = Iox2RpcServer(
            self.service_name,
            max_payload_size=self.max_payload_size,
            poll_ns=self.poll_ns,
            poll_ms=self.poll_ms,
        )
        try:
            transport.open()
        except Exception:
            transport.close()
            raise
        self._transport = transport

    def serve_once(self) -> int:
        if self._closed:
            return 0
        self.open()
        if self._closed or self._transport is None:
            return 0
        return self._transport.drain_requests(self.handle_payload)

    def serve_forever(self) -> None:
        self.open()
        while not self._closed:
            self.serve_once()

    def close(self) -> None:
        self._closed = True
        if self._transport is not None:
            self._transport.close()
            self._transport = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Redis-like JSON server over iceoryx2.")
    parser.add_argument("service", help="iceoryx2 service path, e.g. /your/topic/to/iox2_server/")
    parser.add_argument(
        "--max-payload-size",
        type=int,
        default=DEFAULT_MAX_PAYLOAD_SIZE,
        help=(
            "maximum request and response payload size "
            f"in bytes (default: {DEFAULT_MAX_PAYLOAD_SIZE})"
        ),
    )
    parser.add_argument(
        "--poll-ns",
        type=int,
        default=None,
        help=f"iceoryx2 wait duration in nanoseconds; defaults to {DEFAULT_POLL_NS}",
    )
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=None,
        help="legacy millisecond wait duration; ignored when --poll-ns is set",
    )
    return parser


def _print_server_started(server: Iox2JsonServer) -> None:
    info = server.info
    lines = (
        f"[{info.name}] server started",
        f"  Service:            {info.service_path}",
        f"  Max payload size:   {info.max_payload_size_text} ({info.max_payload_size} bytes)",
        f"  Poll interval:      {info.poll_text}",
        f"  Constant namespace: {info.const_key_prefix}* (write-once)",
        f"  Server information: {info.server_info_key}",
        info.ready_message,
    )
    for line in lines:
        print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = Iox2JsonServer(
        args.service,
        max_payload_size=args.max_payload_size,
        poll_ns=args.poll_ns,
        poll_ms=args.poll_ms,
    )
    stopping = started = False
    exit_code = 0

    def _stop(signum: int, _frame: Any) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        print(
            f"\n[{server.info.name}] received {signal_name}; shutting down...",
            flush=True,
        )
        server.close()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        server.open()
        started = True
        _print_server_started(server)
        server.serve_forever()
    except KeyboardInterrupt:
        server.close()
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        print(
            f"[{server.info.name}] server error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        server.close()
        if started:
            print(f"[{server.info.name}] server stopped", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
