"""Wire codec for Redis-like commands over iceoryx2."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
MAGIC = b"IX2R"

FRAME_COMMAND = 1
FRAME_RESPONSE = 2

TAG_NONE = 0
TAG_BYTES = 1
TAG_STR = 2
TAG_JSON = 3

KIND_TO_CODE = {
    "simple": 1,
    "bulk": 2,
    "array": 3,
    "nil": 4,
    "error": 5,
    "integer": 6,
    "pong": 7,
}
CODE_TO_KIND = {value: key for key, value in KIND_TO_CODE.items()}

U8 = struct.Struct("!B")
U16 = struct.Struct("!H")
U32 = struct.Struct("!I")


class CodecError(ValueError):
    """Raised when a command or response frame is invalid."""


@dataclass(frozen=True)
class CommandFrame:
    command: str
    args: list[Any]


@dataclass(frozen=True)
class ResponseFrame:
    kind: str
    value: Any = None
    message: str | None = None


def _pack_len_prefixed(data: bytes) -> bytes:
    return U32.pack(len(data)) + data


def _read_exact(payload: bytes, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if end > len(payload):
        raise CodecError("truncated frame")
    return payload[offset:end], end


def _read_u8(payload: bytes, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(payload, offset, U8.size)
    return U8.unpack(raw)[0], offset


def _read_u16(payload: bytes, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(payload, offset, U16.size)
    return U16.unpack(raw)[0], offset


def _read_u32(payload: bytes, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(payload, offset, U32.size)
    return U32.unpack(raw)[0], offset


def _read_len_prefixed(payload: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_u32(payload, offset)
    return _read_exact(payload, offset, size)


def _decode_header(payload: bytes, expected_frame_type: int) -> int:
    if len(payload) < 6:
        raise CodecError("truncated frame header")
    if payload[:4] != MAGIC:
        raise CodecError("invalid binary frame magic")
    version = payload[4]
    if version != PROTOCOL_VERSION:
        raise CodecError(f"unsupported protocol version: {version!r}")
    frame_type = payload[5]
    if frame_type != expected_frame_type:
        raise CodecError(f"unexpected frame type: {frame_type!r}")
    return 6


def _decode_json_command(payload: bytes) -> CommandFrame:
    try:
        frame = json.loads(payload.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve details in CodecError
        raise CodecError(f"invalid command frame: {exc}") from exc

    if frame.get("v") != PROTOCOL_VERSION:
        raise CodecError(f"unsupported protocol version: {frame.get('v')!r}")
    command = frame.get("command")
    if not isinstance(command, str) or not command:
        raise CodecError("missing command")
    raw_args = frame.get("args", [])
    if not isinstance(raw_args, list):
        raise CodecError("args must be a list")
    return CommandFrame(command=command.upper(), args=[decode_json_value(arg) for arg in raw_args])


def _decode_json_response(payload: bytes) -> ResponseFrame:
    try:
        frame = json.loads(payload.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CodecError(f"invalid response frame: {exc}") from exc

    if frame.get("v") != PROTOCOL_VERSION:
        raise CodecError(f"unsupported protocol version: {frame.get('v')!r}")
    kind = frame.get("kind")
    if not isinstance(kind, str):
        raise CodecError("missing response kind")

    if kind == "bulk":
        return ResponseFrame(kind=kind, value=decode_json_value(frame["value"]))
    if kind == "array":
        items = frame.get("value", [])
        if not isinstance(items, list):
            raise CodecError("array response value must be a list")
        return ResponseFrame(
            kind=kind,
            value=[None if item is None else decode_json_value(item) for item in items],
        )
    if kind == "error":
        return ResponseFrame(kind=kind, message=str(frame.get("message", "ERR unknown error")))
    return ResponseFrame(kind=kind, value=frame.get("value"), message=frame.get("message"))


def encode_json_value(value: Any) -> dict[str, Any]:
    """Encode one Python value into the legacy JSON-safe tagged value."""

    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        import base64

        return {"type": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, str):
        return {"type": "str", "data": value}
    if value is None or isinstance(value, (bool, int, float, list, dict)):
        return {"type": "json", "data": value}
    return {"type": "str", "data": str(value)}


def decode_json_value(value: dict[str, Any]) -> Any:
    """Decode a tagged value from the legacy JSON wire format."""

    if not isinstance(value, dict):
        raise CodecError(f"tagged value must be a dict, got {type(value).__name__}")
    tag = value.get("type")
    if tag == "bytes":
        import base64

        return base64.b64decode(value["data"].encode("ascii"))
    if tag == "str":
        return value["data"]
    if tag == "json":
        return value.get("data")
    raise CodecError(f"unknown value tag: {tag!r}")


def encode_value(value: Any) -> bytes:
    """Encode one Python value into a binary tagged value."""

    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if value is None:
        return U8.pack(TAG_NONE)
    if isinstance(value, bytes):
        return U8.pack(TAG_BYTES) + _pack_len_prefixed(value)
    if isinstance(value, str):
        return U8.pack(TAG_STR) + _pack_len_prefixed(value.encode("utf-8"))
    if isinstance(value, (bool, int, float, list, dict)):
        data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return U8.pack(TAG_JSON) + _pack_len_prefixed(data)
    return U8.pack(TAG_STR) + _pack_len_prefixed(str(value).encode("utf-8"))


def decode_value(payload: bytes, offset: int) -> tuple[Any, int]:
    """Decode one binary tagged value."""

    tag, offset = _read_u8(payload, offset)
    if tag == TAG_NONE:
        return None, offset

    data, offset = _read_len_prefixed(payload, offset)
    if tag == TAG_BYTES:
        return data, offset
    if tag == TAG_STR:
        return data.decode("utf-8"), offset
    if tag == TAG_JSON:
        return json.loads(data.decode("utf-8")), offset
    raise CodecError(f"unknown value tag: {tag!r}")


def encode_command(args: Iterable[Any]) -> bytes:
    """Encode redis-py command arguments to an iceoryx2 payload."""

    args = list(args)
    if not args:
        raise CodecError("empty command")

    command = args[0]
    if isinstance(command, bytes):
        command = command.decode("utf-8")
    command = str(command).upper()

    command_bytes = command.encode("utf-8")
    if len(command_bytes) > 0xFFFF:
        raise CodecError("command name is too long")
    frame_args = args[1:]
    if len(frame_args) > 0xFFFF:
        raise CodecError("too many command arguments")

    return b"".join(
        [
            MAGIC,
            U8.pack(PROTOCOL_VERSION),
            U8.pack(FRAME_COMMAND),
            U16.pack(len(command_bytes)),
            U16.pack(len(frame_args)),
            command_bytes,
            *(encode_value(arg) for arg in frame_args),
        ]
    )


def decode_command(payload: bytes) -> CommandFrame:
    """Decode an iceoryx2 payload into a command frame."""

    if not payload.startswith(MAGIC):
        return _decode_json_command(payload)

    offset = _decode_header(payload, FRAME_COMMAND)
    command_size, offset = _read_u16(payload, offset)
    argc, offset = _read_u16(payload, offset)
    raw_command, offset = _read_exact(payload, offset, command_size)
    command = raw_command.decode("utf-8")
    if not command:
        raise CodecError("missing command")

    args = []
    for _ in range(argc):
        value, offset = decode_value(payload, offset)
        args.append(value)
    if offset != len(payload):
        raise CodecError("trailing bytes in command frame")
    return CommandFrame(command=command.upper(), args=args)


def encode_response(frame: ResponseFrame) -> bytes:
    try:
        kind = KIND_TO_CODE[frame.kind]
    except KeyError as exc:
        raise CodecError(f"unknown response kind: {frame.kind!r}") from exc

    parts = [MAGIC, U8.pack(PROTOCOL_VERSION), U8.pack(FRAME_RESPONSE), U8.pack(kind)]
    if frame.kind == "nil":
        return b"".join(parts)
    if frame.kind == "error":
        return b"".join(parts + [_pack_len_prefixed((frame.message or "").encode("utf-8"))])
    if frame.kind == "array":
        value = [] if frame.value is None else list(frame.value)
        return b"".join(parts + [U32.pack(len(value)), *(encode_value(item) for item in value)])
    if frame.kind == "pong" and frame.value is None:
        return b"".join(parts + [encode_value(None)])
    return b"".join(parts + [encode_value(frame.value)])


def decode_response(payload: bytes) -> ResponseFrame:
    if not payload.startswith(MAGIC):
        return _decode_json_response(payload)

    offset = _decode_header(payload, FRAME_RESPONSE)
    kind_code, offset = _read_u8(payload, offset)
    try:
        kind = CODE_TO_KIND[kind_code]
    except KeyError as exc:
        raise CodecError(f"unknown response kind: {kind_code!r}") from exc

    if kind == "nil":
        value = None
        if offset != len(payload):
            raise CodecError("trailing bytes in nil response frame")
        return ResponseFrame(kind=kind, value=value)
    if kind == "error":
        raw_message, offset = _read_len_prefixed(payload, offset)
        if offset != len(payload):
            raise CodecError("trailing bytes in error response frame")
        return ResponseFrame(kind=kind, message=raw_message.decode("utf-8") or "ERR unknown error")
    if kind == "array":
        count, offset = _read_u32(payload, offset)
        items = []
        for _ in range(count):
            value, offset = decode_value(payload, offset)
            items.append(value)
        if offset != len(payload):
            raise CodecError("trailing bytes in array response frame")
        return ResponseFrame(kind=kind, value=items)

    value, offset = decode_value(payload, offset)
    if offset != len(payload):
        raise CodecError("trailing bytes in response frame")
    return ResponseFrame(kind=kind, value=value)


def response_to_redis_value(frame: ResponseFrame) -> Any:
    """Map a ResponseFrame to the shape redis-py expects from read_response()."""

    if frame.kind == "simple":
        return str(frame.value).encode("utf-8")
    if frame.kind == "bulk":
        value = frame.value
        if isinstance(value, str):
            return value.encode("utf-8")
        return value
    if frame.kind == "integer":
        return int(frame.value)
    if frame.kind == "array":
        out: list[Any] = []
        for item in frame.value:
            if item is None:
                out.append(None)
            elif isinstance(item, str):
                out.append(item.encode("utf-8"))
            else:
                out.append(item)
        return out
    if frame.kind == "nil":
        return None
    if frame.kind == "pong":
        return b"PONG" if frame.value is None else str(frame.value).encode("utf-8")
    if frame.kind == "error":
        raise CodecError(frame.message or "ERR unknown error")
    raise CodecError(f"unknown response kind: {frame.kind!r}")


def key_to_str(value: Any) -> str:
    """Redis keys are bytes-safe. This demo treats keys as UTF-8 strings."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def value_to_json_text(value: Any) -> str:
    """Convert a command argument into JSON text for JSON.SET."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
