"""Client helpers for redis-py + iceoryx2."""

from __future__ import annotations

import json
from typing import Any

from .codec import CodecError, decode_response, encode_command, response_to_redis_value
from .connection import Iox2Connection
from .transport import Iox2RpcClient, service_name_from_host


def is_iox2_host(host: str | None) -> bool:
    return isinstance(host, str) and (host.startswith("/") or host.startswith("iox2://"))


def _import_redis() -> Any:
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ImportError(
            "redis-py is required. Install with: python -m pip install redis"
        ) from exc
    return redis


class JsonHelpersMixin:
    """Small convenience helpers on top of Redis-like commands."""

    def execute_command(self, *args: Any, **options: Any) -> Any:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, *, nx: bool = False, xx: bool = False) -> Any:
        options: list[str] = []
        if nx:
            options.append("NX")
        if xx:
            options.append("XX")
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return self.execute_command("JSON.SET", key, "$", payload, *options)

    def get_json(self, key: str) -> Any:
        raw = self.execute_command("JSON.GET", key, "$")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def json_set(
        self, key: str, path: str, value: Any, *, nx: bool = False, xx: bool = False
    ) -> Any:
        options: list[str] = []
        if nx:
            options.append("NX")
        if xx:
            options.append("XX")
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return self.execute_command("JSON.SET", key, path, payload, *options)

    def json_get(self, key: str, path: str = "$") -> Any:
        raw = self.execute_command("JSON.GET", key, path)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def dump(self, key: str) -> bytes | None:
        # DUMP is binary. With decode_responses=True, redis-py would normally
        # decode the bulk response to str unless NEVER_DECODE is passed through.
        value = self.execute_command("DUMP", key, NEVER_DECODE=True)
        if isinstance(value, str):
            return value.encode("utf-8")
        if value is not None and not isinstance(value, bytes):
            raise CodecError(f"DUMP returned non-bytes value: {type(value).__name__}")
        return value

    def load(self, key: str, payload: bytes, *, nx: bool = False, xx: bool = False) -> Any:
        options: list[str] = []
        if nx:
            options.append("NX")
        if xx:
            options.append("XX")
        return self.execute_command("LOAD", key, payload, *options)


class Iox2DirectClient:
    """Direct iceoryx2 client that bypasses redis-py compatibility layers."""

    def __init__(
        self,
        host: str,
        *,
        response_timeout: float = 1.0,
        max_payload_size: int = 64 * 1024,
        poll_ns: int | None = None,
        poll_ms: int | None = None,
    ) -> None:
        self.service_name = service_name_from_host(host)
        self._transport = Iox2RpcClient(
            self.service_name,
            timeout=response_timeout,
            max_payload_size=max_payload_size,
            poll_ns=poll_ns,
            poll_ms=poll_ms,
        )

    def close(self) -> None:
        self._transport.close()

    def execute_command(self, *args: Any) -> Any:
        response = decode_response(self._transport.request(encode_command(args)))
        return response_to_redis_value(response)

    def ping(self) -> bool:
        return bool(self.execute_command("PING") == b"PONG")

    def set_bytes(self, key: bytes | str, value: bytes) -> bool:
        return bool(self.execute_command("SET", key, value) == b"OK")

    def get_bytes(self, key: bytes | str) -> bytes | None:
        value = self.execute_command("GET", key)
        if value is not None and not isinstance(value, bytes):
            raise CodecError(f"GET returned non-bytes value: {type(value).__name__}")
        return value

    def keys(self, pattern: bytes | str = "*") -> list[bytes]:
        value = self.execute_command("KEYS", pattern)
        if not isinstance(value, list) or any(not isinstance(key, bytes) for key in value):
            raise CodecError("KEYS returned an invalid response")
        return value

    def set_json_bytes(self, key: bytes | str, json_bytes: bytes) -> bool:
        return bool(self.execute_command("JSON.SET", key, "$", json_bytes) == b"OK")

    def get_json_bytes(self, key: bytes | str) -> bytes | None:
        value = self.execute_command("JSON.GET", key, "$")
        if value is not None and not isinstance(value, bytes):
            raise CodecError(f"JSON.GET returned non-bytes value: {type(value).__name__}")
        return value

    def set_json(self, key: bytes | str, value: Any) -> bool:
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return self.set_json_bytes(key, payload)

    def get_json(self, key: bytes | str) -> Any:
        raw = self.get_json_bytes(key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    def dump(self, key: bytes | str) -> bytes | None:
        value = self.execute_command("DUMP", key)
        if value is not None and not isinstance(value, bytes):
            raise CodecError(f"DUMP returned non-bytes value: {type(value).__name__}")
        return value

    def load(
        self,
        key: bytes | str,
        payload: bytes,
        *,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        options: list[str] = []
        if nx:
            options.append("NX")
        if xx:
            options.append("XX")
        return bool(self.execute_command("LOAD", key, payload, *options) == b"OK")

    def __enter__(self) -> Iox2DirectClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _client_class() -> type[Any]:
    RedisBase: Any = _import_redis().Redis

    class Iox2Redis(JsonHelpersMixin, RedisBase):  # type: ignore[misc]
        def execute_command(self, *args: Any, **options: Any) -> Any:
            return RedisBase.execute_command(self, *args, **options)

    return Iox2Redis


def direct_redis_for(
    host: str,
    *,
    response_timeout: float = 1.0,
    max_payload_size: int = 64 * 1024,
    poll_ns: int | None = None,
    poll_ms: int | None = None,
) -> Iox2DirectClient:
    """Create a direct iceoryx2 client without redis-py compatibility layers."""

    return Iox2DirectClient(
        host,
        response_timeout=response_timeout,
        max_payload_size=max_payload_size,
        poll_ns=poll_ns,
        poll_ms=poll_ms,
    )


def redis_for(
    host: str = "localhost",
    *,
    port: int = 6379,
    response_timeout: float = 1.0,
    max_payload_size: int = 64 * 1024,
    poll_ns: int | None = None,
    poll_ms: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Redis client.

    If host starts with `/` or `iox2://`, returns a redis-py client using the
    iceoryx2 connection class. Otherwise it returns a normal TCP redis-py client
    with the same JSON helper methods mixed in.
    """

    redis = _import_redis()
    RedisClass = _client_class()

    if is_iox2_host(host):
        pool = redis.ConnectionPool(
            connection_class=Iox2Connection,
            host=host,
            port=0,
            response_timeout=response_timeout,
            max_payload_size=max_payload_size,
            poll_ns=poll_ns,
            poll_ms=poll_ms,
            **kwargs,
        )
        return RedisClass(connection_pool=pool)

    return RedisClass(host=host, port=port, **kwargs)
