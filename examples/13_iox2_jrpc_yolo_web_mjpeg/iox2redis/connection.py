"""redis-py Connection implementation backed by iceoryx2."""

from __future__ import annotations

import builtins
from typing import Any

import redis
from redis.connection import Connection
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .codec import CodecError, decode_response, encode_command, response_to_redis_value
from .transport import Iox2RpcClient, service_name_from_host


class Iox2Connection(Connection):
    """A redis-py connection that sends commands over iceoryx2 request-response.

    Only a small command subset is supported by the matching demo server. The
    class intentionally leaves pipelines/packed RESP commands unsupported.
    """

    def __init__(
        self,
        host: str = "/redis/json",
        port: int = 0,
        *,
        response_timeout: float = 1.0,
        max_payload_size: int = 64 * 1024,
        poll_ns: int | None = None,
        poll_ms: int | None = None,
        **kwargs: Any,
    ) -> None:
        if redis is None:  # pragma: no cover
            raise ImportError("redis-py is required. ex: pip install redis")
        self.service_name = service_name_from_host(host)
        self.response_timeout = response_timeout
        self.max_payload_size = max_payload_size
        self.poll_ns = poll_ns
        self.poll_ms = poll_ms
        self._transport: Iox2RpcClient | None = None
        self._pending_response: Any = None
        self._pending_error: Exception | None = None

        kwargs.setdefault("health_check_interval", 0)
        kwargs.setdefault("client_name", None)
        # port is ignored, but redis-py Connection requires it.
        super().__init__(host=host, port=port, **kwargs)  # type: ignore[no-untyped-call]

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_open

    def repr_pieces(self) -> list[tuple[str, Any]]:
        return [("iox2_service", f"/{self.service_name}/"), ("db", self.db)]

    def connect(self) -> None:
        if self.is_connected:
            return
        self._transport = Iox2RpcClient(
            self.service_name,
            timeout=self.response_timeout,
            max_payload_size=self.max_payload_size,
            poll_ns=self.poll_ns,
            poll_ms=self.poll_ms,
        )
        self._transport.open()

    def disconnect(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002 - redis-py signature compatibility
        if self._transport is not None:
            self._transport.close()
        self._transport = None
        self._pending_response = None
        self._pending_error = None

    def check_health(self) -> None:
        return None

    def send_command(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002 - redis-py compatibility
        self.connect()
        assert self._transport is not None
        try:
            request = encode_command(args)
            raw_response = self._transport.request(request)
            frame = decode_response(raw_response)
            self._pending_response = response_to_redis_value(frame)
            self._pending_error = None
        except builtins.TimeoutError as exc:
            raise RedisTimeoutError(str(exc)) from exc
        except CodecError as exc:
            self._pending_response = None
            self._pending_error = ResponseError(str(exc))
        except Exception as exc:  # noqa: BLE001
            self._pending_response = None
            self._pending_error = RedisConnectionError(str(exc))

    def read_response(
        self,
        disable_decoding: bool = False,
        *,
        timeout: Any = None,  # noqa: ARG002
        disconnect_on_error: bool = True,  # noqa: ARG002
        push_request: bool = False,  # noqa: ARG002
    ) -> Any:
        if self._pending_error is not None:
            error = self._pending_error
            self._pending_error = None
            raise error

        response = self._pending_response
        self._pending_response = None

        if not disable_decoding:
            return self._decode_response_value(response)
        return response

    def _decode_response_value(self, response: Any) -> Any:
        if isinstance(response, bytes) and getattr(self.encoder, "decode_responses", False):
            return self.encoder.decode(response)  # type: ignore[no-untyped-call]
        if isinstance(response, list) and getattr(self.encoder, "decode_responses", False):
            return [self.encoder.decode(v) if isinstance(v, bytes) else v for v in response]  # type: ignore[no-untyped-call]
        return response

    def can_read(self, timeout: float = 0) -> bool:  # noqa: ARG002
        return self._pending_response is not None or self._pending_error is not None

    def send_packed_command(self, command: Any, check_health: bool = True) -> None:  # noqa: ARG002
        raise ResponseError("iox2redis-json does not support pipelines or packed RESP commands")
