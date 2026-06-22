"""iceoryx2 request-response transport adapter."""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Callable
from typing import Any

DEFAULT_POLL_NS = 100_000


class Iox2Unavailable(RuntimeError):
    """Raised when the iceoryx2 Python package is not installed."""


def service_name_from_host(host: str) -> str:
    """Convert redis-style host path into an iceoryx2 service name."""

    if host.startswith("iox2://"):
        host = host.removeprefix("iox2://")
    return host.strip("/")


def _import_iox2() -> Any:
    try:
        import iceoryx2 as iox2  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise Iox2Unavailable(
            "iceoryx2 is required for IPC transport. Install with: "
            "python -m pip install -e '.[iox2]'"
        ) from exc
    return iox2


def _service_type(iox2: Any) -> Any:
    name = os.getenv("IOX2REDIS_SERVICE_TYPE", "ipc").lower()
    if name == "ipc":
        return iox2.ServiceType.Ipc
    if name == "local":
        try:
            return iox2.ServiceType.Local
        except AttributeError as exc:
            raise ValueError("iceoryx2 does not expose ServiceType.Local") from exc
    raise ValueError("IOX2REDIS_SERVICE_TYPE must be either 'ipc' or 'local'")


def _poll_ns(*, poll_ns: int | None, poll_ms: int | None) -> int:
    if poll_ns is not None:
        if poll_ns < 0:
            raise ValueError("poll_ns must be >= 0")
        return poll_ns
    if poll_ms is not None:
        if poll_ms < 0:
            raise ValueError("poll_ms must be >= 0")
        return poll_ms * 1_000_000
    return DEFAULT_POLL_NS


def _number_of_elements(obj: Any) -> int | None:
    header = getattr(obj, "header", None)
    header = header() if callable(header) else header
    number = getattr(header, "number_of_elements", None)
    if callable(number):
        number = number()
    return int(number) if number is not None else None


def slice_payload_to_bytes(obj: Any) -> bytes:
    """Read bytes from a payload-bearing iceoryx2 object.

    The Python bindings expose slices as either a Slice-like object with
    len()/as_ptr() or a ctypes pointer depending on context/version. This helper
    handles both shapes.
    """

    payload = obj.payload()

    # Dynamic Slice[T] path.
    if hasattr(payload, "len") and hasattr(payload, "as_ptr"):
        return ctypes.string_at(payload.as_ptr(), int(payload.len()))

    # Pointer path with length in request/response header.
    n = _number_of_elements(obj)
    if n is None:
        try:
            n = len(payload)
        except TypeError as exc:
            raise RuntimeError("cannot determine iceoryx2 payload length") from exc
    return bytes(int(payload[i]) for i in range(n))


def write_bytes_to_uninit(uninit: Any, data: bytes) -> Any:
    payload = uninit.payload()
    if hasattr(payload, "as_ptr"):
        ctypes.memmove(payload.as_ptr(), data, len(data))
    else:
        for idx, byte in enumerate(data):
            payload[idx] = byte
    return uninit.assume_init()


class Iox2RpcClient:
    """Small request-response client using dynamic byte slices."""

    def __init__(
        self,
        service_name: str,
        *,
        max_payload_size: int = 64 * 1024,
        timeout: float = 1.0,
        poll_ns: int | None = DEFAULT_POLL_NS,
        poll_ms: int | None = None,
    ) -> None:
        self.service_name = service_name_from_host(service_name)
        self.max_payload_size = max_payload_size
        self.timeout = timeout
        self.poll_ns = _poll_ns(poll_ns=poll_ns, poll_ms=poll_ms)
        self._iox2: Any | None = None
        self._node: Any | None = None
        self._service: Any | None = None
        self._client: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._client is not None

    def open(self) -> None:
        if self.is_open:
            return
        iox2 = _import_iox2()
        self._iox2 = iox2
        self._node = iox2.NodeBuilder.new().create(_service_type(iox2))
        self._service = (
            self._node.service_builder(iox2.ServiceName.new(self.service_name))
            .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        self._client = (
            self._service.client_builder()
            .initial_max_slice_len(self.max_payload_size)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def request(self, payload: bytes) -> bytes:
        self.open()
        assert self._client is not None
        assert self._node is not None
        assert self._iox2 is not None

        if len(payload) > self.max_payload_size:
            raise ValueError(
                f"request payload is {len(payload)} bytes, max_payload_size={self.max_payload_size}"
            )

        request = self._client.loan_slice_uninit(len(payload))
        pending = write_bytes_to_uninit(request, payload).send()

        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                response = pending.receive()
                if response is not None:
                    try:
                        return slice_payload_to_bytes(response)
                    finally:
                        response.delete()
                self._node.wait(self._iox2.Duration.from_nanos(self.poll_ns))
        finally:
            pending.delete()

        raise TimeoutError(f"timeout waiting for iceoryx2 response from /{self.service_name}/")

    def close(self) -> None:
        for attr in ("_client", "_service", "_node"):
            obj = getattr(self, attr)
            if obj is not None and hasattr(obj, "delete"):
                obj.delete()
            setattr(self, attr, None)


class Iox2RpcServer:
    """Small request-response server using dynamic byte slices."""

    def __init__(
        self,
        service_name: str,
        *,
        max_payload_size: int = 64 * 1024,
        poll_ns: int | None = DEFAULT_POLL_NS,
        poll_ms: int | None = None,
    ) -> None:
        self.service_name = service_name_from_host(service_name)
        self.max_payload_size = max_payload_size
        self.poll_ns = _poll_ns(poll_ns=poll_ns, poll_ms=poll_ms)
        self._iox2: Any | None = None
        self._node: Any | None = None
        self._service: Any | None = None
        self._server: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._server is not None

    def open(self) -> None:
        if self.is_open:
            return
        iox2 = _import_iox2()
        self._iox2 = iox2
        self._node = iox2.NodeBuilder.new().create(_service_type(iox2))
        self._service = (
            self._node.service_builder(iox2.ServiceName.new(self.service_name))
            .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
            .open_or_create()
        )
        self._server = (
            self._service.server_builder()
            .initial_max_slice_len(self.max_payload_size)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def drain_requests(self, handler: Callable[[bytes], bytes]) -> int:
        self.open()
        assert self._node is not None
        assert self._iox2 is not None
        assert self._server is not None

        self._node.wait(self._iox2.Duration.from_nanos(self.poll_ns))
        handled = 0

        while True:
            active_request = self._server.receive()
            if active_request is None:
                break

            try:
                request_bytes = slice_payload_to_bytes(active_request)
                response_bytes = handler(request_bytes)
                if len(response_bytes) > self.max_payload_size:
                    raise ValueError(
                        "response payload is "
                        f"{len(response_bytes)} bytes, max_payload_size={self.max_payload_size}"
                    )
                response = active_request.loan_slice_uninit(len(response_bytes))
                write_bytes_to_uninit(response, response_bytes).send()
                handled += 1
            finally:
                active_request.delete()

        return handled

    def close(self) -> None:
        for attr in ("_server", "_service", "_node"):
            obj = getattr(self, attr)
            if obj is not None and hasattr(obj, "delete"):
                obj.delete()
            setattr(self, attr, None)
