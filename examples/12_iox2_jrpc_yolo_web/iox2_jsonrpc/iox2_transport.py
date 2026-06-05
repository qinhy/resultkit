from __future__ import annotations

import importlib
import itertools
import time
from typing import Any

from .core import JsonRpcProcessor
from .packet import JsonRpcPacket, JsonRpcPacketCodec
from .services import JsonRpcServiceDescriptor


def _load_iox2() -> Any:
    try:
        return importlib.import_module("iceoryx2")
    except ImportError as exc:
        raise RuntimeError(
            "The Python module 'iceoryx2' is not installed or not importable. "
            "This repo intentionally does not install/build iceoryx2. "
            "Install iceoryx2 in your environment, then run this program again."
        ) from exc


class Iceoryx2JsonRpcServer:
    def __init__(
        self,
        *,
        service_name: str,
        processor: JsonRpcProcessor,
        max_payload_bytes: int = 64 * 1024,
        wait_seconds: int = 1,
    ) -> None:
        self.service_name = service_name
        self.processor = processor
        self.codec = JsonRpcPacketCodec(max_payload_bytes)
        self.wait_seconds = wait_seconds
        self._running = False

    def serve_forever(self) -> None:
        iox2 = _load_iox2()
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        service = (
            node.service_builder(iox2.ServiceName.new(self.service_name))
            .request_response(JsonRpcPacket, JsonRpcPacket)
            .open_or_create()
        )
        server = service.server_builder().create()
        wait_time = iox2.Duration.from_secs(self.wait_seconds)

        self._running = True
        print(f"[iox2-jsonrpc] server ready: {self.service_name}")

        try:
            while self._running:
                node.wait(wait_time)
                while True:
                    active_request = server.receive()
                    if active_request is None:
                        break

                    request_packet = active_request.payload().contents
                    request_bytes = self.codec.decode(request_packet)
                    response_bytes = self.processor.handle_bytes(request_bytes)

                    # JSON-RPC notifications have no response. iceoryx2 request-response
                    # still needs an answer, so use an empty packet as the transport ACK.
                    if response_bytes is None:
                        response_bytes = b""

                    response_packet = self.codec.encode(
                        response_bytes,
                        correlation_id=int(request_packet.correlation_id),
                    )
                    active_request.send_copy(response_packet)
                    active_request.delete()
        except iox2.NodeWaitFailure:
            print(f"[iox2-jsonrpc] server stopped by node wait failure: {self.service_name}")

    def stop(self) -> None:
        self._running = False


class Iceoryx2JsonRpcClient:
    _ids = itertools.count(1)

    def __init__(
        self,
        service: JsonRpcServiceDescriptor,
        *,
        max_payload_bytes: int = 64 * 1024,
        wait_seconds: int = 1,
    ) -> None:
        self.service = service
        self.codec = JsonRpcPacketCodec(max_payload_bytes)
        self.wait_seconds = wait_seconds
        self._iox2: Any | None = None
        self._node: Any | None = None
        self._client: Any | None = None

    def call(self, request_payload: bytes) -> bytes:
        self._ensure_open()
        assert self._iox2 is not None
        assert self._node is not None
        assert self._client is not None

        correlation_id = next(self._ids)
        request_packet = self.codec.encode(request_payload, correlation_id=correlation_id)
        pending_response = self._client.send_copy(request_packet)

        wait_time = self._iox2.Duration.from_secs(self.wait_seconds)
        deadline = time.monotonic() + self.service.timeout_seconds

        while time.monotonic() < deadline:
            self._node.wait(wait_time)
            while True:
                response = pending_response.receive()
                if response is None:
                    break
                packet = response.payload().contents
                if int(packet.correlation_id) == correlation_id:
                    return self.codec.decode(packet)

        raise TimeoutError(f"Timed out waiting for iceoryx2 service: {self.service.name}")

    def _ensure_open(self) -> None:
        if self._client is not None:
            return

        self._iox2 = _load_iox2()
        self._iox2.set_log_level_from_env_or(self._iox2.LogLevel.Info)
        self._node = self._iox2.NodeBuilder.new().create(self._iox2.ServiceType.Ipc)
        service = (
            self._node.service_builder(self._iox2.ServiceName.new(self.service.iceoryx2_service))
            .request_response(JsonRpcPacket, JsonRpcPacket)
            .open_or_create()
        )
        self._client = service.client_builder().create()


class Iceoryx2JsonRpcClientPool:
    def __init__(self) -> None:
        self._clients: dict[str, Iceoryx2JsonRpcClient] = {}

    def get(self, service: JsonRpcServiceDescriptor) -> Iceoryx2JsonRpcClient:
        if service.name not in self._clients:
            self._clients[service.name] = Iceoryx2JsonRpcClient(service)
        return self._clients[service.name]
