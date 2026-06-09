from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass
from typing import Any

from .controller import describe_controller, rpc_endpoint_name, schema_endpoint_name
from .endpoint import ControllerRpcEndpoint
from .models import JsonRpcRequest, JsonRpcResponse, RpcServiceDescriptor

try:
    import iceoryx2 as iox2
except ImportError as exc:  # pragma: no cover - depends on optional package
    raise ImportError(
        "Missing optional dependency: iceoryx2. Install with: pip install 'iox2-jsonrpc[iox2]'"
    ) from exc

U8Slice = iox2.Slice[ctypes.c_uint8]


def slice_to_bytes(payload: Any) -> bytes:
    return bytes(int(payload[i]) for i in range(payload.len()))


def write_bytes_to_uninit_slice(uninit_slice: Any, payload: bytes) -> Any:
    dst = uninit_slice.payload()
    for i, byte in enumerate(payload):
        dst[i] = byte
    return uninit_slice.assume_init()


def to_str(value: Any) -> str:
    if callable(value):
        value = value()

    if hasattr(value, "to_string"):
        return value.to_string()

    if hasattr(value, "as_str"):
        attr = value.as_str
        return attr() if callable(attr) else attr

    return str(value)


def call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def make_attributes(values: dict[str, str]) -> Any:
    spec = iox2.AttributeSpecifier.new()

    for key, value in values.items():
        spec = spec.define(
            iox2.AttributeKey.new(key),
            iox2.AttributeValue.new(value),
        )

    return spec


def attribute_set_to_dict(attribute_set: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    raw_values = getattr(attribute_set, "values")
    attributes = call_or_value(raw_values)

    for attribute in attributes:
        key = to_str(getattr(attribute, "key"))
        value = to_str(getattr(attribute, "value"))
        result[key] = value

    return result


class Iox2JsonRpcServer:
    def __init__(
        self,
        controller: Any,
        *,
        poll_ms: int = 10,
        initial_max_slice_len: int = 4096,
    ):
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

        self.controller = controller
        self.poll_time = iox2.Duration.from_millis(poll_ms)
        self.endpoint = ControllerRpcEndpoint(controller)

        self.rpc_endpoint = rpc_endpoint_name(controller)
        self.schema_endpoint = schema_endpoint_name(controller)
        self.descriptor = describe_controller(controller)

        try:
            iox2.Node.try_cleanup_dead_nodes(
                iox2.ServiceType.Ipc,
                iox2.config.global_config(),
            )
        except Exception:
            pass

        self.node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

        rpc_attrs = make_attributes(
            {
                "rpc.protocol": "jsonrpc-2.0",
                "rpc.service": controller.service_name,
                "rpc.controller": controller.controller_name,
                "rpc.kind": "rpc",
                "rpc.schema": self.schema_endpoint,
                "rpc.methods": ",".join(m.jsonrpc_method for m in self.descriptor.methods),
            }
        )

        schema_attrs = make_attributes(
            {
                "rpc.protocol": "jsonrpc-2.0",
                "rpc.service": controller.service_name,
                "rpc.controller": controller.controller_name,
                "rpc.kind": "schema",
            }
        )

        self.rpc_service = (
            self.node.service_builder(iox2.ServiceName.new(self.rpc_endpoint))
            .request_response(U8Slice, U8Slice)
            .max_servers(1)
            .max_clients(32)
            .max_response_buffer_size(4)
            .max_active_requests_per_client(8)
            .create_with_attributes(rpc_attrs)
        )

        self.schema_service = (
            self.node.service_builder(iox2.ServiceName.new(self.schema_endpoint))
            .request_response(U8Slice, U8Slice)
            .max_servers(1)
            .max_clients(32)
            .max_response_buffer_size(4)
            .max_active_requests_per_client(8)
            .create_with_attributes(schema_attrs)
        )

        self.rpc_server = (
            self.rpc_service.server_builder()
            .initial_max_slice_len(initial_max_slice_len)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

        self.schema_server = (
            self.schema_service.server_builder()
            .initial_max_slice_len(initial_max_slice_len)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def _send_response(self, active_request: Any, payload: bytes) -> None:
        response = active_request.loan_slice_uninit(len(payload))
        response = write_bytes_to_uninit_slice(response, payload)
        response.send()

    def _drain_schema_requests(self) -> None:
        while True:
            active_request = self.schema_server.receive()

            if active_request is None:
                break

            try:
                payload = self.descriptor.model_dump_json(indent=2).encode()
                self._send_response(active_request, payload)
            finally:
                active_request.delete()

    def _drain_rpc_requests(self) -> None:
        while True:
            active_request = self.rpc_server.receive()

            if active_request is None:
                break

            try:
                request_bytes = slice_to_bytes(active_request.payload())
                response_bytes = self.endpoint.handle_bytes(request_bytes)
                self._send_response(active_request, response_bytes)
            finally:
                active_request.delete()

    def run_forever(self) -> None:
        print("\n=== iceoryx2 JSON-RPC server started ===")
        print(f"RPC endpoint:    {self.rpc_endpoint}")
        print(f"Schema endpoint: {self.schema_endpoint}")
        print("\n=== Descriptor ===")
        print(self.descriptor.model_dump_json(indent=2))
        print("\nWaiting for requests...")

        try:
            while True:
                self.node.wait(self.poll_time)
                self._drain_schema_requests()
                self._drain_rpc_requests()
        except iox2.NodeWaitFailure:
            print("Server stopped.")


@dataclass
class DiscoveredRpcService:
    service_name: str
    controller_name: str
    rpc_endpoint: str
    schema_endpoint: str
    methods: list[str]
    attributes: dict[str, str]
    descriptor: RpcServiceDescriptor | None = None
    rpc_service: Any | None = None
    rpc_client: Any | None = None


class Iox2RpcRegistry:
    def __init__(self, *, node: Any, services: list[DiscoveredRpcService]):
        self.node = node
        self.services = services
        self.request_counter = 0

    @classmethod
    def discover_all(
        cls,
        *,
        timeout_s: float = 2.0,
        initial_max_slice_len: int = 4096,
        load_descriptors: bool = True,
    ) -> Iox2RpcRegistry:
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        deadline = time.monotonic() + timeout_s
        discovered: dict[str, DiscoveredRpcService] = {}

        while time.monotonic() < deadline:
            services = iox2.Service.list(
                iox2.config.global_config(),
                iox2.ServiceType.Ipc,
            )

            for details in services:
                try:
                    if details.messaging_pattern() != iox2.MessagingPattern.RequestResponse:
                        continue

                    attrs = attribute_set_to_dict(details.attributes())

                    if attrs.get("rpc.protocol") != "jsonrpc-2.0":
                        continue
                    if attrs.get("rpc.kind") != "rpc":
                        continue

                    rpc_endpoint = to_str(details.name())
                    schema_endpoint = attrs.get("rpc.schema")

                    if not schema_endpoint:
                        continue

                    discovered[rpc_endpoint] = DiscoveredRpcService(
                        service_name=attrs.get("rpc.service", ""),
                        controller_name=attrs.get("rpc.controller", ""),
                        rpc_endpoint=rpc_endpoint,
                        schema_endpoint=schema_endpoint,
                        methods=[item for item in attrs.get("rpc.methods", "").split(",") if item],
                        attributes=attrs,
                    )
                except Exception as exc:
                    print(f"Skipping service during discovery: {exc}")

            if discovered:
                break

            node.wait(iox2.Duration.from_millis(100))

        registry = cls(node=node, services=list(discovered.values()))

        if load_descriptors:
            registry.load_all_descriptors(
                timeout_s=timeout_s,
                initial_max_slice_len=initial_max_slice_len,
            )

        return registry

    def load_all_descriptors(
        self,
        *,
        timeout_s: float = 2.0,
        initial_max_slice_len: int = 4096,
    ) -> None:
        for discovered in self.services:
            try:
                schema_service = (
                    self.node.service_builder(iox2.ServiceName.new(discovered.schema_endpoint))
                    .request_response(U8Slice, U8Slice)
                    .open()
                )

                schema_client = (
                    schema_service.client_builder()
                    .initial_max_slice_len(initial_max_slice_len)
                    .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
                    .create()
                )

                descriptor_bytes = self._request_bytes(
                    client=schema_client,
                    payload=b"{}",
                    timeout_s=timeout_s,
                )

                descriptor = RpcServiceDescriptor.model_validate_json(descriptor_bytes)

                rpc_service = (
                    self.node.service_builder(iox2.ServiceName.new(descriptor.rpc_endpoint))
                    .request_response(U8Slice, U8Slice)
                    .open()
                )

                rpc_client = (
                    rpc_service.client_builder()
                    .initial_max_slice_len(initial_max_slice_len)
                    .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
                    .create()
                )

                discovered.descriptor = descriptor
                discovered.rpc_service = rpc_service
                discovered.rpc_client = rpc_client
                discovered.methods = [method.jsonrpc_method for method in descriptor.methods]
            except Exception as exc:
                print(f"Could not load descriptor for {discovered.rpc_endpoint}: {exc}")

    def _request_bytes(
        self,
        *,
        client: Any,
        payload: bytes,
        timeout_s: float,
    ) -> bytes:
        request = client.loan_slice_uninit(len(payload))
        request = write_bytes_to_uninit_slice(request, payload)
        pending_response = request.send()
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            while True:
                response = pending_response.receive()

                if response is None:
                    break

                return slice_to_bytes(response.payload())

            self.node.wait(iox2.Duration.from_millis(10))

        raise TimeoutError("Timed out waiting for iceoryx2 response")

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "service": service.service_name,
                "controller": service.controller_name,
                "rpc_endpoint": service.rpc_endpoint,
                "schema_endpoint": service.schema_endpoint,
                "methods": service.methods,
                "attributes": service.attributes,
            }
            for service in self.services
        ]

    def print_catalog(self) -> None:
        print(json.dumps(self.catalog(), indent=2))

    def find_by_method(self, method: str) -> list[DiscoveredRpcService]:
        return [service for service in self.services if method in service.methods]

    def find_by_endpoint(self, rpc_endpoint: str) -> DiscoveredRpcService:
        for service in self.services:
            if service.rpc_endpoint == rpc_endpoint:
                return service

        raise RuntimeError(f"No RPC service found at endpoint: {rpc_endpoint}")

    def call(
        self,
        service: DiscoveredRpcService,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        if service.rpc_client is None:
            raise RuntimeError(f"Service {service.rpc_endpoint} has no opened RPC client")

        self.request_counter += 1
        request = JsonRpcRequest(
            id=self.request_counter,
            method=method,
            params=params or {},
        )

        raw_response = self._request_bytes(
            client=service.rpc_client,
            payload=request.model_dump_json().encode(),
            timeout_s=timeout_s,
        )

        return JsonRpcResponse.model_validate_json(raw_response)

    def call_unique(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        matches = self.find_by_method(method)

        if not matches:
            raise RuntimeError(f"No discovered service provides method: {method}")

        if len(matches) > 1:
            endpoints = [service.rpc_endpoint for service in matches]
            raise RuntimeError(f"Method {method!r} is ambiguous. Matching endpoints: {endpoints}")

        return self.call(matches[0], method, params, timeout_s=timeout_s)

    def call_endpoint(
        self,
        rpc_endpoint: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        service = self.find_by_endpoint(rpc_endpoint)
        return self.call(service, method, params, timeout_s=timeout_s)
