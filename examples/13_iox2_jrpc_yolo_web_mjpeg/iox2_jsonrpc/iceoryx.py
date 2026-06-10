from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .controller import describe_controller, rpc_endpoint_name, schema_endpoint_name
from .endpoint import ControllerRpcEndpoint
from .models import JsonRpcRequest, JsonRpcResponse, RpcServiceDescriptor
os.environ.setdefault("IOX2_JSONRPC_FORCE_REMOVE_SERVICES","1")
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
        updated = spec.define(
            iox2.AttributeKey.new(key),
            iox2.AttributeValue.new(value),
        )
        if updated is not None:
            spec = updated

    return spec


def make_attribute_verifier(values: dict[str, str]) -> Any:
    verifier = iox2.AttributeVerifier.new()

    for key, value in values.items():
        updated = verifier.require(
            iox2.AttributeKey.new(key),
            iox2.AttributeValue.new(value),
        )
        if updated is not None:
            verifier = updated

    return verifier


def _is_exceeds_max_supported_servers(exc: Exception) -> bool:
    return "ExceedsMaxSupportedServers" in f"{type(exc).__name__}: {exc}"


def _best_effort_cleanup_dead_nodes(
    *,
    node: Any | None = None,
    service: Any | None = None,
    timeout_ms: int = 100,
) -> None:
    timeout = iox2.Duration.from_millis(timeout_ms)

    for target in (service, node):
        if target is None:
            continue

        blocking_cleanup = getattr(target, "blocking_cleanup_dead_nodes", None)
        if callable(blocking_cleanup):
            try:
                blocking_cleanup(timeout)
                continue
            except Exception:
                pass

        try_cleanup = getattr(target, "try_cleanup_dead_nodes", None)
        if callable(try_cleanup):
            try:
                try_cleanup()
            except TypeError:
                # Older/alternate Python bindings expose Node.try_cleanup_dead_nodes
                # with service_type and config arguments.
                if node is not None:
                    try:
                        try_cleanup(iox2.ServiceType.Ipc, node.config)
                    except Exception:
                        pass
            except Exception:
                pass


def _describe_service_nodes(service: Any) -> str:
    try:
        nodes = call_or_value(getattr(service, "nodes"))
        return repr(nodes)
    except Exception as exc:
        return f"<could not inspect service nodes: {exc}>"


def create_server_with_dead_node_cleanup(
    service: Any,
    *,
    initial_max_slice_len: int,
    service_name: str,
    node: Any | None = None,
    max_attempts: int = 20,
    cleanup_timeout_ms: int = 100,
) -> Any:
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return (
                service.server_builder()
                .initial_max_slice_len(initial_max_slice_len)
                .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
                .create()
            )
        except Exception as exc:
            last_exc = exc

            if not _is_exceeds_max_supported_servers(exc):
                raise

            _best_effort_cleanup_dead_nodes(
                node=node,
                service=service,
                timeout_ms=cleanup_timeout_ms,
            )

            if attempt + 1 >= max_attempts:
                break

            time.sleep(min(0.05 * (attempt + 1), 0.5))

    assert last_exc is not None
    nodes = _describe_service_nodes(service)
    raise RuntimeError(
        "Could not create iceoryx2 server port for "
        f"{service_name!r}: {last_exc}.\n"
        "iceoryx2 still counts an existing Server port against max_servers(1). "
        "This usually means another server process is still alive, or a crashed process "
        "is still visible to the OS and cannot be cleaned by iceoryx2 yet.\n"
        f"Known nodes for this service: {nodes}\n"
        "Stop all processes using this endpoint and retry. During local development only, "
        "you can also set IOX2_JSONRPC_FORCE_REMOVE_SERVICES=1 before starting the server "
        "to force-remove the RPC/schema services and recreate them."
    ) from last_exc


def force_remove_request_response_service_if_requested(node: Any, service_name: str) -> None:
    if os.environ.get("IOX2_JSONRPC_FORCE_REMOVE_SERVICES") != "1":
        return

    try:
        node.force_remove_service(
            iox2.ServiceName.new(service_name),
            iox2.MessagingPattern.RequestResponse,
        )
        print(f"Force-removed stale iceoryx2 service: {service_name}")
    except Exception as exc:
        # It is fine if the service does not exist. If it does exist but cannot be
        # removed, opening/creating it below will surface a more specific error.
        print(f"Could not force-remove iceoryx2 service {service_name!r}: {exc}")


def delete_iox2_object(obj: Any) -> None:
    delete = getattr(obj, "delete", None)
    if callable(delete):
        try:
            delete()
        except Exception:
            pass


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

        self.node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        _best_effort_cleanup_dead_nodes(node=self.node)

        force_remove_request_response_service_if_requested(self.node, self.rpc_endpoint)
        force_remove_request_response_service_if_requested(self.node, self.schema_endpoint)

        rpc_attr_values = {
            "rpc.protocol": "jsonrpc-2.0",
            "rpc.service": controller.service_name,
            "rpc.controller": controller.controller_name,
            "rpc.kind": "rpc",
            "rpc.schema": self.schema_endpoint,
            "rpc.methods": ",".join(m.jsonrpc_method for m in self.descriptor.methods),
        }

        schema_attr_values = {
            "rpc.protocol": "jsonrpc-2.0",
            "rpc.service": controller.service_name,
            "rpc.controller": controller.controller_name,
            "rpc.kind": "schema",
        }

        rpc_attr_verifier = make_attribute_verifier(rpc_attr_values)
        schema_attr_verifier = make_attribute_verifier(schema_attr_values)

        self.rpc_service = (
            self.node.service_builder(iox2.ServiceName.new(self.rpc_endpoint))
            .request_response(U8Slice, U8Slice)
            .max_servers(1)
            .max_clients(32)
            .max_response_buffer_size(4)
            .max_active_requests_per_client(8)
            .open_or_create_with_attributes(rpc_attr_verifier)
        )
        _best_effort_cleanup_dead_nodes(node=self.node, service=self.rpc_service)

        self.schema_service = (
            self.node.service_builder(iox2.ServiceName.new(self.schema_endpoint))
            .request_response(U8Slice, U8Slice)
            .max_servers(1)
            .max_clients(32)
            .max_response_buffer_size(4)
            .max_active_requests_per_client(8)
            .open_or_create_with_attributes(schema_attr_verifier)
        )
        _best_effort_cleanup_dead_nodes(node=self.node, service=self.schema_service)

        try:
            self.rpc_server = create_server_with_dead_node_cleanup(
                self.rpc_service,
                initial_max_slice_len=initial_max_slice_len,
                service_name=self.rpc_endpoint,
                node=self.node,
            )

            self.schema_server = create_server_with_dead_node_cleanup(
                self.schema_service,
                initial_max_slice_len=initial_max_slice_len,
                service_name=self.schema_endpoint,
                node=self.node,
            )
        except Exception:
            self.close()
            raise

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

    def close(self) -> None:
        # Explicitly releasing the Server ports on normal shutdown prevents many
        # development-time ExceedsMaxSupportedServers restarts.
        for name in ("schema_server", "rpc_server", "schema_service", "rpc_service"):
            obj = getattr(self, name, None)
            if obj is not None:
                delete_iox2_object(obj)
                setattr(self, name, None)

        node = getattr(self, "node", None)
        if node is not None:
            _best_effort_cleanup_dead_nodes(node=node)

    def __enter__(self) -> Iox2JsonRpcServer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

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
        except (KeyboardInterrupt, iox2.NodeWaitFailure):
            print("Server stopped.")
        finally:
            self.close()


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
        state = node.try_cleanup_dead_nodes(iox2.ServiceType.Ipc,node.config)
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
                state = schema_service.try_cleanup_dead_nodes()

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
                state = rpc_service.try_cleanup_dead_nodes()

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
