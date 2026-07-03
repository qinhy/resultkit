from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from .iceoryx_helpers import attribute_set_to_dict, slice_to_bytes, to_str, write_bytes_to_uninit_slice
from .iceoryx_protocols import (
    CatalogEntry,
    Client,
    JsonObject,
    NodeLike,
    RequestResponseService,
    ServiceDetails,
)
from .iceoryx_runtime import U8Slice, iox2
from .models import JsonRpcRequest, JsonRpcResponse, RpcServiceDescriptor


@dataclass
class DiscoveredRpcService:
    service_name: str
    controller_name: str
    rpc_endpoint: str
    schema_endpoint: str
    methods: list[str]
    attributes: dict[str, str]
    descriptor: RpcServiceDescriptor | None = None
    rpc_service: RequestResponseService | None = None
    rpc_client: Client | None = None


class Iox2RpcRegistry:
    def __init__(self, *, node: NodeLike, services: list[DiscoveredRpcService]) -> None:
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

        node: NodeLike = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        _ = node.try_cleanup_dead_nodes(iox2.ServiceType.Ipc, node.config)
        deadline = time.monotonic() + timeout_s
        discovered: dict[str, DiscoveredRpcService] = {}

        while time.monotonic() < deadline:
            services = iox2.Service.list(
                iox2.config.global_config(),
                iox2.ServiceType.Ipc,
            )

            for details in cast(Iterable[ServiceDetails], services):
                try:
                    if details.messaging_pattern() != iox2.MessagingPattern.RequestResponse:
                        continue

                    attrs = attribute_set_to_dict(details.attributes())

                    if attrs.get("rpc.protocol") != "jsonrpc-2.0":
                        continue
                    if attrs.get("rpc.kind") != "rpc":
                        continue

                    rpc_endpoint = to_str(details.name)
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
                _ = schema_service.try_cleanup_dead_nodes()

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
                _ = rpc_service.try_cleanup_dead_nodes()

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
        client: Client,
        payload: bytes,
        timeout_s: float,
    ) -> bytes:
        request = client.loan_slice_uninit(len(payload))
        request_sender = write_bytes_to_uninit_slice(request, payload)
        pending_response = request_sender.send()
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            while True:
                response = pending_response.receive()

                if response is None:
                    break

                return slice_to_bytes(response.payload())

            self.node.wait(iox2.Duration.from_millis(10))

        raise TimeoutError("Timed out waiting for iceoryx2 response")

    def catalog(self) -> list[CatalogEntry]:
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
        params: JsonObject | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        if service.rpc_client is None:
            raise RuntimeError(f"Service {service.rpc_endpoint} has no opened RPC client")

        self.request_counter += 1
        request = JsonRpcRequest(
            id=self.request_counter,
            method=method,
            params=params if params is not None else {},
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
        params: JsonObject | None = None,
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
        params: JsonObject | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        service = self.find_by_endpoint(rpc_endpoint)
        return self.call(service, method, params, timeout_s=timeout_s)


__all__ = ["DiscoveredRpcService", "Iox2RpcRegistry"]
