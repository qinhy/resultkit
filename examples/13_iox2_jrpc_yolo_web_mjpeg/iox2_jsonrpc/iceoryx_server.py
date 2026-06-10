from __future__ import annotations

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import cast

from .controller import describe_controller, rpc_endpoint_name, schema_endpoint_name
from .endpoint import ControllerRpcEndpoint
from .iceoryx_helpers import (
    best_effort_cleanup_dead_nodes,
    create_server_with_dead_node_cleanup,
    delete_iox2_object,
    force_remove_request_response_service_if_requested,
    make_attribute_verifier,
    slice_to_bytes,
    write_bytes_to_uninit_slice,
)
from .iceoryx_protocols import (
    ActiveRequest,
    Deletable,
    JsonRpcController,
    NodeLike,
    RequestResponseService,
    Server,
)
from .iceoryx_runtime import U8Slice, iox2


class Iox2JsonRpcServer:
    def __init__(
        self,
        controller: JsonRpcController,
        *,
        poll_ms: int = 10,
        initial_max_slice_len: int = 4096,
    ) -> None:
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)

        self.controller = controller
        self.poll_time = iox2.Duration.from_millis(poll_ms)
        self.endpoint = ControllerRpcEndpoint(controller)

        self.rpc_endpoint = rpc_endpoint_name(controller)
        self.schema_endpoint = schema_endpoint_name(controller)
        self.descriptor = describe_controller(controller)

        self.rpc_service: RequestResponseService | None = None
        self.schema_service: RequestResponseService | None = None
        self.rpc_server: Server | None = None
        self.schema_server: Server | None = None

        self.node: NodeLike = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        best_effort_cleanup_dead_nodes(node=self.node)

        for endpoint in (self.rpc_endpoint, self.schema_endpoint):
            force_remove_request_response_service_if_requested(self.node, endpoint)

        self.rpc_service = self._create_request_response_service(
            self.rpc_endpoint,
            self._rpc_attributes(controller),
        )
        self.schema_service = self._create_request_response_service(
            self.schema_endpoint,
            self._schema_attributes(controller),
        )

        try:
            self.rpc_server = self._create_server(
                self.rpc_service,
                self.rpc_endpoint,
                initial_max_slice_len,
            )
            self.schema_server = self._create_server(
                self.schema_service,
                self.schema_endpoint,
                initial_max_slice_len,
            )
        except Exception:
            self.close()
            raise

    def _base_attributes(self, controller: JsonRpcController, kind: str) -> dict[str, str]:
        return {
            "rpc.protocol": "jsonrpc-2.0",
            "rpc.service": controller.service_name,
            "rpc.controller": controller.controller_name,
            "rpc.kind": kind,
        }

    def _rpc_attributes(self, controller: JsonRpcController) -> dict[str, str]:
        return self._base_attributes(controller, "rpc") | {
            "rpc.schema": self.schema_endpoint,
            "rpc.methods": ",".join(m.jsonrpc_method for m in self.descriptor.methods),
        }

    def _schema_attributes(self, controller: JsonRpcController) -> dict[str, str]:
        return self._base_attributes(controller, "schema")

    def _create_request_response_service(
        self,
        endpoint: str,
        attr_values: Mapping[str, str],
    ) -> RequestResponseService:
        service = (
            self.node.service_builder(iox2.ServiceName.new(endpoint))
            .request_response(U8Slice, U8Slice)
            .max_servers(1)
            .max_clients(32)
            .max_response_buffer_size(4)
            .max_active_requests_per_client(8)
            .open_or_create_with_attributes(make_attribute_verifier(dict(attr_values)))
        )
        best_effort_cleanup_dead_nodes(node=self.node, service=service)
        return service

    def _create_server(
        self,
        service: RequestResponseService,
        endpoint: str,
        initial_max_slice_len: int,
    ) -> Server:
        return create_server_with_dead_node_cleanup(
            service,
            initial_max_slice_len=initial_max_slice_len,
            service_name=endpoint,
            node=self.node,
        )

    def _send_response(self, active_request: ActiveRequest, payload: bytes) -> None:
        response = active_request.loan_slice_uninit(len(payload))
        response_sender = write_bytes_to_uninit_slice(response, payload)
        response_sender.send()

    def _drain_requests(
        self,
        server: Server | None,
        build_response: Callable[[ActiveRequest], bytes],
    ) -> None:
        if server is None:
            return

        while True:
            active_request = server.receive()

            if active_request is None:
                break

            try:
                self._send_response(active_request, build_response(active_request))
            finally:
                active_request.delete()

    def _schema_response(self, _active_request: ActiveRequest) -> bytes:
        return self.descriptor.model_dump_json(indent=2).encode()

    def _rpc_response(self, active_request: ActiveRequest) -> bytes:
        request_bytes = slice_to_bytes(active_request.payload())
        return self.endpoint.handle_bytes(request_bytes)

    def _drain_schema_requests(self) -> None:
        self._drain_requests(self.schema_server, self._schema_response)

    def _drain_rpc_requests(self) -> None:
        self._drain_requests(self.rpc_server, self._rpc_response)

    def close(self) -> None:
        # Explicitly releasing the Server ports on normal shutdown prevents many
        # development-time ExceedsMaxSupportedServers restarts.
        for name in ("schema_server", "rpc_server", "schema_service", "rpc_service"):
            obj = getattr(self, name, None)
            if obj is not None:
                delete_iox2_object(cast(Deletable, obj))
                setattr(self, name, None)

        node = getattr(self, "node", None)
        if node is not None:
            best_effort_cleanup_dead_nodes(node=cast(NodeLike, node))

    def __enter__(self) -> Iox2JsonRpcServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
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


__all__ = ["Iox2JsonRpcServer"]
