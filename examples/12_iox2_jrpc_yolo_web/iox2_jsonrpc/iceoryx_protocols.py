from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Self, TypeAlias, TypeVar

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
CatalogEntry: TypeAlias = dict[str, str | list[str] | dict[str, str]]

_T = TypeVar("_T")
_TInitializedSlice = TypeVar("_TInitializedSlice")
_TInitializedSlice_co = TypeVar("_TInitializedSlice_co", covariant=True)


class ByteWritable(Protocol):
    def __setitem__(self, index: int, value: int) -> None: ...


class SliceLike(Protocol):
    def __getitem__(self, index: int) -> int: ...

    def len(self) -> int: ...


class UninitSlice(Protocol[_TInitializedSlice_co]):
    def payload(self) -> ByteWritable: ...

    def assume_init(self) -> _TInitializedSlice_co: ...


class PendingResponse(Protocol):
    def receive(self) -> Response | None: ...


class RequestSender(Protocol):
    def send(self) -> PendingResponse: ...


class ResponseSender(Protocol):
    def send(self) -> object: ...


class Response(Protocol):
    def payload(self) -> SliceLike: ...


class ActiveRequest(Protocol):
    def loan_slice_uninit(self, length: int) -> UninitSlice[ResponseSender]: ...

    def payload(self) -> SliceLike: ...

    def delete(self) -> object: ...


class Server(Protocol):
    def receive(self) -> ActiveRequest | None: ...


class Client(Protocol):
    def loan_slice_uninit(self, length: int) -> UninitSlice[RequestSender]: ...


class ServerBuilder(Protocol):
    def initial_max_slice_len(self, length: int) -> Self: ...

    def allocation_strategy(self, strategy: object) -> Self: ...

    def create(self) -> Server: ...


class ClientBuilder(Protocol):
    def initial_max_slice_len(self, length: int) -> Self: ...

    def allocation_strategy(self, strategy: object) -> Self: ...

    def create(self) -> Client: ...


class RequestResponseService(Protocol):
    def server_builder(self) -> ServerBuilder: ...

    def client_builder(self) -> ClientBuilder: ...

    def try_cleanup_dead_nodes(self, *args: object) -> object: ...


class RequestResponseBuilder(Protocol):
    def max_servers(self, value: int) -> Self: ...

    def max_clients(self, value: int) -> Self: ...

    def max_response_buffer_size(self, value: int) -> Self: ...

    def max_active_requests_per_client(self, value: int) -> Self: ...

    def open_or_create_with_attributes(self, attributes: object) -> RequestResponseService: ...

    def open(self) -> RequestResponseService: ...


class ServiceBuilder(Protocol):
    def request_response(self, request_type: object, response_type: object) -> RequestResponseBuilder: ...


class NodeLike(Protocol):
    config: object

    def service_builder(self, service_name: object) -> ServiceBuilder: ...

    def force_remove_service(self, service_name: object, messaging_pattern: object) -> object: ...

    def wait(self, duration: object) -> object: ...

    def try_cleanup_dead_nodes(self, *args: object) -> object: ...


class ServiceDetails(Protocol):
    def messaging_pattern(self) -> object: ...

    def attributes(self) -> object: ...

    def name(self) -> object: ...


class JsonRpcController(Protocol):
    service_name: str
    controller_name: str


class Deletable(Protocol):
    def delete(self) -> object: ...


__all__ = [
    "ActiveRequest",
    "ByteWritable",
    "Callable",
    "CatalogEntry",
    "Client",
    "ClientBuilder",
    "Deletable",
    "JsonObject",
    "JsonRpcController",
    "JsonValue",
    "NodeLike",
    "PendingResponse",
    "RequestResponseBuilder",
    "RequestResponseService",
    "RequestSender",
    "Response",
    "ResponseSender",
    "Server",
    "ServerBuilder",
    "ServiceBuilder",
    "ServiceDetails",
    "SliceLike",
    "UninitSlice",
    "_T",
    "_TInitializedSlice",
]
