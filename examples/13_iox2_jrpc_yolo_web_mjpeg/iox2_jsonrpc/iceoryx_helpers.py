from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from typing import TypeVar, cast

from .iceoryx_protocols import (
    Deletable,
    NodeLike,
    RequestResponseService,
    Server,
    SliceLike,
    UninitSlice,
)
from .iceoryx_runtime import iox2

_T = TypeVar("_T")
_TInitializedSlice = TypeVar("_TInitializedSlice")


def slice_to_bytes(payload: SliceLike) -> bytes:
    return bytes(int(payload[i]) for i in range(payload.len()))


def write_bytes_to_uninit_slice(
    uninit_slice: UninitSlice[_TInitializedSlice],
    payload: bytes,
) -> _TInitializedSlice:
    dst = uninit_slice.payload()
    for i, byte in enumerate(payload):
        dst[i] = byte
    return uninit_slice.assume_init()


def to_str(value: object | Callable[[], object]) -> str:
    if callable(value):
        value = value()

    to_string = getattr(value, "to_string", None)
    if callable(to_string):
        return str(to_string())

    as_str = getattr(value, "as_str", None)
    if as_str is not None:
        return str(as_str() if callable(as_str) else as_str)

    return str(value)


def call_or_value(value: _T | Callable[[], _T]) -> _T:
    return value() if callable(value) else value


def make_attributes(values: dict[str, str]) -> object:
    spec = iox2.AttributeSpecifier.new()

    for key, value in values.items():
        updated = spec.define(
            iox2.AttributeKey.new(key),
            iox2.AttributeValue.new(value),
        )
        if updated is not None:
            spec = updated

    return spec


def make_attribute_verifier(values: dict[str, str]) -> object:
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


def best_effort_cleanup_dead_nodes(
    *,
    node: NodeLike | None = None,
    service: RequestResponseService | None = None,
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


def describe_service_nodes(service: RequestResponseService) -> str:
    try:
        nodes_accessor = cast(object | Callable[[], object], getattr(service, "nodes"))
        nodes = call_or_value(nodes_accessor)
        return repr(nodes)
    except Exception as exc:
        return f"<could not inspect service nodes: {exc}>"


def create_server_with_dead_node_cleanup(
    service: RequestResponseService,
    *,
    initial_max_slice_len: int,
    service_name: str,
    node: NodeLike | None = None,
    max_attempts: int = 20,
    cleanup_timeout_ms: int = 100,
) -> Server:
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

            best_effort_cleanup_dead_nodes(
                node=node,
                service=service,
                timeout_ms=cleanup_timeout_ms,
            )

            if attempt + 1 >= max_attempts:
                break

            time.sleep(min(0.05 * (attempt + 1), 0.5))

    assert last_exc is not None
    nodes = describe_service_nodes(service)
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


def force_remove_request_response_service_if_requested(node: NodeLike, service_name: str) -> None:
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


def delete_iox2_object(obj: Deletable) -> None:
    delete = getattr(obj, "delete", None)
    if callable(delete):
        try:
            delete()
        except Exception:
            pass


def attribute_set_to_dict(attribute_set: object) -> dict[str, str]:
    result: dict[str, str] = {}

    raw_values = cast(Iterable[object] | Callable[[], Iterable[object]], getattr(attribute_set, "values"))
    attributes = call_or_value(raw_values)

    for attribute in attributes:
        key = to_str(cast(object | Callable[[], object], getattr(attribute, "key")))
        value = to_str(cast(object | Callable[[], object], getattr(attribute, "value")))
        result[key] = value

    return result


__all__ = [
    "attribute_set_to_dict",
    "best_effort_cleanup_dead_nodes",
    "call_or_value",
    "create_server_with_dead_node_cleanup",
    "delete_iox2_object",
    "describe_service_nodes",
    "force_remove_request_response_service_if_requested",
    "make_attribute_verifier",
    "make_attributes",
    "slice_to_bytes",
    "to_str",
    "write_bytes_to_uninit_slice",
]
