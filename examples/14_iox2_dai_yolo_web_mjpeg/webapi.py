from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from inspect import Parameter, Signature
from threading import RLock
from typing import Any, TYPE_CHECKING

from iox2_jsonrpc.endpoint import INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR
from iox2_jsonrpc.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    RpcMethodDescriptor,
    RpcModel,
    RpcServiceDescriptor,
)

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from fastapi import FastAPI
    from fastapi.routing import APIRouter


class ControllerRouteInfo(RpcModel):
    """HTTP routes exposed for one discovered controller."""

    service: str
    controller: str
    rpc_endpoint: str
    schema_endpoint: str
    method_routes: list[str]


class ControllerApiInfo(RpcModel):
    """Summary returned by the auto-discovery controller HTTP API root."""

    controllers: list[ControllerRouteInfo]
    routes: list[str]


class DynamicGatewayStatus(RpcModel):
    """Runtime status for the dynamic RPC gateway."""

    ok: bool
    refresh_count: int
    service_count: int
    last_error: str | None = None


@dataclass
class DynamicRpcGateway:
    """Discovery-backed JSON-RPC gateway.

    The gateway never registers concrete controller instances in the FastAPI
    process. It periodically calls Iox2RpcRegistry.discover_all(), stores the
    latest registry snapshot, and forwards HTTP calls to whichever RPC services
    are currently discovered.

    Tests or alternate transports can provide registry_factory. The returned
    object only needs the same small surface used here: services, catalog(),
    call(), and call_unique().
    """

    timeout_s: float = 0.2
    initial_max_slice_len: int = 4096
    load_descriptors: bool = True
    registry_factory: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        self._lock = RLock()
        self._registry: Any | None = None
        self._last_error: str | None = None
        self._refresh_count = 0

    @property
    def refresh_count(self) -> int:
        with self._lock:
            return self._refresh_count

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def status(self) -> DynamicGatewayStatus:
        with self._lock:
            service_count = len(getattr(self._registry, "services", []) or [])
            return DynamicGatewayStatus(
                ok=self._last_error is None,
                refresh_count=self._refresh_count,
                service_count=service_count,
                last_error=self._last_error,
            )

    def _default_registry_factory(self) -> Any:
        # Import lazily so importing iox2_jsonrpc.fastapi does not require iceoryx2.
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

        return Iox2RpcRegistry.discover_all(
            timeout_s=self.timeout_s,
            initial_max_slice_len=self.initial_max_slice_len,
            load_descriptors=self.load_descriptors,
        )

    def refresh(self) -> DynamicGatewayStatus:
        """Rediscover RPC services and atomically replace the active registry."""

        factory = self.registry_factory or self._default_registry_factory

        try:
            registry = factory()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._refresh_count += 1
            return self.status()

        with self._lock:
            self._registry = registry
            self._last_error = None
            self._refresh_count += 1

        return self.status()

    def _registry_snapshot(self) -> Any | None:
        with self._lock:
            return self._registry

    def _services_snapshot(self) -> list[Any]:
        registry = self._registry_snapshot()
        if registry is None:
            return []
        return list(getattr(registry, "services", []) or [])

    def catalog(self) -> list[dict[str, Any]]:
        registry = self._registry_snapshot()
        if registry is None:
            return []

        catalog_fn = getattr(registry, "catalog", None)
        if callable(catalog_fn):
            return catalog_fn()

        return [self._service_summary(service) for service in self._services_snapshot()]

    def descriptors(self) -> list[RpcServiceDescriptor]:
        descriptors: list[RpcServiceDescriptor] = []

        for service in self._services_snapshot():
            descriptor = getattr(service, "descriptor", None)
            if isinstance(descriptor, RpcServiceDescriptor):
                descriptors.append(descriptor)

        return descriptors

    def describe_all(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for service in self._services_snapshot():
            key = self._service_key(service)
            descriptor = getattr(service, "descriptor", None)

            if isinstance(descriptor, RpcServiceDescriptor):
                result[key] = descriptor.model_dump(mode="json")
            else:
                result[key] = self._service_summary(service)

        return result

    def controllers(self) -> list[ControllerRouteInfo]:
        return [self._route_info_for_service(service) for service in self._services_snapshot()]

    def _service_key(self, service: Any) -> str:
        service_name = str(getattr(service, "service_name", ""))
        controller_name = str(getattr(service, "controller_name", ""))
        if service_name:
            return f"{service_name}/{controller_name}"
        return controller_name

    def _service_summary(self, service: Any) -> dict[str, Any]:
        return {
            "service": str(getattr(service, "service_name", "")),
            "controller": str(getattr(service, "controller_name", "")),
            "rpc_endpoint": str(getattr(service, "rpc_endpoint", "")),
            "schema_endpoint": str(getattr(service, "schema_endpoint", "")),
            "methods": list(getattr(service, "methods", []) or []),
            "attributes": dict(getattr(service, "attributes", {}) or {}),
        }

    def _route_info_for_service(self, service: Any) -> ControllerRouteInfo:
        controller = str(getattr(service, "controller_name", ""))
        descriptor = getattr(service, "descriptor", None)

        if isinstance(descriptor, RpcServiceDescriptor):
            local_methods = [method.name for method in descriptor.methods]
        else:
            local_methods = []
            for method in getattr(service, "methods", []) or []:
                method = str(method)
                local_methods.append(method.split(".", 1)[1] if "." in method else method)

        return ControllerRouteInfo(
            service=str(getattr(service, "service_name", "")),
            controller=controller,
            rpc_endpoint=f"POST /controllers/{controller}/rpc",
            schema_endpoint=f"GET /controllers/{controller}",
            method_routes=[
                route
                for method in local_methods
                for route in (
                    f"GET /controllers/{controller}/{method}",
                    f"POST /controllers/{controller}/{method}",
                )
            ],
        )

    def _matching_services(self, controller: str, service_name: str | None = None) -> list[Any]:
        matches: list[Any] = []

        for service in self._services_snapshot():
            if str(getattr(service, "controller_name", "")) != controller:
                continue
            if service_name is not None and str(getattr(service, "service_name", "")) != service_name:
                continue
            matches.append(service)

        return matches

    def _find_service(self, controller: str, service_name: str | None = None) -> Any | JsonRpcResponse:
        matches = self._matching_services(controller, service_name=service_name)

        if not matches:
            detail = f"No discovered RPC controller named {controller!r}"
            if service_name is not None:
                detail += f" for service {service_name!r}"
            return JsonRpcResponse.fail(id=None, code=METHOD_NOT_FOUND, message=detail)

        if len(matches) > 1:
            choices = [self._service_key(service) for service in matches]
            return JsonRpcResponse.fail(
                id=None,
                code=METHOD_NOT_FOUND,
                message=(
                    f"Controller {controller!r} is ambiguous. Add ?service=... "
                    f"to choose one of: {choices}"
                ),
            )

        return matches[0]

    def _call_service(
        self,
        *,
        service: Any,
        request_id: Any,
        method: str,
        params: dict[str, Any] | None,
        timeout_s: float,
    ) -> JsonRpcResponse:
        registry = self._registry_snapshot()
        if registry is None:
            return JsonRpcResponse.fail(
                id=request_id,
                code=METHOD_NOT_FOUND,
                message="No RPC registry has been loaded yet.",
            )

        try:
            response = registry.call(
                service,
                method,
                params or {},
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return JsonRpcResponse.fail(
                id=request_id,
                code=INTERNAL_ERROR,
                message="Gateway call failed",
                data=str(exc),
            )

        return self._rewrite_response_id(response, request_id)

    def _rewrite_response_id(self, response: JsonRpcResponse, request_id: Any) -> JsonRpcResponse:
        if response.error is not None:
            return JsonRpcResponse(id=request_id, error=response.error)
        return JsonRpcResponse(id=request_id, result=response.result)

    def call_rpc(
        self,
        request: JsonRpcRequest,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        registry = self._registry_snapshot()
        if registry is None:
            return JsonRpcResponse.fail(
                id=request.id,
                code=METHOD_NOT_FOUND,
                message="No RPC registry has been loaded yet.",
            )

        try:
            response = registry.call_unique(
                request.method,
                request.params or {},
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return JsonRpcResponse.fail(
                id=request.id,
                code=METHOD_NOT_FOUND,
                message=str(exc),
            )

        return self._rewrite_response_id(response, request.id)

    def call_controller_rpc(
        self,
        controller: str,
        request: JsonRpcRequest,
        *,
        service_name: str | None = None,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        service = self._find_service(controller, service_name=service_name)
        if isinstance(service, JsonRpcResponse):
            service.id = request.id
            return service

        method = request.method
        if "." not in method:
            method = f"{controller}.{method}"

        return self._call_service(
            service=service,
            request_id=request.id,
            method=method,
            params=request.params,
            timeout_s=timeout_s,
        )

    def call_method(
        self,
        controller: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        service_name: str | None = None,
        request_id: Any = 1,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        service = self._find_service(controller, service_name=service_name)
        if isinstance(service, JsonRpcResponse):
            service.id = request_id
            return service

        return self._call_service(
            service=service,
            request_id=request_id,
            method=f"{controller}.{method}",
            params=params,
            timeout_s=timeout_s,
        )

    def schema_for_controller(
        self,
        controller: str,
        *,
        service_name: str | None = None,
    ) -> Any | JsonRpcResponse:
        service = self._find_service(controller, service_name=service_name)
        if isinstance(service, JsonRpcResponse):
            return service

        descriptor = getattr(service, "descriptor", None)
        if isinstance(descriptor, RpcServiceDescriptor):
            return descriptor

        return self._service_summary(service)

    def methods_for_controller(
        self,
        controller: str,
        *,
        service_name: str | None = None,
    ) -> list[RpcMethodDescriptor] | list[str] | JsonRpcResponse:
        service = self._find_service(controller, service_name=service_name)
        if isinstance(service, JsonRpcResponse):
            return service

        descriptor = getattr(service, "descriptor", None)
        if isinstance(descriptor, RpcServiceDescriptor):
            return descriptor.methods

        return list(getattr(service, "methods", []) or [])


def _query_params_without_service(request: Any) -> dict[str, Any]:
    return {key: value for key, value in request.query_params.items() if key != "service"}


def _raise_gateway_response(response: JsonRpcResponse) -> None:
    if response.error is None:
        return

    status_code = 400
    if response.error.code == METHOD_NOT_FOUND:
        status_code = 404
    elif response.error.code == INVALID_PARAMS:
        status_code = 422
    elif response.error.code == INTERNAL_ERROR:
        status_code = 500
    elif response.error.code == PARSE_ERROR:
        status_code = 400

    from fastapi import HTTPException

    raise HTTPException(
        status_code=status_code,
        detail=response.error.model_dump(mode="json"),
    )


def _openapi_query_parameters(params_schema: dict[str, Any]) -> list[dict[str, Any]]:
    required = set(params_schema.get("required", []) or [])
    properties = params_schema.get("properties", {}) or {}
    parameters: list[dict[str, Any]] = []

    for name, schema in properties.items():
        # A service discriminator for the gateway is transport metadata, not RPC params.
        if name == "service":
            continue

        parameters.append(
            {
                "name": name,
                "in": "query",
                "required": name in required,
                "schema": schema,
            }
        )

    parameters.append(
        {
            "name": "service",
            "in": "query",
            "required": False,
            "schema": {"type": "string"},
            "description": "Optional service_name when multiple discovered services expose the same controller.",
        }
    )
    return parameters


def _dynamic_openapi_schema_for_method(method: RpcMethodDescriptor) -> dict[str, Any]:
    return {
        "post": {
            "tags": ["dynamic-rpc"],
            "summary": f"Call {method.jsonrpc_method}",
            "operationId": f"dynamic_{method.jsonrpc_method.replace('.', '_')}_post",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": method.params_schema,
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Successful Response",
                    "content": {"application/json": {"schema": method.result_schema}},
                }
            },
        },
        "get": {
            "tags": ["dynamic-rpc"],
            "summary": f"Call {method.jsonrpc_method} with query parameters",
            "operationId": f"dynamic_{method.jsonrpc_method.replace('.', '_')}_get",
            "parameters": _openapi_query_parameters(method.params_schema),
            "responses": {
                "200": {
                    "description": "Successful Response",
                    "content": {"application/json": {"schema": method.result_schema}},
                }
            },
        },
    }


def _install_dynamic_openapi(app: "FastAPI", gateway: DynamicRpcGateway) -> None:
    from fastapi.openapi.utils import get_openapi

    def custom_openapi() -> dict[str, Any]:
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        paths = schema.setdefault("paths", {})

        for descriptor in gateway.descriptors():
            controller = descriptor.controller

            paths.setdefault(
                f"/controllers/{controller}",
                {
                    "get": {
                        "tags": ["dynamic-rpc"],
                        "summary": f"Describe {controller}",
                        "operationId": f"dynamic_{controller}_detail_get",
                        "responses": {"200": {"description": "Successful Response"}},
                    }
                },
            )
            paths.setdefault(
                f"/controllers/{controller}/methods",
                {
                    "get": {
                        "tags": ["dynamic-rpc"],
                        "summary": f"List {controller} methods",
                        "operationId": f"dynamic_{controller}_methods_get",
                        "responses": {"200": {"description": "Successful Response"}},
                    }
                },
            )
            paths.setdefault(
                f"/controllers/{controller}/rpc",
                {
                    "post": {
                        "tags": ["dynamic-rpc"],
                        "summary": f"JSON-RPC bridge for {controller}",
                        "operationId": f"dynamic_{controller}_rpc_post",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/JsonRpcRequest"}
                                }
                            },
                        },
                        "responses": {"200": {"description": "Successful Response"}},
                    }
                },
            )

            for method in descriptor.methods:
                paths[f"/controllers/{controller}/{method.name}"] = _dynamic_openapi_schema_for_method(method)

        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


async def _run_gateway_refresh_loop(
    gateway: DynamicRpcGateway,
    *,
    refresh_interval_s: float,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        await asyncio.to_thread(gateway.refresh)

        try:
            await asyncio.wait_for(stop.wait(), timeout=refresh_interval_s)
        except TimeoutError:
            pass


def create_dynamic_rpc_gateway_fastapi_router(
    gateway: DynamicRpcGateway | None = None,
    *,
    tags: list[str] | None = None,
    default_timeout_s: float = 5.0,
) -> "APIRouter":
    """Create a dynamic FastAPI router backed by service discovery.

    The router does not receive or register controller instances. It forwards
    HTTP/JSON-RPC calls to discovered RPC services, so new services appear under
    /controllers/** after the gateway refreshes.
    """

    from fastapi import APIRouter, Body, Query, Request

    gateway = gateway or DynamicRpcGateway()
    route_tags = tags or ["dynamic-rpc"]
    router = APIRouter()

    def root() -> dict[str, Any]:
        return {
            "status": gateway.status().model_dump(mode="json"),
            "controllers": [item.model_dump(mode="json") for item in gateway.controllers()],
            "routes": [
                "GET /health",
                "GET /refresh",
                "GET /controllers",
                "GET /controllers/{controller}",
                "GET /controllers/{controller}/methods",
                "GET /controllers/{controller}/{method}",
                "POST /controllers/{controller}/{method}",
                "POST /rpc",
                "POST /refresh",
            ],
        }

    def status() -> DynamicGatewayStatus:
        return gateway.status()

    async def refresh_now() -> DynamicGatewayStatus:
        return await asyncio.to_thread(gateway.refresh)

    def catalog() -> list[dict[str, Any]]:
        return gateway.catalog()

    def controllers_list() -> list[ControllerRouteInfo]:
        return gateway.controllers()

    def describe_all() -> dict[str, Any]:
        return gateway.describe_all()

    async def rpc(request: JsonRpcRequest) -> JsonRpcResponse:
        return await asyncio.to_thread(gateway.call_rpc, request, timeout_s=default_timeout_s)

    async def controller_detail(
        controller: str,
        service: str | None = Query(default=None),
    ) -> Any:
        result = gateway.schema_for_controller(controller, service_name=service)
        if isinstance(result, JsonRpcResponse):
            _raise_gateway_response(result)
        return result

    async def controller_methods(
        controller: str,
        service: str | None = Query(default=None),
    ) -> Any:
        result = gateway.methods_for_controller(controller, service_name=service)
        if isinstance(result, JsonRpcResponse):
            _raise_gateway_response(result)
        return result

    async def controller_rpc(
        controller: str,
        request: JsonRpcRequest,
        service: str | None = Query(default=None),
    ) -> JsonRpcResponse:
        return await asyncio.to_thread(
            gateway.call_controller_rpc,
            controller,
            request,
            service_name=service,
            timeout_s=default_timeout_s,
        )

    async def post_method(
        controller: str,
        method: str,
        params: dict[str, Any] | None = Body(default=None),
        service: str | None = Query(default=None),
    ) -> Any:
        response = await asyncio.to_thread(
            gateway.call_method,
            controller,
            method,
            params or {},
            service_name=service,
            timeout_s=default_timeout_s,
        )
        _raise_gateway_response(response)
        return response.result

    async def get_method(controller: str, method: str, request: Any) -> Any:
        service = request.query_params.get("service")
        response = await asyncio.to_thread(
            gateway.call_method,
            controller,
            method,
            _query_params_without_service(request),
            service_name=service,
            timeout_s=default_timeout_s,
        )
        _raise_gateway_response(response)
        return response.result

    # Avoid a postponed local ForwardRef("Request") in generated OpenAPI.
    get_method.__signature__ = Signature(
        parameters=[
            Parameter("controller", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            Parameter("method", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            Parameter("request", Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
        ],
        return_annotation=Any,
    )

    router.add_api_route("/", root, methods=["GET"], tags=route_tags, summary="RPC gateway info")
    router.add_api_route(
        "/health",
        status,
        methods=["GET"],
        response_model=DynamicGatewayStatus,
        response_model_exclude_none=True,
        tags=["health"],
        summary="RPC gateway health",
    )
    router.add_api_route(
        "/refresh",
        refresh_now,
        methods=["GET", "POST"],
        response_model=DynamicGatewayStatus,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Refresh discovered RPC services",
    )
    router.add_api_route(
        "/controllers/refresh",
        refresh_now,
        methods=["GET", "POST"],
        response_model=DynamicGatewayStatus,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Refresh discovered RPC services",
        include_in_schema=False,
    )
    router.add_api_route(
        "/catalog",
        catalog,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="List discovered RPC services",
        include_in_schema=False,
    )
    router.add_api_route(
        "/describe",
        describe_all,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Describe discovered controllers",
        include_in_schema=False,
    )
    router.add_api_route(
        "/controllers",
        controllers_list,
        methods=["GET"],
        response_model=list[ControllerRouteInfo],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="List discovered controllers",
    )
    router.add_api_route(
        "/rpc",
        rpc,
        methods=["POST"],
        response_model=JsonRpcResponse,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="JSON-RPC bridge",
    )

    # Register fixed subpaths before the simple /controllers/{controller}/{method}
    # route so names like methods/rpc/schema are not treated as controller methods.
    router.add_api_route(
        "/controllers/{controller}/methods",
        controller_methods,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="List discovered controller methods",
    )
    router.add_api_route(
        "/controllers/{controller}/rpc",
        controller_rpc,
        methods=["POST"],
        response_model=JsonRpcResponse,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Controller-scoped JSON-RPC bridge",
    )
    router.add_api_route(
        "/controllers/{controller}/schema",
        controller_detail,
        methods=["GET"],
        response_model_exclude_none=True,
        include_in_schema=False,
    )
    router.add_api_route(
        "/controllers/{controller}",
        controller_detail,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Describe a discovered controller",
    )
    router.add_api_route(
        "/controllers/{controller}/{method}",
        post_method,
        methods=["POST"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Call a discovered controller method",
    )
    router.add_api_route(
        "/controllers/{controller}/{method}",
        get_method,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Call a discovered controller method from query parameters",
    )

    # Backward-compatible verbose aliases. Hidden from OpenAPI so the visible API stays simple.
    router.add_api_route(
        "/controllers/{controller}/methods/{method}",
        post_method,
        methods=["POST"],
        response_model_exclude_none=True,
        include_in_schema=False,
    )
    router.add_api_route(
        "/controllers/{controller}/methods/{method}",
        get_method,
        methods=["GET"],
        response_model_exclude_none=True,
        include_in_schema=False,
    )

    return router


def create_dynamic_rpc_gateway_fastapi_app(
    *,
    gateway: DynamicRpcGateway | None = None,
    title: str = "Auto-discovered JSON-RPC API",
    version: str = "1.0.0",
    description: str = "FastAPI gateway for RPC services discovered at runtime.",
    refresh_on_startup: bool = True,
    refresh_interval_s: float | None = None,
    install_dynamic_openapi: bool = True,
    tags: list[str] | None = None,
    default_timeout_s: float = 5.0,
) -> "FastAPI":
    """Create a zero-registration FastAPI app from runtime discovery.

    - No controller instances are passed into the API process.
    - The optional lifespan refresh loop discovers services that start later.
    - /controllers/** routes are path-parameter based, so they do not require
      app.routes mutation or a Uvicorn reload when a new RPC service appears.
    - When install_dynamic_openapi is true, /openapi.json is generated from the
      latest descriptors so /docs reflects newly discovered controller methods.
    """

    from fastapi import FastAPI

    gateway = gateway or DynamicRpcGateway()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task: asyncio.Task[None] | None = None

        if refresh_on_startup:
            await asyncio.to_thread(gateway.refresh)

        if refresh_interval_s is not None and refresh_interval_s > 0:
            task = asyncio.create_task(
                _run_gateway_refresh_loop(
                    gateway,
                    refresh_interval_s=refresh_interval_s,
                    stop=stop,
                )
            )

        try:
            yield
        finally:
            stop.set()
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title=title, version=version, description=description, lifespan=lifespan)
    app.state.rpc_gateway = gateway
    app.include_router(
        create_dynamic_rpc_gateway_fastapi_router(
            gateway,
            tags=tags,
            default_timeout_s=default_timeout_s,
        )
    )

    if install_dynamic_openapi:
        _install_dynamic_openapi(app, gateway)

    return app


def create_auto_discover_fastapi_app(
    *,
    title: str = "Auto-discovered JSON-RPC API",
    version: str = "1.0.0",
    description: str = "FastAPI gateway that automatically discovers RPC controllers at runtime.",
    refresh_interval_s: float | None = None,
    refresh_on_startup: bool = True,
    install_dynamic_openapi: bool = True,
    default_timeout_s: float = 5.0,
    gateway: DynamicRpcGateway | None = None,
) -> "FastAPI":
    """Create the recommended zero-registration FastAPI gateway."""

    return create_dynamic_rpc_gateway_fastapi_app(
        gateway=gateway,
        title=title,
        version=version,
        description=description,
        refresh_on_startup=refresh_on_startup,
        refresh_interval_s=refresh_interval_s,
        install_dynamic_openapi=install_dynamic_openapi,
        default_timeout_s=default_timeout_s,
    )
