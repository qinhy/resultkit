from __future__ import annotations

import tomllib
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class JsonRpcServiceDescriptor(BaseModel):
    name: str
    iceoryx2_service: str
    timeout_seconds: float = Field(default=5.0, gt=0)


class JsonRpcServiceRegistry:
    def __init__(self, services: list[JsonRpcServiceDescriptor] | None = None) -> None:
        self._services: dict[str, JsonRpcServiceDescriptor] = {}
        for service in services or []:
            self.register(service)

    @classmethod
    def from_toml(cls, path: str | Path) -> "JsonRpcServiceRegistry":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        services = []
        for name, values in raw.get("services", {}).items():
            services.append(JsonRpcServiceDescriptor(name=name, **values))
        return cls(services)

    @classmethod
    def from_list(cls, services:List[JsonRpcServiceDescriptor]) -> "JsonRpcServiceRegistry":
        return cls(services)
    
    def register(self, service: JsonRpcServiceDescriptor) -> None:
        self._services[service.name] = service

    def get(self, name: str) -> JsonRpcServiceDescriptor:
        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(f"Unknown JSON-RPC service: {name}") from exc

    def list(self) -> list[JsonRpcServiceDescriptor]:
        return [self._services[name] for name in sorted(self._services)]
