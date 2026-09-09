from __future__ import annotations

import os
import sys

from pathlib import Path
import threading
import time

from pydantic import BaseModel

EXAMPLE_DIR = os.path.dirname(os.path.dirname(Path(__file__).absolute()))

if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from iox2_jsonrpc import EmptyParams, RpcModel
from store.custom_record_store import CustomRecord, CustomStore, RecordMode, RecordPath

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Sequence

import requests


logger = logging.getLogger(__name__)

HookChain = Sequence[str]


@dataclass
class HookDispatcher:
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than 0")

    def dispatch(
        self,
        db_record: Any,
        hook_chains: Sequence[HookChain] | None,
    ) -> None:
        record_payload = self._serialize_record(db_record)

        for chain in hook_chains or ():
            urls = [
                url.strip()
                for url in chain
                if isinstance(url, str) and url.strip()
            ]

            if not urls:
                continue

            current_url = urls[0]
            remaining_urls = urls[1:]

            payload = self.build_payload(
                db_record=record_payload,
                remaining_urls=remaining_urls,
            )

            logger.info(
                "Dispatching hook immediately: url=%s",
                current_url,
            )

            threading.Thread(
                target=self._post,
                args=(current_url, payload),
                daemon=True,
                name="hook-dispatch",
            ).start()

    def _post(
        self,
        url: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            logger.info("Sending hook POST: url=%s", url)

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_s,
            )

            logger.info(
                "Hook response: url=%s status=%s body=%s",
                url,
                response.status_code,
                response.text[:500],
            )

            response.raise_for_status()

        except Exception:
            logger.exception(
                "Hook request failed: url=%s",
                url,
            )

    @staticmethod
    def _serialize_record(
        db_record: Any,
    ) -> dict[str, Any]:
        if hasattr(db_record, "model_dump"):
            return db_record.model_dump(mode="json")

        if hasattr(db_record, "dict"):
            return db_record.dict()

        if isinstance(db_record, Mapping):
            return dict(db_record)

        raise TypeError(
            "db_record must be a mapping or a "
            "Pydantic-compatible model; "
            f"received {type(db_record).__name__}"
        )

    @staticmethod
    def build_payload(
        db_record: Mapping[str, Any],
        remaining_urls: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "db_record": dict(db_record),
            "hook_urls": (
                [list(remaining_urls)]
                if remaining_urls
                else []
            ),
        }

    
def openapi_doc(key="yolo_status", id=1, params={}):
    """Return OpenAPI document for the given key."""
    service_name = key.split("_")[0]
    func_name = key.replace(service_name+"_", "")
    return {key: {
                "summary": f"{service_name}: {func_name}",
                "description": f"Use service with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": f"{service_name}.{func_name}",
                    "params": params,
                },
            }}


__all__ = ["EmptyParams", "RpcModel",
           "openapi_doc",
           "CustomRecord", "CustomStore", "RecordMode", "RecordPath"]
