from __future__ import annotations

import os
import sys

from pathlib import Path

from pydantic import BaseModel

EXAMPLE_DIR = os.path.dirname(os.path.dirname(Path(__file__).absolute()))

if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from iox2_jsonrpc import EmptyParams, RpcModel
from store.custom_record_store import CustomRecord, CustomStore, RecordMode

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Sequence

import requests


logger = logging.getLogger(__name__)

HookChain = Sequence[str]


@dataclass
class HookDispatcher:
    max_workers: int = 8
    timeout_s: float = 5.0
    session: requests.Session = field(default_factory=requests.Session)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than 0")

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="hook-dispatch",
        )

    def dispatch(
        self,
        db_record: Any,
        hook_chains: Sequence[HookChain] | None,
    ) -> list[Future[requests.Response]]:
        futures: list[Future[requests.Response]] = []

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
                "Submitting hook POST: url=%s remaining_hooks=%s",
                current_url,
                remaining_urls,
            )

            future = self._executor.submit(
                self._post,
                current_url,
                payload,
            )
            future.add_done_callback(self._log_future_result)
            futures.append(future)

        return futures

    @staticmethod
    def _serialize_record(db_record: Any) -> dict[str, Any]:
        if hasattr(db_record, "model_dump"):
            return db_record.model_dump(mode="json")

        if hasattr(db_record, "dict"):
            return db_record.dict()

        if isinstance(db_record, Mapping):
            return dict(db_record)

        raise TypeError(
            "db_record must be a mapping or a Pydantic-compatible model; "
            f"received {type(db_record).__name__}"
        )

    @staticmethod
    def build_payload(
        db_record: Mapping[str, Any],
        remaining_urls: Sequence[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "db_record": dict(db_record),
        }

        # hook_urls belongs to StartYoloParams, not CustomRecord.
        if remaining_urls:
            payload["hook_urls"] = [list(remaining_urls)]
        else:
            payload["hook_urls"] = []

        return payload

    def _post(
        self,
        url: str,
        payload: Mapping[str, Any],
    ) -> requests.Response:
        logger.info("Sending hook POST to %s", url)
        logger.debug("Hook payload for %s: %r", url, payload)

        response = self.session.post(
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
        return response

    @staticmethod
    def _log_future_result(
        future: Future[requests.Response],
    ) -> None:
        try:
            response = future.result()
            logger.info(
                "Hook completed successfully: status=%s",
                response.status_code,
            )
        except Exception:
            logger.exception("Hook request failed")

    def close(self, wait: bool = True) -> None:
        self._executor.shutdown(
            wait=wait,
            cancel_futures=not wait,
        )
        self.session.close()


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
           "CustomRecord", "CustomStore", "RecordMode"]
