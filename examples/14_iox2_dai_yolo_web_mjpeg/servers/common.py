from __future__ import annotations

import os
import sys

from pathlib import Path
import threading
import time
from typing import Any

EXAMPLE_DIR = os.path.dirname(os.path.dirname(Path(__file__).absolute()))

if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from iox2_jsonrpc import EmptyParams, RpcModel

class StoppableLoop:
    """Small base class for loops that can run blocking or in a worker thread."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._exception: BaseException | None = None
        self._pause_event = threading.Event()
        self.pause_sleep_seconds: float = 0.01

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    def start(self, *, blocking: bool = True) -> "StoppableLoop":
        """Start the loop.

        Args:
            blocking: When True, run on the current thread. When False, run in a
                daemon worker thread and return immediately.
        """
        if self._running:
            return self

        self._stop_event.clear()
        self._exception = None

        if blocking:
            self._run_guarded()
        else:
            self._thread = threading.Thread(target=self._run_guarded, daemon=True)
            self._thread.start()

        return self

    def stop(self, *, join: bool = True, timeout: float | None = None) -> None:
        """Request the loop to stop and optionally wait for the worker thread."""
        self._stop_event.set()

        thread = self._thread
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def pause(self) -> None:
        """Pause after the current frame finishes processing."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume frame processing."""
        self._pause_event.clear()

    def _should_stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _wait_if_paused(self) -> bool:
        """Return True when the loop was stopped while waiting in pause state."""
        while self._pause_event.is_set():
            if self._should_stop():
                return True
            time.sleep(self.pause_sleep_seconds)
        return False
    
    def join(self, timeout: float | None = None) -> None:
        """Wait for a non-blocking start() worker thread to finish."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        if self._exception is not None:
            raise self._exception

    def _run_guarded(self) -> None:
        self._running = True
        try:
            self._run()
        except BaseException as exc:
            self._exception = exc
            raise
        finally:
            self._running = False

    def _run(self) -> None:
        raise NotImplementedError    
        # try:
        #     self.startup()
        #     while not self._should_stop():
        #         if self._wait_if_paused():
        #             return
        #         # run one step
        # finally:
        #     self.close()


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


__all__ = ["EmptyParams", "RpcModel"]
