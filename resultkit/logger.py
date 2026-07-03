from datetime import datetime

def logger(msg, level="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [{level.upper()}] {msg}")

try:

    import atexit
    import os
    import queue
    import re
    import socket
    import threading
    import time
    import traceback
    from typing import Any, Optional

    import redis


    class RedisLogger:
        """
        Singleton Redis Stream logger.

        Usage:
            logger = RedisLogger()

            logger("[App:init] started")
            logger("[Auth:login] failed", level="warning", extra={"user_id": 123})

            try:
                1 / 0
            except Exception:
                logger.exception("[Calc:divide] failed")

            logger.close()

        Check Redis:
            redis-cli XRANGE "log:App:init" - +
            redis-cli XRANGE "log:Auth:login" - +
        """

        _instance = None
        _instance_lock = threading.Lock()
        _parse_pattern = re.compile(r"^\[([^\]]+)\]\s*(.*)$")

        def __new__(cls, *args, **kwargs):
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(
            self,
            redis_url: str = "redis://localhost:6379/0",
            key_prefix: str = "",#"log:",
            max_queue_size: int = 10_000,
            max_stream_len: int = 100_000,
            print_also: bool = True,
            redis_socket_timeout: float = 2.0,
        ):
            # Prevent singleton from initializing multiple times.
            if getattr(self, "_initialized", False):
                return

            self.redis_url = redis_url
            self.key_prefix = key_prefix
            self.max_stream_len = max_stream_len
            self.print_also = print_also

            self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

            self.redis = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=redis_socket_timeout,
                socket_timeout=redis_socket_timeout,
            )

            if not self.redis.ping():
                raise RuntimeError("Failed to connect to Redis")

            self.hostname = socket.gethostname()
            self.pid = str(os.getpid())

            self.closed = False
            self._sentinel = object()

            self.worker_thread = threading.Thread(
                target=self._worker,
                name="RedisLoggerWorker",
                daemon=True,
            )
            self.worker_thread.start()

            atexit.register(self.close)

            self._initialized = True

        def __call__(
            self,
            raw_msg: str,
            level: str = "info",
            extra: Optional[dict[str, Any]] = None,
        ):
            """
            Very light hot path.

            This does NOT parse, format, build Redis keys, build Redis records,
            or write to Redis. It only puts a small tuple into the queue.
            """
            if self.closed:
                return

            try:
                # Capture timestamp here so it means "time log was called",
                # not "time worker processed the log".
                self.queue.put_nowait((time.time(), raw_msg, level, extra))
            except queue.Full:
                print("[LOGGER WARNING] queue full; dropped log:", raw_msg)

        def exception(
            self,
            raw_msg: str,
            extra: Optional[dict[str, Any]] = None,
        ):
            """
            Use inside an except block.

            Example:
                try:
                    1 / 0
                except Exception:
                    logger.exception("[Calc:divide] failed")
            """
            if extra is None:
                extra = {}

            # Copy to avoid mutating caller's dict.
            extra = dict(extra)
            extra["traceback"] = traceback.format_exc()

            self(raw_msg, level="error", extra=extra)

        def _worker(self):
            """
            Heavy work happens here, not in __call__().
            """
            while True:
                item = self.queue.get()

                try:
                    if item is self._sentinel:
                        return

                    ts, raw_msg, level, extra = item

                    parsed_key, message = self.parse(raw_msg)
                    redis_key = self.make_redis_key(parsed_key)
                    level = str(level).upper()

                    record = {
                        # "key": parsed_key,
                        "msg": message,
                        # "raw": str(raw_msg),
                        "level": level,
                        "ts": str(ts),
                        # "hostname": self.hostname,
                        # "pid": self.pid,
                    }

                    if extra:
                        for k, v in extra.items():
                            record[f"extra:{k}"] = str(v)

                    if self.print_also:
                        print(f"[{level}] [{parsed_key}] {message}")

                    self.redis.xadd(
                        redis_key,
                        record,
                        maxlen=self.max_stream_len,
                        approximate=True,
                    )

                except Exception as exc:
                    print("[LOGGER ERROR] failed to write to Redis:", repr(exc))

                finally:
                    self.queue.task_done()

        def close(self):
            """
            Flush queued logs and close Redis.

            Safe to call multiple times.
            """
            if getattr(self, "closed", True):
                return

            self.closed = True

            try:
                # Wait until all real log items are processed.
                self.queue.join()
            except Exception:
                pass

            try:
                # Stop worker.
                self.queue.put_nowait(self._sentinel)
            except queue.Full:
                try:
                    self.queue.put(self._sentinel, timeout=1)
                except Exception:
                    pass

            try:
                self.worker_thread.join(timeout=2)
            except Exception:
                pass

            try:
                self.redis.close()
            except Exception:
                pass

        def parse(self, raw_msg: str) -> tuple[str, str]:
            """
            Parse:
                "[Amodule:member1] say something"

            Into:
                key = "Amodule:member1"
                message = "say something"
            """
            raw_msg = str(raw_msg).strip()

            match = self._parse_pattern.match(raw_msg)

            if not match:
                return "unknown", raw_msg

            key = match.group(1).strip() or "unknown"
            message = match.group(2).strip()

            return key, message

        def make_redis_key(self, key: str) -> str:
            return f"{self.key_prefix}{key}"

    logger = RedisLogger()
    logger("[RedisLogger:init] start")

except:
    pass



