"""Redis-like JSON SET/GET over iceoryx2 request-response IPC."""

from __future__ import annotations

from typing import Any

__all__ = ["direct_redis_for", "redis_for"]
__version__ = "0.1.0"


def redis_for(*args: Any, **kwargs: Any) -> Any:
    """Create either a normal redis-py client or an iceoryx2-backed client.

    This is intentionally imported lazily so that protocol/store tests do not
    need redis-py or iceoryx2 installed.
    """

    from .client import redis_for as _redis_for

    return _redis_for(*args, **kwargs)


def direct_redis_for(*args: Any, **kwargs: Any) -> Any:
    """Create a direct iceoryx2 client without redis-py compatibility layers."""

    from .client import direct_redis_for as _direct_redis_for

    return _direct_redis_for(*args, **kwargs)
