from __future__ import annotations

import ctypes
import os
from typing import TypeAlias

os.environ.setdefault("IOX2_JSONRPC_FORCE_REMOVE_SERVICES", "1")

try:
    import iceoryx2 as iox2
except ImportError as exc:  # pragma: no cover - depends on optional package
    raise ImportError(
        "Missing optional dependency: iceoryx2. Install with: pip install 'iox2-jsonrpc[iox2]'"
    ) from exc

U8Slice: TypeAlias = iox2.Slice[ctypes.c_uint8]

__all__ = ["U8Slice", "iox2"]
