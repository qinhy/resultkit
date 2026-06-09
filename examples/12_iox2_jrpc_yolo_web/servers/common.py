from __future__ import annotations

import os
import sys

from pathlib import Path

EXAMPLE_DIR = os.path.dirname(os.path.dirname(Path(__file__).absolute()))

if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from iox2_jsonrpc import EmptyParams, RpcModel


__all__ = ["EmptyParams", "RpcModel"]
