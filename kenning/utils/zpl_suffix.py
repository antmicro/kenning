# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides enumeration of Zephelin related file extensions.
"""

from enum import Enum
from pathlib import Path


class ZplSuffix(Enum):
    """
    List of common report file suffixes.
    """

    CTF = ".ctf"
    TVM_GRAPH_JSON = ".graph.json"
    TVM_METADATA = ".tvm_metadata"
    TFLITE = ".tflite"
    TRACE_JSON = ".trace.json"

    def _get_path_with_suffix(self, path: Path) -> Path:
        return path.with_suffix(self.value)
