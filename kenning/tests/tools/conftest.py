# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Makes scripts from the tools directory importable in tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
