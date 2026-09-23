# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for CoralNPU protobuf trace parsing.
"""

import os
import sys

flags = sys.getdlopenflags()

try:
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    from ._parser import parse
finally:
    sys.setdlopenflags(flags)

__all__ = ["parse"]
