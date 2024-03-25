# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for benchmarking the Deep Learning frameworks.

This module consists of benchmarking tools and scenarios for checking:

* Deep Learning frameworks,
* Deep Learning models' optimizers,
* Deep Learning models' compilers.
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

from kenning.utils.excepthook import kenning_missing_import_excepthook
from kenning.utils.logger import KLogger

# Extend execpthook function to handle ModuleNotFoundError
if os.environ.get("KENNING_USE_DEFAULT_EXCEPTHOOK", None) is None:
    sys.excepthook = kenning_missing_import_excepthook

sys.path.insert(0, os.path.abspath(__file__ + "../"))

try:
    __version__ = version("kenning")
except PackageNotFoundError:
    # assumes Kenning source code is used
    import re
    from pathlib import Path

    try:
        __version__ = re.search(
            r'version = "([^"]+)"',
            (Path(__file__).parent.parent / "pyproject.toml").read_text(),
        ).group(1)
    except Exception:
        KLogger.error(
            "Version of Kenning cannot be retrieved, "
            "please install Kenning through `pip`"
        )
        __version__ = None
