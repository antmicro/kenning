# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
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

from kenning.utils.excepthook import kenning_missing_import_excepthook


# Extend execpthook function to handle ModuleNotFoundError
if os.environ.get("KENNING_USE_DEFAULT_EXCEPTHOOK", None) is None:
    sys.excepthook = kenning_missing_import_excepthook

sys.path.insert(0, os.path.abspath(__file__ + '../'))
