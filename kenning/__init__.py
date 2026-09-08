# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for benchmarking the Deep Learning frameworks.

This module consists of benchmarking tools and scenarios for checking:

* Deep Learning frameworks,
* Deep Learning models' optimizers,
* Deep Learning models' compilers.
"""

import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from kenning.temporary_fixes.triton import install_triton_preloader
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


# TODO: pinned version of triton is compiled
# against LLVM 23 while tensorflow is compiled against
# LLVM 18, which causes symbol collision errors.
install_triton_preloader()

# Importing onnx2tf sets TF_USE_LEGACY_KERAS, which makes `tf.keras` resolve
# to tf_keras (Keras 2). The variable has to be set before TensorFlow
# is first imported. onnx2tf is imported lazily, so set
# it here rather than relying on that import's side effect.
if importlib.util.find_spec("onnx2tf") is not None:
    # If kenning[onnx2tf] is installed
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
