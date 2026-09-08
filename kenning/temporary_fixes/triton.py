# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The module contains a fix for the symbol collision between triton
and TensorFlow.
"""

import contextlib
import sys
from types import ModuleType
from typing import Optional, Sequence


def find_spec(
    fullname: str,
    path: Optional[Sequence[str]] = None,
    target: Optional[ModuleType] = None,
) -> None:
    """
    Loads triton when TensorFlow is about to be imported.

    Parameters
    ----------
    fullname : str
        Name of the imported module.
    path : Optional[Sequence[str]]
        Search path for the imported module.
    target : Optional[ModuleType]
        Module to be reloaded, if any.
    """
    if fullname.split(".")[0] != "tensorflow":
        return None

    sys.meta_path.remove(sys.modules[__name__])
    with contextlib.suppress(ImportError):
        import triton  # noqa: F401

    return None


def install_triton_preloader():
    """
    Registers this module as a finder.
    """
    if sys.modules[__name__] not in sys.meta_path:
        sys.meta_path.insert(0, sys.modules[__name__])
