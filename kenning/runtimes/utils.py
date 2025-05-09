# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides helper functions for runtimes.
"""

from kenning.core.runtime import Runtime
from kenning.utils.resource_manager import PathOrURI


def get_default_runtime(
    model_framework: str, model_path: PathOrURI
) -> Runtime:
    """
    Returns default Runtime for given model framework.

    Parameters
    ----------
    model_framework : str
        Framework of the model.
    model_path : PathOrURI
        Path to the model.

    Returns
    -------
    Runtime
        Default Runtime for given model.
    """
    if model_framework == "tvm":
        from kenning.runtimes.tvm import TVMRuntime

        return TVMRuntime(model_path)

    if model_framework == "tflite":
        from kenning.runtimes.tflite import TFLiteRuntime

        return TFLiteRuntime(model_path)

    if model_framework == "torch":
        from kenning.runtimes.pytorch import PyTorchRuntime

        return PyTorchRuntime(model_path)

    if model_framework == "iree":
        from kenning.runtimes.iree import IREERuntime

        return IREERuntime(model_path)

    if model_framework == "ai8x_c":
        from kenning.runtimes.ai8x import Ai8xRuntime

        return Ai8xRuntime(model_path)
