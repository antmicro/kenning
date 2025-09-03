# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides helper functions for runtimes.
"""

import itertools
from typing import Iterable, List, TypeVar

import numpy as np

from kenning.utils.resource_manager import PathOrURI

Runtime = TypeVar("kenning.core.runtime.Runtime")


def get_default_runtime(
    model_framework: str, model_path: PathOrURI
) -> "Runtime":
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

    if model_framework == "executorch":
        from kenning.runtimes.executorch import ExecuTorchRuntime

        return ExecuTorchRuntime(model_path)

    if model_framework == "iree":
        from kenning.runtimes.iree import IREERuntime

        return IREERuntime(model_path)

    if model_framework == "ai8x_c":
        from kenning.runtimes.ai8x import Ai8xRuntime

        return Ai8xRuntime(model_path)


def zero_pad_batch(
    batch: Iterable[np.ndarray], target_batch_size: int
) -> List[np.ndarray]:
    """
    Pad the provided batch with zero-filled tensors to match
    the target batch size.

    Parameters
    ----------
    batch : Iterable[np.ndarray]
        The batch of samples to be zero-padded.
    target_batch_size : int
        The target batch size, to which the batch will be extended
        by padding with zeros.

    Returns
    -------
    List[np.ndarray]
        The zero-padded batch with the target size.
    """
    iterator_with_first_sample = itertools.islice(batch, 0, 1)
    first_sample = next(iterator_with_first_sample, None)
    return [
        sample if sample is not None else mimic_sample(first_sample)
        for sample, _ in itertools.zip_longest(batch, range(target_batch_size))
    ]


def mimic_sample(sample: np.ndarray) -> np.ndarray:
    """
    Mimic samples shape with zero-filled NumPy array.

    Parameters
    ----------
    sample : np.ndarray
        Sample to infer the data type and shape from.

    Returns
    -------
    np.ndarray
        Zero-filled copy of a sample.

    Raises
    ------
    TypeError
        Raised if the input is not a `numpy.ndarray`.
    """
    if isinstance(sample, np.ndarray):
        return np.zeros_like(sample)

    raise TypeError(
        "Samples of inputs have to be of `numpy.ndarray`."
        f"The sample is of type: `{type(sample)}`."
    )
