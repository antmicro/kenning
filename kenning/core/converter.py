# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for model converters.
"""

from abc import ABC

from kenning.utils.resource_manager import PathOrURI


class ModelConverter(ABC):
    """
    Loads or converts model to specific format.

    """

    source_format: str

    def __init__(
        self,
        source_model_path: PathOrURI,
    ):
        """
        Prepares the ModelConverter object.

        Parameters
        ----------
        source_model_path : PathOrURI
            Path to file where the source model is located.
        """
        self.source_model_path = source_model_path

    def to_ai8x(self, *args, **kwargs):
        """
        Converts model to ai8x format.
        """
        raise NotImplementedError

    def to_torch(self, *args, **kwargs):
        """
        Converts model to PyTorch format.
        """
        raise NotImplementedError

    def to_onnx(self, *args, **kwargs):
        """
        Converts model to ONNX format.
        """
        raise NotImplementedError

    def to_tflite(self, *args, **kwargs):
        """
        Converts model to TensorFlowLite format.
        """
        raise NotImplementedError

    def to_tvm(self, *args, **kwargs):
        """
        Converts model to TVM format.
        """
        raise NotImplementedError

    def to_keras(self, *args, **kwargs):
        """
        Converts model to Keras format.
        """
        raise NotImplementedError

    def to_tinygrad(self, *args, **kwargs):
        """
        Convert model to Tinygrad format.
        """
        raise NotImplementedError
