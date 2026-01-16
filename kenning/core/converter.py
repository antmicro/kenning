# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for model converters.
"""

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from kenning.utils.resource_manager import PathOrURI

if TYPE_CHECKING:
    import onnx
    import tensorflow as tf
    import tflite
    import torch
    import tvm

    from kenning.optimizers.ai8x import Ai8xTools


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

    def to_ai8x(
        self,
        ai8x_model_path: Path,
        ai8x_tools: "Ai8xTools",
        model: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Converts model to ai8x format.

        Parameters
        ----------
        ai8x_model_path : Path
            Path where ai8x-compatible model will be saved.
        ai8x_tools : Ai8xTools
            Ai8X tools wrapper.
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.
        """
        ...

    def to_torch(
        self,
        model: Optional[Any] = None,
        **kwargs,
    ) -> "torch.nn.Module":
        """
        Converts model to PyTorch format.

        Parameters
        ----------
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        torch.nn.Module
            Torch Model.
        """
        ...

    def to_onnx(
        self, model: Optional[Any] = None, **kwargs
    ) -> "onnx.ModelProto":
        """
        Converts model to ONNX format.

        Parameters
        ----------
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model.
        """
        ...

    def to_tflite(
        self,
        model: Optional[Any] = None,
        **kwargs,
    ) -> Union["tf.lite.TFLiteConverter", "tflite.Model.Model"]:
        """
        Converts model to TensorFlowLite format.

        Parameters
        ----------
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        Union[tf.lite.TFLiteConverter, tflite.Model.Model]
            Either a TFLite model, or a TFLiteConverer object.
        """
        ...

    def to_tvm(
        self,
        io_spec: Dict[str, List[Dict]],
        model: Optional[Any] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts model to TVM format.

        Parameters
        ----------
        io_spec: Dict[str, List[Dict]]
            Input and output specification.
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        mod: tvm.IRModule
            The relay module.
        params: Union[Dict, str]
            Parameters dictionary to be used by relay module.
        """
        ...

    def to_keras(
        self,
        model: Optional[Any] = None,
        **kwargs,
    ) -> "tf.keras.Model":
        """
        Converts model to Keras format.

        Parameters
        ----------
        model : Optional[Any]
            Optional model object. If not provided, model is
            loaded from file.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        tf.keras.Model
        """
        ...
