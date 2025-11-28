# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading TFLite models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Optional

from kenning.core.converter import ModelConverter
from kenning.utils.logger import KLogger

if TYPE_CHECKING:
    import onnx
    import tvm
    import tflite
    import tensorflow as tf


class TFLiteConverter(ModelConverter):
    """
    The TFLite model converter.
    """

    source_format: str = "tflite"

    def to_tflite(
        self,
        model: Optional["tflite.Model.Model"] = None,
        **kwargs,
    ) -> "tflite.Model.Model":
        """
        Loads the TFLite model.

        Parameters
        ----------
        model : Optional["tflite.Model.Model"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        tflite.Model.Model
            Loaded TFLiteConverter.
        """
        if model is not None:
            return model

        with open(self.source_model_path, "rb") as f:
            tflite_model_buf = f.read()

        try:
            import tflite

            return tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model

            return tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    def to_tvm(
        self,
        input_shapes: Dict,
        dtypes: Dict,
        model: Optional["tf.lite.TFLiteConverter"] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts TFLite model to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        dtypes: Dict
            Mapping from input name to input dtype.
        model : Optional["tf.lite.TFLiteConverter"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        mod: tvm.IRModule
            The relay module.
        params: Union[Dict, str]
            Parameters dictionary to be used by relay module.
        """
        import tvm.relay as relay

        if model is None:
            model = self.to_tflite(**kwargs)

        return relay.frontend.from_tflite(
            model,
            shape_dict=input_shapes,
            dtype_dict=dtypes,
        )

    def to_onnx(
        self,
        input_spec: List[Dict],
        output_names: List,
        model: Optional["tf.lite.TFLiteConverter"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Converts TFLite model to ONNX format.

        Parameters
        ----------
        model : Optional["tf.lite.TFLiteConverter"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model, a variant of ModelProto.

        Raises
        ------
        ConversionError
            Raised when model could not be loaded.
        """
        import tf2onnx

        from kenning.core.exceptions import ConversionError

        if model is not None:
            KLogger.warning(
                "TFLite to onnx conversion requires reading from file."
            )

        try:
            converted_model, _ = tf2onnx.convert.from_tflite(
                str(self.source_model_path)
            )
            return converted_model
        except ValueError as e:
            raise ConversionError(e)

