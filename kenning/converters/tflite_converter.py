# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading TFLite models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import ConversionError

if TYPE_CHECKING:
    import onnx
    import tvm


class TFLiteConverter(ModelConverter):
    """
    The TFLite model converter.
    """

    source_format: str = "tflite"

    def to_tvm(
        self,
        input_shapes: Dict,
        dtypes: Dict,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts TFLite model to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        dtypes: Dict
            Mapping from input name to input dtype.

        Returns
        -------
        mod: tvm.IRModule
            The relay module.
        params: Union[Dict, str]
            Parameters dictionary to be used by relay module.
        """
        import tvm.relay as relay

        with open(self.source_model_path, "rb") as f:
            tflite_model_buf = f.read()

        try:
            import tflite

            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model

            tflite_model = tflite.Model.Model.GetRootAsModel(
                tflite_model_buf, 0
            )

        return relay.frontend.from_tflite(
            tflite_model, shape_dict=input_shapes, dtype_dict=dtypes
        )

    def to_onnx(
        self,
        input_spec: List[Dict],
        output_names: List,
    ) -> "onnx.ModelProto":
        """
        Converts TFLite model to ONNX format.

        Parameters
        ----------
        input_spec: List[Dict]
            List of Dictionaries representing inputs.
        output_names: List
            Names of outputs to include in the final model.

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

        try:
            modelproto, _ = tf2onnx.convert.from_tflite(
                str(self.source_model_path)
            )
        except ValueError as e:
            raise ConversionError(e)

        return modelproto
