# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Keras models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Optional

from kenning.core.converter import ModelConverter
from kenning.utils.logger import KLogger
from kenning.utils.update_h5_file import update_h5_file

if TYPE_CHECKING:
    import onnx
    import tensorflow as tf
    import tvm


class KerasConverter(ModelConverter):
    """
    The Keras model converter.
    """

    source_format: str = "keras"

    def to_keras(
        self,
        model: Optional["tf.keras.Model"] = None,
        **kwargs,
    ) -> "tf.keras.Model":
        """
        Loads Keras model.

        Parameters
        ----------
        model : Optional["tf.keras.Model"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        tf.keras.Model
        """
        if not model:
            import tensorflow as tf

            model = tf.keras.models.load_model(
                str(self.source_model_path), compile=False
            )
        return model

    def to_tflite(
        self,
        model: Optional["tf.keras.Model"] = None,
        **kwargs: Dict,
    ) -> "tf.lite.TFLiteConverter":
        """
        Converts Keras model to TFLiteConverter.

        Parameters
        ----------
        model : Optional["tf.keras.Model"]
            Optional model object.
        **kwargs: Dict
            Keyword arguments passed between conversions.

        Returns
        -------
        tf.lite.TFLiteConverter
            TFLite converter for model.

        """
        import tensorflow as tf

        if self.source_model_path.suffix in (".h5", ".hdf5"):
            updated_path = update_h5_file(self.source_model_path)
            self.source_model_path = updated_path

        model = tf.keras.models.load_model(
            str(self.source_model_path), compile=False
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter

    def to_onnx(
        self,
        input_spec: Optional[List[Dict]] = None,
        output_names: List = ["output"],
        model: Optional["tf.keras.Model"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Converts Keras model to ONNX format.

        Parameters
        ----------
        input_spec: Optional[List[Dict]]
            List of dictionaries representing inputs.
        output_names: List
            Names of outputs to include in the final model.
        model : Optional["tf.keras.Model"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model, a variant of ModelProto.
        """
        import tensorflow as tf
        import tf2onnx

        if not model:
            model = tf.keras.models.load_model(
                str(self.source_model_path), compile=False
            )

        model.output_names = [
            o["name"] if isinstance(o, dict) else o for o in output_names
        ]

        signature = None
        if input_spec:
            signature = [
                tf.TensorSpec(spec["shape"], spec["dtype"], name=spec["name"])
                for spec in input_spec
            ]

        if signature:
            modelproto, _ = tf2onnx.convert.from_keras(
                model, input_signature=signature
            )
        else:
            modelproto, _ = tf2onnx.convert.from_keras(model)

        return modelproto

    def to_tvm(
        self,
        input_shapes: Dict,
        model: Optional["tf.keras.Model"] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts Keras model to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        model : Optional["tf.keras.Model"]
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
        import tensorflow as tf
        import tvm.relay as relay

        tf.keras.backend.clear_session()
        if not model:
            model = tf.keras.models.load_model(
                str(self.source_model_path), compile=False
            )
        KLogger.info(model.summary())

        mod, params = relay.frontend.from_keras(
            model, shape=input_shapes, layout="NHWC"
        )
        return mod, params
