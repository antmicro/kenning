# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading TFLite models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from kenning.core.converter import ModelConverter
from kenning.utils.logger import KLogger

if TYPE_CHECKING:
    import onnx
    import tensorflow as tf
    import tflite
    import tvm


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

            model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model

            model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

        try:
            import tflite

            is_model = isinstance(model, tflite.Model)

        except AttributeError:
            import tflite.Model

            is_model = isinstance(model, tflite.Model.Model)

        if not is_model:
            target = kwargs.get("targets", "default")
            inferenceinputtype = kwargs.get("inferenceinputtype", "float32")
            inferenceoutputtype = kwargs.get("inferenceoutputtype", "float32")
            use_tf_select_ops = kwargs.get("use_tf_select_ops", False)

            if target in ["int8", "edgetpu"]:
                model.optimizations = [tf.lite.Optimize.DEFAULT]
                if inferenceinputtype in [
                    "int8",
                    "uint8",
                ] and inferenceinputtype in ["int8", "uint8"]:
                    model.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
            elif target == "float16":
                model.optimizations = [tf.lite.Optimize.DEFAULT]
                model.target_spec.supported_types = [tf.float16]
            else:
                model.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS
                ]
            if use_tf_select_ops:
                model.target_spec.supported_ops.append(
                    tf.lite.OpsSet.SELECT_TF_OPS
                )
            model.inference_input_type = tf.as_dtype(inferenceinputtype)
            model.inference_output_type = tf.as_dtype(inferenceoutputtype)

        return model

    def to_tvm(
        self,
        io_spec: Dict[str, List[Dict]],
        model: Optional["tf.lite.TFLiteConverter"] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts TFLite model to TVM format.

        Parameters
        ----------
        io_spec: Dict[str, List[Dict]]
            Input and output specification.
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

        Raises
        ------
        ValueError
            Raised if no shapes provided in the input specification.
        IOSpecificationNotFoundError
            Raised if input specification is not provided.
        """
        import tvm.relay as relay

        if model is None:
            model = self.to_tflite(**kwargs)

        input_shapes = {
            spec["name"]: spec["shape"] for spec in io_spec["input"]
        }
        if not input_shapes:
            raise ValueError("No shapes in the input specification")
        dtypes = {spec["name"]: spec["dtype"] for spec in io_spec["input"]}

        return relay.frontend.from_tflite(
            model,
            shape_dict=input_shapes,
            dtype_dict=dtypes,
        )

    def to_onnx(
        self,
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
