# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading LiteRT (TFLite) models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import ConversionError
from kenning.utils.logger import KLogger

if TYPE_CHECKING:
    import ai_edge_litert
    import onnx
    import tensorflow as tf
    import tvm


class LiteRTConverter(ModelConverter):
    """
    The LiteRT model converter.
    """

    source_format: str = "litert"

    def to_litert(
        self,
        model: Optional["ai_edge_litert.compiled_model.CompiledModel"] = None,
        **kwargs,
    ) -> "ai_edge_litert.schema_py_generated.ModelT":
        """
        Loads the LiteRT (TFLite) model.

        Parameters
        ----------
        model : Optional["ai_edge_litert.compiled_model.CompiledModel"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        ai_edge_litert.schema_py_generated.ModelT
            Loaded LiteRTConverter.
        """
        if model is not None:
            return model

        from ai_edge_litert import schema_py_generated as schema_fb
        from ai_edge_litert.tools import flatbuffer_utils as fbu

        model = fbu.read_model(self.source_model_path)
        is_model = isinstance(model, schema_fb.ModelT)

        if not is_model:
            import tensorflow as tf

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
        Converts LiteRT model to TVM format.

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
        import tensorflow as tf
        import tvm.relay as relay

        if model is None or not isinstance(model, tf.lite.TFLiteConverter):
            with open(self.source_model_path, "rb") as f:
                tflite_model_buf = f.read()
            import tflite

            model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        if isinstance(model, tf.lite.TFLiteConverter):
            model_data = model.convert()
            model = tflite.Model.GetRootAsModel(model_data, 0)

        if isinstance(model, tf.lite.TFLiteConverter):
            import tflite

            model_bytes = model.convert()
            model = tflite.Model.GetRootAsModel(model_bytes, 0)

        input_shapes = {
            spec["name"]: spec["shape"] for spec in io_spec["processed_input"]
        }
        if not input_shapes:
            raise ValueError("No shapes in the input specification")
        dtypes = {
            spec["name"]: spec["dtype"] for spec in io_spec["processed_input"]
        }

        return relay.frontend.from_tflite(
            model,
            shape_dict=input_shapes,
            dtype_dict=dtypes,
        )

    def to_tvm_relax(
        self,
        io_spec: Dict[str, List[Dict]],
        model: Optional["tf.lite.TFLiteConverter"] = None,
        **kwargs,
    ) -> "tvm.IRModule":
        """
        Converts TFLite model to TVM Relax module.

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
        tvm.IRModule
            The Relax module, with weights embedded as constants.

        Raises
        ------
        ValueError
            Raised if no shapes provided in the input specification.
        """
        import tflite
        from tvm.relax.frontend.tflite import from_tflite

        model = self.to_tflite(model=model, **kwargs)

        if not isinstance(model, tflite.Model):
            model_data = model.convert()
            model = tflite.Model.GetRootAsModel(model_data, 0)

        input_shapes = {
            spec["name"]: spec["shape"] for spec in io_spec["processed_input"]
        }
        if not input_shapes:
            raise ValueError("No shapes in the input specification")
        dtypes = {
            spec["name"]: spec["dtype"] for spec in io_spec["processed_input"]
        }

        return from_tflite(model, shape_dict=input_shapes, dtype_dict=dtypes)

    def to_onnx(
        self,
        model: Optional["tf.lite.TFLiteConverter"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Converts LiteRT (TFLite) model to ONNX format.

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

        # Register custom op handlers
        import kenning.utils.tf2onnx  # noqa: F401

        if model is not None:
            KLogger.warning(
                "LiteRT (TFLite) to onnx "
                "conversion requires reading from file."
            )

        try:
            converted_model, _ = tf2onnx.convert.from_tflite(
                str(self.source_model_path)
            )
            return converted_model
        except ValueError as e:
            raise ConversionError(e)
