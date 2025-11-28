# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of ONNX models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from kenning.core.converter import ModelConverter

if TYPE_CHECKING:
    import onnx
    import tensorflow as tf
    import torch
    import tvm


class OnnxConverter(ModelConverter):
    """
    The ONNX model converter.
    """

    source_format = "onnx"

    def to_onnx(
        self,
        input_spec: Optional[List[Dict]] = None,
        output_names: List = ["output"],
        model: Optional["onnx.ModelProto"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Loads ONNX model.

        Parameters
        ----------
        input_spec: Optional[List[Dict]]
            Input specification.
        output_names: List
            Names of the outputs.
        model : Optional["onnx.ModelProto"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model.

        """
        import onnx

        if not model:
            model = onnx.load_model(str(self.source_model_path))
        return model

    def to_torch(
        self,
        model: Optional["onnx.ModelProto"] = None,
        **kwargs,
    ) -> "torch.nn.Module":
        """
        Converts ONNX model to PyTorch.

        Parameters
        ----------
        model : Optional["onnx.ModelProto"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        torch.nn.Module
            Loaded PyTorch model.

        """
        import onnx
        from onnx2torch import convert

        if not model:
            model = onnx.load(str(self.source_model_path))

    def to_tflite(
        self,
        model: Optional["onnx.ModelProto"] = None,
        **kwargs,
    ) -> "tf.lite.TFLiteConverter":
        """
        Converts ONNX file to TFLite format.

        Parameters
        ----------
        model : Optional["onnx.ModelProto"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        tf.lite.TFLiteConverter
            TFLite converter for model.

        Raises
        ------
        TypeError
            If conversion fails.
        ValueError
            If conversion fails.
        RuntimeError
            If export or converter initialization fails.
        """
        import tempfile
        from datetime import datetime

        import onnx
        import onnx2tf
        import tensorflow as tf

        if model:
            model_path = tempfile.NamedTemporaryFile(suffix=".onnx").name
            onnx.save(model, model_path)
        else:
            model_path = self.source_model_path

        input_names = [input.name for input in model.graph.input]

        # Use multiple options to prevent dynamic shape and symbolic
        #  tensor issues
        # - keep_nwc_or_nhwc_or_ndhwc_input_names: for NHWC input
        #  arrangements
        # - keep_shape_absolutely_input_names: force all shapes to
        #  remain static
        # - disable_strict_mode: skip strict accuracy correction for
        #   speed/compatibility
        # - batch_size: fix dynamic batch to static batch size of 1
        try:
            model = onnx2tf.convert(
                str(model_path),
                keep_nwc_or_nhwc_or_ndhwc_input_names=input_names,
                keep_shape_absolutely_input_names=input_names,
                disable_strict_mode=True,
                batch_size=1,
            )
        except (TypeError, ValueError) as e:
            # Fallback: try simpler conversion without input name preservation
            # This may lose some shape information but should avoid symbolici
            # issues
            if "symbolic inputs/outputs do not implement `__len__`" in str(e):
                model = onnx2tf.convert(
                    str(model_path),
                    disable_strict_mode=True,
                    batch_size=1,
                )
            else:
                raise
        converted_path = self.source_model_path.with_suffix(
            f'.{datetime.now().strftime("%Y%m%d-%H%M%S")}.pb'
        )
        model.export(str(converted_path))
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(converted_path)
        )
        return converter

    def to_tvm(
        self,
        input_shapes: Dict,
        dtypes: Dict,
        model: Optional["onnx.ModelProto"],
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts ONNX model to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        dtypes: Dict
            Mapping from input name to input dtype.
        model : Optional["onnx.ModelProto"]
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
        IndexError
            Raised when no dtype was provided in the IO specification.
        """
        import onnx
        import tvm.relay as relay

        try:
            dtype = list(dtypes.values())[0]
        except IndexError:
            raise IndexError("No dtype in the input specification")
        if not model:
            model = onnx.load(self.source_model_path)

        input_shapes = {
            k: [_v if _v > 0 else 1 for _v in v]
            for k, v in input_shapes.items()
        }
        return relay.frontend.from_onnx(
            model, shape=input_shapes, freeze_params=True, dtype=dtype
        )
