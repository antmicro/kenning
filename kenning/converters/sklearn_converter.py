# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of SKLearn models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import (
    ConversionError,
)

if TYPE_CHECKING:
    import onnx

_DEFAULT_DEVICE = "cpu"


class SKLearnConverter(ModelConverter):
    """
    The SKLearn model converter.
    """

    source_format: str = "sklearn"

    def to_sklearn(
        self,
        model: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Loads model.

        Parameters
        ----------
        model : Optional[Any]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        Any
            SKLearn model.
        """
        import joblib

        if not model:
            model = joblib.load(
                self.source_model_path,
            )
        return model

    def to_onnx(
        self,
        io_spec: Dict[str, List[Dict]],
        model: Optional[Any] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Converts model to ONNX.

        Parameters
        ----------
        io_spec: Dict[str, List[Dict]]
            Input and output specification.
        model : Optional[Any]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model.
        """
        model = self.to_sklearn(model)

        def type_lookup(type_string, shape):
            import skl2onnx.common.data_types as dtypes

            map = {
                "float64": dtypes.DoubleTensorType(shape),
                "float32": dtypes.FloatTensorType(shape),
                "int64": dtypes.Int64TensorType(shape),
                "int16": dtypes.Int64TensorType(shape),
                "int8": dtypes.Int64TensorType(shape),
                "uint64": dtypes.UInt64TensorType(shape),
                "uint16": dtypes.UInt64TensorType(shape),
                "uint8": dtypes.UInt64TensorType(shape),
                "float16": dtypes.Float16TensorType(shape),
                "bool": dtypes.BooleanTensorType(shape),
            }
            if type_string in map.keys():
                return map[type_string]
            raise ConversionError(
                "This type does not allow for conversion to ONNX"
            )

        from skl2onnx import to_onnx

        options = {id(model): {"zipmap": False}}
        initial_types = [
            (input["name"], type_lookup(input["dtype"], input["shape"]))
            for input in (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )
        ]
        onnx_model = to_onnx(
            model,
            initial_types=initial_types,
            target_opset=12,
            options=options,
        )
        return onnx_model
