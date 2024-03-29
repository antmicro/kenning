# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for ONNX related functions.
"""
__all__ = ["try_extracting_input_shape_from_onnx"]
from typing import List, Optional

import onnx

from kenning.utils.logger import KLogger


def try_extracting_input_shape_from_onnx(
    model_onnx: onnx.ModelProto
) -> Optional[List[List]]:
    """
    Function for extracting ONNX model's input shape.

    Parameters
    ----------
    model_onnx : onnx.ModelProto
        Loaded ONNX model

    Returns
    -------
    Optional[List[List]]
        List of tensors input shapes or None if extracting was impossible
    """
    try:
        initializers = set(
            [node.name for node in model_onnx.graph.initializer]
        )
        inputs = model_onnx.graph.input
        shapes = []
        for input_ in inputs:
            if input_.name in initializers:
                continue
            if input_.type.tensor_type.elem_type != 1:
                KLogger.error("Input type differ from Tensor")
                return None
            dims = []
            for dim in input_.type.tensor_type.shape.dim:
                if not dim.dim_value and dim.dim_param != "":  # batch size
                    dims.append(1)
                elif dim.dim_value > 0:  # normal dimension
                    dims.append(dim.dim_value)
                else:
                    KLogger.error(
                        "Input's dimension not known, missing dim_value or "
                        "dim_param attribute"
                    )
                    return None
            shapes.append(dims)
    except AttributeError:
        KLogger.error(
            "ONNX model's graph don't have necessary attributes to extract "
            "input shape",
            stack_info=True,
        )
        return None
    return shapes
