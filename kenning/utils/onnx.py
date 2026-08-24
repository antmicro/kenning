# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for ONNX related functions.
"""
__all__ = ["try_extracting_input_shape_from_onnx"]
from typing import Dict, Iterable, List, Optional

import onnx

from kenning.core.exceptions import DynamicIOSpecError
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


def apply_io_spec_to_model(
    model: onnx.ModelProto, io_spec: Dict[str, List[Dict]]
):
    """
    Function for modifying ONNX graph by setting input/output sizes from
    io_spec.

    This function will raise if not all dimension sizes are specified.

    Parameters
    ----------
    model : onnx.ModelProto
        Loaded ONNX model.
    io_spec : Dict[str, List[Dict]]
        The io specification.
    """
    input_spec = io_spec.get("processed_input", None) or io_spec["input"]

    apply_io_spec_to_node_list(model.graph.input, input_spec)
    apply_io_spec_to_node_list(model.graph.output, io_spec["output"])


def apply_io_spec_to_node_list(
    nodes: Iterable[onnx.TensorProto], sub_io_spec: List[Dict]
):
    """
    Takes a list of ONNX nodes (such as inputs or outputs) and overrides their
    sizes with values specified in sub_io_spec.

    If there is a node that doesn't have a corresponding io_spec entry, it will
    be left untouched.

    Parameters
    ----------
    nodes : Iterable[onnx.TensorProto]
        Nodes with shapes.
    sub_io_spec : List[Dict]
        A list containing part of io_spec, e.g. "input", "processed_input",
        etc. It will be used to override the nodes' shapes.

    Raises
    ------
    ValueError
        Raised when sub_io_spec contains unspecified dimension sizes.
    """
    for node in nodes:
        node_spec = next(
            (spec for spec in sub_io_spec if spec["name"] == node.name), None
        )

        if node_spec is None:
            continue

        for dim, size_from_spec in zip(
            node.type.tensor_type.shape.dim, node_spec["shape"]
        ):
            if (
                size_from_spec is None
                or not isinstance(size_from_spec, int)
                or size_from_spec < 1
            ):
                raise DynamicIOSpecError("Can't set None dimension size")

            dim.dim_value = size_from_spec
