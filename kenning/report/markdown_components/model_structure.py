# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for model layers structure analysis generation.
"""

from functools import cache
from typing import Any, Iterable, Optional

import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto

from kenning.utils.logger import KLogger

BITS_PER_ELEMENT: dict[int, int] = {
    TensorProto.FLOAT: 32,
    TensorProto.UINT8: 8,
    TensorProto.INT8: 8,
    TensorProto.UINT16: 16,
    TensorProto.INT16: 16,
    TensorProto.INT32: 32,
    TensorProto.INT64: 64,
    TensorProto.BOOL: 8,
    TensorProto.FLOAT16: 16,
    TensorProto.DOUBLE: 64,
    TensorProto.UINT32: 32,
    TensorProto.UINT64: 64,
    TensorProto.COMPLEX64: 64,
    TensorProto.COMPLEX128: 128,
    TensorProto.BFLOAT16: 16,
    TensorProto.FLOAT8E4M3FN: 8,
    TensorProto.FLOAT8E4M3FNUZ: 8,
    TensorProto.FLOAT8E5M2: 8,
    TensorProto.FLOAT8E5M2FNUZ: 8,
    TensorProto.UINT4: 4,
    TensorProto.INT4: 4,
}

_ROWS_COLUMNS: tuple[str, ...] = (
    "Layer #",
    "Name",
    "Operation type",
    "Parameters types",
    "Parameters count",
    "Parameters bytes",
)


def _data_type_name(element_type: int) -> str:
    """
    Casts tensor DataType to string.

    Parameters
    ----------
    element_type : int
        TensorProto.DataType enum value.

    Returns
    -------
    str
        DataType enum name. If any enum name matched `UNDEFINED` returned.
    """
    try:
        return TensorProto.DataType.Name(element_type)
    except (TypeError, ValueError):
        return "UNDEFINED"


def _tensor_count(tensor: TensorProto) -> int:
    """
    Counts elements in tensor.

    Parameters
    ----------
    tensor : TensorProto
        Tensor with weights.

    Returns
    -------
    int
        Element count in tensor.
    """
    size = 1
    for d in tensor.dims:
        size *= d
    return size


@cache
def emit_unknown_datatype_warning(data_type: int):
    """
    Emit single logger warning about unknown memory size of given data type.
    """
    KLogger.error(
        f"Unknown memory size for TensorProto.{_data_type_name(data_type)}."
        " Update `BITS_PER_ELEMENT` dictionary."
    )


def _tensor_bytes(tensor: TensorProto) -> int | str:
    """
    Counts elements memory size in tensor.

    Parameters
    ----------
    tensor : TensorProto
        Tensor with weights.

    Returns
    -------
    int | str
        Elements memory size in tensor.
        Returns "Unknown" when failed to get memory size for tensor.
    """
    bits = BITS_PER_ELEMENT.get(tensor.data_type)
    if bits is None:
        emit_unknown_datatype_warning(tensor.data_type)
        return "Unknown"
    return (_tensor_count(tensor) * bits + 7) // 8


def _subgraphs(node: NodeProto) -> Iterable[GraphProto]:
    """
    Checks if node is a subgraph. If yes, node treated as graph.

    Parameters
    ----------
    node: NodeProto
        Operation in model graph.

    Yields
    ------
    GraphProto
        Subgraph inside node.
    """
    for a in node.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            yield a.g
        elif a.type == onnx.AttributeProto.GRAPHS:
            yield from a.graphs


def unknown_add(a: int | str, b: int | str) -> int | str:
    """
    Adds two integers.
    If any of them equals to "Unknown", "Unknown" will be returned.

    Parameters
    ----------
    a: int | str
        First value to add.

    b: int | str
        Second value to add.

    Returns
    -------
    int | str
        Addition result. Returns "Unknown" when a or b is not an integer.
    """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    else:
        return "Unknown"


class ModelStructure:
    """
    Class responsible for model layers structure analysis.
    """

    def __init__(self, onnx_model: ModelProto):
        """
        Initializes the ModelStructure object.

        Parameters
        ----------
        onnx_model: ModelProto
            Neural network model in ONNX format.
        """
        self._onnx_model = onnx_model
        self._visited_initializers = set()

    def _walk(
        self, graph: GraphProto, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """
        Calculates model parameters by iteration over nodes inside
        computational graph.

        Parameters
        ----------
        graph: GraphProto
            Model computation graph or subgraph.
        prefix: str
            Prefix name for subgraphs to distinguish them from main model.

        Returns
        -------
        list[dict[str, Any]]
            Measurements dictionary list.
        """
        layer_nr = 0
        rows = []
        subtables = []
        total: dict[str, Any] = {"layer_count": 0, "parameters": 0, "bytes": 0}

        def add_row(
            layer_name: str,
            layer_op_type: str,
            layer_output_count: Optional[int],
            layer_output_size: int | str,
            tensor_type: str,
        ) -> None:
            nonlocal layer_nr
            layer_nr += 1
            rows.append(
                (
                    layer_nr,
                    layer_name,
                    layer_op_type,
                    tensor_type,
                    layer_output_count,
                    layer_output_size,
                )
            )

        initializers = {tensor.name: tensor for tensor in graph.initializer}

        for value_info in graph.input:
            if value_info.name in initializers:
                continue

            add_row(
                layer_name=prefix + value_info.name,
                layer_op_type="Input",
                layer_output_count=0,
                layer_output_size=0,
                tensor_type=_data_type_name(
                    value_info.type.tensor_type.elem_type
                ),
            )

        for i, node in enumerate(graph.node):
            input_tensor_types = []
            layer_parameters = 0
            layer_parameters_size = 0
            node_label = node.name or f"{node.op_type}_{i}"

            for input_tensor_name in node.input:
                if not input_tensor_name:
                    # unused tensor input
                    continue

                tensor = initializers.get(input_tensor_name)
                if tensor is None:
                    # tensor not stored in initializers
                    continue

                input_tensor_types.append(_data_type_name(tensor.data_type))
                tensor_parameters = _tensor_count(tensor)
                tensor_parameters_size = _tensor_bytes(tensor)

                layer_parameters += tensor_parameters
                layer_parameters_size = unknown_add(
                    layer_parameters_size, tensor_parameters_size
                )

                # shared weight is only counted once
                key = prefix + input_tensor_name
                if key not in self._visited_initializers:
                    self._visited_initializers.add(key)
                    total["parameters"] += tensor_parameters
                    total["bytes"] = unknown_add(
                        total["bytes"], tensor_parameters_size
                    )

            # unique elements in first-seen order
            tensor_type = ", ".join(dict.fromkeys(input_tensor_types)) or "N/A"
            add_row(
                layer_name=prefix + node_label,
                layer_op_type=node.op_type,
                layer_output_count=layer_parameters,
                layer_output_size=layer_parameters_size,
                tensor_type=tensor_type,
            )

            for subgraph in _subgraphs(node):
                subgraph_label = f"{node_label}/{subgraph.name}"
                subtables.extend(
                    self._walk(subgraph, prefix=f"{prefix}{subgraph_label}")
                )

        # unvisited tensors in initializers
        for name, tensor in initializers.items():
            key = prefix + name
            if key in self._visited_initializers:
                continue

            self._visited_initializers.add(key)
            tensor_parameters = _tensor_count(tensor)
            tensor_parameters_size = _tensor_bytes(tensor)
            total["parameters"] += tensor_parameters
            total["bytes"] = unknown_add(
                total["bytes"], tensor_parameters_size
            )

            add_row(
                layer_name=prefix + name,
                layer_op_type="InitializerUnused",
                layer_output_count=tensor_parameters,
                layer_output_size=tensor_parameters_size,
                tensor_type=_data_type_name(tensor.data_type),
            )

        for value_info in graph.output:
            add_row(
                layer_name=prefix + value_info.name,
                layer_op_type="Output",
                layer_output_count=0,
                layer_output_size=0,
                tensor_type=_data_type_name(
                    value_info.type.tensor_type.elem_type
                ),
            )

        total["layer_count"] = layer_nr
        table = {
            "model_name": prefix or "Main",
            "columns": _ROWS_COLUMNS,
            "rows": rows,
            "total": total,
        }
        return [table, *subtables]

    def build_layer_table(self) -> list[dict[str, Any]]:
        """
        Creates table with model layer details and model parameter information.

        Returns
        -------
        list[dict[str, Any]]
            Measurements dictionary list.
        """
        self._visited_initializers = set()
        return self._walk(self._onnx_model.graph)
