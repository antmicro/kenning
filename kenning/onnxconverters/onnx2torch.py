# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for converting ONNX models to PyTorch framework with help of onnx2torch.
It contains custom converters to enable pruning with NNI.
WARNING: importing this module will change onnx2torch default converters,
to reverte changes see restore_default_converters function.
"""

__all__ = [
    "convert",
    "restore_default_converters",
    "restore_custom_converters",
]
import copy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import onnx
import onnx2torch
import torch
from onnx2torch.node_converters import onnx_mapping_from_node
from onnx2torch.node_converters.batch_norm import _ as _batch_converter
from onnx2torch.node_converters.binary_math_operations import (
    _ as _binary_math_converter,
)
from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters.matmul import _ as _matmul_converter
from onnx2torch.node_converters.max_pool import _ as _max_pool_converter
from onnx2torch.node_converters.registry import (
    _CONVERTER_REGISTRY,
    OperationDescription,
    add_converter,
)
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode, OnnxTensor
from onnx2torch.utils.common import (
    OnnxMapping,
    OperationConverterResult,
)

from kenning.core.exceptions import NotSupportedError
from kenning.utils.logger import KLogger

CONST_NODES = dict()

# Backing up default converters
_CONVERTER_REGISTRY_BACKUP = copy.deepcopy(_CONVERTER_REGISTRY)
_REMOVED_DEFAULT_CONVERTER = (
    ("Gemm", (9, 11, 13)),
    ("MatMul", (1, 9, 13)),
    ("BatchNormalization", (9, 14, 15)),
    ("Dropout", (10, 12, 13)),
    ("Reshape", (5, 13, 14)),
    ("MaxPool", (8, 10, 11, 12)),
    ("Add", (1, 6, 7, 13, 14)),
    ("Sub", (1, 6, 7, 13, 14)),
    ("Shape", (1, 13, 15)),
)
# Removing default converters
for op_type, versions in _REMOVED_DEFAULT_CONVERTER:
    for version in versions:
        del _CONVERTER_REGISTRY[
            OperationDescription(
                operation_type=op_type,
                version=version,
                domain=onnx.defs.ONNX_DOMAIN,
            )
        ]


def create_linear_from_weights(
    weights: torch.Tensor,
    bias: Optional[Union[torch.Tensor, bool]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.nn.Linear:
    """
    The function for creating torch Linear layer based on provided
    weights and biases.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor with layer weights
    bias : Optional[Union[torch.Tensor, bool]]
        Tensor with layer bias or None/False when layer don't have bias
    alpha : float
        Scaling factor for weights
    beta : float
        Scaling factor for bias

    Returns
    -------
    torch.nn.Linear
        Created Linear layer
    """
    if bias is False:
        bias = None
    in_feature, out_feature = weights.shape[1], weights.shape[0]
    linear = torch.nn.Linear(
        in_features=in_feature,
        out_features=out_feature,
        bias=bias is not None,
    )
    with torch.no_grad():
        weights = weights * alpha
        linear.weight.data = weights
        if bias is not None:
            bias = bias * beta
            linear.bias.data = bias
    return linear


def extract_value_from_graph(
    node: OnnxNode,
    graph: OnnxGraph,
    value_name: str,
    default_value: Optional = None,
) -> Optional[torch.Tensor]:
    """
    The function for extracting values from graph of converted model.

    Parameters
    ----------
    node : OnnxNode
        Currently processed node
    graph : OnnxGraph
        Graph representing model
    value_name : str
        Name of the value which will be extracted
    default_value : Optional
        Any value which will be returned if value_name cannot be extracted

    Returns
    -------
    Optional[torch.Tensor]
        Extracted value or default_value
    """
    value = None
    if value_name in node.attributes:
        value = node.attributes.get(value_name, default_value)
    elif value_name in graph.initializers:
        value = graph.initializers[value_name]
    elif value_name in graph._node_output_values:
        # Value is output of other node, checking if node is constant
        value_node, _ = graph.value_as_node_output(value_name)
        if value_node.operation_type == "Constant":
            value = value_node.attributes.get("value", default_value)
        elif value_node.name in CONST_NODES:
            value = CONST_NODES[value_node.name].value

    if value is None:
        return default_value

    # Converting onnx2torch's types to PyTorch and Python types
    if isinstance(value, OnnxTensor):
        value = value.to_torch()
    elif isinstance(value, list):
        if isinstance(value[0], OnnxTensor):
            value = [v.to_torch() for v in value]
        elif isinstance(value, (int, float)):
            value = torch.tensor(value)
    return value


class Transposition(torch.nn.Module):
    """
    Artificial torch Module for transposing input.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.transpose(x, dim0=1, dim1=-1)


@add_converter(operation_type="Gemm", version=9)
@add_converter(operation_type="Gemm", version=11)
@add_converter(operation_type="Gemm", version=13)
def gemm_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Conversion from Gemm to torch Linear layer.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting Gemm

    Raises
    ------
    RuntimeError
        Raised when weights were not found
    """
    a_name = node.input_values[0]
    b_name = node.input_values[1]
    c_name = node.input_values[2] if len(node.input_values) > 2 else None

    # Extracting nodes attributes
    alpha = extract_value_from_graph(node, graph, "alpha", 1.0)
    beta = extract_value_from_graph(node, graph, "beta", 1.0)
    trans_a = extract_value_from_graph(node, graph, "transA", 0) != 0
    trans_b = extract_value_from_graph(node, graph, "transB", 0) != 0

    sequence = []
    if trans_a:
        sequence.append(Transposition())

    bias = extract_value_from_graph(node, graph, c_name)
    weights = extract_value_from_graph(node, graph, b_name)

    if weights is None:
        raise RuntimeError("Missing weights for linear layer")

    # Creating and returning Linear layer
    if not trans_b:
        weights = weights.T
    linear = create_linear_from_weights(weights, bias, alpha, beta)
    sequence.append(linear)
    mapping = OnnxMapping(inputs=(a_name,), outputs=node.output_values)
    if len(sequence) > 1:
        return OperationConverterResult(
            torch_module=torch.nn.Sequential(*sequence),
            onnx_mapping=mapping,
        )
    return OperationConverterResult(
        torch_module=sequence[0], onnx_mapping=mapping
    )


@add_converter(operation_type="MatMul", version=1)
@add_converter(operation_type="MatMul", version=9)
@add_converter(operation_type="MatMul", version=13)
def matmul_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Conversion from MatMul to (if possible) torch Linear layer
    or OnnxMatmul.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting MatMul
    """
    in_name_0 = node.input_values[0]
    in_name_1 = node.input_values[1]
    sequence = []
    matrix_0 = extract_value_from_graph(node, graph, in_name_0)
    matrix_1 = extract_value_from_graph(node, graph, in_name_1)
    if matrix_0 is not None:
        # Linear layer works as X*W, where X is input, so to give the same
        # result as MatMul A*B, where B is input, some transposition
        # have to be made: A*B=B.T*A.T
        # matrix_0 is not transposed as Linear layer store transposed weights
        weights = matrix_0
        sequence.append(Transposition())
        in_name_0, in_name_1 = in_name_1, in_name_0
    elif matrix_1 is not None:
        weights = matrix_1.T
    else:
        # If both matrices aren't known, returns default conversion
        return _matmul_converter(node, graph)

    sequence.append(create_linear_from_weights(weights))
    mapping = OnnxMapping(inputs=(in_name_0,), outputs=node.output_values)
    if len(sequence) > 1:
        return OperationConverterResult(
            torch_module=torch.nn.Sequential(*sequence),
            onnx_mapping=mapping,
        )
    return OperationConverterResult(
        torch_module=sequence[0], onnx_mapping=mapping
    )


@add_converter(operation_type="BatchNormalization", version=9)
@add_converter(operation_type="BatchNormalization", version=14)
@add_converter(operation_type="BatchNormalization", version=15)
def batch_norm_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Extension of onnx2torch's BatchNormalization conversion with
    reducing (if needed) number of inputs.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting BatchNormalization
    """
    if len(node.output_values) > 1:
        KLogger.warning("Number of BatchNormalization outputs reduced to one")
        node._output_values = (node.output_values[0],)
    return _batch_converter(node, graph)


@add_converter(operation_type="Dropout", version=10)
@add_converter(operation_type="Dropout", version=12)
@add_converter(operation_type="Dropout", version=13)
def dropout_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Extension of onnx2torch's Dropout conversion with reducing
    (if needed) number of node's inputs and output.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting Dropout

    Raises
    ------
    NotImplementedError
        Raised when no seeds for Dropout are provided
    """
    if len(node.input_values) > 1:
        KLogger.warning("Number of Dropout inputs reduced to one")
        node._input_values = (node.input_values[0],)
    if len(node.output_values) > 1:
        KLogger.warning("Number of Dropout outputs reduced to one")
        node._output_values = (node.output_values[0],)
    ratio = extract_value_from_graph(node, graph, "ratio", 0.5)
    seed = extract_value_from_graph(node, graph, "seed")
    if seed is not None:
        raise NotSupportedError("Dropout nodes seeds are not supported")

    dropout = torch.nn.Dropout(ratio)
    return OperationConverterResult(
        torch_module=dropout, onnx_mapping=onnx_mapping_from_node(node)
    )


class ReshapeWithConstShape(torch.nn.Module):
    """
    Artificial torch Reshaping module with constant reshaping shape.
    """

    def __init__(self, size: Tuple[int, ...]) -> None:
        super().__init__()
        self.shape = size

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x[0], torch.Size((x[0].shape[0], *self.shape)))


class Reshape(OnnxReshape):
    """
    Extension of OnnxReshape with correcting inputs order.
    """

    def flatten(self, input_, shape):
        return torch.flatten(input_, startt_dim=shape.shape[0] - 1)

    def forward(self, *inputs):
        if len(inputs) <= 2:
            return torch.flatten(inputs[0], start_dim=1)
        if inputs[0].dim() == 1:
            if inputs[0].shape[0] <= 2:
                return self.flatten(inputs[1], inputs[0])
            return super().forward(inputs[1], inputs[0])
        if inputs[1].shape[0] <= 2:
            return self.flatten(inputs[0], inputs[1])
        return super().forward(*inputs)


@add_converter(operation_type="Reshape", version=5)
@add_converter(operation_type="Reshape", version=13)
@add_converter(operation_type="Reshape", version=14)
def reshape_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Conversion of Reshape to:
    - ReshapeWithConstantShape, if shape is constant
    - torch.nn.Flatten, if shape is constant and it's length is smaller than 3
    - Reshape, in other cases.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting Reshape
    """
    shape = extract_value_from_graph(node, graph, node.input_values[1])
    mapping = OnnxMapping(
        inputs=(node.input_values[0],),
        outputs=(node.output_values[0],),
    )
    if shape is None:
        return OperationConverterResult(
            torch_module=Reshape(), onnx_mapping=onnx_mapping_from_node(node)
        )

    if shape.shape[0] <= 2:
        return OperationConverterResult(
            torch_module=torch.nn.Flatten(start_dim=shape.shape[0] - 1),
            onnx_mapping=mapping,
        )
    return OperationConverterResult(
        torch_module=ReshapeWithConstShape(tuple(shape[1:])),
        onnx_mapping=mapping,
    )


@add_converter(operation_type="MaxPool", version=12)
@add_converter(operation_type="MaxPool", version=11)
@add_converter(operation_type="MaxPool", version=10)
@add_converter(operation_type="MaxPool", version=8)
def max_pool_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Extension of onnx2torch's MaxPool conversion with forcing
    symmetical padding.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting MaxPool

    Raises
    ------
    RuntimeError
        Raised when paddings can't be symmetrical
    """
    padding = extract_value_from_graph(node, graph, "pads")
    if padding is not None:
        KLogger.warning("Forcing symmetric paddings")
        half_len = len(padding) // 2
        begin_pads, end_pads = padding[:half_len], padding[half_len:]
        new_padding = []
        for begin_pad, end_pad in zip(begin_pads, end_pads):
            if (begin_pad + end_pad) % 2 != 0:
                raise RuntimeError("Cannot make paddings symmetrical")
            new_padding.append((begin_pad + end_pad) // 2)
        new_padding.extend(new_padding)
        node._proto_attributes["pads"] = new_padding

    return _max_pool_converter(node, graph)


@add_converter(operation_type="Add", version=1)
@add_converter(operation_type="Add", version=6)
@add_converter(operation_type="Add", version=7)
@add_converter(operation_type="Add", version=13)
@add_converter(operation_type="Add", version=14)
@add_converter(operation_type="Sub", version=1)
@add_converter(operation_type="Sub", version=6)
@add_converter(operation_type="Sub", version=7)
@add_converter(operation_type="Sub", version=13)
@add_converter(operation_type="Sub", version=14)
def add_sub_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Extension of onnx2torch's Add and Sub conversion, if one input is not
    an output from other node, then convert to Linear layer with bias.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting Add or Sub
    """
    in_name_0 = node.input_values[0]
    in_name_1 = node.input_values[1]
    input_0 = extract_value_from_graph(node, graph, in_name_0)
    input_1 = extract_value_from_graph(node, graph, in_name_1)
    if input_0 is not None:
        bias = input_0
        input_name = in_name_1
    elif input_1 is not None:
        bias = input_1
        input_name = in_name_0
    else:
        # If both input is not known return default conversion
        return _binary_math_converter(node, graph)
    if node.operation_type == "Sub":
        # Subtraction is addition of reversed value
        bias = bias * -1
    # Creating Linear layer with frozen weights as identity matrix
    weights = torch.eye(bias.shape[-1], requires_grad=False)
    linear = create_linear_from_weights(weights, bias)
    mapping = OnnxMapping(inputs=(input_name,), outputs=node.output_values)
    return OperationConverterResult(torch_module=linear, onnx_mapping=mapping)


class ShapeWithMemory(torch.nn.Module):
    """
    Artificial torch Shape module, returning shape of the input Tensor
    or, if input is not provided, previously returned value.
    """

    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        super().__init__()
        self.start = start
        self.end = end
        self.output = None

    def __call__(self, *input_tensor):
        if input_tensor[0] is not None:
            self.output = self.forward(input_tensor[0])
        return self.output

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        shape = input_tensor.shape
        return torch.tensor(shape, device=input_tensor.device)


@add_converter(operation_type="Shape", version=1)
@add_converter(operation_type="Shape", version=13)
@add_converter(operation_type="Shape", version=15)
def shape_converter(
    node: OnnxNode, graph: OnnxGraph
) -> OperationConverterResult:
    """
    Conversion of Shape to ShapeWithMemory.

    Parameters
    ----------
    node : OnnxNode
        The Gemm node for conversion
    graph : OnnxGraph
        The whole model wrapped in OnnxGraph

    Returns
    -------
    OperationConverterResult
        Scheme for converting Shape
    """
    return OperationConverterResult(
        torch_module=ShapeWithMemory(),
        onnx_mapping=onnx_mapping_from_node(node),
    )


def fill_none(
    module: torch.nn.Module, *inputs: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    The function for combining input list from inputs and constants.

    Parameters
    ----------
    module : torch.nn.Module
        Module which will receive the input, has to have template attribute
    *inputs : List[torch.Tensor]
        Inputs for the forward method

    Returns
    -------
    List[torch.Tensor]
        Completed list of inputs
    """
    i = 0
    c = 0
    result = []
    for id, constant in enumerate(module.template):
        if not constant:
            result.append(inputs[i] if len(inputs) > i else None)
            i += 1
        else:
            result.append(module.get_buffer(f"const{c}"))
            c += 1
    return result


class FunctionWrapperForCheckingConst:
    """
    Class wrapping existing onnx2torch converters, for checking
    constant inputs and converting nodes to OnnxConstant if necessary.
    """

    # Operation types with different behavior depending on test/eval
    op_type_train_eval = {
        "Dropout",
        "BatchNormalization",
        "LayerNormalization",
    }

    def __init__(self, default_converter):
        self.default_converter = default_converter

    def __call__(self, node: OnnxNode, graph: OnnxGraph):
        default_conversion = self.default_converter(node, graph)
        params = list(default_conversion.torch_module.parameters())

        if node.operation_type == "Constant":
            CONST_NODES[node.name] = default_conversion.torch_module
            return default_conversion

        # Trying to extract input values from model's graph
        inputs = [
            extract_value_from_graph(node, graph, name)
            for name in default_conversion.onnx_mapping.inputs
        ]
        constant_input = [_input is not None for _input in inputs]

        if all(constant_input):
            # All inputs are constant, so node will be converted
            # to OnnxConstant
            const = OnnxConstant(default_conversion.torch_module(*inputs))
            CONST_NODES[node.name] = const
            return OperationConverterResult(
                torch_module=const,
                onnx_mapping=OnnxMapping(
                    inputs=tuple(), outputs=node.output_values
                ),
            )

        if (any(constant_input) and len(params) > 0) or len(params) == 0:
            # Some inputs are constants or module don't have
            # parameters - fixing number of inputs
            inputs_name = tuple(
                (
                    name
                    for name, const in zip(
                        default_conversion.onnx_mapping.inputs, constant_input
                    )
                    if not const
                )
            )
            module: torch.nn.Module = type(
                node.operation_type,
                (torch.nn.Module,),
                {
                    "template": constant_input,
                    "forward": lambda self, *inputs: self.wrapped_module(
                        *fill_none(self, *inputs)
                    ),
                },
            )()
            if (
                len(params) > 0
                or node.operation_type in self.op_type_train_eval
            ):
                # Creating module with registered submodule - NNI can see it
                # and prune it separately
                module.register_module(
                    "wrapped_module", default_conversion.torch_module
                )
            else:
                # Creating module without registered submodule
                module.__dict__[
                    "wrapped_module"
                ] = default_conversion.torch_module
            # Registering constant values as module buffer,
            # so it will be moved to specific device when method .to() invoked
            i = 0
            for const in inputs:
                if isinstance(const, torch.Tensor):
                    module.register_buffer(f"const{i}", const)
                    i += 1
            return OperationConverterResult(
                torch_module=module,
                onnx_mapping=OnnxMapping(
                    inputs=inputs_name,
                    outputs=default_conversion.onnx_mapping.outputs,
                ),
            )

        return default_conversion


# Wrapping all converters in FunctionWrapperForCheckingConst
for description, converter in _CONVERTER_REGISTRY.items():
    _CONVERTER_REGISTRY[description] = FunctionWrapperForCheckingConst(
        converter
    )


def convert(
    onnx_model: Union[Path, onnx.ModelProto]
) -> torch.fx.graph_module.GraphModule:
    """
    Function for converting model from ONNX framework to PyTorch.

    Parameters
    ----------
    onnx_model : Union[Path, onnx.ModelProto]
        Path to ONNX model or loaded ONNX model

    Returns
    -------
    torch.fx.graph_module.GraphModule
        Model converted to PyTorch framework
    """
    global CONST_NODES
    CONST_NODES = dict()

    converted_model = onnx2torch.convert(onnx_model)

    CONST_NODES = None
    return converted_model


def restore_default_converters():
    """
    Restoring default onnx2torch converters.
    """
    for description, func in _CONVERTER_REGISTRY_BACKUP.items():
        _CONVERTER_REGISTRY[description] = func


def restore_custom_converters():
    """
    Restoring custom converters for NNI pruning.
    """
    import importlib

    from kenning.onnxconverters import onnx2torch

    importlib.reload(onnx2torch)
