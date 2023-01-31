# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ONNXConversion for PyTorch models.
"""

from copy import deepcopy
import torchvision.models as models
import torch
import onnx
from pathlib import Path
from typing import Optional, Tuple, Union, List

from kenning.core.onnxconversion import ONNXConversion
from kenning.core.onnxconversion import SupportStatus


class PyTorchONNXConversion(ONNXConversion):
    def __init__(self):
        super().__init__('pytorch', torch.__version__)

    def prepare(self):
        self.add_entry(
            'DenseNet201',
            lambda: models.densenet201(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        self.add_entry(
            'MobileNetV2',
            lambda: models.mobilenet_v2(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        self.add_entry(
            'ResNet50',
            lambda: models.resnet50(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        # self.add_entry(
        #     'VGG16',
        #     lambda: models.vgg16(True),
        #     input_tensor=torch.randn((1, 3, 224, 224))
        # )
        self.add_entry(
            'DeepLabV3 ResNet50',
            lambda: models.segmentation.deeplabv3_resnet50(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        self.add_entry(
            'Faster R-CNN ResNet50 FPN',
            lambda: models.detection.fasterrcnn_resnet50_fpn(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        self.add_entry(
            'RetinaNet ResNet50 FPN',
            lambda: models.detection.retinanet_resnet50_fpn(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )
        self.add_entry(
            'Mask R-CNN',
            lambda: models.detection.maskrcnn_resnet50_fpn(True),
            input_tensor=torch.randn((1, 3, 224, 224))
        )

    def onnx_export(self, modelentry, exportpath):
        model = modelentry.modelgenerator()
        input_tensor = modelentry.parameters['input_tensor']
        torch.onnx.export(model, input_tensor, exportpath, opset_version=11)
        del model
        return SupportStatus.SUPPORTED

    def onnx_import(self, modelentry, importpath):
        model_onnx = onnx.load(str(importpath))
        if model_onnx.ir_version <= 3:
            self.logger.error("Model unsupported due to the not suficient "
                              f"ir_version {model_onnx.ir_version}, have to be"
                              " grater than 3")
            return SupportStatus.UNSUPPORTED

        if modelentry is not None:
            input_tensor = modelentry.parameters["input_tensor"]
        else:
            input_tensor = self.try_extracting_input_shape_from_onnx(
                model_onnx
            )
            input_tensor = (
                [torch.rand(shape) for shape in input_tensor]
                if input_tensor
                else None
            )

        if input_tensor is None:
            self.logger.error("Cannot get properties of input tensor")
            return SupportStatus.UNSUPPORTED

        try:
            model_torch = self.onnx_to_torch(model_onnx)
        except RuntimeError or NotImplementedError as e:
            self.logger.error(f"Conversion: {e}")
            del model_onnx
            return SupportStatus.UNSUPPORTED

        model_torch.train()
        model_torch(*input_tensor)

        del model_onnx  # noqa: F821
        del model_torch
        return SupportStatus.SUPPORTED

    def onnx_to_torch(self, onnx_model: Union[Path, onnx.ModelProto]):
        """
        Function for converting model from ONNX framework to PyTorch

        Parameters
        ----------
        onnx_model: Path | ModelProto
            Path to ONNX model or loaded ONNX model

        Returns
        -------
        Model converted to PyTorch framework
        """
        import onnx2torch
        from onnx2torch.node_converters.registry import (
            add_converter,
            get_converter,
            _CONVERTER_REGISTRY,
            OperationDescription,
        )
        from onnx2torch.onnx_node import OnnxNode, OnnxTensor
        from onnx2torch.onnx_graph import OnnxGraph
        from onnx2torch.node_converters import onnx_mapping_from_node
        from onnx2torch.node_converters.batch_norm import _ as _batch_converter
        from onnx2torch.node_converters.reshape import OnnxReshape
        from onnx2torch.node_converters.matmul import _ as _matmul_converter
        from onnx2torch.node_converters.gather import OnnxGather
        from onnx2torch.node_converters.constant import OnnxConstant
        from onnx2torch.node_converters.activations import OnnxSoftmaxV1V11
        from onnx2torch.node_converters.concat import (
            OnnxConcat,
            _ as _concat_converter
        )
        from onnx2torch.node_converters.max_pool import (
            _ as _max_pool_converter
        )
        from onnx2torch.node_converters.binary_math_operations import (
            _ as _binary_math_converter,
        )
        from onnx2torch.utils.common import (
            OperationConverterResult,
            OnnxMapping,
        )
        _unsqueeze_converter = get_converter('Unsqueeze', 11)

        # Backup copy and removing default converters
        converter_registy_copy = deepcopy(_CONVERTER_REGISTRY)
        default_versions = (
            ("Gemm", (9, 11, 13)),
            ("MatMul", (1, 9, 13)),
            ("BatchNormalization", (9, 14, 15)),
            ("Dropout", (10, 12, 13)),
            ("Reshape", (5, 13, 14)),
            ("MaxPool", (8, 10, 11, 12)),
            ("Add", (1, 6, 7, 13, 14)),
            ("Sub", (1, 6, 7, 13, 14)),
            ("Shape", (1, 13, 15)),
            ("Gather", (1, 11, 13)),
            ("Concat", (4, 11, 13)),
            ("LRN", (1, 13)),
            ("Softmax", (1, 11)),
            ("Unsqueeze", (1, 11)),
        )
        for op_type, versions in default_versions:
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
        ):
            """
            The function for creating torch Linear layer based on provided
            weights and biases

            Parameters
            ----------
            weights: torch.Tensor
                Tensor with layer weights
            bias: torch.Tensor | bool | None
                Tensor with layer bias or None/False when layer don't have bias
            alpha: float
                Scaling factor for weights
            beta: float
                Scaling factor for bias

            Returns
            -------
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
            default_value: Optional = None
        ) -> Optional[torch.Tensor]:
            """
            The function for extracting values from graph of converted model

            Parameters
            ----------
            node: OnnxNode
                Currently processed node
            graph: OnnxGraph
                Graph representing model
            value_name: str
                Name of the value which will be extracted
            default_value: Optional
                Any value which will be returned if value_name cannot
                be extracted

            Returns
            -------
            Extracted value or default_value
            """
            value = None
            if value_name in node.attributes:
                value = node.attributes.get(value_name, default_value)
            elif value_name in graph.initializers:
                value = graph.initializers[value_name]
            elif value_name in graph._node_output_values:
                value_node, _ = graph.value_as_node_output(value_name)
                value = value_node.attributes.get('value', default_value)

            if value is None:
                return default_value

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
            Artifficial torch Module for transposing input
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
            Conversion from Gemm to torch Linear layer

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            a_name = node.input_values[0]
            b_name = node.input_values[1]
            c_name = (
                node.input_values[2] if len(node.input_values) > 2 else None
            )

            alpha = extract_value_from_graph(node, graph, 'alpha', 1.0)
            beta = extract_value_from_graph(node, graph, 'beta', 1.0)
            trans_a = extract_value_from_graph(node, graph, 'transA', 0) != 0
            trans_b = extract_value_from_graph(node, graph, 'transB', 0) != 0

            sequence = []
            if trans_a:
                sequence.append(Transposition())

            bias = extract_value_from_graph(node, graph, c_name)
            weights = extract_value_from_graph(node, graph, b_name)

            if weights is not None:
                if not trans_b:
                    weights = weights.T
                linear = create_linear_from_weights(weights, bias, alpha, beta)
                sequence.append(linear)
            else:  # weights are missing
                raise RuntimeError("Missing weights for linear layer")
            mapping = OnnxMapping(inputs=(a_name,), outputs=node.output_values)
            if len(sequence) > 1:
                return OperationConverterResult(
                    torch_module=torch.nn.Sequential(*sequence),
                    onnx_mapping=mapping,
                )
            else:
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
            or OnnxMatmul

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            in_name_0 = node.input_values[0]
            in_name_1 = node.input_values[1]
            sequence = []
            matrix_0 = extract_value_from_graph(node, graph, in_name_0)
            matrix_1 = extract_value_from_graph(node, graph, in_name_1)
            if matrix_0 is not None:
                weights = matrix_0
                sequence.append(Transposition())
                in_name_0, in_name_1 = in_name_1, in_name_0
            elif matrix_1 is not None:
                weights = matrix_1.T
            else:
                return _matmul_converter(node, graph)

            sequence.append(create_linear_from_weights(weights))
            mapping = OnnxMapping(
                inputs=(in_name_0,), outputs=node.output_values
            )
            if len(sequence) == 1:
                return OperationConverterResult(
                    torch_module=sequence[0], onnx_mapping=mapping
                )
            else:
                return OperationConverterResult(
                    torch_module=torch.nn.Sequential(*sequence),
                    onnx_mapping=mapping,
                )

        @add_converter(operation_type="BatchNormalization", version=9)
        @add_converter(operation_type="BatchNormalization", version=14)
        @add_converter(operation_type="BatchNormalization", version=15)
        def batch_norm_converter(
            node: OnnxNode, graph: OnnxGraph
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's BatchNormalization conversion with
            reducing (if needed) number of inputs

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            if len(node.output_values) > 1:
                self.logger.warning(
                    "Number of BatchNormalization outputs reduced to one"
                )
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
            (if needed) number of node's inputs and output

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            if len(node.input_values) > 1:
                self.logger.warning("Number of Dropout inputs reduced to one")
                node._input_values = (node.input_values[0],)
            if len(node.output_values) > 1:
                self.logger.warning("Number of Dropout outputs reduced to one")
                node._output_values = (node.output_values[0],)
            ratio = extract_value_from_graph(node, graph, "ratio", 0.5)
            seed = extract_value_from_graph(node, graph, "seed")
            if seed is not None:
                raise NotImplementedError(
                    "Dropout nodes seeds are not supported"
                )

            dropout = torch.nn.Dropout(ratio)
            return OperationConverterResult(
                torch_module=dropout, onnx_mapping=onnx_mapping_from_node(node)
            )

        class ReshapeWithConstShape(torch.nn.Module):
            """
            Artificial torch Reshaping module with constant reshaping shape
            """

            def __init__(self, size: Tuple[int, ...]) -> None:
                super().__init__()
                self.shape = size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, torch.Size((x.shape[0], *self.shape)))

        class Reshape(OnnxReshape):
            """
            Extension of OnnxReshape with correcting inputs order
            """

            def forward(self, *inputs):
                if inputs[0].dim() == 1:
                    return super().forward(inputs[1], inputs[0])
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
            - torch.nn.Flatten, if shape is constant and it's length is smaller
                than 3
            - Reshape, in other cases

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            shape = extract_value_from_graph(node, graph, node.input_values[1])
            mapping = OnnxMapping(
                inputs=(node.input_values[0],),
                outputs=(node.output_values[0],),
            )
            if shape is None:
                return OperationConverterResult(
                    torch_module=Reshape(),
                    onnx_mapping=onnx_mapping_from_node(node)
                )

            if shape.shape[0] <= 2:
                return OperationConverterResult(
                    torch_module=torch.nn.Flatten(
                        start_dim=shape.shape[0] - 1
                    ),
                    onnx_mapping=mapping,
                )
            return OperationConverterResult(
                torch_module=ReshapeWithConstShape(tuple(shape[1:])),
                onnx_mapping=mapping
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
            symmetical padding

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            padding = extract_value_from_graph(node, graph, "pads")
            if padding is not None:
                self.logger.warning("Forcing symmetric paddings")
                print(padding)
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
            node: OnnxNode, graph: OnnxGraph, scale: float = 1.0
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's Add and Sub conversion, if one input is
            not an output from other node, then convert to Linear layer
            with bias

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
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
                return _binary_math_converter(node, graph)
            if node.operation_type == 'Sub':
                bias = bias * -1
            weights = torch.eye(bias.shape[-1], requires_grad=False)
            linear = create_linear_from_weights(weights, bias)
            mapping = OnnxMapping(
                inputs=(input_name,), outputs=node.output_values
            )
            return OperationConverterResult(
                torch_module=linear, onnx_mapping=mapping
            )

        class ShapeWithMemory(torch.nn.Module):
            """
            Artificial torch Shape module, returning shape of the input Tensor
            or, if input is not provided, previously returned value
            """

            def __init__(
                self, start: Optional[int] = None, end: Optional[int] = None
            ):
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
            Conversion of Shape to ShapeWithMemory

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            return OperationConverterResult(
                torch_module=ShapeWithMemory(),
                onnx_mapping=onnx_mapping_from_node(node)
            )

        class GatherWithConstIndicies(OnnxGather):
            """
            Extension of OnnxGather with constant indicies
            """

            def __init__(self, indicies: torch.Tensor, axis: int = 0):
                super().__init__(axis)
                self.indicies = indicies

            def forward(self, input_tensor):
                return super().forward(input_tensor, self.indicies)

        @add_converter(operation_type="Gather", version=1)
        @add_converter(operation_type="Gather", version=11)
        @add_converter(operation_type="Gather", version=13)
        def gather_converter(
            node: OnnxNode, graph: OnnxGraph
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's Gather conversion, if one input is not
            an output from other node, then convert to GatherWithConstIndicies

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            axis = extract_value_from_graph(node, graph, 'axis', 0)
            indicies = extract_value_from_graph(
                node, graph, node.input_values[1]
            )

            if indicies is not None:
                if indicies.dim() == 0:
                    indicies = int(indicies)
                return OperationConverterResult(
                    torch_module=GatherWithConstIndicies(indicies, axis),
                    onnx_mapping=OnnxMapping(
                        inputs=(node.input_values[0],),
                        outputs=node.output_values
                    )
                )
            return OperationConverterResult(
                torch_module=OnnxGather(axis),
                onnx_mapping=onnx_mapping_from_node(node)
            )

        class ConcatWithConstInputs(OnnxConcat):
            """
            Extension of OnnxConcat with part of input as a constant
            """

            def __init__(self, axis: int, const_input: List[torch.Tensor]):
                super().__init__(axis)
                self.const_input = const_input

            def _apply(self, fn):
                self.const_input = [fn(const) for const in self.const_input]
                return super()._apply(fn)

            def forward(self, *input_tensors):
                return super().forward(*input_tensors, *self.const_input)

        @add_converter(operation_type="Concat", version=4)
        @add_converter(operation_type="Concat", version=11)
        @add_converter(operation_type="Concat", version=13)
        def concat_converter(
            node: OnnxNode, graph: OnnxGraph
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's Concat conversion, if at least one input
            is constant convert to ConcatWithConstInputs

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            const_inputs = []
            const_input_names = []
            axis = extract_value_from_graph(node, graph, 'axis', 0)
            for input_name in node.input_values:
                _input = extract_value_from_graph(node, graph, input_name)
                if _input is not None:
                    const_inputs.append(_input)
                    const_input_names.append(input_name)
            if len(const_inputs) > 0:
                new_inputs = list(node.input_values)
                [new_inputs.remove(v) for v in const_input_names]
                return OperationConverterResult(
                    torch_module=ConcatWithConstInputs(axis, const_inputs),
                    onnx_mapping=OnnxMapping(
                        inputs=new_inputs,
                        outputs=node.output_values
                    )
                )
            else:
                return _concat_converter(node, graph)

        class WrapperForUniqueInputs(torch.nn.Module):
            """
            Wrapper for torch.nn.Module class, which ensure only one input
            """

            def __init__(self, module: torch.nn.Module):
                super().__init__()
                self.wrapped_module = module

            def forward(self, *inputs):
                return self.wrapped_module.forward(inputs[0])

        @add_converter(operation_type='LRN', version=1)
        @add_converter(operation_type='LRN', version=13)
        def lrn_converter(
            node: OnnxNode, graph: OnnxGather
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's LRN conversion, wrapping
            LocalResponseNorn with WrapperForUniqueInputs to ensure right
            number of inputs

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            size = extract_value_from_graph(node, graph, 'size')
            alpha = extract_value_from_graph(node, graph, 'alpha', 0.0001)
            beta = extract_value_from_graph(node, graph, 'beta', 0.75)
            k = extract_value_from_graph(node, graph, 'bias', 1)

            return OperationConverterResult(
                torch_module=WrapperForUniqueInputs(
                    torch.nn.LocalResponseNorm(size, alpha, beta, k)),
                onnx_mapping=onnx_mapping_from_node(node)
            )

        @add_converter(operation_type='Softmax', version=1)
        @add_converter(operation_type='Softmax', version=11)
        def softmax_converter(
            node: OnnxNode, graph: OnnxGraph
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's Softmax conversion, wrapping
            OnnxSoftmaxV1V11 with WrapperForUniqueInputs to ensure right
            number of inputs

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            dim = extract_value_from_graph(node, graph, 'axis', 1)
            return OperationConverterResult(
                torch_module=WrapperForUniqueInputs(OnnxSoftmaxV1V11(dim)),
                onnx_mapping=onnx_mapping_from_node(node)
            )

        @add_converter(operation_type='Unsqueeze', version=1)
        @add_converter(operation_type='Unsqueeze', version=11)
        def unsqueeze_converter(
            node: OnnxNode, graph: OnnxGather
        ) -> OperationConverterResult:
            """
            Extension of onnx2torch's Unsqueeze conversion, changing node to
            OnnxConstant if both inputs are constants

            Parameters
            ----------
            node: OnnxNode
                The Gemm node for conversion
            graph: OnnxGraph
                The whole model wrapped in OnnxGraph

            Returns
            -------
            OperationConverterResult which is a scheme for converting Gemm
            """
            _input = extract_value_from_graph(
                node, graph, node.input_values[0])
            dims = extract_value_from_graph(node, graph, 'axes', -1)
            if _input is not None and dims is not None:
                for dim in dims:
                    _input = torch.unsqueeze(_input, dim)
                return OperationConverterResult(
                    torch_module=OnnxConstant(_input),
                    onnx_mapping=OnnxMapping(
                        inputs=tuple(),
                        outputs=node.output_values
                    )
                )
            else:
                return _unsqueeze_converter(node, graph)

        # Convert module and restore default Gemm conversions
        converted_model = onnx2torch.convert(onnx_model)
        _CONVERTER_REGISTRY = converter_registy_copy
        return converted_model
