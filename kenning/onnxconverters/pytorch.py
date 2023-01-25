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
from typing import Union

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
            return SupportStatus.UNSUPPORTED

        if modelentry is not None:
            input_tensor = modelentry.parameters['input_tensor']
        else:
            input_tensor = self.try_extracting_input_shape_from_onnx(
                model_onnx)
            input_tensor = [torch.rand(shape) for shape in input_tensor] \
                if input_tensor else None

        if input_tensor is None:
            return SupportStatus.UNSUPPORTED

        try:
            model_torch = self.onnx_to_torch(model_onnx)
            # print(str(model_torch))
        except RuntimeError or NotImplementedError:
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
            _CONVERTER_REGISTRY,
            OperationDescription
        )
        from onnx2torch.onnx_node import OnnxNode
        from onnx2torch.onnx_graph import OnnxGraph
        from onnx2torch.node_converters import onnx_mapping_from_node
        from onnx2torch.node_converters.batch_norm import (
            _ as batch_converter
        )
        from onnx2torch.utils.common import (
            OperationConverterResult,
            OnnxMapping,
        )

        # Backup copy and removing default conversions
        converter_registy_copy = deepcopy(_CONVERTER_REGISTRY)
        for version in (9, 11, 13):
            del _CONVERTER_REGISTRY[OperationDescription(
                operation_type='Gemm',
                version=version,
                domain=onnx.defs.ONNX_DOMAIN)]
        for version in (9, 14, 15):
            del _CONVERTER_REGISTRY[OperationDescription(
                operation_type='BatchNormalization',
                version=version,
                domain=onnx.defs.ONNX_DOMAIN)]
        for version in (10, 12, 13):
            del _CONVERTER_REGISTRY[OperationDescription(
                operation_type='Dropout',
                version=version,
                domain=onnx.defs.ONNX_DOMAIN)]

        class Transposition(torch.nn.Module):
            """
            Artifficial torch Module for transposing input
            """
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor):
                return torch.transpose(x, dim0=1, dim1=-1)

        @add_converter(operation_type='Gemm', version=9)
        @add_converter(operation_type='Gemm', version=11)
        @add_converter(operation_type='Gemm', version=13)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
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
            c_name = node.input_values[2] \
                if len(node.input_values) > 2 else None

            alpha = node.attributes.get('alpha', 1.)
            beta = node.attributes.get('beta', 1.)
            trans_a = node.attributes.get('transA', 0) != 0
            trans_b = node.attributes.get('transB', 0) != 0

            sequence = []
            if trans_a:
                sequence.append(Transposition())

            if c_name is None:
                bias = None
            else:
                bias = graph.initializers[c_name].to_torch()

            if b_name in graph.initializers:
                weights = graph.initializers[b_name].to_torch()
                if not trans_b:
                    weights = weights.T
                in_feature, out_feature = weights.shape[1], weights.shape[0]
                linear = torch.nn.Linear(
                    in_features=in_feature,
                    out_features=out_feature,
                    bias=bias is not None
                )
                with torch.no_grad():
                    weights = weights * alpha
                    linear.weight.data = weights
                    if bias is not None:
                        bias = bias * beta
                        linear.bias.data = bias
                sequence.append(linear)
            else:  # weights are missing
                raise RuntimeError("Missing weights for linear layer")
            if len(sequence) > 1:
                return OperationConverterResult(
                    torch_module=torch.nn.Sequential(*sequence),
                    onnx_mapping=OnnxMapping(
                        inputs=(a_name,),
                        outputs=node.output_values)
                )
            else:
                return OperationConverterResult(
                    torch_module=sequence[0],
                    onnx_mapping=OnnxMapping(
                        inputs=(a_name,),
                        outputs=node.output_values)
                )

        @add_converter(operation_type='BatchNormalization', version=9)
        @add_converter(operation_type='BatchNormalization', version=14)
        @add_converter(operation_type='BatchNormalization', version=15)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
            if len(node.output_values) > 1:
                self.logger.warning(
                    "Number of BatchNormalization outputs reduced to one")
                node._output_values = (node.output_values[0],)
            return batch_converter(node, graph)

        @add_converter(operation_type='Dropout', version=10)
        @add_converter(operation_type='Dropout', version=12)
        @add_converter(operation_type='Dropout', version=13)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
            if len(node.input_values) > 1:
                self.logger.warning("Number of Dropout inputs reduced to one")
                node._input_values = (node.input_values[0],)
            if len(node.output_values) > 1:
                self.logger.warning("Number of Dropout outputs reduced to one")
                node._output_values = (node.output_values[0],)
            ratio = node.attributes.get('ratio', 0.5)
            seed = node.attributes.get('seed', None)
            if seed is not None:
                raise NotImplementedError(
                    'Dropout nodes seeds are not supported')

            dropout = torch.nn.Dropout(ratio)
            return OperationConverterResult(
                torch_module=dropout,
                onnx_mapping=onnx_mapping_from_node(node)
            )

        # Convert module and restore default Gemm conversions
        converted_model = onnx2torch.convert(onnx_model)
        _CONVERTER_REGISTRY = converter_registy_copy
        return converted_model
