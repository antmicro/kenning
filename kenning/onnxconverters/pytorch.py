# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ONNXConversion for PyTorch models.
"""

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
        except RuntimeError or NotImplementedError:
            del model_onnx
            return SupportStatus.UNSUPPORTED

        if len(input_tensor) == 1:
            model_torch(input_tensor[0])
        else:  # input_tensor: List[Tensor]
            model_torch(*input_tensor)

        del model_onnx  # noqa: F821
        del model_torch
        return SupportStatus.SUPPORTED

    @staticmethod
    def onnx_to_torch(onnx_model: Union[Path, onnx.ModelProto]):
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
        return onnx2torch.convert(onnx_model)
