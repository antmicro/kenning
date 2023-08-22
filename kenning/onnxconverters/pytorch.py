# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ONNXConversion for PyTorch models.
"""

import torchvision.models as models
import torch
import onnx

from kenning.core.onnxconversion import ONNXConversion
from kenning.core.onnxconversion import SupportStatus
from kenning.utils.onnx import try_extracting_input_shape_from_onnx


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_onnx = onnx.load(str(importpath))
        if model_onnx.ir_version <= 3:
            self.logger.error("Model unsupported due to the not sufficient "
                              f"ir_version {model_onnx.ir_version}, have to be"
                              " greater than 3")
            return SupportStatus.UNSUPPORTED

        # Preparing dummy input for testing model
        if modelentry is not None:
            input_tensor = modelentry.parameters["input_tensor"].to(device)
        else:
            input_tensor = try_extracting_input_shape_from_onnx(model_onnx)
            input_tensor = (
                [torch.rand(shape).to(device) for shape in input_tensor]
                if input_tensor
                else None
            )

        if input_tensor is None:
            self.logger.error("Cannot get properties of input tensor")
            return SupportStatus.UNSUPPORTED

        # Converting model
        try:
            from kenning.onnxconverters import onnx2torch
            model_torch = onnx2torch.convert(model_onnx)
        except RuntimeError or NotImplementedError as e:
            self.logger.error(f"Conversion: {e}")
            del model_onnx
            return SupportStatus.UNSUPPORTED

        # Testing converted model
        model_torch = model_torch.to(device)
        model_torch.train()
        model_torch(*input_tensor)

        # Cleanup
        del model_onnx  # noqa: F821
        del model_torch
        return SupportStatus.SUPPORTED
