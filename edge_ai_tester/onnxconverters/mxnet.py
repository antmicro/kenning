"""
ONNXConversion for MXNet models.
"""

import mxnet
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
from gluoncv import model_zoo as model_zoo
from tempfile import TemporaryDirectory
from pathlib import Path
from collections import namedtuple

from edge_ai_tester.core.onnxconversion import ONNXConversion
from edge_ai_tester.core.onnxconversion import SupportStatus

Batch = namedtuple('Batch', ['data'])


class MXNetONNXConversion(ONNXConversion):
    def __init__(self):
        super().__init__('mxnet', mxnet.__version__)

    def prepare(self):
        self.add_entry(
            'DenseNet201',
            lambda: model_zoo.densenet201(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'MobileNetV2',
            lambda: model_zoo.mobilenet_v2_1_0(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'ResNet50',
            lambda: model_zoo.resnet50_v1b(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'VGG16',
            lambda: model_zoo.vgg16(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'DeepLabV3 ResNet50',
            lambda: model_zoo.get_deeplab_resnet50_citys(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'Faster R-CNN ResNet50 FPN',
            lambda: model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )
        self.add_entry(
            'Mask R-CNN',
            lambda: model_zoo.mask_rcnn_resnet50_v1b_coco(pretrained=True),
            input_shape=(1, 3, 224, 224)
        )

    def onnx_export(self, modelentry, exportpath):
        model = modelentry.modelgenerator()
        model.hybridize()
        inp = mxnet.ndarray.random.randn(*modelentry.parameters['input_shape'])
        model(inp)
        with TemporaryDirectory() as tempdir:
            tempd = Path(tempdir)
            modelname = tempd / 'mxnet-model'
            model.export(modelname)
            params = str(tempd / 'mxnet-model-0000.params')
            symbol = str(tempd / 'mxnet-model-symbol.json')
            onnx_mxnet.export_model(
                symbol,
                params,
                [modelentry.parameters['input_shape']],
                np.float32,
                exportpath
            )
        return SupportStatus.SUPPORTED

    def onnx_import(self, modelentry, importpath):
        sym, arg, aux = onnx_mxnet.import_model(importpath)
        data_names = [inp for inp in sym.list_inputs() if inp not in arg and inp not in aux]  # noqa: E501
        mod = mxnet.mod.Module(
            symbol=sym,
            data_names=data_names,
            label_names=None
        )
        inputdata = mxnet.ndarray.random.randn(
            *modelentry.parameters['input_shape']
        )
        mod.bind(
            for_training=False,
            data_shapes=[(data_names[0], inputdata.shape)],
            label_shapes=None
        )
        mod.set_params(
            arg_params=arg,
            aux_params=aux,
            allow_missing=True,
            allow_extra=True
        )
        mod.forward(Batch([inputdata]))
        return SupportStatus.SUPPORTED
