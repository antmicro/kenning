import tensorflow as tf
import tf2onnx

from dl_framework_analyzer.core.onnxconversion import ONNXConversion
from dl_framework_analyzer.core.onnxconversion import SupportStatus
import tensorflow.keras.applications as apps

class TensorFlowONNXConversion(ONNXConversion):
    def __init__(self):
        super().__init__('tensorflow', tf.__version__)

    def prepare(self):
        self.add_entry(
            'DenseNet201',
            lambda: apps.DenseNet201(),
            tensor_spec= (tf.TensorSpec(
                (None, 224, 224, 3),
                tf.float32,
                name="input"
            ),))
        self.add_entry(
            'MobileNetV2',
            lambda: apps.MobileNetV2(),
            tensor_spec= (tf.TensorSpec(
                (None, 224, 224, 3),
                tf.float32,
                name="input"
            ),))
        self.add_entry(
            'ResNet50',
            lambda: apps.ResNet50(),
            tensor_spec= (tf.TensorSpec(
                (None, 224, 224, 3),
                tf.float32,
                name="input"
            ),))
        self.add_entry(
            'VGG16',
            lambda: apps.VGG16(),
            tensor_spec= (tf.TensorSpec(
                (None, 224, 224, 3),
                tf.float32,
                name="input"
            ),))

    def onnx_export(self, modelentry, exportpath):
        model = modelentry.modelgenerator()
        spec = modelentry.parameters['tensor_spec'] if 'tensor_spec' in modelentry.parameters else None
        modelproto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            output_path=exportpath)
        return SupportStatus.SUPPORTED
