"""
Contains Tensorflow models for the classification problem.

Pretrained on ImageNet dataset.
"""

from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset
from kenning.utils.class_loader import load_class
import tensorflow as tf
import numpy as np


class TensorFlowImageNet(TensorFlowWrapper):
    def __init__(
            self,
            modelpath,
            dataset: Dataset,
            modelcls: str,
            numclasses: int,
            from_file=False):
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.modelcls = modelcls
        self.numclasses = 1000
        super().__init__(
            modelpath,
            dataset,
            from_file,
            (tf.TensorSpec((1, 224, 224, 3), name='input_1'),)
        )

    def get_input_spec(self):
        return {'input_1': (1, 224, 224, 3)}, 'float32'

    def prepare_model(self):
        if self.from_file:
            self.load_model(self.modelpath)
        else:
            self.model = load_class(self.modelcls)()
            self.save_model(self.modelpath)

    def preprocess_input(self, X):
        return tf.keras.applications.resnet.preprocess_input(np.array(X))

    def run_inference(self, X):
        return self.model.predict(X)

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--model-cls',
            help='The Keras model class'
        )
        group.add_argument(
            '--num-classes',
            type=int,
            help='The number of classifier classes'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset, args, from_file=False):
        return cls(
            args.model_path,
            dataset,
            args.model_cls,
            args.num_classes,
            from_file
        )
