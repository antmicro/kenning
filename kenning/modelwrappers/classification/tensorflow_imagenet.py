"""
Contains Tensorflow models for the classification problem.

Pretrained on ImageNet dataset.
"""

from pathlib import Path
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset
from kenning.utils.class_loader import load_class

import tensorflow as tf


class TensorFlowImageNet(TensorFlowWrapper):

    arguments_structure = {
        'modelcls': {
            'argparse_name': '--model-cls',
            'description': 'The Keras model class',
            'type': str
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = True,
            modelcls: str = ''):
        """
        Creates model wrapper for TensorFlow classification
        model pretrained on ImageNet dataset.

        Parameters
        ----------
        modelpath : Path
            The path to the model
        dataset : Dataset
            The dataset to verify the inference
        from_file: bool
            True if model should be loaded from file
        modelcls : str
            The model class import path
            Used for loading keras.applications pretrained models
        from_file: bool
            True if model should be loaded from file
        """
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

    @classmethod
    def from_argparse(cls, dataset, args, from_file=False):
        return cls(
            args.model_path,
            dataset,
            args.model_cls,
            args.num_classes,
            from_file
        )
