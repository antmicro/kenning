"""
Contains Tensorflow models for the classification problem.

Pretrained on ImageNet dataset.
"""

from pathlib import Path
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset
from kenning.utils.class_loader import load_class
from typing import List, Dict

import tensorflow as tf


class TensorFlowImageNet(TensorFlowWrapper):

    arguments_structure = {
        'modelcls': {
            'argparse_name': '--model-cls',
            'description': 'The Keras model class',
            'type': str
        },
        'modelinputname': {
            'argparse_name': '--model-input-name',
            'description': 'Name of the input in the TensorFlow model',
            'type': str,
            'default': 'input'
        },
        'modeloutputname': {
            'argparse_name': '--model-output-name',
            'description': 'Name of the output in the TensorFlow model',
            'type': str,
            'default': 'output'
        },
        'inputshape': {
            'argparse_name': '--input-shape',
            'description': 'Input shape',
            'type': int,
            'is_list': True,
            'default': [1, 224, 224, 3]
        },
        'numclasses': {
            'argparse_name': '--num-classes',
            'description': 'Output shape',
            'type': int,
            'default': 1000
        },
        'disablebuiltinpreprocessing': {
            'argparse_name': '--disable-builtin-preprocessing',
            'description': 'Removes (if possible) internal preprocessing in the model',  # noqa: E501
            'type': bool,
            'default': False
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = False,
            modelcls: str = '',
            modelinputname: str = 'input',
            modeloutputname: str = 'output',
            inputshape: List[int] = [1, 224, 224, 3],
            numclasses: int = 1000,
            disablebuiltinpreprocessing: bool = False):
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
        modelinputname : str
            The name of the model input
        modeloutputname : str
            The name of the model output
        inputshape : List[int]
            The shape of the input
        numclasses : int
            Number of classes in the model
        disablebuiltinpreprocessing : bool
            Tells if the input preprocessing should be removed from the model
        """
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.modelcls = modelcls
        self.modelinputname = modelinputname
        self.modeloutputname = modeloutputname
        self.inputshape = inputshape
        self.numclasses = numclasses
        self.outputshape = [inputshape[0], numclasses]
        self.disablebuiltinpreprocessing = disablebuiltinpreprocessing

        super().__init__(
            modelpath,
            dataset,
            from_file
        )

    def get_io_specification_from_model(self):
        return {
            'input': [{'name': self.modelinputname, 'shape': self.inputshape, 'dtype': 'float32'}],  # noqa: E501
            'output': [{'name': self.modeloutputname, 'shape': self.outputshape, 'dtype': 'float32'}]  # noqa: E501
        }

    def prepare_model(self):
        if self.from_file:
            self.load_model(self.modelpath)
        else:
            if self.disablebuiltinpreprocessing:
                self.model = load_class(self.modelcls)(
                    input_shape=tuple(self.inputshape[1:]),
                    include_preprocessing=False
                )
            else:
                self.model = load_class(self.modelcls)(
                    input_shape=tuple(self.inputshape[1:])
                )
            self.save_model(self.modelpath)
            self.model.summary()

    @classmethod
    def from_argparse(cls, dataset, args, from_file=False):
        return cls(
            args.model_path,
            dataset,
            from_file,
            args.model_cls,
            args.num_classes,
            args.model_input_name,
            args.model_output_name,
            args.input_shape,
            args.num_classes,
            args.disable_builtin_preprocessing
        )

    @classmethod
    def from_json(
            cls,
            dataset: Dataset,
            json_dict: Dict,
            from_file: bool = False):
        return super(TensorFlowImageNet, cls).from_json(
            dataset,
            json_dict,
            from_file
        )
