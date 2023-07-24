# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Tensorflow models for the classification problem.

Pretrained on ImageNet dataset.
"""

from typing import List

from kenning.utils.logger import get_logger
from kenning.core.dataset import Dataset
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.utils.class_loader import load_class
from kenning.utils.resource_manager import PathOrURI


class TensorFlowImageNet(TensorFlowWrapper):

    default_dataset = ImageNetDataset
    pretrained_model_uri = 'kenning:///models/classification/tensorflow_imagenet_mobilenetv3small.h5'  # noqa: 501
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
            model_path: PathOrURI,
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
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Dataset
            The dataset to verify the inference.
        from_file : bool
            True if model should be loaded from file.
        modelcls : str
            The model class import path.
            Used for loading keras.applications pretrained models.
        modelinputname : str
            The name of the model input.
        modeloutputname : str
            The name of the model output.
        inputshape : List[int]
            The shape of the input.
        numclasses : int
            Number of classes in the model.
        disablebuiltinpreprocessing : bool
            Tells if the input preprocessing should be removed from the model.
        """
        super().__init__(model_path, dataset, from_file)
        import tensorflow as tf
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

        if dataset and dataset.batch_size != self.inputshape[0]:
            logger = get_logger()
            logger.error('Dataset batch size and ModelWrapper inputshape '
                         f'mismatch: batch size: {dataset.batch_size} '
                         f'input shape[0]: {self.inputshape[0]}')
            raise ValueError

    @classmethod
    def _get_io_specification(
            cls, modelinputname, inputshape, modeloutputname, outputshape):
        return {
            'input': [{'name': modelinputname, 'shape': inputshape, 'dtype': 'float32'}],  # noqa: E501
            'output': [{'name': modeloutputname, 'shape': outputshape, 'dtype': 'float32'}]  # noqa: E501
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        input_shape = json_dict['inputshape']
        output_shape = [input_shape[0], json_dict['numclasses']]
        return cls._get_io_specification(
            json_dict['modelinputname'],
            input_shape,
            json_dict['modeloutputname'],
            output_shape
        )

    def get_io_specification_from_model(self):
        return self._get_io_specification(
            self.modelinputname,
            self.inputshape,
            self.modeloutputname,
            self.outputshape
        )

    def prepare_model(self):
        if self.model_prepared:
            return None
        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
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
            self.model_prepared = True
            self.save_model(self.model_path)
            self.model.summary()
