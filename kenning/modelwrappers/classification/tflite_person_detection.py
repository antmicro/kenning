# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Tensorflow Lite model for the person detection.

Trained on VisualWakeWords dataset.
"""

from typing import List, Tuple
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
import cv2

if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files

from kenning.core.model import ModelWrapper
from kenning.core.dataset import Dataset
from kenning.datasets.visual_wake_words_dataset import VisualWakeWordsDataset
from kenning.resources.models import classification


class PersonDetectionModelWrapper(ModelWrapper):

    default_dataset = VisualWakeWordsDataset
    pretrained_modelpath = files(classification) / 'person_detect.tflite'

    arguments_structure = {
        'central_fraction': {
            'argparse_name': '--central-fraction',
            'description': 'Fraction used to crop images during preprocessing',
            'default': .875,
            'type': float
        },
        'image_width': {
            'description': 'Width of the input images',
            'type': int,
            'default': 96
        },
        'image_height': {
            'description': 'Height of the input images',
            'type': int,
            'default': 96
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = True,
            central_fraction: float = .875,
            image_width: int = 96,
            image_height: int = 96):
        """
        Creates the Person Detection model wrapper.

        Parameters
        ----------
        modelpath : Path
            The path to the model.
        dataset : Dataset
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        central_fraction : float
            Fraction used to crop images during preprocessing.
        image_width : int
            Width of the input images.
        image_height : int
            Height of the input images.
        """
        super().__init__(
            modelpath,
            dataset,
            from_file
        )
        self.central_fraction = central_fraction
        self.image_width = image_width
        self.image_height = image_height
        self.numclasses = 2
        self.interpreter = None
        if dataset is not None:
            class_names = self.dataset.get_class_names()
            assert len(class_names) == 2
            self.class_names = class_names
            self.save_io_specification(self.modelpath)

    @classmethod
    def _get_io_specification(
            cls, img_width=96, img_height=96, class_names=None):
        io_spec = {
            'input': [{
                'name': 'input_1',
                'shape': (1, img_width, img_height, 1),
                'dtype': 'int8',
                'prequantized_dtype': 'float32',
                'zero_point': -1,
                'scale': .007843137718737125
            }],
            'output': [{
                'name': 'out_layer',
                'shape': (1, 2),
                'dtype': 'int8',
                'prequantized_dtype': 'float32',
                'zero_point': -128,
                'scale': .00390625
            }]
        }
        if class_names is not None:
            io_spec['output'][0]['class_names'] = class_names
        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification()

    def get_io_specification_from_model(self):
        return self._get_io_specification(
            self.image_width, self.image_height, self.class_names
        )

    def prepare_model(self):
        if self.model_prepared:
            return None

        if self.from_file:
            self.model_prepared = True

    def get_output_formats(self) -> List[str]:
        return ['tflite']

    def get_framework_and_version(self) -> Tuple[str, str]:
        return ('tensorflow', tf.__version__)

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        data = bytes()
        for x in inputdata:
            data += x.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        io_spec = self.get_io_specification_from_model()
        dtype = np.dtype(io_spec['output'][0]['dtype'])
        shape = io_spec['output'][0]['shape']

        tensor_size = dtype.itemsize*np.prod(shape)

        assert len(outputdata) % tensor_size == 0

        y = []
        for i in range(len(outputdata)//tensor_size):
            y.append(np.frombuffer(
                outputdata[tensor_size*i: tensor_size*(i + 1)],
                dtype=dtype
            ))

        return y

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        io_spec = self.get_io_specification_from_model()
        zero_point = io_spec['input'][0]['zero_point']
        scale = io_spec['input'][0]['scale']
        dtype = np.dtype(io_spec['input'][0]['dtype'])

        result = []
        for img in X:
            w, h = img.shape[:2]
            img = img[int((w/2)*(1 - self.central_fraction)):
                      int((w/2)*(1 + self.central_fraction)),
                      int((h/2)*(1 - self.central_fraction)):
                      int((h/2)*(1 + self.central_fraction))]
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            img = img*2. - 1
            img = np.expand_dims(img, -1)

            # quantization
            img = np.around(img/scale + zero_point).astype(dtype)

            result.append(img)

        return result

    def postprocess_outputs(self, y: List[np.ndarray]) -> List[np.ndarray]:
        io_spec = self.get_io_specification_from_model()
        zero_point = io_spec['output'][0]['zero_point']
        scale = io_spec['output'][0]['scale']
        dtype = np.dtype(io_spec['output'][0]['prequantized_dtype'])

        return [(output.astype(dtype) - zero_point)*scale for output in y]
