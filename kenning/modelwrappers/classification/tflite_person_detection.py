# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

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

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = True):
        """
        Creates the Person Detection model wrapper.

        Parameters
        ----------
        modelpath : Path
            The path to the model
        dataset : Dataset
            The dataset to verify the inference
        from_file : bool
            True if the model should be loaded from file
        """
        super().__init__(
            modelpath,
            dataset,
            from_file
        )
        self.numclasses = 2
        self.interpreter = None
        if dataset is not None:
            class_names = self.dataset.get_class_names()
            assert len(class_names) == 2
            self.class_names = class_names
            self.save_io_specification(self.modelpath)

    @classmethod
    def from_argparse(cls, dataset, args, from_file=False):
        return cls(
            args.modelpath,
            dataset,
            from_file
        )

    @classmethod
    def _get_io_specification(
            cls, class_names=None):
        io_spec = {
            'input': [{
                'name': 'input_1',
                'shape': (1, 96, 96, 1),
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
        return self._get_io_specification(self.class_names)

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
        io_spec = self.get_io_specification_from_model()
        inputdata = inputdata[0]
        inputdata /= io_spec['input'][0]['scale']
        inputdata += io_spec['input'][0]['zero_point']
        inputdata = np.around(inputdata)
        inputdata = inputdata.astype(io_spec['input'][0]['dtype'])
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        io_spec = self.get_io_specification_from_model()
        outputdata = np.frombuffer(
            outputdata,
            dtype=io_spec['output'][0]['dtype']
        )
        outputdata = outputdata.astype(
            io_spec['output'][0]['prequantized_dtype']
        )
        outputdata -= io_spec['output'][0]['zero_point']
        outputdata *= io_spec['output'][0]['scale']
        return [outputdata]
