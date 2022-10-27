import tensorflow as tf
import numpy as np
from pathlib import Path

from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset


class MagicWandModelWrapper(TensorFlowWrapper):
    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool):
        self.modelpath = modelpath
        self.dataset = dataset
        self.from_file = from_file
        self.numclasses = len(self.dataset.get_class_names())

        super.__init__(
            modelpath,
            dataset,
            from_file,
            (tf.TensorSpec(1, 128, 3, 1, name='input_1'),),
        )

    def get_input_spec(self):
        return {'input_1': (1, 128, 3, 1)}, 'float32'

