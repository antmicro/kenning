# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using TensorFlow models in Kenning.
"""

from abc import ABC
from typing import Optional

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.utils.resource_manager import PathOrURI


class TensorFlowWrapper(ModelWrapper, ABC):
    """
    Base model wrapper for TensorFlow models.
    """

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool,
        model_name: Optional[str] = None,
    ):
        """
        Creates the TensorFlow model wrapper.

        TensorFlow models require input shape specification in a form of
        TensorSpec to serialize the model to ONNX.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Dataset
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report
        """
        super().__init__(model_path, dataset, from_file, model_name)

    def load_model(self, model_path: PathOrURI):
        import tensorflow as tf

        tf.keras.backend.clear_session()
        if hasattr(self, "model") and self.model is not None:
            del self.model
        self.model = tf.keras.models.load_model(str(model_path))
        self.model.summary()

    def save_model(self, model_path: PathOrURI):
        self.prepare_model()
        self.model.save(model_path)

    def preprocess_input(self, X):
        return np.array(X, dtype="float32")

    def run_inference(self, X):
        self.prepare_model()
        return self.model.predict(X, verbose=0)

    def get_framework_and_version(self):
        import tensorflow as tf

        return ("tensorflow", tf.__version__)

    def get_output_formats(self):
        return ["onnx", "keras"]

    def save_to_onnx(self, model_path: PathOrURI):
        import tensorflow as tf
        import tf2onnx

        self.prepare_model()
        x = tuple(
            tf.TensorSpec(
                spec["shape"],
                spec["dtype"],
                name=spec["name"],
            )
            for spec in self.get_io_specification()["input"]
        )

        modelproto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=x, output_path=model_path, opset=11
        )

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata):
        result = []
        singleoutputsize = self.numclasses * np.dtype(np.float32).itemsize
        for ind in range(0, len(outputdata), singleoutputsize):
            arr = np.frombuffer(
                outputdata[ind : ind + singleoutputsize], dtype=np.float32
            )
            result.append(arr)
        return result
