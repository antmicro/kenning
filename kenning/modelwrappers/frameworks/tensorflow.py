# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using TensorFlow models in Kenning.
"""

from abc import ABC
from typing import List, Optional

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import ModelNotLoadedError
from kenning.core.model import ModelWrapper
from kenning.utils.logger import KLogger
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

        self.model = None

        try:
            self.model = tf.keras.models.load_model(
                str(model_path), compile=False
            )
        except Exception:
            KLogger.warning(
                "The model %s could not be loaded with tf.keras loader.",
                str(model_path),
            )
            KLogger.info("Attempting another method to load the model.")

        if self.model is None:
            try:
                import tf_keras

                self.model = tf_keras.models.load_model(
                    str(model_path), compile=False
                )
            except Exception as e:
                KLogger.error(
                    "All methods to load the model %s failed.", str(model_path)
                )
                raise ModelNotLoadedError(f"Cannot load a model: {e}")

        self.model.summary()

    def save_model(self, model_path: PathOrURI):
        self.prepare_model()
        self.model.export(model_path)

    def run_inference(self, X: List[np.ndarray]) -> List[np.ndarray]:
        self.prepare_model()
        if 1 == len(X):
            X = X[0]
        y = self.model.predict(X, verbose=0)
        if not isinstance(y, (list, tuple)):
            y = [y]
        return y

    def get_framework_and_version(self):
        import tensorflow as tf

        return ("tensorflow", tf.__version__)

    @classmethod
    def get_output_formats(cls):
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

        tf2onnx.convert.from_keras(
            self.model, input_signature=x, output_path=model_path, opset=11
        )

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        return b"".join(inp.tobytes() for inp in inputdata)

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        out_spec = self.get_io_specification()["output"]

        result = []
        data_idx = 0
        for spec in out_spec:
            dtype = np.dtype(spec["dtype"])
            shape = spec["shape"]

            out_size = np.prod(shape) * np.dtype(dtype).itemsize
            arr = np.frombuffer(
                outputdata[data_idx : data_idx + out_size], dtype=dtype
            )
            data_idx += out_size
            result.append(arr.reshape(shape))

        return result
