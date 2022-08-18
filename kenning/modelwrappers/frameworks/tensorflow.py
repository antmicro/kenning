import numpy as np
import tensorflow as tf
from pathlib import Path

from kenning.core.model import ModelWrapper
from kenning.core.dataset import Dataset


class TensorFlowWrapper(ModelWrapper):
    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool,
            inputspec: tf.TensorSpec):
        """
        Creates the TensorFlow model wrapper.

        TensorFlow models require input shape specification in a form of
        TensorSpec to serialize the model to ONNX.

        Parameters
        ----------
        modelpath : Path
            The path to the model
        dataset : Dataset
            The dataset to verify the inference
        from_file : bool
            True if the model should be loaded from file
        inputspec : tf.TensorSpec
            Specification of the input tensor dimensionality and type (used for
            ONNX conversion)
        """
        self.inputspec = inputspec
        super().__init__(modelpath, dataset, from_file)

    def load_model(self, modelpath):
        tf.keras.backend.clear_session()
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        self.model = tf.keras.models.load_model(str(modelpath))
        print(self.model.summary())

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def preprocess_input(self, X):
        return np.array(X, dtype='float32')

    def run_inference(self, X):
        return self.model.predict(X, verbose=0)

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)

    def get_output_formats(self):
        return ['onnx', 'keras']

    def save_to_onnx(self, modelpath):
        import tf2onnx
        modelproto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=self.inputspec,
            output_path=modelpath,
            opset=11
        )

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata):
        result = []
        singleoutputsize = self.numclasses * np.dtype(np.float32).itemsize
        for ind in range(0, len(outputdata), singleoutputsize):
            arr = np.frombuffer(
                outputdata[ind:ind + singleoutputsize],
                dtype=np.float32
            )
            result.append(arr)
        return result
