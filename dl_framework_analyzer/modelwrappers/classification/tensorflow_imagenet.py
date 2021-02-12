from ...core.model import ModelWrapper
from tensorflow import keras
import numpy as np


"""
Contains Tensorflow models for the classification problem.
"""


class TensorflowImagenetResNet152(ModelWrapper):
    def prepare_model(self):
        self.model = keras.applications.ResNet152()

    def preprocess_input(self, X):
        return keras.applications.resnet.preprocess_input(np.array(X))

    def run_inference(self, X):
        return self.model.predict(X)
