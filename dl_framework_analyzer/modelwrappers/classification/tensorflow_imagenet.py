from dl_framework_analyzer.core.model import ModelWrapper
import tensorflow as tf
import numpy as np


"""
Contains Tensorflow models for the classification problem.
"""


class TensorflowImagenetResNet152(ModelWrapper):
    def prepare_model(self):
        self.model = tf.keras.applications.ResNet152()

    def preprocess_input(self, X):
        return tf.keras.applications.resnet.preprocess_input(np.array(X))

    def run_inference(self, X):
        return self.model.predict(X)

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)
