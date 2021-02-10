from ...core.inference import InferenceTester
from tensorflow import keras
import numpy as np

class TensorflowImagenetClassifier(InferenceTester):
    def prepare_model(self):
        self.model = keras.applications.ResNet152()

    def preprocess_input(self, X):
        return keras.applications.resnet.preprocess_input(np.array(X))

    def run_inference(self, X):
        return self.model.predict(X)
