from dl_framework_analyzer.core.model import ModelWrapper

import numpy as np
import tensorflow as tf

class TensorFlowWrapper(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
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
        return np.array(X)

    def run_inference(self, X):
        return self.model.predict(X)

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)
