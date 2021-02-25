from dl_framework_analyzer.core.model import ModelWrapper

import numpy as np
import torch

class PyTorchWrapper(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # noqa: E501
        super().__init__(modelpath, dataset, from_file)

    def load_model(self, modelpath):
        self.model = torch.load(modelpath)
        self.model.eval()

    def save_model(self, modelpath):
        torch.save(self.model, modelpath)

    def preprocess_input(self, X):
        return torch.Tensor(np.array(X)).to(self.device).permute(0, 3, 1, 2)

    def postprocess_outputs(self, y):
        return y.detach().cpu().numpy()

    def run_inference(self, X):
        return self.model(X)

    def get_framework_and_version(self):
        return ('torch', torch.__version__)
