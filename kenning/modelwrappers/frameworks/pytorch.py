from kenning.core.model import ModelWrapper

import numpy as np
import torch
import copy


class PyTorchWrapper(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # noqa: E501
        super().__init__(modelpath, dataset, from_file)

    def load_model(self, modelpath):
        self.model.load_state_dict(copy.deepcopy(torch.load(modelpath)))
        self.model.to(self.device)
        self.model.eval()

    def save_model(self, modelpath):
        torch.save(self.model.state_dict(), modelpath)

    def preprocess_input(self, X):
        return torch.Tensor(np.array(X)).to(self.device)

    def postprocess_outputs(self, y):
        return y.detach().cpu().numpy()

    def run_inference(self, X):
        return self.model(X)

    def get_framework_and_version(self):
        return ('torch', torch.__version__)
