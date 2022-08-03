from kenning.core.model import ModelWrapper

import numpy as np
import torch
import copy
from collections import OrderedDict


class PyTorchWrapper(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # noqa: E501
        super().__init__(modelpath, dataset, from_file)

    def load_weights(self, weights: OrderedDict):
        """
        Loads the model state using ``weights``.
        It assumes that the model structure is already defined.

        Parameters
        ----------
        weights : OrderedDict
            Dictionary used to load model weights
        """
        self.model.load_state_dict(
            copy.deepcopy(weights)
        )
        self.model.eval()

    def create_model_structure(self):
        """
        Recreates the model structure.
        Every PyTorchWrapper subclass has to implement its own architecture.
        """
        raise NotImplementedError

    def load_model(self, modelpath):
        input_data = torch.load(
            self.modelpath,
            map_location=self.device
        )

        # If the file constains only the weights
        # we have to recreate the model's structure
        # Otherwise we just load the model
        if isinstance(input_data, OrderedDict):
            self.create_model_structure()
            self.load_weights(input_data)
        elif isinstance(input_data, torch.nn.Module):
            self.model = input_data

    def save_to_onnx(self, modelpath):
        x = tuple(torch.randn(
            spec['shape'],
            device='cpu'
        ) for spec in self.get_io_specs()['input'])

        torch.onnx.export(
            self.model.to(device='cpu'),
            x,
            modelpath,
            opset_version=11,
            input_names=[
                spec['name'] for spec in self.get_io_specs()['input']
            ],
            output_names=[
                spec['name'] for spec in self.get_io_specs()['output']
            ]
        )

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

    def get_output_formats(self):
        return ['onnx', 'torch']
