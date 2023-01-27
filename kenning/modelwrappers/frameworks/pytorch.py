# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.core.model import ModelWrapper

import numpy as np
import copy
from collections import OrderedDict


class PyTorchWrapper(ModelWrapper):
    def __init__(self, modelpath, dataset, from_file):
        import torch
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
        import torch
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
        import torch
        self.prepare_model()
        x = tuple(torch.randn(
            spec['shape'],
            device='cpu'
        ) for spec in self.get_io_specification()['input'])

        torch.onnx.export(
            self.model.to(device='cpu'),
            x,
            modelpath,
            opset_version=11,
            input_names=[
                spec['name'] for spec in self.get_io_specification()['input']
            ],
            output_names=[
                spec['name'] for spec in self.get_io_specification()['output']
            ]
        )

    def save_model(self, modelpath, export_dict=True):
        import torch
        self.prepare_model()
        if export_dict:
            torch.save(self.model.state_dict(), modelpath)
        else:
            torch.save(self.model, modelpath)

    def preprocess_input(self, X):
        import torch
        return torch.Tensor(np.array(X)).to(self.device)

    def postprocess_outputs(self, y):
        return y.detach().cpu().numpy()

    def run_inference(self, X):
        self.prepare_model()
        return self.model(X)

    def get_framework_and_version(self):
        import torch
        return ('torch', torch.__version__)

    def get_output_formats(self):
        return ['onnx', 'torch']
