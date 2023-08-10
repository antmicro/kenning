# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import uuid
import torch
from kenning.optimizers.tvm import TVMCompiler
from pathlib import Path
from torch.autograd import Variable
from typing import Dict, Tuple
from pytest import FixtureRequest

from kenning.utils.resource_manager import ResourceURI


def create_onnx_model(path: Path) -> Tuple[Path, Dict[str, Tuple[int, ...]]]:
    """
    Creates simple convolutional onnx model at given path.

    Parameters
    ----------
    path: Path
        The path to folder where model will be located

    Returns
    -------
    Tuple[Path, Dict[str, Tuple[int, ...]]]:
        The tuple containing path to created Onnx model
        and inputshapes
    """

    modelname = str(uuid.uuid4().hex) + ".onnx"

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3))
    for param in model.parameters():
        param.requires_grad = False
    model[0].weight[0, 0, 0, 0] = 0.0
    model[0].weight[0, 0, 0, 1] = -1.0
    model[0].weight[0, 0, 0, 2] = 0.0
    model[0].weight[0, 0, 1, 0] = -1.0
    model[0].weight[0, 0, 1, 1] = 4.0
    model[0].weight[0, 0, 1, 2] = -1.0
    model[0].weight[0, 0, 2, 0] = 0.0
    model[0].weight[0, 0, 2, 1] = -1.0
    model[0].weight[0, 0, 2, 2] = 0.0
    model[0].bias[:] = 0.0
    data = Variable(torch.FloatTensor(
        [[[[i for i in range(5)] for _ in range(5)]]])
                    )

    model_path = path / modelname
    torch.onnx.export(model, data, model_path,
                      input_names=['input.1'],
                      output_names=['output'], verbose=True)
    return (
        model_path,
        {
            'input': [{'name': 'input.1', 'shape': (1, 1, 5, 5), 'dtype': 'float32'}],  # noqa: E501
            'output': [{'name': 'output', 'shape': (1, 1, 3, 3), 'dtype': 'float32'}]   # noqa: E501
        }
    )


@pytest.fixture(scope='function')
def runtimemodel(request: FixtureRequest, tmpfolder: Path):
    """
    Fixture that creates simple, runtime specific model.

    In order to use fixture, add: `@pytest.mark.usefixtures('runtimemodel')`
    to testing class

    The optimizer class for model compiling should be passed using:
    `@pytest.mark.parametrize('runtimemodel', [Optimizer], indirect=True)`

    Returned objects can be accessed using `self.objectname` variables


    Parameters
    ----------
    request: RequestFixture
        Fixture that allows get Optimizer class through parameter
        and share returned object with testing class
    tmpfolder: Path
        Fixture that provides temporary folder.

    Returns
    -------
    runtimemodel: PathOrURI
        The path to created model for runtime
    inputshapes: Tuple[int, ...]
        The input shape of model
    outputshapes: Tuple[int, ...]
        The output shape of model
    """

    onnxmodel, io_specs = create_onnx_model(tmpfolder)
    compiledmodelname = (uuid.uuid4().hex)
    if issubclass(request.param, TVMCompiler):
        compiledmodelname += '.so'
    compiled_model_path = tmpfolder / compiledmodelname
    optimizer = request.param(None, compiled_model_path)
    optimizer.compile(onnxmodel, io_specs)
    request.cls.runtimemodel = ResourceURI(compiled_model_path)
    request.cls.inputshapes = (1, 1, 5, 5)
    request.cls.outputshapes = (1, 1, 3, 3)
