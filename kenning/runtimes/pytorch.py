# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for PyTorch models
"""
from typing import Optional, List
import gc

from kenning.core.runtime import (
    InputNotPreparedError,
    ModelNotPreparedError,
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class PyTorchRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on PyTorch models.
    """

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.pth",
        },
        "skip_jit": {
            "argparse_name": "--skip-jit",
            "description": "Do not execute Just-In-Time compilation of the model",   # noqa: E501
            "type": bool,
            "default": False,
        }
    }

    def __init__(
        self,
        model_path: PathOrURI,
        disable_performance_measurements: bool = True,
        skip_jit: bool = False,
    ):
        """
        Constructs PyTorch runtime

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics
        skip_jit : bool
            Do not execute Just-In-Time compilation of the model
        """
        import torch

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.model = None
        self.skip_jit = skip_jit
        self.input: Optional[List] = None
        self.output: Optional[List] = None
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        KLogger.info('Loading model')
        import torch
        from torch.jit.frontend import UnsupportedNodeError
        # Make sure GPU doesn't store redundant data
        gc.collect()
        torch.cuda.empty_cache()

        if input_data:
            with open(self.model_path, "wb") as fd:
                fd.write(input_data)

        try:
            self.model = torch.load(
                self.model_path,
                map_location=self.device
            )
        except Exception:
            import dill
            try:
                self.model = torch.load(
                    self.model_path,
                    map_location=self.device,
                    pickle_module=dill
                )
            except Exception:
                with open(self.model_path, 'rb') as fd:
                    self.model = dill.load(fd)

        if not isinstance(self.model,
                          (torch.nn.Module, torch.jit.ScriptModule)):
            KLogger.error(
                f'Loaded model is type {type(self.model).__name__}, only '
                'torch.nn.Module and torch.jit.ScriptModule supported'
            )
            return False
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            if not self.skip_jit:
                try:
                    self.model = torch.jit.script(self.model)
                    self.model = torch.jit.freeze(self.model)
                except (UnsupportedNodeError, RuntimeError):
                    KLogger.error(
                        'Model contains unsupported nodes, conversion to '
                        'TorchScript aborted',
                        stack_info=True
                    )
                except Exception:
                    KLogger.error(
                        'Model cannot be converted to TorchScript',
                        stack_info=True
                    )
        elif (isinstance(self.model, torch.jit.ScriptModule)
                and not self.skip_jit):
            self.model = torch.jit.freeze(self.model)
        KLogger.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data: bytes):
        KLogger.debug(f'Preparing inputs of size {len(input_data)}')
        import torch

        try:
            self.input = self.preprocess_input(input_data)
        except ValueError as ex:
            KLogger.error(f'Failed to load input: {ex}', stack_info=True)
            return False

        for id, input in enumerate(self.input):
            self.input[id] = torch.from_numpy(input.copy()).to(self.device)
        return True

    def run(self):
        if self.model is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError
        import torch

        with torch.no_grad():
            self.output = [self.model(data) for data in self.input]
        self.input = None

    def extract_output(self):
        import torch

        results = []
        for id, output in enumerate(self.output):
            if isinstance(output, torch.Tensor):
                results.append(output.detach().cpu().numpy())
            elif isinstance(output, list):
                results.extend([out.detach().cpu().numpy() for out in output])
            else:
                results.append(output)

        return self.postprocess_output(results)
