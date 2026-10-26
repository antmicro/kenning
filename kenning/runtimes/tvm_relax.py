# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TVM Relax-compiled models.
"""

from typing import List, Optional

import numpy as np

from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class TVMRelaxRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TVM Relax models.
    """

    inputtypes = ["tvm_relax"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.so",
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        disable_performance_measurements: bool = False,
        batch_size: int = 1,
    ):
        """
        Constructs TVM Relax runtime.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        batch_size : int
            Batch size for inference, which is a number of sample
            in a single batch.
        """
        self.model = None
        super().__init__(
            model_path=model_path,
            disable_performance_measurements=disable_performance_measurements,
            batch_size=batch_size,
        )

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        raise NotImplementedError

    def prepare_model(self, input_data: Optional[bytes]):
        import tvm
        from tvm.runtime.vm import VirtualMachine

        KLogger.info("Loading model")
        if input_data:
            with open(self.model_path, "wb") as outmodel:
                outmodel.write(input_data)
        module = tvm.runtime.load_module(str(self.model_path))
        self.model = VirtualMachine(module, tvm.cpu())
        KLogger.info("Model loading ended successfully")

    def run(self):
        raise NotImplementedError

    def extract_output(self) -> List[np.ndarray]:
        raise NotImplementedError
