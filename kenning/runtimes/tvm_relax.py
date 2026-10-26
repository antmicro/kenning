# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TVM Relax-compiled models.
"""

from typing import List, Optional

import numpy as np

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotPreparedError,
)
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
        self._input_prepared = False
        super().__init__(
            model_path=model_path,
            disable_performance_measurements=disable_performance_measurements,
            batch_size=batch_size,
        )

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        import tvm

        KLogger.debug(f"Number of loaded inputs: {len(input_data)}")
        if self.model is None:
            raise ModelNotPreparedError
        if input_data is None or len(input_data) == 0:
            KLogger.error("Received empty input data")
            return False

        self.model.set_input(
            "main", *[tvm.runtime.tensor(inp) for inp in input_data]
        )
        self._input_prepared = True
        return True

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
        if self.model is None:
            raise ModelNotPreparedError
        if not self._input_prepared:
            raise InputNotPreparedError
        self.model.invoke_stateful("main")

    def extract_output(self) -> List[np.ndarray]:
        if self.model is None:
            raise ModelNotPreparedError

        outputs = self.model.get_outputs("main")
        if hasattr(outputs, "numpy"):
            outputs = [outputs]
        return [output.numpy() for output in outputs]
