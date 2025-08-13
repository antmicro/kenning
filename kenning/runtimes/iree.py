# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for IREE models.
"""

from typing import List, Optional

import numpy as np
from iree import runtime as ireert

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotPreparedError,
)
from kenning.core.runtime import (
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class IREERuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on IREE models.
    """

    inputtypes = ["iree"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.vmfb",
        },
        "driver": {
            "argparse_name": "--driver",
            "description": "Name of the runtime target",
            "enum": ireert.HalDriver.query(),
            "required": True,
        },
        "llext_binary_path": {
            "argparse_name": "--llext-binary-path",
            "description": "Path to the LLEXT binary",
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        driver: str = "local-sync",
        disable_performance_measurements: bool = False,
        llext_binary_path: Optional[PathOrURI] = None,
    ):
        """
        Constructs IREE runtime.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        driver : str
            Name of the deployment target on the device.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        llext_binary_path : Optional[PathOrURI]
            Path to the LLEXT binary.
        """
        self.model_path = model_path
        self.model = None
        self.input = None
        self.driver = driver
        self.llext_binary_path = llext_binary_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def load_input(self, input_data: List[List[np.ndarray]]) -> bool:
        KLogger.debug(f"Loading inputs of size {len(input_data)}")
        if self.model is None:
            raise ModelNotPreparedError
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        self.input = input_data
        return True

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        KLogger.info("loading model")
        if input_data:
            with open(self.model_path, "wb") as outmodel:
                outmodel.write(input_data)

        with open(self.model_path, "rb") as outmodel:
            compiled_buffer = outmodel.read()

        self.model = ireert.load_vm_flatbuffer(
            compiled_buffer, driver=self.driver
        )

        KLogger.info("Model loading ended successfully")
        return True

    def run(self):
        if self.model is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError
        self.output = self.model.main(*self.input)

    def extract_output(self) -> List[np.ndarray]:
        if self.model is None:
            raise ModelNotPreparedError

        results = []
        try:
            results.append(self.output.to_host())
        except AttributeError:
            for out in self.output:
                results.append(out.to_host())
        return results
