# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for IREE models.
"""

from iree import runtime as ireert

from kenning.core.runtime import (
    InputNotPreparedError,
    ModelNotPreparedError,
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
    }

    def __init__(
        self,
        model_path: PathOrURI,
        driver: str = "local-sync",
        disable_performance_measurements: bool = False,
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
        """
        self.model_path = model_path
        self.model = None
        self.input = None
        self.driver = driver
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def prepare_input(self, input_data):
        KLogger.debug(f"Preparing inputs of size {len(input_data)}")
        if self.model is None:
            raise ModelNotPreparedError

        try:
            self.input = self.preprocess_input(input_data)
        except ValueError as ex:
            KLogger.error(f"Failed to load input: {ex}", stack_info=True)
            return False
        return True

    def prepare_model(self, input_data):
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

    def extract_output(self):
        if self.model is None:
            raise ModelNotPreparedError

        results = []
        try:
            results.append(self.output.to_host())
        except AttributeError:
            for out in self.output:
                results.append(out.to_host())
        return self.postprocess_output(results)
