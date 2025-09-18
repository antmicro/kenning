# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TFLite models.
"""

from typing import List, Optional

import numpy as np

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotPreparedError,
)
from kenning.core.runtime import (
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class TFLiteRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TFLite models.
    """

    inputtypes = ["tflite"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.tar",
        },
        "delegates": {
            "argparse_name": "--delegates-list",
            "description": "List of runtime delegates for the TFLite runtime",
            "type": list[str],
            "default": None,
            "nullable": True,
        },
        "num_threads": {
            "description": "Number of threads to use for inference",
            "default": 4,
            "type": int,
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
        delegates: Optional[List] = None,
        num_threads: int = 4,
        disable_performance_measurements: bool = False,
        llext_binary_path: Optional[PathOrURI] = None,
    ):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        delegates : Optional[List]
            List of TFLite acceleration delegate libraries.
        num_threads : int
            Number of threads to use for inference.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        llext_binary_path : Optional[PathOrURI]
            Path to the LLEXT binary.
        """
        self.model_path = model_path
        self.interpreter = None
        self._input_prepared = False
        self.num_threads = num_threads
        self.delegates = delegates
        self.llext_binary_path = llext_binary_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        KLogger.info("Loading model")
        if input_data:
            with open(self.model_path, "wb") as outmodel:
                outmodel.write(input_data)
        delegates = None
        if self.delegates:
            delegates = [
                tflite.load_delegate(delegate) for delegate in self.delegates
            ]
        self.interpreter = tflite.Interpreter(
            str(self.model_path),
            experimental_delegates=delegates,
            num_threads=self.num_threads,
        )
        self.interpreter.allocate_tensors()
        KLogger.info("Model loading ended successfully")
        return True

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        KLogger.debug(f"Loading inputs of size {len(input_data)}")
        if self.interpreter is None:
            raise ModelNotPreparedError
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        for i, spec in enumerate(
            self.processed_input_spec
            if self.processed_input_spec
            else self.input_spec
        ):
            self.interpreter.resize_tensor_input(
                i, [s if s > 0 else 1 for s in spec["shape"]]
            )
        self.interpreter.allocate_tensors()

        for det, inp in zip(self.interpreter.get_input_details(), input_data):
            self.interpreter.set_tensor(det["index"], inp)
        self._input_prepared = True
        return True

    def run(self):
        if self.interpreter is None:
            raise ModelNotPreparedError
        if not self._input_prepared:
            raise InputNotPreparedError
        self.interpreter.invoke()

    def extract_output(self) -> List[np.ndarray]:
        if self.interpreter is None:
            raise ModelNotPreparedError

        results = []
        for det in self.interpreter.get_output_details():
            out = self.interpreter.tensor(det["index"])()
            results.append(out.copy())

        return results
