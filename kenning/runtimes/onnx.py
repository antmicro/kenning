# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for ONNX models.
"""

from typing import List, Optional

import numpy as np
import onnxruntime as ort

from kenning.core.runtime import (
    InputNotPreparedError,
    ModelNotPreparedError,
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ONNXRuntime(Runtime):
    """
    Runtime subclass that provides an API for testing inference on ONNX models.
    """

    inputtypes = ["onnx"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.tar",
        },
        "execution_providers": {
            "description": "List of execution providers ordered by priority",
            "type": list[str],
            "default": ["CPUExecutionProvider"],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        execution_providers: List[str] = ["CPUExecutionProvider"],
        disable_performance_measurements: bool = False,
    ):
        """
        Constructs ONNX runtime.

        Parameters
        ----------
        model_path : PathOrURI
            URI for the model file.
        execution_providers : List[str]
            List of execution providers ordered by priority.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        """
        self.model_path = model_path
        self.session = None
        self.input = None
        self.execution_providers = execution_providers
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        if self.session is None:
            raise ModelNotPreparedError
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        self.input = {}
        for spec, inp in zip(self.input_spec, input_data):
            self.input[spec["name"]] = inp
        return True

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        KLogger.info("Loading model")
        if input_data:
            with open(self.model_path, "wb") as outmodel:
                outmodel.write(input_data)

        self.session = ort.InferenceSession(
            str(self.model_path), providers=self.execution_providers
        )

        # Input dtype can come either as a valid np.dtype
        # or as a string that need to be parsed
        def onnx_to_np_dtype(s):
            if s == "tensor(float)":
                return "float32"
            if isinstance(s, np.dtype):
                return s.name

        def update_io_spec(read_spec, session_spec):
            model_spec = []
            for input in session_spec:
                model_spec.append(
                    {
                        "name": input.name,
                        "shape": np.array(
                            [
                                s if isinstance(s, int) else -1
                                for s in input.shape
                            ]
                        ),
                        "dtype": onnx_to_np_dtype(input.type),
                    }
                )

            if not read_spec:
                return model_spec
            else:
                for s, m in zip(read_spec, model_spec):
                    if "name" not in s:
                        s["name"] = m["name"]
                    if "shape" not in s:
                        s["shape"] = m["shape"]
                    if "dtype" not in s:
                        s["dtype"] = m["dtype"]

            return read_spec

        self.input_spec = update_io_spec(
            self.input_spec, self.session.get_inputs()
        )

        self.output_spec = update_io_spec(
            self.output_spec, self.session.get_outputs()
        )

        KLogger.info("Model loading ended successfully")
        return True

    def run(self):
        if self.session is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError
        # Use actual output names from the session instead of spec names.
        output_names = [output.name for output in self.session.get_outputs()]
        self.scores = self.session.run(output_names, self.input)

    def extract_output(self) -> List[np.ndarray]:
        if self.session is None:
            raise ModelNotPreparedError

        results = []
        for i in range(len(self.session.get_outputs())):
            results.append(self.scores[i])

        return results
