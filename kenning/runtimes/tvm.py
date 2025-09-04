# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TVM-compiled models.
"""

from typing import List, Optional

import numpy as np
import tvm
from tvm.contrib import graph_executor
from tvm.runtime.vm import Executable, VirtualMachine

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotPreparedError,
)
from kenning.core.runtime import (
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class TVMRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TVM models.
    """

    inputtypes = ["tvm"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.tar",
        },
        "contextname": {
            "argparse_name": "--target-device-context",
            "description": "What accelerator should be used on target device",
            "default": "cpu",
            "enum": list(tvm.runtime.Device.STR2MASK.keys()),
        },
        "contextid": {
            "argparse_name": "--target-device-context-id",
            "description": "ID of the device to run the inference on",
            "type": int,
            "default": 0,
        },
        "use_tvm_vm": {
            "argparse_name": "--runtime-use-vm",
            "description": "At runtime use the TVM Relay VirtualMachine",
            "type": bool,
            "default": False,
        },
        "llext_binary_path": {
            "argparse_name": "--llext-binary-path",
            "description": "Path to the LLEXT binary",
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "The number of samples in a single batch.",
            "type": int,
            "default": 1,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        contextname: str = "cpu",
        contextid: int = 0,
        use_tvm_vm: bool = False,
        disable_performance_measurements: bool = False,
        llext_binary_path: Optional[PathOrURI] = None,
        batch_size: int = 1,
    ):
        """
        Constructs TVM runtime.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        contextname : str
            Name of the runtime context on the target device.
        contextid : int
            ID of the runtime context device.
        use_tvm_vm : bool
            Use the TVM Relay VirtualMachine.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        llext_binary_path : Optional[PathOrURI]
            Path to the LLEXT binary.
        batch_size : int
            Batch size for inference, which is a number of sample
            in a single batch.
        """
        self.model_path = model_path
        self.contextname = contextname
        self.contextid = contextid
        self.module = None
        self.func = None
        self.model = None
        self._input_prepared = False
        self.use_tvm_vm = use_tvm_vm
        self.llext_binary_path = llext_binary_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements,
            batch_size=batch_size,
        )

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        KLogger.debug(f"Loading inputs of size {len(input_data)}")
        if self.model is None:
            raise ModelNotPreparedError
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        input = {}
        try:
            for spec, inp in zip(
                self.processed_input_spec
                if self.processed_input_spec
                else self.input_spec,
                input_data,
            ):
                input[spec["name"]] = tvm.nd.array(inp)

            if self.use_tvm_vm:
                self.model.set_input("main", **input)
            else:
                self.model.set_input(**input)
            KLogger.debug("Inputs are ready")
            self._input_prepared = True
            return True
        except (TypeError, tvm.TVMError) as ex:
            KLogger.error(f"Failed to load input: {ex}", stack_info=True)
            return False

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        KLogger.info("Loading model")
        ctx = tvm.runtime.device(self.contextname, self.contextid)
        if self.use_tvm_vm:
            self.module = tvm.runtime.load_module(
                str(
                    self.model_path.with_suffix(self.model_path.suffix + ".so")
                )
            )
            loaded_bytecode = bytearray(
                open(str(self.model_path) + ".ro", "rb").read()
            )
            loaded_vm_exec = Executable.load_exec(loaded_bytecode, self.module)

            self.model = VirtualMachine(loaded_vm_exec, ctx)
        else:
            if input_data:
                with open(self.model_path, "wb") as outmodel:
                    outmodel.write(input_data)
            else:
                self.model_path
            self.module = tvm.runtime.load_module(str(self.model_path))
            self.func = self.module.get_function("default")
            self.model = graph_executor.GraphModule(self.func(ctx))
        KLogger.info("Model loading ended successfully")
        return True

    def run(self):
        if self.model is None:
            raise ModelNotPreparedError
        if not self._input_prepared:
            raise InputNotPreparedError
        self.model.run()

    def extract_output(self) -> List[np.ndarray]:
        if self.model is None:
            raise ModelNotPreparedError

        results = []
        if self.use_tvm_vm:
            for output in self.model.get_outputs():
                results.append(output.asnumpy())
        else:
            for i in range(self.model.get_num_outputs()):
                results.append(self.model.get_output(i).asnumpy())
        return results
