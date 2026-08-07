# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
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
from kenning.core.platform import Platform
from kenning.core.runtime import (
    Runtime,
)
from kenning.platforms.cuda import CUDAPlatform
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
            "required": False,
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
        driver: str = "local-sync",
        disable_performance_measurements: bool = False,
        llext_binary_path: Optional[PathOrURI] = None,
        batch_size: int = 1,
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
        batch_size : int
            Batch size for inference, which is a number of sample
            in a single batch.
        """
        self.model_path = model_path
        self.model = None
        self.entry_func = None
        self.io_spec = None
        self.input = None
        self.driver = driver
        self.llext_binary_path = llext_binary_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements,
            batch_size=batch_size,
        )

    def load_input(self, input_data: List[List[np.ndarray]]) -> bool:
        KLogger.debug(f"Loading inputs of size {len(input_data)}")
        if self.entry_func is None:
            raise ModelNotPreparedError
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        self.input = input_data
        return True

    def prepare_model(self, input_data: Optional[bytes]):
        KLogger.info(f"Loading model, driver: {self.driver}")

        if self.driver == "coralnpu":
            self._prepare_model_coralnpu(input_data)
        else:
            self._prepare_model(input_data)

        KLogger.info("Model loading ended successfully")

    def run(self):
        if self.entry_func is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError

        self.output = self.entry_func(*self.input)

    def extract_output(self) -> List[np.ndarray]:
        if self.entry_func is None:
            raise ModelNotPreparedError

        results = []
        try:
            results.append(self.output.to_host())
        except AttributeError:
            for out in self.output:
                results.append(out.to_host())
        return results

    def read_platform(self, platform: Platform):
        super().read_platform(platform)
        if isinstance(platform, CUDAPlatform):
            self.driver = "cuda"

    def _prepare_model_coralnpu(self, input_data: Optional[bytes]):
        instance = ireert.VmInstance()

        try:
            cpu_device = ireert.get_device("local-sync")
            KLogger.info("Created CPU device")
        except Exception as e:
            KLogger.error(f"Failed to create CPU device: {e}")
            raise

        try:
            npu_device = ireert.get_device("coralnpu")
            KLogger.info("Created NPU device")
        except Exception as e:
            KLogger.error(f"Failed to create NPU device: {e}")
            raise

        hal_module = ireert.create_hal_module(
            instance, devices=[cpu_device, npu_device]
        )

        class MultiDeviceConfig:
            def __init__(self, device, instance, hal_module):
                self.device = device  # Used by FunctionInvoker for arguments
                self.vm_instance = instance
                self.default_vm_modules = (hal_module,)

        config = MultiDeviceConfig(cpu_device, instance, hal_module)

        KLogger.info(f"Loading VMFB from {self.model_path}")
        try:
            vm_module = ireert.VmModule.mmap(
                instance, str(self.model_path.resolve())
            )
        except Exception as e:
            KLogger.warning(f"mmap failed: {e}. Trying from_flatbuffer...")
            with open(self.model_path, "rb") as f:
                vm_module = ireert.VmModule.from_flatbuffer(
                    config.vm_instance, f.read()
                )
            KLogger.info("Successfully loaded VMFB from flatbuffer")

        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        self.entry_func = ctx.modules.jit__lambda.main

    def _prepare_model(self, input_data: Optional[bytes]):
        if input_data:
            with open(self.model_path, "wb") as outmodel:
                outmodel.write(input_data)

        with open(self.model_path, "rb") as outmodel:
            compiled_buffer = outmodel.read()

        self.model = ireert.load_vm_flatbuffer(
            compiled_buffer, driver=self.driver
        )

        # We are retrieving entry function name from iospec and taking only
        # the function name itself (discaring the module name).
        if hasattr(self, "entry_function_name"):
            entry_func_name = (
                self.entry_function_name.rsplit(".", 1)[-1]
                if self.entry_function_name
                else ""
            )
        else:
            entry_func_name = ""
            KLogger.warning(
                "IO specification not loaded, using default entry"
                f" function name: {entry_func_name}"
            )
        self.entry_func = getattr(self.model, entry_func_name)
