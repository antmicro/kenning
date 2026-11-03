# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an wrapper for compiling models with Vela compiler.
"""

import os
import shutil
import subprocess
import sysconfig
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class VelaCompiler(Optimizer):
    """
    The Vela Compiler.
    """

    outputtypes = ["litert"]

    inputtypes = ["litert"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "litert",
            "enum": inputtypes,
        },
        "output_type": {
            "description": "Output type for the compiled model.",
            "type": str,
            "default": "tflite",
            "enum": ["raw", "tflite"],
        },
        "accelerator_config": {
            "description": "Accelerator configuration to use.",
            "type": str,
            "default": "ethos-u55-256",
        },
        "config": {"description": "Vela configuration.", "type": Path},
        "system_config": {
            "description": "System configuration to select "
            "from the Vela configuration file.",
            "type": str,
            "default": "Ethos_U55_High_End_Embedded",
        },
        "memory_mode": {
            "description": "Memory mode to select from the "
            "Vela configuration file.",
            "type": str,
            "default": "Shared_Sram",
        },
        "cop_format": {
            "description": "Custom Operator Payload format.",
            "type": str,
            "enum": ["COP1", "COP2"],
            "default": "COP1",
        },
        "tensor_allocator": {
            "description": "Tensor allocation method.",
            "type": str,
            "enum": ["LinearAlloc", "Greedy", "HillClimb"],
            "default": "Greedy",
        },
        "show_cpu_operations": {
            "description": "Show the operations that fall back to the CPU.",
            "type": bool,
            "default": False,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
        output_type: str = "tflite",
        model_framework: str = "litert",
        accelerator_config: str = "ethos-u55-256",
        config: Optional[Path] = None,
        system_config: str = "Ethos_U55_High_End_Embedded",
        memory_mode: str = "Shared_Sram",
        cop_format: str = "COP1",
        tensor_allocator: str = "Greedy",
        show_cpu_operations: bool = False,
    ):
        """
        Prepares the Optimizer object.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage.
        compiled_model_path : PathOrURI
            Path to directory where the compiled model should be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        output_type : str
            Output type for the compiled model.
        model_framework : str
            The input type of the model, framework-wise.
        accelerator_config : str
            Accelerator configuration to use.
        config : Optional[Path]
            Vela configuration.
        system_config : str
            System configuration to select from the Vela configuration file.
        memory_mode : str
            Memory mode to select from the Vela configuration file.
        cop_format : str
            Custom Operator Payload format.
        tensor_allocator : str
            Tensor allocation method.
        show_cpu_operations : bool
            Show the operations that fall back to the CPU.
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        compiled_model_path = Path(compiled_model_path)
        super().__init__(dataset, compiled_model_path, location, model_wrapper)
        self.output_type = output_type
        self.accelerator_config = accelerator_config
        if config is None:
            self.config = "Arm/vela.ini"
        else:
            self.config = config
        self.system_config = system_config
        self.memory_mode = memory_mode
        self.cop_format = cop_format
        self.tensor_allocator = tensor_allocator
        self.show_cpu_operations = show_cpu_operations

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        input_model_path = ResourceURI(input_model_path)
        KLogger.debug(f"Model input path: {input_model_path}")

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        vela_exe = Path(sysconfig.get_path("scripts")) / "vela"

        output_dir_name = Path(tempfile.TemporaryDirectory().name)

        cmd = [
            str(vela_exe),
            str(input_model_path),
            "--accelerator-config",
            str(self.accelerator_config),
            "--config",
            str(self.config),
            "--system-config",
            str(self.system_config),
            "--memory-mode",
            str(self.memory_mode),
            "--cop-format",
            str(self.cop_format),
            "--tensor-allocator",
            str(self.tensor_allocator),
            "--verbose-cycle-estimate",
            "--verbose-performance",
            "--output-format",
            str(self.output_type),
            "--output-dir",
            str(output_dir_name),
        ] + (["--show-cpu-operations"] if self.show_cpu_operations else [])

        subprocess.run(
            cmd,
            check=True,
        )

        if os.path.exists(self.compiled_model_path):
            if os.path.isdir(self.compiled_model_path):
                shutil.rmtree(self.compiled_model_path)
            else:
                os.remove(self.compiled_model_path)

        self.compiled_model_path.mkdir(parents=True, exist_ok=True)
        for file in os.listdir(output_dir_name):
            shutil.move(output_dir_name / file, self.compiled_model_path)

        self.save_io_specification(self.compiled_model_path, io_spec)

    def get_framework(self) -> str:
        return "litert"

    @classmethod
    def get_framework_version(cls) -> str:
        return version("ethos-u-vela")
