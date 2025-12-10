# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for ai8x accelerator compiler.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from kenning.converters.ai8x_converter import Ai8xConverter
from kenning.converters.torch_converter import TorchConverter
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    KenningOptimizerError,
    OptimizedModelSizeError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    Optimizer,
)
from kenning.core.platform import Platform
from kenning.core.runtime import Runtime
from kenning.optimizers.ai8x_codegen import (
    generate_model_bin,
    generate_model_source,
)
from kenning.utils.class_loader import append_to_sys_path
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class Ai8xIzerError(KenningOptimizerError):
    """
    Raised when ai8xizer.py script fails.
    """

    def __init__(self, model_size: Optional[float] = None, *args):
        super().__init__(*args)
        self.model_size = model_size


class Ai8xTools(object):
    """
    Wrapper for ai8x tools.
    """

    def __init__(
        self,
        ai8x_training_path: Optional[Path] = None,
        ai8x_synthesis_path: Optional[Path] = None,
    ):
        """
        Wrapper for ai8x tools.

        Parameters
        ----------
        ai8x_training_path : Optional[Path]
            Path to the ai8x_training.
        ai8x_synthesis_path : Optional[Path]
            Path to the ai8x_synthesis.

        Raises
        ------
        FileNotFoundError
            Raised if any of the provided path does not exist.
        """
        if ai8x_training_path is None and "AI8X_TRAINING_PATH" in os.environ:
            ai8x_training_path = Path(os.environ["AI8X_TRAINING_PATH"])
        if ai8x_synthesis_path is None and "AI8X_SYNTHESIS_PATH" in os.environ:
            ai8x_synthesis_path = Path(os.environ["AI8X_SYNTHESIS_PATH"])

        if not ai8x_training_path:
            raise ValueError("ai8x_training_path not specified")
        if not ai8x_training_path.exists():
            raise FileNotFoundError(f"{ai8x_training_path} not found")
        if not (ai8x_training_path / ".venv").exists():
            raise FileNotFoundError(
                f"Python venv in {ai8x_training_path / '.venv'} not found. "
                "Create venv and install all dependencies."
            )

        if not ai8x_synthesis_path:
            raise ValueError("ai8x_synthesis_path not specified")
        if not ai8x_synthesis_path.exists():
            raise FileNotFoundError(f"{ai8x_synthesis_path} not found")
        if not (ai8x_synthesis_path / ".venv").exists():
            raise FileNotFoundError(
                f"Python venv in {ai8x_synthesis_path / '.venv.'} not found. "
                "Create venv and install all dependencies."
            )

        self.ai8x_training_path = ai8x_training_path
        self.ai8x_synthesis_path = ai8x_synthesis_path

    def yamlwriter(
        self,
        model_path: Path,
        model_input_shape: List[int],
        device_id: int,
        output_path: Path,
    ):
        """
        Executes yamlwriter.py from ai8x_training which generates YAML
        configuration of the model.

        Parameters
        ----------
        model_path : Path
            Path to the input model.
        model_input_shape : List[int]
            Shape of the model input.
        device_id : int
            ID of the ai8x device.
        output_path : Path
            Path where the YAML configuration will be saved.

        Raises
        ------
        CalledProcessError
            Raised when executed script fails.
        """
        try:
            KLogger.info("Running yamlwriter")
            outp = subprocess.check_output(
                [
                    str(self.ai8x_training_path / ".venv/bin/python"),
                    str(self.ai8x_training_path / "yamlwriter.py"),
                    "--model-path",
                    str(model_path.resolve()),
                    "--input-shape",
                    ",".join(map(str, model_input_shape)),
                    "--device-id",
                    str(device_id),
                    "--output-path",
                    str(output_path.resolve()),
                ],
                stderr=subprocess.STDOUT,
                cwd=str(self.ai8x_training_path),
            )
            KLogger.debug(f"yamlwriter output: {outp.decode().strip()}")
        except subprocess.CalledProcessError as e:
            KLogger.error(f"yamlwriter failed: {e}")
            KLogger.error(f"output: {e.output.decode().strip()}")
            raise

    def quantize(
        self,
        input_model_path: Path,
        output_model_path: Path,
        device: str,
    ):
        """
        Executes quantize.py from ai8x_synthesis which quantizes the model.

        Parameters
        ----------
        input_model_path : Path
            Path to the input model.
        output_model_path : Path
            Path where the quantized model will be saved.
        device : str
            Name of the ai8x device.

        Raises
        ------
        CalledProcessError
            Raised when executed script fails.
        """
        try:
            KLogger.info("Running quantize")
            outp = subprocess.check_output(
                [
                    str(self.ai8x_synthesis_path / ".venv/bin/python"),
                    str(self.ai8x_synthesis_path / "quantize.py"),
                    str(input_model_path.resolve()),
                    str(output_model_path.resolve()),
                    "--device",
                    device,
                ],
                stderr=subprocess.STDOUT,
                cwd=str(self.ai8x_synthesis_path),
            )
            KLogger.debug(f"quantize output: {outp.decode().strip()}")
        except subprocess.CalledProcessError as e:
            KLogger.error(f"quantize failed: {e}")
            KLogger.error(f"output: {e.output.decode().strip()}")
            raise

    def _extract_size_from_ai8xize(self, logs: str) -> Optional[float]:
        """
        Extracts size of the optimized model
        from the output of executed ai8xize.py script.

        Parameters
        ----------
        logs : str
            The output of the ai8xize.py script.

        Returns
        -------
        Optional[float]
            The optimized model size.
        """
        match = re.search(
            r"TOTAL: (?P<layers>[0-9,]+) parameter layers,"
            r" (?P<parameters>[0-9,]+) parameters,"
            r" (?P<bytes>[0-9,]+) bytes",
            logs,
        )
        if not match:
            KLogger.warning(
                "Cannot find compiled model size in the ai8xize output"
            )
            return
        KLogger.info(
            f"ai8xize processed model with {match.group('layers')} layers "
            f"and {match.group('parameters')}"
        )
        return int(match.group("bytes").replace(",", "")) / 1024

    def ai8xize(
        self,
        test_dir: Path,
        checkpoint_file: Path,
        config_file: Path,
        sample_input_path: Path,
        device: str,
    ) -> Optional[float]:
        """
        Executes ai8xize.py from ai8x_synthesis which generates CNN accelerator
        configuration code.

        Parameters
        ----------
        test_dir : Path
            Path where the output will be saved.
        checkpoint_file : Path
            Checkpoint file which contains quantized model.
        config_file : Path
            YAML configuration file.
        sample_input_path: Path
            Path to the sample input.
        device : str
            Name of the ai8x device.

        Raises
        ------
        Ai8xIzerError
            Raised when executed script fails.

        Return
        ------
        Optional[float]
            Size of the generated model.
        """
        try:
            KLogger.info("Running ai8xize")
            outp = subprocess.check_output(
                [
                    str(self.ai8x_synthesis_path / ".venv/bin/python"),
                    str(self.ai8x_synthesis_path / "ai8xize.py"),
                    "--verbose",
                    "--log",
                    "--test-dir",
                    str(test_dir.resolve()),
                    "--prefix",
                    "ai8xize",
                    "--checkpoint-file",
                    str(checkpoint_file.resolve()),
                    "--config-file",
                    str(config_file.resolve()),
                    "--sample-input",
                    str(sample_input_path.resolve()),
                    "--device",
                    device,
                    "--compact-data",
                    "--mexpress",
                    "--timer",
                    "0",
                    "--display-checkpoint",
                    "--no-scale-output",
                ],
                stderr=subprocess.STDOUT,
                cwd=str(self.ai8x_synthesis_path),
            )
            KLogger.debug(f"ai8xize output: {outp.decode().strip()}")
        except subprocess.CalledProcessError as e:
            KLogger.error(f"ai8xize failed: {e}")
            KLogger.error(f"output: {e.output.decode().strip()}")
            raise Ai8xIzerError(
                self._extract_size_from_ai8xize(e.output.decode().strip())
            ) from e
        return self._extract_size_from_ai8xize(outp.decode().strip())


class Ai8xCompiler(Optimizer):
    """
    The ai8x accelerator compiler.
    """

    outputtypes = ["ai8x_c"]

    inputtypes = {
        "ai8x": Ai8xConverter,
        "torch": TorchConverter,
    }

    SUPPORTED_DEVICE_IDS = [84, 85, 87]
    SUPPORTED_DEVICE_NAMES = ["MAX78002", "MAX78000"]

    arguments_structure = {
        "ai8x_synthesis_path": {
            "description": "Path to the ai8x-synthesis tool",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "ai8x_training_path": {
            "description": "Path to the ai8x-training tool",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "config_file": {
            "description": "Path to YAML file with model config",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        ai8x_synthesis_path: Optional[Path] = None,
        ai8x_training_path: Optional[Path] = None,
        config_file: Optional[PathOrURI] = None,
        location: Literal["host", "target"] = "host",
        model_framework: str = "any",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        The ai8x accelerator compiler.

        Compiler converts input models into C source code, that can be used to
        tun them on ai8x accelerators.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Dataset used to train the model - may be used for quantization
            during compilation stage.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        ai8x_synthesis_path : Optional[Path]
            Path to the ai8x-synthesis tool.
        ai8x_training_path : Optional[Path]
            Path to the ai8x-training tool.
        config_file : Optional[PathOrURI]
            Path to model ai8x config file in YAML format.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.config_file = config_file
        self.set_input_type(model_framework)
        self.device = None
        self.device_id = None
        self.ai8x_tools = Ai8xTools(ai8x_training_path, ai8x_synthesis_path)
        self.ai8x_model_size = None

        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
            model_wrapper=model_wrapper,
        )

    def get_framework_and_version(self) -> Tuple[str, str]:
        return ("ai8x", 1.0)

    def read_platform(self, platform: Platform):
        device = getattr(platform, "ai8x_device", None)
        device_id = getattr(platform, "ai8x_device_id", None)
        if device is not None and device_id is not None:
            self.device = device
            self.device_id = device_id
        else:
            raise ValueError(f"Unsupported platform {platform.name}")

    def run_compatibility_checks(
        self,
        platform: Platform,
        runtime: Optional[Runtime],
        input_model_path: PathOrURI,
    ) -> bool:
        device = getattr(platform, "ai8x_device", None)
        device_id = getattr(platform, "ai8x_device_id", None)
        if device is None or device_id is None:
            return False
        if device_id not in Ai8xCompiler.SUPPORTED_DEVICE_IDS:
            return False
        if device not in Ai8xCompiler.SUPPORTED_DEVICE_NAMES:
            return False
        return True

    def get_optimized_model_size(self):
        if self.ai8x_model_size is None:
            raise OptimizedModelSizeError(
                "Cannot retrieve model size before compilation"
            )
        return self.ai8x_model_size

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        if self.device_id is None:
            KLogger.warning("Ai8x device_id not specified. Defaulting to 85.")
            self.device_id = 85

        if self.device is None:
            KLogger.warning(
                "Ai8x device name not specified. Defaulting to MAX78000."
            )
            self.device = "MAX78000"
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            tmp_input_model_path = tmp_dir / f"{input_model_path.stem}_tmp.pth"

            if self.model_wrapper is not None:
                input_model = self.model_wrapper
                input_model.load_model(input_model_path)
                input_model.model_prepared = True

                with append_to_sys_path([self.ai8x_tools.ai8x_training_path]):
                    import ai8x

                # Check if batch norm is fused
                bn_fused = True
                for module in input_model.model.modules():
                    if (
                        isinstance(module, ai8x.QuantizationAwareModule)
                        and module.bn is not None
                    ):
                        KLogger.debug("Unfused batch norm found")
                        bn_fused = False
                        break
                if not bn_fused:
                    KLogger.info("Fusing batch norm layers")
                    ai8x.fuse_bn_layers(input_model.model)

                input_model.save_model(tmp_input_model_path, export_dict=False)
                input_model.save_io_specification(tmp_input_model_path)
            else:
                shutil.copy(input_model_path, tmp_input_model_path)
                shutil.copy(
                    input_model_path.with_suffix(
                        input_model_path.suffix + ".json"
                    ),
                    tmp_input_model_path.with_suffix(
                        tmp_input_model_path.suffix + ".json"
                    ),
                )

            # convert model
            converted_model_path = tmp_dir / f"{input_model_path.stem}_c.pth"

            converter = self.inputtypes[self.inputtype](tmp_input_model_path)
            if self.inputtype == "torch":
                converter.to_ai8x(
                    converted_model_path,
                    self.ai8x_tools,
                    self.device_id,
                )
            elif self.inputtype == "ai8x":
                converter.to_ai8x(
                    converted_model_path,
                    self.ai8x_tools,
                )
            else:
                converter.to_ai8x(
                    converted_model_path,
                )

            config_file = (
                self.config_file
                if self.config_file is not None
                else converted_model_path.with_suffix(
                    converted_model_path.suffix + ".yaml"
                )
            )
            input_shape = io_spec.get("processed_input", io_spec["input"])[0][
                "shape"
            ]

            if not config_file or not config_file.exists():
                self.ai8x_tools.yamlwriter(
                    tmp_input_model_path,
                    input_shape,
                    self.device_id,
                    config_file,
                )
                KLogger.debug(
                    f"Generated YAML layers config:\n{config_file.read_text()}"
                )

            # quantize model
            quantized_model_path = tmp_dir / f"{input_model_path.stem}_q.pth"
            self.ai8x_tools.quantize(
                converted_model_path, quantized_model_path, self.device
            )

            sample_input = np.random.randint(
                -128, 127, input_shape[1:], dtype=np.int64
            )
            sample_input_path = (
                quantized_model_path.parent / "sample_input.npy"
            )
            np.save(sample_input_path, sample_input)

            # compile model
            try:
                self.ai8x_model_size = self.ai8x_tools.ai8xize(
                    tmp_dir,
                    quantized_model_path,
                    config_file,
                    sample_input_path,
                    self.device,
                )
            except KenningOptimizerError as e:
                # Retrieve calculated model size and reraise the exception
                self.ai8x_model_size = e.model_size
                raise

            # update quantization params in IO spec
            io_spec["processed_input"][0]["dtype"] = "int8"
            io_spec["processed_input"][0]["prequantized_dtype"] = "float32"
            io_spec["processed_input"][0]["scale"] = 1.0 / 128
            io_spec["processed_input"][0]["zero_point"] = 0

            io_spec["output"][0]["dtype"] = "int8"
            io_spec["output"][0]["prequantized_dtype"] = "float32"
            io_spec["output"][0]["scale"] = 1.0
            io_spec["output"][0]["zero_point"] = 0

            self.compiled_model_path.parent.mkdir(exist_ok=True, parents=True)

            self.save_io_specification(self.compiled_model_path, io_spec)

            model_dir = self.compiled_model_path.with_suffix("")
            if model_dir == self.compiled_model_path:
                msg = "Compiled model path requires an extension"
                raise ValueError(msg)

            model_dir.mkdir(exist_ok=True, parents=True)

            generate_model_source(
                tmp_dir / "ai8xize" / "main.c",
                tmp_dir / "ai8xize" / "cnn.c",
                model_dir / "cnn_model.c",
            )

            generate_model_bin(
                tmp_dir / "ai8xize" / "cnn.c",
                tmp_dir / "ai8xize" / "weights.h",
                self.compiled_model_path,
            )

            KLogger.info(f"Compiled model saved in {self.compiled_model_path}")
