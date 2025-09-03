# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of ExecuTorch runtime.
"""

from typing import List, Optional

import numpy as np

from kenning.core.exceptions import (
    InputNotPreparedError,
    KenningRuntimeError,
    ModelNotLoadedError,
    ModelNotPreparedError,
    ModulesIncompatibleError,
)
from kenning.core.runtime import Runtime
from kenning.modelwrappers.frameworks.pytorch import Tensor
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ExecuTorchRuntime(Runtime):
    """
    Class implementing Kenning runtime API for the ExecuTorch runtime.
    """

    SUPPORTED_MEMORY_LAYOUTS = ["NCHW", "NHWC"]

    inputtypes = ["executorch"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be saved.",
            "type": ResourceURI,
            "default": "model.pth",
        },
        "image_memory_layout": {
            "argparse_name": "--image-memory-layout",
            "description": "The memory layout of an image, where ",
            "type": str,
            "default": "NCHW",
            "enum": SUPPORTED_MEMORY_LAYOUTS,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        disable_performance_measurements: bool = False,
        image_memory_layout: str = "NCHW",
        batch_size: int = 1,
    ):
        import torch

        super().__init__(disable_performance_measurements, batch_size)
        # Currently, only CPU-capable backends are supported.
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.model = None
        self.input = None
        self.output = None

        image_memory_layout = image_memory_layout.upper()
        if image_memory_layout not in self.SUPPORTED_MEMORY_LAYOUTS:
            raise KenningRuntimeError(
                f"Unsupported memory memory layout `{image_memory_layout}`"
                f" layouts: {', '.join(self.SUPPORTED_MEMORY_LAYOUTS)}"
            )
        self._memory_layout = image_memory_layout

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        KLogger.info("Loading a model to the ExecuTorch runtime.")

        from executorch.runtime import Runtime as ExecuTorch

        self.runtime = ExecuTorch.get()
        if input_data:
            with open(self.model_path, "wb") as fd:
                fd.write(input_data)

        self._load_pte_method()
        KLogger.info("Successfully loaded the model.")
        return True

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        import torch

        KLogger.debug(f"Loading inputs of size {len(input_data)}.")
        if not input_data:
            KLogger.error("Received an empty input.")
            return False
        if self.model is None:
            raise ModelNotPreparedError()

        self.input = [
            torch.from_numpy(tensor.copy()).to(self.device)
            for tensor in input_data
        ]

        return True

    @staticmethod
    def _convert_NHWC_to_NCHW(tensor: Tensor) -> Tensor:
        """
        Convert a tensor in the NHWC format to the NCHW format.

        Parameters
        ----------
        tensor : Tensor
            A 3- or 4-dimensional input tensor to be converted.

        Returns
        -------
        Tensor
            An output tensor in the proper shape.

        Raises
        ------
        ModelNotPreparedError
            Raised if the input tensor has too many or
            too few dimensions.
        """
        if tensor.dim() == 4:
            # NHWC: (N, H, W, C) -> NCHW: (N, C, H, W)
            return tensor.permute(0, 3, 1, 2).contiguous()
        elif tensor.dim() == 3:
            # NHWC: (H, W, C) -> NCHW: (C, H, W)
            return tensor.permute(2, 0, 1).contiguous()
        else:
            raise ModulesIncompatibleError(
                "The input tensor must be 3D or 4D for NHWC to NCHW "
                f"conversion. It has {tensor.dim()} dimensions."
            )

    def _load_pte_method(
        self, method_name: str = "forward"
    ) -> Optional[object]:
        """
        Load an executable method from PTE binary.

        The function loads a method performing some operation
        (usually, inference) from the PTE binary file.
        The PTE binary is the ExecuTorch format for storing the models.

        Parameters
        ----------
        method_name : str, optional
            Name of a method, by default "forward".

        Returns
        -------
        Optional[object]
            Either object of type Method or None
            if the method is missing.

        Raises
        ------
        ModelNotLoadedError
            Raised if a PTE lacks the desired method.
        """
        from executorch.runtime import Verification

        program = self.runtime.load_program(
            self.model_path,
            verification=Verification.Minimal,
        )
        self.model = program.load_method(method_name)
        if self.model is None:
            raise ModelNotLoadedError(
                f"The loaded PTE binary lacks `{method_name}` method."
            )

    def run(self):
        import torch

        if self.model is None:
            raise ModelNotPreparedError

        if self.input is None:
            raise InputNotPreparedError("Input data for inference is None.")

        self.output = []
        reshaped_input = []
        for tensor in self.input:
            if not isinstance(tensor, torch.Tensor):
                raise ModulesIncompatibleError(
                    "Input must be a `torch.Tensor` for ExecuTorch."
                )
            KLogger.debug(
                "The shape of the received input tensor: %s", str(tensor.shape)
            )
            if self._memory_layout == "NHWC":
                tensor = ExecuTorchRuntime._convert_NHWC_to_NCHW(tensor)
            KLogger.debug(
                f"Running inference with tensor shape: {tensor.shape}"
            )
            reshaped_input.append(tensor)

        self.output = self.model.execute((*reshaped_input,))
        KLogger.debug("The inference was run successfully.")

    def extract_output(self) -> List[np.ndarray]:
        import torch

        if self.model is None or self.output is None:
            raise ModelNotPreparedError(
                "Cannot extract output from the non-prepared model."
            )

        results = []
        for output in self.output:
            if isinstance(output, torch.Tensor):
                results.append(output.detach().numpy())
            elif isinstance(output, list):
                results.extend([out.detach().numpy() for out in output])
            else:
                results.append(output)

        return results
