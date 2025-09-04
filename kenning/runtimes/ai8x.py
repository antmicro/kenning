# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for ai8x models.
"""

from typing import List, Optional

import numpy as np

from kenning.core.platform import Platform
from kenning.core.runtime import Runtime
from kenning.utils.resource_manager import PathOrURI


class Ai8xRuntime(Runtime):
    """
    Dummy runtime subclass that provides an API for testing inference on AI8X
    models.
    """

    inputtypes = ["ai8x_c"]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": PathOrURI,
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
        model_path: Optional[PathOrURI] = None,
        disable_performance_measurements: bool = False,
        batch_size: int = 1,
    ):
        """
        Constructs AI8X runtime.

        Parameters
        ----------
        model_path : Optional[PathOrURI]
            Path or URI to the model file.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        batch_size : int
            Batch size for inference, which is a number of sample
            in a single batch.
        """
        self.model_path = model_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements,
            batch_size=batch_size,
        )

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        ...

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        ...

    def run(self):
        ...

    def extract_output(self) -> List[np.ndarray]:
        ...

    @staticmethod
    def get_available_ram(platform: Platform) -> Optional[float]:
        return getattr(platform, "ai8x_weights_memory_kb", None)
