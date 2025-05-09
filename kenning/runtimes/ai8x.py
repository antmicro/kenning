# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for ai8x models.
"""

from typing import List, Optional

import numpy as np

from kenning.core.runtime import Runtime
from kenning.utils.resource_manager import PathOrURI


class Ai8xRuntime(Runtime):
    """
    Dummy runtime subclass that provides an API for testing inference on AI8X
    models.
    """

    inputtypes = ["ai8x_c"]

    arguments_structure = {}

    def __init__(self, model_path: PathOrURI = None):
        self.model_path = model_path

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        ...

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        ...

    def run(self):
        ...

    def extract_output(self) -> List[np.ndarray]:
        ...
