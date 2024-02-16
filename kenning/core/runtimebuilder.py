# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for runtime builders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from kenning.utils.args_manager import ArgumentsHandler


class RuntimeBuilder(ArgumentsHandler, ABC):
    """
    Builds the given model for a selected model framework.
    """

    arguments_structure = {
        "workspace": {
            "description": "The path to the runtime source",
            "type": Path,
            "required": True,
        },
        "runtime_location": {
            "description": "Specifies where built runtime should be stored",
            "type": Path,
            "required": True,
        },
        "model_framework": {
            "description": "Model framework",
            "type": str,
            "default": None,
            "nullable": True,
        },
    }

    allowed_frameworks = []

    def __init__(
        self,
        workspace: Path,
        runtime_location: Path,
        model_framework: Optional[str] = None,
    ):
        """
        Prepares the RuntimeBuilder object.

        Parameters
        ----------
        workspace: Path
            Location of the project directory.
        runtime_location: Path
            Destination of the built runtime

        model_framework: Optional[str]
            Selected model framework
        """
        self.workspace = workspace
        self.runtime_location = runtime_location

        self.model_framework = None
        if model_framework is not None:
            self.set_input_framework(model_framework)

    @abstractmethod
    def build(self):
        ...

    def set_input_framework(self, model_framework, force=False):
        if model_framework not in self.allowed_frameworks and (
            self.model_framework is None or force
        ):
            msg = f"Unsupported input type '{model_framework}.'"
            raise ValueError(msg)

        self.model_framework = model_framework
