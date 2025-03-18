# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for runtime builders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from kenning.core.platform import Platform
from kenning.utils.args_manager import ArgumentsHandler


class RuntimeBuilder(ArgumentsHandler, ABC):
    """
    Builds the given model for a selected model framework.
    """

    arguments_structure = {
        "workspace": {
            "description": "Path to the runtime source",
            "type": Path,
            "required": True,
        },
        "output_path": {
            "description": "Specifies where built binaries should be stored",
            "type": Path,
            "default": None,
            "nullable": True,
        },
        "model_framework": {
            "description": "Name of the target model framework",
            "type": str,
            "default": None,
            "nullable": True,
        },
    }

    allowed_frameworks = []

    def __init__(
        self,
        workspace: Path,
        output_path: Optional[Path] = None,
        model_framework: Optional[str] = None,
    ):
        """
        Prepares the RuntimeBuilder object.

        Parameters
        ----------
        workspace : Path
            Location of the project directory.
        output_path : Optional[Path]
            Destination of the built binaries.
        model_framework : Optional[str]
            Selected model framework
        """
        self.workspace = workspace
        self.output_path = output_path

        self.model_framework = None
        if model_framework is not None:
            self.set_input_framework(model_framework)
        self.model_path = None

    @abstractmethod
    def build(self) -> Path:
        """
        Builds the runtime for the selected model framework
        and stores the result in the chosen location.
        """
        ...

    def set_input_framework(self, model_framework, force=False):
        if model_framework not in self.allowed_frameworks and (
            self.model_framework is None or force
        ):
            msg = f"Unsupported input type '{model_framework}.'"
            raise ValueError(msg)

        self.model_framework = model_framework

    def set_model_path(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def read_platform(self, platform: Platform):
        """
        Reads and integrates Platform data to configure model building.

        By default no data is neither read or integrate.

        It is important to take into account that different
        Platform-based classes come with a different sets of attributes.

        Parameters
        ----------
        platform: Platform
            object with platform details
        """
        pass
