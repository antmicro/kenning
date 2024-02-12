from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.resource_manager import PathOrURI


class RuntimeBuilder(ArgumentsHandler, ABC):
    arguments_structure = {
        "workspace": {
            "description": "The path to the runtime source",
            "type": Path,
            "required": True,
        },
        "runtime_location": {
            "description": "Specifies where built runtime should be stored",
            "type": PathOrURI,
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
        runtime_location: PathOrURI,
        model_framework: Optional[str] = None,
    ):
        self.workspace = workspace
        self.runtime_location = runtime_location

        self.model_framework = None
        self.set_input_framework(model_framework)

    @abstractmethod
    def build(self):
        ...

    def set_input_framework(self, model_framework):
        if model_framework in self.allowed_frameworks:
            self.model_framework = model_framework
