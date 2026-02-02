# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An Optimizer-based block for inserting models
in specified format to an existing flow.
"""

import shutil
from typing import Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ModelInserter(Optimizer):
    """
    Mock Optimizer-based class for inserting model into flow.

    This Optimizer does not perform any optimizations, it
    only fetches the model from a given path and returns
    it for further Optimizer blocks.
    """

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "type": str,
            "required": True,
        },
        "input_model_path": {
            "argparse_name": "--input-model-path",
            "description": "Path to the model to be inserted",
            "type": ResourceURI,
            "required": True,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        model_framework: str,
        input_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        A mock Optimizer for model injection.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        model_framework : str
            Framework of the input model to be inserted.
        input_model_path : PathOrURI
            URI to the input model to be inserted.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.model_framework = model_framework
        self.input_model_path = input_model_path
        self.outputtypes = [self.model_framework]
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        KLogger.warning("Inserting the model into pipeline")
        KLogger.warning("The input model from previous block is ignored")
        KLogger.warning(f"The used model is from {self.input_model_path}")

        shutil.copy(self.input_model_path, self.compiled_model_path)
        self.save_io_specification(self.input_model_path, None)

    def set_input_type(self, inputtype: str):
        self.inputtype = inputtype

    @classmethod
    def get_framework(cls) -> str:
        return "kenning"

    @classmethod
    def get_framework_version(cls) -> str:
        import kenning

        if hasattr(kenning, "__version__"):
            return kenning.__version__
        else:
            return "dev"
