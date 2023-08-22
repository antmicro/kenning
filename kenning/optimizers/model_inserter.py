# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An Optimizer-based block for inserting models
in specified format to an existing flow.
"""

from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
from typing import Optional, Dict, List, Union
import shutil
from kenning.utils.logger import get_logger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ModelInserter(Optimizer):
    """
    Mock Optimizer-based class for inserting model into flow.

    This Optimizer does not perform any optimizations, it
    only fetches the model from a given path and returns
    it for further Optimizer blocks.
    """

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'type': str,
            'required': True
        },
        'input_model_path': {
            'argparse_name': '--input-model-path',
            'description': 'Path to the model to be inserted',
            'type': ResourceURI,
            'required': True
        },
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: PathOrURI,
            modelframework: str,
            input_model_path: PathOrURI):
        """
        A mock Optimizer for model injection.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        modelframework : str
            Framework of the input model to be inserted.
        input_model_path : PathOrURI
            URI to the input model to be inserted.
        """
        self.modelframework = modelframework
        self.input_model_path = input_model_path
        self.outputtypes = [self.modelframework]
        super().__init__(dataset, compiled_model_path)

    def compile(
            self,
            input_model_path: PathOrURI,
            io_spec: Optional[Dict[str, List[Dict]]] = None):
        log = get_logger()
        log.warn('Inserting the model into pipeline')
        log.warn('The input model from previous block is ignored')
        log.warn(f'The used model is from {self.input_model_path}')

        shutil.copy(self.input_model_path, self.compiled_model_path)
        self.save_io_specification(self.input_model_path, None)

    def consult_model_type(
            self,
            previous_block: Union['ModelWrapper', 'Optimizer'],
            force_onnx=False) -> str:
        """
        Returns the first type supported by the previous block.

        Override of the original consult_model_type, simplifying
        the consulting process due to class nature.

        Parameters
        ----------
        previous_block : Union[ModelWrapper, Optimizer]
            Previous block in the optimization chain.
        force_onnx : bool
            Forces ONNX format.

        Returns
        -------
        str :
            Matching format.
        """
        possible_outputs = previous_block.get_output_formats()

        if force_onnx and self.modelframework != 'onnx':
            raise ValueError(
                '"onnx" format is not supported by ModelInserter'
            )
        return possible_outputs[0]

    def set_input_type(self, inputtype: str):
        self.inputtype = inputtype

    def get_framework_and_version(self):
        import kenning
        if hasattr(kenning, '__version__'):
            return ("kenning", kenning.__version__)
        else:
            return ("kenning", "dev")