"""
Provides an API for model compilers.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from kenning.core.dataset import Dataset
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument, get_parsed_json_dict  # noqa: E501


class CompilationError(Exception):
    pass


class Optimizer(object):
    """
    Compiles the given model to a different format or runtime.
    """

    outputtypes = []

    inputtypes = {}

    arguments_structure = {
        'compiled_model_path': {
            'argparse_name': '--compiled-model-path',
            'description': 'The path to the compiled model output',
            'type': Path,
            'required': True
        },
        'dataset_percentage': {
            'argparse_name': '--dataset-percentage',
            'description': 'Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset',  # noqa: E501
            'type': float,
            'default': 1.0
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            dataset_percentage: float = 1.0):
        """
        Prepares the Optimizer object.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path to file where the compiled model should be saved
        dataset_percentage : float
            If the dataset is used for optimization (quantization), the
            dataset_percentage determines how much of data samples is going
            to be used
        """
        self.dataset = dataset
        self.compiled_model_path = compiled_model_path
        self.dataset_percentage = dataset_percentage

    @classmethod
    def form_argparse(cls, quantizes_model: bool = False):
        """
        Creates argparse parser for the Optimizer object.

        Parameters
        ----------
        quantizes_model : bool
            Tells if the compiler quantizes model - if so, flags for
            calibration dataset are enabled

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Compiler arguments')
        add_argparse_argument(
            group,
            Optimizer.arguments_structure,
            'compiled_model_path',
        )
        if quantizes_model:
            add_argparse_argument(
                group,
                Optimizer.arguments_structure,
                'dataset_percentage'
            )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset: Dataset, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Dataset
            The dataset object that is optionally used for optimization
        args : Dict
            arguments from ArgumentParser object

        Returns
        -------
        Optimizer : object of class Optimizer
        """
        if hasattr(args, 'dataset_percentage'):
            return cls(
                dataset,
                args.compiled_model_path,
                args.dataset_percentage
            )
        else:
            return cls(
                dataset,
                args.compiled_model_path
            )

    @classmethod
    def form_parameterschema(cls, quantizes_model: bool = True):
        """
        Creates schema for the Optimizer class

        Parameters
        ----------
        quantizes_model : bool
            Tells if the compiler quantizes model - if so, flags for
            calibration dataset are enabled

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = {
            "type": "object",
            "additionalProperties": False
        }

        add_parameterschema_argument(
            parameterschema,
            Optimizer.arguments_structure,
            'compiled_model_path',
        )

        if quantizes_model:
            add_parameterschema_argument(
                parameterschema,
                Optimizer.arguments_structure,
                'dataset_percentage'
            )

        return parameterschema

    @classmethod
    def from_json(cls, dataset: Dataset, json_dict: Dict):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        dataset : Dataset
            The dataset object that is optionally used for optimization
        json_dict : Dict
            Arguments for the constructor

        Returns
        -------
        Optimizer : object of class Optimizer
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            dataset=dataset,
            **parsed_json_dict
        )

    def set_input_type(self, inputtype: str):
        """
        Sets input type of the model for the compiler.
        """
        assert inputtype in self.inputtypes.keys()
        self.inputtype = inputtype

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):
        """
        Compiles the given model to a target format.

        The function compiles the model and saves it to the output file.

        The model can be compiled to a binary, a different framework or a
        different programming language.

        The additional compilation parameters that are not derivable from
        the input and output format should be passed in the constructor or via
        argument parsing.

        The compiled model is saved to compiled_model_path

        Parameters
        ----------
        inputmodelpath : Path
            Path to the input model
        inputshapes : Dict[str, Tuple[int, ...]]
            The dictionary with mapping (input name) -> (input shape)
        dtype : str
            The type of input tensors
        """
        raise NotImplementedError

    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.
        """
        raise NotImplementedError

    def get_input_formats(self) -> List[str]:
        """
        Returns list of names of possible input formats.
        """
        return list(self.inputtypes.keys())

    def get_output_formats(self) -> List[str]:
        """
        Returns list of names of possible output formats.
        """
        return self.outputtypes

    def consult_model_type(self, previous_block) -> str:
        """
        Finds output format of the previous block in the chain
        matching with an input format of the current block.

        Parameters
        ----------
        previous_block : Optimizer or ModelWrapper
            Previous block in the optimization chain.

        Raises
        ------
        ValueError : Raised if there is no matching format.

        Returns
        -------
        str : Matching format.
        """

        possible_outputs = previous_block.get_output_formats()

        for input in self.get_input_formats():
            if input in possible_outputs:
                return input

        raise ValueError(
            f'No matching formats between two objects: {self} and ' +
            f'{previous_block}'
        )

    def get_inputdtype(self) -> str:
        """
        Returns dtype of the input of the compiled model.
        Should be set during compilation.
        """
        assert hasattr(self, 'inputdtype')
        return self.inputdtype
