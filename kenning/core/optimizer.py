"""
Provides an API for model compilers.
"""

import argparse
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union
import json

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument, get_parsed_json_dict  # noqa: E501


class CompilationError(Exception):
    """
    General purpose exception raised when the compilation proccess fails.
    """
    pass


class Optimizer(object):
    """
    Compiles the given model to a different format or runtime.
    """

    outputtypes = []

    inputtypes = {}

    arguments_structure = {
        'compiled_model_path': {
            'description': 'The path to the compiled model output',
            'type': Path,
            'required': True
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path):
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

        self.actions = {
            'compile': self.action_compile
        }

    @classmethod
    def _form_argparse(cls):
        """
        Wrapper for creating argparse structure for the Optimizer class.

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
            Optimizer.arguments_structure
        )
        return parser, group

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Optimizer object.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser, group = cls._form_argparse()
        if cls.arguments_structure != Optimizer.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
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
        return cls(
            dataset,
            args.compiled_model_path
        )

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure for the Optimizer class.

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
            Optimizer.arguments_structure
        )
        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the Optimizer class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != Optimizer.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
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

    def set_compiled_model_path(self, compiled_model_path: Path):
        """
        Sets path for compiled model.
        """
        self.compiled_model_path = compiled_model_path

    def set_input_type(self, inputtype: str):
        """
        Sets input type of the model for the compiler.
        """
        assert inputtype in self.inputtypes.keys()
        self.inputtype = inputtype

    def compile(
            self,
            inputmodelpath: Path,
            io_specs: Optional[dict[list[dict]]] = None):
        """
        Compiles the given model to a target format.

        The function compiles the model and saves it to the output file.

        The model can be compiled to a binary, a different framework or a
        different programming language.

        If `io_specs` is passed, then the function uses it during the
        compilation, otherwise `load_io_specification` is used to fetch the
        specification saved in `inputmodelpath` + `.json`.

        The compiled model is saved to compiled_model_path and
        the specification is saved to compiled_model_path + .json

        Parameters
        ----------
        inputmodelpath : Path
            Path to the input model
        io_specs : Optional[dict[list[dict]]]
            Dictionary that has `input` and `output` keys that contain list
            of dictionaries mapping (property name) -> (property value)
            for the layers
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

    def consult_model_type(
            self,
            previous_block: Union['ModelWrapper', 'Optimizer'],
            force_onnx=False) -> str:
        """
        Finds output format of the previous block in the chain
        matching with an input format of the current block.

        Parameters
        ----------
        previous_block : Union[ModelWrapper, Optimizer]
            Previous block in the optimization chain.

        Raises
        ------
        ValueError : Raised if there is no matching format.

        Returns
        -------
        str : Matching format.
        """

        possible_outputs = previous_block.get_output_formats()

        if force_onnx:
            if (('onnx' in self.get_input_formats())
                    and ('onnx' in possible_outputs)):
                return 'onnx'
            else:
                raise ValueError(
                    '"onnx" format is not supported by at least one block\n' +
                    f'Input block supported formats: {", ".join(possible_outputs)}\n' +  # noqa: E501
                    f'Output block supported formats: {", ".join(self.get_input_formats())}'  # noqa: E501
                )

        for input in self.get_input_formats():
            if input in possible_outputs:
                return input

        raise ValueError(
            f'No matching formats between two objects: {self} and ' +
            f'{previous_block}\n' +
            f'Input block supported formats: {", ".join(possible_outputs)}\n' +  # noqa: E501
            f'Output block supported formats: {", ".join(self.get_input_formats())}'  # noqa: E501
        )

    def get_spec_path(self, modelpath: Path) -> Path:
        """
        Returns input/output specification path for the model
        saved in `modelpath`. It concatenates `modelpath` and `.json`.

        Parameters
        ----------
        modelpath : Path
            Path where the model is saved

        Returns
        -------
        Path : Path to the input/output specification of a given model.
        """
        spec_path = modelpath.parent / (modelpath.name + '.json')
        return Path(spec_path)

    def action_compile(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def save_io_specification(
            self,
            inputmodelpath: Path,
            io_specs: Optional[dict[list[dict]]] = None):
        """
        Internal function that saves input/output model specification
        which is used during both inference and compilation. If `io_specs`
        is None, the function uses specification of an input model
        stored in `inputmodelpath` + `.json`. Otherwise `io_specs` is used.

        The input/output specification is a list of dictionaries mapping
        properties names to their values. Legal properties names are `dtype`,
        `prequantized_dtype`, `shape`, `name`, `scale`, `zero_point`.

        The order of the layers has to be preserved.

        Parameters
        ----------
        inputmodelpath : Path
            Path to the input model
        io_specs : Optional[dict[list[dict]]]
            Specification of the input/ouput layers
        """
        if io_specs:
            model_spec = io_specs
        else:
            model_spec = self.load_io_specification(inputmodelpath)

        with open(self.get_spec_path(self.compiled_model_path), 'w') as f:  # noqa: E501
            json.dump(
                model_spec,
                f
            )

    def load_io_specification(self, modelpath: Path) -> dict[list[dict]]:
        """
        Returns saved input and output specification of a model
        saved in `modelpath` if there is one. Otherwise return an empty
        template of a specification.

        Parameters
        ----------
        modelpath : Path
            Path to the model which specification the function should read

        Returns
        -------
        dict : Specification of a model saved in `modelpath` if there is one.
            Empty template otherwise
        """
        spec_path = self.get_spec_path(modelpath)
        if spec_path.exists():
            with open(spec_path, 'r') as f:
                spec = json.load(f)

            return spec
        return {'input': [], 'output': []}
