# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for model compilers.
"""

import json
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from kenning.core.dataset import Dataset
from kenning.core.helpers.utils import _get_model_size
from kenning.core.model import ModelWrapper
from kenning.core.platform import Platform
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI

EXT_TO_FRAMEWORK = {
    ".onnx": "onnx",
    ".h5": "keras",
    ".pt": "torch",
    ".pth": "torch",
    ".tflite": "tflite",
}


class ConversionError(Exception):
    """
    General purpose exception raised when the model conversion process fails.
    """

    pass


class CompilationError(Exception):
    """
    General purpose exception raised when the compilation process fails.
    """

    pass


class IOSpecificationNotFoundError(Exception):
    """
    Exception raised when needed input/output specification can not be found.
    """

    pass


class OptimizedModelSizeError(Exception):
    """
    Exception raised when retrieving size of the optimized model failed.
    """

    pass


class Optimizer(ArgumentsHandler, ABC):
    """
    Compiles the given model to a different format or runtime.
    """

    outputtypes = []

    inputtypes = {}

    locations = ["host", "target"]

    arguments_structure = {
        "compiled_model_path": {
            "description": "The path to the compiled model output",
            "type": ResourceURI,
            "required": True,
        },
        "location": {
            "description": "Specifies where optimization should be performed "
            "in client-server scenario",
            "default": "host",
            "enum": locations,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        Prepares the Optimizer object.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Dataset used to train the model - may be used for quantization
            during compilation stage.
        compiled_model_path : PathOrURI
            Path to file where the compiled model should be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        assert location in Optimizer.locations, f"Invalid location: {location}"
        self.dataset = dataset
        self.compiled_model_path = compiled_model_path
        self.location = location
        self.model_wrapper = model_wrapper

    def init(self):
        """
        Initializes optimizer, should be called before compilation.
        """
        ...

    @classmethod
    def from_argparse(
        cls,
        dataset: Optional[Dataset],
        args: Namespace,
    ) -> "Optimizer":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Optional[Dataset]
            The dataset object that is optionally used for optimization.
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        Optimizer
            Object of class Optimizer.
        """
        return super().from_argparse(args, dataset=dataset)

    @classmethod
    def from_json(
        cls,
        json_dict: Dict,
        dataset: Optional[Dataset] = None,
        model_wrapper: Optional[ModelWrapper] = None,
    ) -> "Optimizer":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.
        dataset : Optional[Dataset]
            The dataset object that is optionally used for optimization.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).

        Returns
        -------
        Optimizer
            Object of class Optimizer.
        """
        return super().from_json(
            json_dict, dataset=dataset, model_wrapper=model_wrapper
        )

    def set_compiled_model_path(self, compiled_model_path: Path):
        """
        Sets path for compiled model.

        compiled_model_path : PathOrURI
            Path to be set.
        """
        self.compiled_model_path = compiled_model_path

    def set_input_type(self, inputtype: str):
        """
        Sets input type of the model for the compiler.

        inputtype : str
            Path to be set.
        """
        assert inputtype in list(self.inputtypes.keys()) + ["any"], (
            f"Unsupported input type {inputtype}, only "
            f"{', '.join(self.inputtypes.keys())} are supported"
        )
        self.inputtype = inputtype

    @abstractmethod
    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        """
        Compiles the given model to a target format.

        The function compiles the model and saves it to the output file.

        The model can be compiled to a binary, a different framework or a
        different programming language.

        If `io_spec` is passed, then the function uses it during the
        compilation, otherwise `load_io_specification` is used to fetch the
        specification saved in `input_model_path` + `.json`.

        The compiled model is saved to compiled_model_path and
        the specification is saved to compiled_model_path + .json

        Parameters
        ----------
        input_model_path : PathOrURI
            Path to the input model.
        io_spec : Optional[Dict[str, List[Dict]]]
            Dictionary that has `input` and `output` keys that contain list
            of dictionaries mapping (property name) -> (property value)
            for the layers.
        """
        ...

    @abstractmethod
    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.

        Returns
        -------
        Tuple[str, str]
            Framework name and version.
        """
        ...

    @classmethod
    def get_input_formats(cls) -> List[str]:
        """
        Returns list of names of possible input formats.

        Returns
        -------
        List[str]
            Names of possible input formats.
        """
        return list(cls.inputtypes.keys())

    @classmethod
    def get_output_formats(cls) -> List[str]:
        """
        Returns list of names of possible output formats.

        Returns
        -------
        List[str]
            List of possible output formats.
        """
        return cls.outputtypes

    def consult_model_type(
        self,
        previous_block: Union[
            "ModelWrapper",
            "Optimizer",
            Type["ModelWrapper"],
            Type["Optimizer"],
        ],
        force_onnx: bool = False,
    ) -> str:
        """
        Finds output format of the previous block in the chain
        matching with an input format of the current block.

        Parameters
        ----------
        previous_block : Union["ModelWrapper", "Optimizer", Type["ModelWrapper"], Type["Optimizer"]]
            Previous block in the optimization chain.
        force_onnx : bool
            Forces ONNX format.

        Returns
        -------
        str
            Matching format.

        Raises
        ------
        ValueError
            Raised if there is no matching format.
        """  # noqa: E501
        possible_outputs = previous_block.get_output_formats()

        if force_onnx:
            KLogger.warning("Forcing ONNX conversion")
            if (
                "onnx" in self.get_input_formats()
                and "onnx" in possible_outputs
            ):
                return "onnx"
            else:
                raise ValueError(
                    '"onnx" format is not supported by at least one block\n'
                    f"Input block supported formats: {', '.join(possible_outputs)}\n"  # noqa: E501
                    f"Output block supported formats: {', '.join(self.get_input_formats())}"  # noqa: E501
                )

        for input in self.get_input_formats():
            if input in possible_outputs:
                return input

        raise ValueError(
            f"No matching formats between two object: {self} and "
            f"{previous_block}\n"
            f"Input block supported formats: {', '.join(possible_outputs)}\n"
            f"Output block supported formats: {', '.join(self.get_input_formats())}"  # noqa: E501
        )

    @staticmethod
    def get_spec_path(model_path: PathOrURI) -> PathOrURI:
        """
        Returns input/output specification path for the model
        saved in `model_path`. It concatenates `model_path` and `.json`.

        Parameters
        ----------
        model_path : PathOrURI
            Path where the model is saved.

        Returns
        -------
        PathOrURI
            Path to the input/output specification of a given model.
        """
        spec_path = model_path.with_suffix(model_path.suffix + ".json")

        return spec_path

    def save_io_specification(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        """
        Internal function that saves input/output model specification
        which is used during both inference and compilation. If `io_spec`
        is None, the function uses specification of an input model
        stored in `input_model_path` + `.json`. If there is no specification
        stored in this path the function does not do anything.

        The input/output specification is a list of dictionaries mapping
        properties names to their values. Legal properties names are `dtype`,
        `prequantized_dtype`, `shape`, `name`, `scale`, `zero_point`.

        The order of the layers has to be preserved.

        Parameters
        ----------
        input_model_path : PathOrURI
            Path to the input model.
        io_spec : Optional[Dict[str, List[Dict]]]
            Specification of the input/ouput layers.
        """
        if not io_spec:
            io_spec = self.load_io_specification(input_model_path)

        if io_spec:
            with open(self.get_spec_path(self.compiled_model_path), "w") as f:
                json.dump(io_spec, f)
        else:
            KLogger.warning(f"{self} did not save io_specification")

    def load_io_specification(
        self,
        model_path: PathOrURI,
    ) -> Optional[Dict[str, List[Dict]]]:
        """
        Returns saved input and output specification of a model
        saved in `model_path` if there is one. Otherwise returns None.

        Parameters
        ----------
        model_path : PathOrURI
            Path to the model which specification the function should read.

        Returns
        -------
        Optional[Dict[str, List[Dict]]]
            Specification of a model saved
            in `model_path` if there is one. None otherwise.
        """
        spec_path = self.get_spec_path(model_path)
        if spec_path.exists():
            with open(spec_path, "r") as f:
                spec = json.load(f)
            return spec

        KLogger.warning(
            f"{self} did not find io_specification in path: {spec_path}"
        )
        return None

    def get_input_type(
        self,
        model_path: PathOrURI,
    ) -> str:
        """
        Return input model type. If input type is set to "any", then it is
        derived from model file extension.

        Parameters
        ----------
        model_path : PathOrURI
            Path to the input model.

        Returns
        -------
        str
            Input model type.

        Raises
        ------
        Exception
            Raised if input model type cannot be determined.
        """
        input_type = (
            self.inputtype
            if self.inputtype != "any"
            else EXT_TO_FRAMEWORK.get(model_path.suffix, None)
        )

        if input_type is None:
            raise Exception("Could not determine input model type")

        return input_type

    def get_optimized_model_size(self) -> float:
        """
        Returns the optimized model size.

        By default, the size of file with optimized
        model is returned.

        Returns
        -------
        float
            The size of the optimized model in KB.
        """
        return _get_model_size(
            self.compiled_model_path,
            OptimizedModelSizeError(
                "Compiled model path does not exist:"
                f" {self.compiled_model_path}"
            ),
        )

    def read_platform(self, platform: Platform):
        """
        Reads Platform data to configure optimization/compilation.

        Platform-based entities come with lots of information on hardware
        architecture that can be used by the Optimizer class.

        By default no data is read.

        It is important to take into account that different
        Platform-based classes come with a different sets of attributes.

        Parameters
        ----------
        platform: Platform
            object with platform details
        """
        pass
