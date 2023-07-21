# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for deep learning models.
"""

import json
from typing import List, Any, Tuple, Dict, Type, Optional
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from urllib.request import HTTPError

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
from kenning.core.measurements import timemeasurements
from kenning.core.measurements import systemstatsmeasurements
from kenning.interfaces.io_interface import IOInterface
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.args_manager import get_parsed_json_dict
from kenning.utils.args_manager import get_parsed_args_dict
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ModelWrapper(IOInterface, ArgumentsHandler, ABC):
    """
    Wraps the given model.
    """
    pretrained_model_uri: Optional[str] = None
    default_dataset: Optional[Type[Dataset]] = None
    arguments_structure = {
        'model_path': {
            'argparse_name': '--model-path',
            'description': 'Path to the model',
            'type': ResourceURI,
            'required': True
        }
    }

    def __init__(
            self,
            model_path: PathOrURI,
            dataset: Optional[Dataset],
            from_file: bool = True):
        """
        Creates the model wrapper.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Optional[Dataset]
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        """
        self.model_path = model_path
        self.dataset = dataset
        self.from_file = from_file
        self.model_prepared = False

    def get_path(self) -> PathOrURI:
        """
        Returns path to the model in a form of a Path or ResourceURI object.

        Returns
        -------
        PathOrURI :
            Path or URI to the model.
        """
        return self.model_path

    @classmethod
    def from_argparse(
            cls,
            dataset: Dataset,
            args: Namespace,
            from_file: bool = True):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to feed to the model.
        args : Namespace
            Arguments from ArgumentParser object.
        from_file : bool
            Determines if the model should be loaded from model_path.

        Returns
        -------
        ModelWrapper :
            Object of class ModelWrapper.
        """

        parsed_args_dict = get_parsed_args_dict(cls, args)

        return cls(
            dataset=dataset,
            **parsed_args_dict,
            from_file=from_file
        )

    @classmethod
    def from_json(
            cls,
            dataset: Optional[Dataset],
            json_dict: Dict,
            from_file: bool = True):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        dataset : Optional[Dataset]
            The dataset object to feed to the model.
        json_dict : Dict
            Arguments for the constructor.
        from_file : bool
            Determines if the model should be loaded from model_path.

        Returns
        -------
        ModelWrapper :
            Object of class ModelWrapper.
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            dataset=dataset,
            **parsed_json_dict,
            from_file=from_file
        )

    @abstractmethod
    def prepare_model(self):
        """
        Downloads the model (if required) and loads it to the device.

        Should be used whenever the model is actually required.

        The prepare_model method should check model_prepared field
        to determine if the model is not already loaded.

        It should also set model_prepared field to True
        once the model is prepared.
        """
        raise NotImplementedError

    def load_model(self, model_path: PathOrURI):
        """
        Loads the model from file.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        raise NotImplementedError

    def save_model(self, model_path: PathOrURI):
        """
        Saves the model to file.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        raise NotImplementedError

    def save_to_onnx(self, model_path: PathOrURI):
        """
        Saves the model in the ONNX format.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        raise NotImplementedError

    def preprocess_input(self, X: List) -> Any:
        """
        Preprocesses the inputs for a given model before inference.

        By default no action is taken, and the inputs are passed unmodified.

        Parameters
        ----------
        X : List
            The input data from the Dataset object.

        Returns
        -------
        Any :
            The preprocessed inputs that are ready to be fed to the model.
        """
        return X

    def _preprocess_input(self, X):
        return self.preprocess_input(X)

    def postprocess_outputs(self, y: List[Any]) -> Any:
        """
        Processes the outputs for a given model.

        By default no action is taken, and the outputs are passed unmodified.

        Parameters
        ----------
        y : List[Any]
            The list of output data from the model.

        Returns
        -------
        Any :
            The post processed outputs from the model that need to be in
            format requested by the Dataset object.
        """
        return y

    def _postprocess_outputs(self, y):
        return self.postprocess_outputs(y)

    def run_inference(self, X: List) -> Any:
        """
        Runs inference for a given preprocessed input.

        Parameters
        ----------
        X : List
            The preprocessed inputs for the model.

        Returns
        -------
        Any :
            The results of the inference.
        """
        raise NotImplementedError

    @abstractmethod
    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.

        Returns
        -------
        Tuple[str, str] :
            Framework name and version.
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_formats(self) -> List[str]:
        """
        Returns list of names of possible output formats.

        Returns
        -------
        List[str] :
            List of possible output format names.
        """
        raise NotImplementedError

    @timemeasurements('target_inference_step')
    def _run_inference(self, X):
        return self.run_inference(X)

    @systemstatsmeasurements('session_utilization')
    @timemeasurements('inference')
    def test_inference(self) -> 'Measurements':
        """
        Runs the inference with a given dataset.

        Returns
        -------
        Measurements :
            The inference results.
        """
        from tqdm import tqdm

        measurements = Measurements()

        for X, y in tqdm(iter(self.dataset)):
            prepX = self._preprocess_input(X)
            preds = self._run_inference(prepX)
            posty = self._postprocess_outputs(preds)
            measurements += self.dataset.evaluate(posty, y)

        MeasurementsCollector.measurements += measurements

        return measurements

    def train_model(
            self,
            batch_size: int,
            learning_rate: float,
            epochs: int,
            logdir: Path):
        """
        Trains the model with a given dataset.

        This method should implement training routine for a given dataset and
        save a working model to a given path in a form of a single file.

        The training should be performed with given batch size, learning rate,
        and number of epochs.

        The model needs to be saved explicitly.

        Parameters
        ----------
        batch_size : int
            The batch size for the training.
        learning_rate : float
            The learning rate for the training.
        epochs : int
            The number of epochs for training.
        logdir : Path
            Path to the logging directory.
        """
        raise NotImplementedError

    @abstractmethod
    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        """
        Returns a new instance of dictionary with `input` and `output`
        keys that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`.

        It is later used in optimization and compilation steps.

        It is used by `get_io_specification` function to get the
        specification and save it for later use.

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output
            layers specification.
        """

        raise NotImplementedError

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        """
        Returns a saved dictionary with `input` and `output` keys
        that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`.

        It is later used in optimization and compilation steps.

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output
            layers specification.
        """
        if not hasattr(self, 'io_specification'):
            self.io_specification = self.get_io_specification_from_model()
        return self.io_specification

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        model_path = ResourceURI(parsed_json_dict['model_path'])
        io_spec = model_path.with_suffix(model_path.suffix + '.json')
        try:
            with open(io_spec, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, HTTPError):
            return cls.derive_io_spec_from_json_params(parsed_json_dict)

    @classmethod
    def derive_io_spec_from_json_params(
        cls,
        json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Creates IO specification by deriving parameters from parsed JSON
        dictionary. The resulting IO specification may differ from the results
        of `get_io_specification`, information that couldn't be retrieved from
        JSON parameters are absent from final IO spec or are filled with
        general value (example: '-1' for unknown dimension shape).

        Parameters
        ----------
        json_dict : Dict
            JSON dictionary formed by parsing the input JSON with
            ModelWrapper's parameterschema.

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output
            layers specification.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_input_to_bytes(self, inputdata: Any) -> bytes:
        """
        Converts the input returned by the ``preprocess_input`` method
        to bytes.

        Parameters
        ----------
        inputdata : Any
            The preprocessed inputs.

        Returns
        -------
        bytes :
            Input data as byte stream.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        """
        Converts bytes array to the model output format.

        The converted output should be compatible with ``postprocess_outputs``
        method.

        Parameters
        ----------
        outputdata : bytes
            Output data in raw bytes.

        Returns
        -------
        List[Any] :
            List of output data from a model. The converted data should be
            compatible with the ``postprocess_outputs`` method.
        """
        raise NotImplementedError
