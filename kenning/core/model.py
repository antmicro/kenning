"""
Provides a wrapper for deep learning models.
"""

from typing import List, Any, Tuple, Dict
import argparse
from pathlib import Path
from collections import defaultdict
import json

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
from kenning.core.measurements import timemeasurements
from kenning.core.measurements import systemstatsmeasurements
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument, get_parsed_json_dict  # noqa: E501


class ModelWrapper(object):
    """
    Wraps the given model.
    """

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--model-path',
            'description': 'Path to the model',
            'type': Path,
            'required': True
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = True):
        """
        Creates the model wrapper.

        Parameters
        ----------
        modelpath : Path
            The path to the model
        dataset : Dataset
            The dataset to verify the inference
        from_file : bool
            True if the model should be loaded from file
        """
        self.modelpath = Path(modelpath)
        self.dataset = dataset
        self.data = defaultdict(list)
        self.from_file = from_file
        self.prepare_model()

        self.actions = {
            'infer': self.action_infer,
            'preprocess': self.action_preprocess,
            'train': self.action_train,
            'postprocess': self.action_postprocess
        }

    def get_path(self) -> Path:
        """
        Returns path to the model in a form of a Path object.

        Returns
        -------
        modelpath : Path
            The path to the model
        """
        return self.modelpath

    @classmethod
    def _form_argparse(cls):
        """
        Wrapper for creating argparse structure for the ModelWrapper class.

        Returns
        -------
        ArgumentParser :
            The argument parser object that can act as parent for program's
            argument parser
        """
        parser = argparse.ArgumentParser(
            add_help=False,
            conflict_handler='resolve'
        )
        group = parser.add_argument_group(title='Inference model arguments')
        add_argparse_argument(
            group,
            ModelWrapper.arguments_structure
        )
        return parser, group

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the ModelWrapper object.

        Returns
        -------
        ArgumentParser :
            the argument parser object that can act as parent for program's
            argument parser
        """
        parser, group = cls._form_argparse()
        if cls.arguments_structure != ModelWrapper.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
        return parser, group

    @classmethod
    def from_argparse(
            cls,
            dataset: Dataset,
            args,
            from_file: bool = True):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to feed to the model
        args : Dict
            Arguments from ArgumentParser object
        from_file : bool
            Determines if the model should be loaded from modelspath

        Returns
        -------
        ModelWrapper : object of class ModelWrapper
        """
        return cls(args.model_path, dataset, from_file)

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure
        for the ModelWrapper class.

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
            ModelWrapper.arguments_structure,
        )

        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the ModelWrapper class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != ModelWrapper.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

    @classmethod
    def from_json(
            cls,
            dataset: Dataset,
            json_dict: Dict,
            from_file: bool = True):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to feed to the model
        json_dict : Dict
            Arguments for the constructor
        from_file : bool
            Determines if the model should be loaded from modelspath

        Returns
        -------
        ModelWrapper : object of class ModelWrapper
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            dataset=dataset,
            **parsed_json_dict,
            from_file=from_file
        )

    def prepare_model(self):
        """
        Downloads the model (if required) and loads it to the device.
        """
        raise NotImplementedError

    def load_model(self, modelpath: Path):
        """
        Loads the model from file.

        Parameters
        ----------
        modelpath : Path
            Path to the model file
        """
        raise NotImplementedError

    def save_model(self, modelpath: Path):
        """
        Saves the model to file.

        Parameters
        ----------
        modelpath : Path
            Path to the model file
        """
        raise NotImplementedError

    def save_to_onnx(self, modelpath: Path):
        """
        Saves the model in the ONNX format.

        Parameters
        ----------
        modelpath : Path
            Path to the ONNX file
        """
        raise NotImplementedError

    def preprocess_input(self, X: List) -> Any:
        """
        Preprocesses the inputs for a given model before inference.

        By default no action is taken, and the inputs are passed unmodified.

        Parameters
        ----------
        X : List
            The input data from the Dataset object

        Returns
        -------
        Any: The preprocessed inputs that are ready to be fed to the model
        """
        return X

    def _preprocess_input(self, X):
        return self.preprocess_input(X)

    def postprocess_outputs(self, y: Any) -> List:
        """
        Processes the outputs for a given model.

        By default no action is taken, and the outputs are passed unmodified.

        Parameters
        ----------
        y : Any
            The output from the model

        Returns
        -------
        List:
            The postprocessed outputs from the model that need to be in
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
            The preprocessed inputs for the model

        Returns
        -------
        Any: The results of the inference.
        """
        raise NotImplementedError

    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.
        """
        raise NotImplementedError

    def get_output_formats(self) -> List[str]:
        """
        Returns list of names of possible output formats.
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
        Measurements : The inference results
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
            The batch size for the training
        learning_rate : float
            The learning rate for the training
        epochs : int
            The number of epochs for training
        logdir : Path
            Path to the logging directory
        """
        raise NotImplementedError

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        """
        Returns a new instance of dictionary with `input` and `output`
        keys that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`

        It is later used in optimization and compilation steps.

        It is used by `get_io_specification` function to get the
        specification and save it for later use.

        Returns
        -------
        Dict[str, List[Dict]] : Dictionary that conveys input and output
            layers specification
        """

        raise NotImplementedError

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        """
        Returns a saved dictionary with `input` and `output` keys
        that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`

        It is later used in optimization and compilation steps

        Returns
        -------
        Dict[str, List[Dict]] : Dictionary that conveys input and output
            layers specification
        """
        if not hasattr(self, 'io_specification'):
            self.io_specification = self.get_io_specification_from_model()
        return self.io_specification

    def save_io_specification(self, modelpath: Path):
        """
        Saves input/output model specification to a file named
        `modelpath` + `.json`. This function uses `get_io_specification()`
        function to get the properties.

        It is later used in optimization and compilation steps.

        Parameters
        ----------
        modelpath : Path
            Path that is used to store the model input/output specification
        """
        spec_path = modelpath.parent / (modelpath.name + '.json')
        spec_path = Path(spec_path)

        with open(spec_path, 'w') as f:
            json.dump(
                self.get_io_specification(),
                f
            )

    def convert_input_to_bytes(self, inputdata: Any) -> bytes:
        """
        Converts the input returned by the preprocess_input method to bytes.

        Parameters
        ----------
        inputdata : Any
            The preprocessed inputs

        Returns
        -------
        bytes : Input data as byte stream
        """
        raise NotImplementedError

    def convert_output_from_bytes(self, outputdata: bytes) -> Any:
        """
        Converts bytes array to the model output format.

        The converted bytes are later passed to postprocess_outputs method.

        Parameters
        ----------
        outputdata : bytes
            Output data in raw bytes

        Returns
        -------
        Any : Output data to feed to postprocess_outputs
        """
        raise NotImplementedError

    def action_infer(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # get_io_specification returns dictionary with multiple possible inputs
        # Currently we do not expect to support more than single input though
        # Hence we assume every model will have its working input at index 0
        # TODO add support for multiple inputs / outputs
        default_name = self.get_io_specification()['input'][0]['name']
        return {
            'out_infer': self.run_inference(
                self.preprocess_input(input[default_name])
            )
        }

    def action_preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        default_name = self.get_io_specification()['input'][0]['name']
        return {
            'out_pre': self.preprocess_input(input[default_name])
        }

    def action_train(self, input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def action_postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        default_name = self.get_io_specification()['input'][0]['name']
        return {
            'out_post': self.postprocess_outputs(input[default_name])
        }
