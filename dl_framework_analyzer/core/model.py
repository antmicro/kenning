"""
Provides a wrapper for deep learning models.
"""

from typing import List, Any, Tuple
import argparse
from pathlib import Path
from collections import defaultdict

from dl_framework_analyzer.core.dataset import Dataset
from dl_framework_analyzer.core.measurements import Measurements
from dl_framework_analyzer.core.measurements import MeasurementsCollector
from dl_framework_analyzer.core.measurements import timemeasurements
from dl_framework_analyzer.core.measurements import systemstatsmeasurements


class ModelWrapper(object):
    """
    Wraps the given model.
    """

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
        """
        self.modelpath = modelpath
        self.dataset = dataset
        self.data = defaultdict(list)
        self.from_file = from_file
        self.prepare_model()

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
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Inference model arguments')
        group.add_argument(
            '--model-path',
            help='Path to the model',
            required=True,
            type=Path
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
        Dataset : object of class Dataset
        """
        return cls(args.model_path, dataset, from_file)

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

    def load_from_onnx(self, modelpath: Path):
        """
        Loads the model in the ONNX format.

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
        Any: the preprocessed inputs that are ready to be fed to the model
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
            the postprocessed outputs from the model that need to be in
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
        Any: the results of the inference.
        """
        raise NotImplementedError

    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.
        """
        raise NotImplementedError

    @timemeasurements('inference_step')
    def _run_inference(self, X):
        return self.run_inference(X)

    @systemstatsmeasurements('inferencesysstats')
    @timemeasurements('inference')
    def test_inference(self) -> List:
        """
        Runs the inference with a given dataset.

        Returns
        -------
        List : The inference results
        """

        measurements = Measurements()

        for X, y in iter(self.dataset):
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
