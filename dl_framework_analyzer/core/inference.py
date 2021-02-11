from typing import List, Any
from .dataset import Dataset
from .measurements import Measurements, statistics
from collections import defaultdict

"""
Provides an API for inference tests of a model.
"""


class InferenceTester(object):
    """
    Runs inference on a given model.
    """

    def __init__(self, dataset: Dataset):
        """
        Creates the inference tester.

        Parameters
        ----------
        dataset : Dataset
            The dataset to verify the inference
        """
        self.dataset = dataset
        self.data = defaultdict(list)
        self.prepare_model()

    def prepare_model(self):
        """
        Downloads/loads the model for the inference.
        """
        return NotImplementedError

    def preprocess_input(self, X: List) -> Any:
        """
        Preprocesses the inputs for a given model.

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

    def postprocess_outputs(self, y: Any) -> List:
        """
        Preprocesses the inputs for a given model.

        By default no action is taken, and the inputs are passed unmodified.

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

    @statistics('inferencetime')
    def test_inference(self) -> List:
        """
        Runs the inference with a given dataset.

        Returns
        -------
        List : The inference results
        """

        measurements = Measurements()

        for X, y in iter(self.dataset):
            prepX = self.preprocess_input(X)
            preds = self.run_inference(prepX)
            measurements += self.dataset.evaluate(preds, y)

        return measurements
