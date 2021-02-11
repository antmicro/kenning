"""
Provides an API for dataset loading, creation and configuration.
"""

from typing import Tuple, List
from .measurements import Measurements


class Dataset(object):
    """
    Prepares data for training and testing of deep learning models.

    Attributes
    ----------
    dataX : List[Any]
        List of input data (or data representing input data, i.e. file paths)
    dataY : List[Any]
        List of output data (or data representing output data)
    batch_size : int
        The batch size for the dataset
    _dataindex : int
        ID of the next data to be delivered for inference
    """

    def __init__(self, root: str, batch_size: int = 1):
        """
        Prepares all structures and data required for providing data samples.

        Parameters
        ----------
        root : str
            The path to the dataset data
        batch_size : int
            The batch size
        """
        self.root = root
        self._dataindex = 0
        self.dataX = []
        self.dataY = []
        self.batch_size = 1
        self.prepare()

    def __iter__(self) -> 'Dataset':
        """
        Provides iterator over data samples' tuples.

        Each data sample is a tuple (X, y), where X are the model inputs,
        and y are the model outputs.

        Returns
        -------
        Dataset : this object
        """
        self._dataindex = 0
        return self

    def __next__(self) -> Tuple[List, List]:
        """
        Returns next data sample in a form of a (X, y) tuple.

        X contains the list of inputs for the model.
        y contains the list of outputs for the model.

        Returns
        -------
        Tuple[List, List] :
            Tuple containing list of input data for inference and output data
            for comparison.
        """
        if self._dataindex < len(self.dataX):
            prev = self._dataindex
            self._dataindex += self.batch_size
            return (
                self.prepare_input_samples(self.dataX[prev:self._dataindex]),
                self.prepare_output_samples(self.dataY[prev:self._dataindex])
            )
        raise StopIteration

    def prepare_input_samples(self, samples: List) -> List:
        """
        Preprocesses input samples, i.e. load images from files, converts them.

        By default the method returns data as is - without any conversions.
        Since the input samples can be large, it does not make sense to load
        all data to the memory - this method handles loading data for a given
        data batch.

        Parameters
        ----------
        samples : List
            List of input samples to be processed

        Returns
        -------
        List : preprocessed input samples
        """
        return samples

    def prepare_output_samples(self, samples: List) -> List:
        """
        Preprocesses output samples.

        Parameters
        ----------
        samples : List
            List of output samples to be processed

        Returns
        -------
        List : preprocessed output samples
        """
        return samples

    def set_batch_size(self, batch_size):
        """
        Sets the batch size of the data in the iterator batches.

        Parameters
        ----------
        batch_size : int
            Number of input samples per batch
        """
        self.batch_size = batch_size

    def get_data(self) -> Tuple[List, List]:
        """
        Returns the tuple of all inputs and outputs for the dataset.

        Returns
        -------
        Tuple[List, List] : the list of data samples
        """
        return (
            self.prepare_input_samples(self.dataX),
            self.prepare_output_samples(self.dataY)
        )

    def download_dataset(self):
        """
        Downloads the dataset to the root directory.
        """
        raise NotImplementedError

    def prepare(self):
        """
        Prepares dataX and dataY attributes based on the dataset directory.
        """
        raise NotImplementedError

    def evaluate(self, predictions: List, truth: List) -> 'Measurements':
        """
        Evaluates the model based on the predictions.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model
        truth: List
            The ground truth for given batch

        Returns
        -------
        Measurements : The dictionary containing the evaluation results
        """
        raise NotImplementedError
