"""
Provides an API for dataset loading, creation and configuration.
"""

from typing import Tuple, List
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

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

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False):
        """
        Prepares all structures and data required for providing data samples.

        Parameters
        ----------
        root : Path
            The path to the dataset data
        batch_size : int
            The batch size
        download_dataset : bool
            True if dataset should be downloaded first
        """
        self.root = Path(root)
        self._dataindex = 0
        self.dataX = []
        self.dataY = []
        self.batch_size = batch_size
        if download_dataset:
            self.download_dataset()
        self.prepare()

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Dataset object.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Dataset arguments')
        group.add_argument(
            '--dataset-root',
            help='Path to the dataset directory',
            required=True,
            type=Path
        )
        group.add_argument(
            '--download-dataset',
            help='Downloads the dataset before taking any action',
            action='store_true'
        )
        group.add_argument(
            '--inference-batch-size',
            help='The batch size for providing the input data',
            type=int,
            default=1
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : arguments from ArgumentParser object

        Returns
        -------
        Dataset : object of class Dataset
        """
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset
        )

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

        *WARNING*: It loads all entries with prepare_input_samples and
        prepare_output_samples to the memory - for large datasets it may result
        in filling the whole space.

        Returns
        -------
        Tuple[List, List] : the list of data samples
        """
        return (
            self.prepare_input_samples(self.dataX),
            self.prepare_output_samples(self.dataY)
        )

    def get_data_unloaded(self) -> Tuple[List, List]:
        """
        Returns the input and output representations before loading.

        The representations can be opened using prepare_input_samples and
        prepare_output_samples.

        Returns
        -------
        Tuple[List, List] : the list of data samples representations
        """
        return (self.dataX, self.dataY)

    def train_test_split_representations(
            self,
            test_fraction: float = 0.25,
            seed: int = 12345):
        """
        Splits the data representations into train dataset and test dataset.

        Parameters
        ----------
        test_fraction : float
            The fraction of data to leave for model validation
        seed : int
            The seed for random state
        """
        dataXtrain, dataXtest, dataYtrain, dataYtest = train_test_split(
            self.dataX,
            self.dataY,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=self.dataY
        )
        return (dataXtrain, dataXtest, dataYtrain, dataYtest)

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
