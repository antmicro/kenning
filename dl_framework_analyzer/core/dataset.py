from typing import Tuple, List, Dict

"""
Provides an API for dataset loading, creation and configuration.
"""

class Dataset(object):
    """
    Prepares data for training and testing of deep learning models.
    """

    def __init__(self, root, batch_size=1):
        """
        Prepares all structures and data required for providing data samples.
        """
        self.root = root
        self._dataindex = 0
        self.dataX = []
        self.dataY = []
        self.batch_size = 1

    def __iter__(self) -> Dataset:
        """
        Provides iterator over data samples' tuples.

        Each data sample is a tuple (X, y), where X are the model inputs,
        and y are the model outputs.
        """
        self._dataindex = 0
        return self

    def __next__(self) -> Tuple[List, List]:
        """
        Returns next data sample in a form of a (X, y) tuple.

        X contains the list of inputs for the model.
        y contains the list of outputs for the model.
        """
        if self._dataindex < len(self.data):
            prev = self._dataindex
            self._dataindex += self.batch_size
            return (
                self.dataX[prev:self._dataindex],
                self.dataY[prev:self._dataindex]
            )
        raise StopIteration

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
        raise NotImplementedError

    def evaluation(self, predictions: List) -> Dict[str, Any]:
        """
        Evaluates the model based on the predictions.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model

        Returns
        -------
        Dict[str, Any] : The dictionary containing the evaluation results
        """
        raise NotImplementedError
