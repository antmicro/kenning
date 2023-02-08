# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for dataset loading, creation and configuration.
"""

from typing import Tuple, List, Any, Dict, Optional, Generator
import random
import argparse
from pathlib import Path
from tqdm import tqdm

from .measurements import Measurements
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument, get_parsed_json_dict  # noqa: E501


class Dataset(object):
    """
    Wraps the datasets for training, evaluation and optimization.

    This class provides an API for datasets used by models, compilers (i.e. for
    calibration) and benchmarking scripts.

    Each Dataset object should implement methods for:

    * processing inputs and outputs from dataset files,
    * downloading the dataset,
    * evaluating the model based on dataset's inputs and outputs.

    The Dataset object provides routines for iterating over dataset samples
    with configured batch size, splitting the dataset into subsets and
    extracting loaded data from dataset files for training purposes.

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

    arguments_structure = {
        'root': {
            'argparse_name': '--dataset-root',
            'description': 'Path to the dataset directory',
            'type': Path,
            'required': True
        },
        'batch_size': {
            'argparse_name': '--inference-batch-size',
            'description': 'The batch size for providing the input data',
            'type': int,
            'default': 1
        },
        'download_dataset': {
            'description': 'Downloads the dataset before taking any action',
            'type': bool,
            'default': False
        },
        'external_calibration_dataset': {
            'argparse_name': '--external-calibration-dataset',
            'description': 'Path to the directory with the external calibration dataset',  # noqa: E501
            'type': Path,
            'nullable': True,
            'default': None
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            external_calibration_dataset: Optional[Path] = None):
        """
        Initializes dataset object.

        Prepares all structures and data required for providing data samples.

        If download_dataset is True, the dataset is downloaded first using
        download_dataset_fun method.

        Parameters
        ----------
        root : Path
            The path to the dataset data
        batch_size : int
            The batch size
        download_dataset : bool
            True if dataset should be downloaded first
        external_calibration_dataset : Optional[Path]
            Path to the external calibration dataset that can be used for
            quantizing the model. If it is not provided, the calibration
            dataset is generated from the actual dataset.
        """
        assert batch_size > 0
        self.root = Path(root)
        self._dataindex = 0
        self.dataX = []
        self.dataY = []
        self.batch_size = batch_size
        self.download_dataset = download_dataset
        self.external_calibration_dataset = None if external_calibration_dataset is None else Path(external_calibration_dataset)  # noqa: E501
        if download_dataset:
            self.download_dataset_fun()
        self.prepare()

        self.actions = {
            'stream': self.action_stream
        }

    @classmethod
    def _form_argparse(cls):
        """
        Wrapper for creating argparse structure for the Dataset class.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(
            add_help=False,
            conflict_handler='resolve'
        )
        group = parser.add_argument_group(title='Dataset arguments')
        add_argparse_argument(
            group,
            Dataset.arguments_structure
        )
        return parser, group

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Dataset object.

        This method is used to create a list of arguments for the object so
        it is possible to configure the object from the level of command
        line.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser, group = cls._form_argparse()
        if cls.arguments_structure != Dataset.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        This method takes the arguments created in form_argparse and uses them
        to create the object.

        Parameters
        ----------
        args : Dict
            arguments from ArgumentParser object

        Returns
        -------
        Dataset :
            object of class Dataset
        """
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset
        )

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure for the Dataset class.

        Returns
        -------
        Dict :
            schema for the class
        """
        parameterschema = {
            "type": "object",
            "additionalProperties": False
        }

        add_parameterschema_argument(
            parameterschema,
            Dataset.arguments_structure,
        )

        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the Dataset class.

        Returns
        -------
        Dict :
            schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != Dataset.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

    @classmethod
    def from_json(cls, json_dict: Dict):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor

        Returns
        -------
        Dataset :
            object of class Dataset
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            **parsed_json_dict
        )

    def __iter__(self) -> 'Dataset':
        """
        Provides iterator over data samples' tuples.

        Each data sample is a tuple (X, y), where X are the model inputs,
        and y are the model outputs.

        Returns
        -------
        Dataset :
            this object
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

    def __len__(self) -> int:
        """
        Returns the number of data samples.

        Returns
        -------
        int :
            Number of input samples
        """
        return len(self.dataX)

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
        List :
            preprocessed input samples
        """
        return samples

    def prepare_output_samples(self, samples: List) -> List:
        """
        Preprocesses output samples.

        By default the method returns data as is.
        It can be used i.e. to create the one-hot output vector with class
        association based on a given sample.

        Parameters
        ----------
        samples : List
            List of output samples to be processed

        Returns
        -------
        List :
            preprocessed output samples
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
        assert batch_size > 0
        self.batch_size = batch_size

    def get_data(self) -> Tuple[List, List]:
        """
        Returns the tuple of all inputs and outputs for the dataset.

        .. warning::
            It loads all entries with prepare_input_samples and
            prepare_output_samples to the memory - for large datasets it may
            result in filling the whole memory.

        Returns
        -------
        Tuple[List, List] :
            the list of data samples
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
        Tuple[List, List] :
            the list of data samples representations
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
        from sklearn.model_selection import train_test_split
        dataXtrain, dataXtest, dataYtrain, dataYtest = train_test_split(
            self.dataX,
            self.dataY,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=self.dataY
        )
        return (dataXtrain, dataXtest, dataYtrain, dataYtest)

    def calibration_dataset_generator(
            self,
            percentage: float = 0.25,
            seed: int = 12345) -> Generator[List[Any], None, None]:
        """
        Creates generator for the calibration data.

        Parameters
        ----------
        percentage : float
            The fraction of data to use for calibration
        seed : int
            The seed for random state
        """
        if self.external_calibration_dataset is None:
            _, X, _, _ = self.train_test_split_representations(
                percentage,
                seed
            )
        else:
            X = self.prepare_external_calibration_dataset(percentage, seed)

        for x in tqdm(X):
            yield self.prepare_input_samples([x])

    def prepare_external_calibration_dataset(
            self,
            percentage: float = 0.25,
            seed: int = 12345) -> List[Path]:
        """
        Prepares the data for external calibration dataset.

        This method is supposed to scan external_calibration_dataset directory
        and prepares the list of entries that are suitable for the
        prepare_input_samples method.

        This method is called by the calibration_dataset_genereator method to
        get the data for calibration when external_calibration_dataset is
        provided.

        By default, this method scans for all files in the directory and
        returns the list of those files.

        Returns
        -------
        List[Any] :
            List of objects that are usable by the prepare_input_samples method
        """
        data = [
            x for x in self.external_calibration_dataset.rglob('*') if x.is_file()  # noqa: E501
        ]
        random.Random(seed).shuffle(data)
        return data[:int(percentage * len(data) + 0.5)]

    def download_dataset_fun(self):
        """
        Downloads the dataset to the root directory defined in the constructor.
        """
        raise NotImplementedError

    def prepare(self):
        """
        Prepares dataX and dataY attributes based on the dataset contents.

        This can i.e. store file paths in dataX and classes in dataY that
        will be later loaded using prepare_input_samples and
        prepare_output_samples.
        """
        raise NotImplementedError

    def evaluate(self, predictions: list, truth: list) -> 'Measurements':
        """
        Evaluates the model based on the predictions.

        The method should compute various quality metrics fitting for the
        problem the model solves - i.e. for classification it may be
        accuracy, precision, G-mean, for detection it may be IoU and mAP.

        The evaluation results should be returned in a form of Measurements
        object.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model
        truth: List
            The ground truth for given batch

        Returns
        -------
        Measurements :
            The dictionary containing the evaluation results
        """
        raise NotImplementedError

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        """
        Returns mean and std values for input tensors.

        The mean and std values returned here should be computed using
        ``compute_input_mean_std`` method.

        Returns
        -------
        Tuple[Any, Any] :
            the standardization values for a given train dataset.
            Tuple of two variables describing mean and std values
        """
        raise NotImplementedError

    def get_class_names(self) -> List[str]:
        """
        Returns list of class names in order of their IDs.

        Returns
        -------
        List[str] :
            List of class names
        """
        raise NotImplementedError

    def action_stream(self, _: Dict[str, Any]) -> Dict[str, Any]:
        next = self.__next__()
        return {
            'data_x': next[0],
            'data_y': next[1]
        }


class CannotDownloadDatasetError(Exception):
    pass
