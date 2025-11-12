# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for dataset loading, creation and configuration.
"""

import datetime
import hashlib
import random
import shutil
import struct
from abc import ABC, abstractmethod
from argparse import Namespace
from binascii import hexlify
from math import ceil
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from kenning.interfaces.io_interface import IOInterface
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import Resources

from .measurements import Measurements


class DatasetIterator:
    """
    Provides an iterator for dataset samples.

    This class is used by Dataset objects to provide data samples for
    inference, training and validation.
    """

    def __init__(self, dataset: "Dataset", indices: List[int]):
        """
        Initializes the iterator.

        Parameters
        ----------
        dataset : Dataset
            The dataset object that provides the data samples.
        indices : List[int]
            The list of indices of samples to be provided by the iterator.
        """
        self.dataset = dataset
        self.indices = indices

    def __iter__(self) -> "Dataset":
        """
        Returns the iterator object.

        Returns
        -------
        Dataset
            The iterator object.
            If the indices list is empty, an empty iterator is returned.
        """
        if len(self.indices) == 0:
            return iter([])

        self.dataset._dataindex = 0
        self.dataset._dataindices = self.indices
        return self.dataset

    def __len__(self) -> int:
        """
        Returns iterator length.

        Returns
        -------
        int
            The length of the iterator.
        """
        return ceil(len(self.indices) / self.dataset.batch_size)


class Dataset(ArgumentsHandler, ABC):
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
        List of input data (or data representing input data, i.e. file paths).
    dataY : List[Any]
        List of output data (or data representing output data).
    batch_size : int
        The batch size for the dataset.
    _dataindex : int
        ID of the next data to be delivered for inference.
    dataXtrain: List[Any]
        dataX subset representing a training set. Available after executing
        train_test_split_representations, otherwise empty.
    dataYtrain: List[Any]
        dataY subset representing a training set. Available after executing
        train_test_split_representations, otherwise empty.
    dataXtest: List[Any]
        dataX subset representing a testing set. Available after executing
        train_test_split_representations, otherwise empty.
    dataYtest: List[Any]
        dataY subset representing a testing set. Available after executing
        train_test_split_representations, otherwise empty.
    dataXval: List[Any]
        Optional dataX subset representing a validation set. Available after
        executing train_test_split_representations, otherwise empty.
    dataYval: List[Any]
        Optional dataY subset representing a validation set. Available after
        executing train_test_split_representations, otherwise empty.
    """

    resources: Resources = Resources(dict())
    arguments_structure = {
        "root": {
            "argparse_name": "--dataset-root",
            "description": "Path to the dataset directory",
            "type": Path,
            "required": True,
        },
        "batch_size": {
            "argparse_name": "--inference-batch-size",
            "description": "The batch size for providing the input data",
            "type": int,
            "default": 1,
        },
        "download_dataset": {
            "argparse_name": "--download-dataset",
            "description": "Downloads the dataset before taking any action. "
            "If the dataset files are already downloaded and "
            "the checksum is correct then they are not "
            "downloaded again. Is enabled by default.",
            "type": bool,
            "default": True,
        },
        "force_download_dataset": {
            "description": "Forces dataset download",
            "type": bool,
            "default": False,
        },
        "external_calibration_dataset": {
            "argparse_name": "--external-calibration-dataset",
            "description": "Path to the directory with the external "
            "calibration dataset",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "split_fraction_test": {
            "argparse_name": "--split-fraction-test",
            "description": "Default fraction of data to leave for model "
            "testing",
            "type": float,
            "default": 0.2,
        },
        "split_fraction_val": {
            "argparse_name": "--split-fraction-val",
            "description": "Default fraction of data to leave for model "
            "valdiation",
            "type": float,
            "nullable": True,
            "default": None,
        },
        "split_seed": {
            "argparse_name": "--split-seed",
            "description": "Default seed used for dataset split",
            "type": int,
            "default": 1234,
        },
        "dataset_percentage": {
            "argparse_name": "--reduce-dataset",
            "type": float,
            "default": 1,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = True,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        dataset_percentage: float = 1,
        shuffle_data: bool = True,
    ):
        """
        Initializes dataset object.

        Prepares all structures and data required for providing data samples.

        If download_dataset is True, the dataset is downloaded first using
        download_dataset_fun method.

        Parameters
        ----------
        root : Path
            The path to the dataset data.
        batch_size : int
            The batch size.
        download_dataset : bool
            Downloads the dataset before taking any action. If the dataset
            files are already downloaded then they are not downloaded again.
        force_download_dataset : bool
            Forces dataset download.
        external_calibration_dataset : Optional[Path]
            Path to the external calibration dataset that can be used for
            quantizing the model. If it is not provided, the calibration
            dataset is generated from the actual dataset.
        split_fraction_test : float
            Default fraction of data to leave for model testing.
        split_fraction_val : Optional[float]
            Default fraction of data to leave for model validation.
        split_seed : int
            Default seed used for dataset split.
        dataset_percentage : float
            Use given percentage of the dataset.
        shuffle_data : bool
            Shuffle dataset data when loading.
        """
        assert batch_size > 0
        assert (
            0 < dataset_percentage <= 1
        ), "Percentage has to be between 0 and 1"
        self.root = Path(root)
        self._dataindex = 0
        self._dataindices = []
        self.dataX = []
        self.dataY = []
        self.dataXtrain = []
        self.dataXtest = []
        self.dataXval = []
        self.dataYtrain = []
        self.dataYtest = []
        self.dataYval = []
        self.batch_size = batch_size
        self.download_dataset = download_dataset
        self.force_download_dataset = force_download_dataset
        self.external_calibration_dataset = (
            None
            if external_calibration_dataset is None
            else Path(external_calibration_dataset)
        )
        self.shuffle_data = shuffle_data
        self.split_fraction_test = split_fraction_test
        self.split_fraction_val = split_fraction_val
        self.split_seed = split_seed
        self.dataset_percentage = dataset_percentage
        if force_download_dataset or (
            download_dataset and not self.verify_dataset_checksum()
        ):
            shutil.rmtree(self.root, ignore_errors=True)
            self.download_dataset_fun()
            self.save_dataset_checksum()

        self.prepare()
        if self.dataset_percentage < 1:
            self._reduce_dataset()

    @classmethod
    def from_argparse(cls, args: Namespace) -> "Dataset":
        """
        Constructor wrapper that takes the parameters from argparse args.

        This method takes the arguments created in ``form_argparse``
        and uses them to create the object.

        Parameters
        ----------
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        Dataset
            Object of class Dataset.
        """
        return super().from_argparse(args)

    @classmethod
    def from_json(cls, json_dict: Dict) -> "Dataset":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Dataset
            Object of class Dataset.
        """
        return super().from_json(json_dict)

    def __iter__(self) -> "Dataset":
        """
        Provides iterator over data samples' tuples.

        Each data sample is a tuple (X, y), where X are the model inputs,
        and y are the model outputs.

        Returns
        -------
        Dataset
            This object.
        """
        self._dataindex = 0
        self._dataindices = list(range(len(self.dataX)))
        return self

    def __next__(self) -> Tuple[List, List]:
        """
        Returns next data sample in a form of a (X, y) tuple.

        X contains the list of inputs for the model.
        Y contains the list of outputs for the model.

        Returns
        -------
        Tuple[List, List]
            Tuple containing list of input data for inference and output data
            for comparison.

        Raises
        ------
        StopIteration
            Raised when iterations are finished
        """
        # Handle the case when the next method is called without the iterator
        if self._dataindex == 0 and not self._dataindices:
            self._dataindices = list(range(len(self.dataX)))

        if self._dataindex < len(self._dataindices):
            samples = self._dataindices[
                self._dataindex : self._dataindex + self.batch_size
            ]
            self._dataindex += len(samples)
            dataX = [self.dataX[i] for i in samples]
            dataY = [self.dataY[i] for i in samples]
            return (
                self.prepare_input_samples(dataX),
                self.prepare_output_samples(dataY),
            )
        raise StopIteration

    def __len__(self) -> int:
        """
        Returns the number of data samples. Takes the batch size into account.

        Returns
        -------
        int
            Number of input samples in a single batch.
        """
        return ceil(len(self.dataX) / self.batch_size)

    def iter_train(self) -> "DatasetIterator":
        """
        Iterates over train data obtained from split.

        Returns
        -------
        DatasetIterator
            Iterator over data samples.
        """
        split = self.train_test_split_representations(append_index=True)
        indices = split[4]
        return DatasetIterator(self, indices)

    def iter_test(self, shuffle: bool = True) -> "DatasetIterator":
        """
        Iterates over test data obtained from split.

        Parameters
        ----------
        shuffle : bool
            Shuffle samples

        Returns
        -------
        DatasetIterator
            Iterator over data samples.
        """
        if shuffle:
            split = self.train_test_split_representations(append_index=True)
            indices = split[5]
        else:
            indices = range(
                len(self.dataX)
            )  # TODO: add proper val and test fractions

        return DatasetIterator(self, indices)

    def iter_val(self) -> "DatasetIterator":
        """
        Iterates over validation data obtained from split.

        Returns
        -------
        DatasetIterator
            Iterator over data samples.
        """
        split = self.train_test_split_representations(append_index=True)
        assert len(split) == 9, "No validation data in split"
        indices = split[8]
        return DatasetIterator(self, indices)

    def prepare_input_samples(self, samples: List) -> List:
        """
        Prepares input samples, i.e. load images from files, converts them.

        By default the method returns data as is - without any conversions.
        Since the input samples can be large, it does not make sense to load
        all data to the memory - this method handles loading data for a given
        data batch.

        Parameters
        ----------
        samples : List
            List of input samples to be processed.

        Returns
        -------
        List
            Preprocessed input samples.
        """
        return [samples]

    def prepare_output_samples(self, samples: List) -> List:
        """
        Prepares output samples.

        By default the method returns data as is.
        It can be used i.e. to create the one-hot output vector with class
        association based on a given sample.

        Parameters
        ----------
        samples : List
            List of output samples to be processed.

        Returns
        -------
        List
            Preprocessed output samples.
        """
        return [samples]

    def set_batch_size(self, batch_size: int):
        """
        Sets the batch size of the data in the iterator batches.

        Parameters
        ----------
        batch_size : int
            Number of input samples per batch.
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
        Tuple[List, List]
            The list of data samples.
        """
        return (
            self.prepare_input_samples(self.dataX),
            self.prepare_output_samples(self.dataY),
        )

    def get_data_unloaded(self) -> Tuple[List, List]:
        """
        Returns the input and output representations before loading.

        The representations can be opened using prepare_input_samples and
        prepare_output_samples.

        Returns
        -------
        Tuple[List, List]
            The list of data samples representations.
        """
        return (self.dataX, self.dataY)

    def _subset_len(self, subset: List[Any]) -> int:
        """
        Returns the length of a given subset (e.g. train, test or val). Takes
        batch_size into account.

        Parameters
        ----------
        subset: List[Any]
            A list representing given dataset subset

        Returns
        -------
        int
            The length of a single batch from a given subset
        """
        return ceil(len(subset) / self.batch_size)

    def train_subset_len(self) -> Optional[int]:
        """
        Returns the length of a single batch from the training set.

        Returns
        -------
        Optional[int]
            The number of samples in a single batch from the training set or
            None if the dataset has not been split
        """
        if not self.dataXtrain:
            return None
        return self._subset_len(self.dataXtrain)

    def test_subset_len(self) -> Optional[int]:
        """
        Returns the length of a single batch from the training set.

        Returns
        -------
        Optional[int]
            The number of samples in a single batch from the testing set or
            None if the dataset has not been split
        """
        if not self.dataXtest:
            return None
        return self._subset_len(self.dataXtest)

    def val_subset_len(self) -> Optional[int]:
        """
        Returns the length of a single batch from the training set.

        Returns
        -------
        Optional[int]
            The number of samples in a single batch from the validation set or
            None if the dataset has not been split
        """
        if not self.dataXval:
            return None
        return self._subset_len(self.dataXval)

    def train_test_split_representations(
        self,
        test_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        append_index: bool = False,
    ) -> Tuple[List, ...]:
        """
        Splits the data representations into train dataset and test dataset.

        Parameters
        ----------
        test_fraction : Optional[float]
            The fraction of data to leave for model testing.
        val_fraction : Optional[float]
            The fraction of data to leave for model validation.
        seed : Optional[int]
            The seed for random state.
        stratify : bool
            Whether to stratify the split.
        append_index : bool
            Whether to return the indices of the split. If True, the returned
            tuple will have indices appended at the end.
            For example, if the split is (X_train, X_test, y_train, y_test),
            the returned tuple will be (X_train, X_test, y_train, y_test,
            train_indices, test_indices).

        Returns
        -------
        Tuple[List, ...]
            Split data into train, test and optionally validation subsets.
        """
        from sklearn.model_selection import train_test_split

        if test_fraction is None:
            test_fraction = self.split_fraction_test
        if val_fraction is None:
            val_fraction = self.split_fraction_val
        if seed is None:
            seed = self.split_seed

        indices = list(range(len(self.dataX)))
        np.random.seed(seed)
        if self.shuffle_data:
            np.random.shuffle(indices)

        if not val_fraction:
            # All data is for testing
            if test_fraction == 1.0:
                ret = ([], self.dataX, [], self.dataY)
                if append_index:
                    ret += ([], indices)
                return ret

            # All data is for training
            elif test_fraction == 0.0:
                ret = (self.dataX, [], self.dataY, [])
                if append_index:
                    ret += (indices, [])
                return ret

        # Verify fraction sizes
        if val_fraction is not None:
            assert test_fraction + val_fraction < 1.0
        else:
            assert test_fraction < 1.0

        if stratify and self.shuffle_data:
            stratify_arg = self.dataY
        else:
            stratify_arg = None

        (
            self.dataXtrain,
            self.dataXtest,
            self.dataYtrain,
            self.dataYtest,
            dataItrain,
            dataItest,
        ) = train_test_split(
            self.dataX,
            self.dataY,
            indices,
            test_size=test_fraction,
            random_state=seed,
            shuffle=self.shuffle_data,
            stratify=stratify_arg,
        )

        ret = (
            self.dataXtrain,
            self.dataXtest,
            self.dataYtrain,
            self.dataYtest,
        )
        if append_index:
            ret += (dataItrain, dataItest)

        if val_fraction is not None and val_fraction != 0:
            if stratify and self.shuffle_data:
                stratify_arg = self.dataYtrain
            else:
                stratify_arg = None
            (
                self.dataXtrain,
                self.dataXval,
                self.dataYtrain,
                self.dataYval,
                dataItrain,
                dataIval,
            ) = train_test_split(
                self.dataXtrain,
                self.dataYtrain,
                dataItrain,
                test_size=val_fraction / (1 - test_fraction),
                random_state=seed,
                shuffle=self.shuffle_data,
                stratify=stratify_arg,
            )
            ret = (
                self.dataXtrain,
                self.dataXtest,
                self.dataYtrain,
                self.dataYtest,
                self.dataXval,
                self.dataYval,
            )
            if append_index:
                ret += (dataItrain, dataItest, dataIval)

        return ret

    def calibration_dataset_generator(
        self,
        percentage: float = 0.25,
        seed: Optional[int] = None,
    ) -> Generator[List[Any], None, None]:
        """
        Creates generator for the calibration data.

        Parameters
        ----------
        percentage : float
            The fraction of data to use for calibration.
        seed : Optional[int]
            The seed for random state,
            by default seed used for the dataset split.

        Yields
        ------
        List[Any]
            List with batch input samples for calibration
        """
        if seed is None:
            seed = self.split_seed
        if self.external_calibration_dataset is None:
            X = self.train_test_split_representations(percentage, seed=seed)[1]
        else:
            X = self.prepare_external_calibration_dataset(percentage, seed)
        with LoggerProgressBar() as logger_progress_bar:
            for x in tqdm(X, **logger_progress_bar.kwargs):
                yield self.prepare_input_samples([x])

    def prepare_external_calibration_dataset(
        self, percentage: float = 0.25, seed: int = 12345
    ) -> List[Path]:
        """
        Prepares the data for external calibration dataset.

        This method is supposed to scan external_calibration_dataset directory
        and prepares the list of entries that are suitable for the
        prepare_input_samples method.

        This method is called by the ``calibration_dataset_generator`` method
        to get the data for calibration when external_calibration_dataset is
        provided.

        By default, this method scans for all files in the directory and
        returns the list of those files.

        Parameters
        ----------
        percentage : float
            Percentage of dataset to be used.
        seed : int
            Random state seed.

        Returns
        -------
        List[Path]
            List of objects that are usable by the ``prepare_input_samples``
            method.
        """
        data = [
            x
            for x in self.external_calibration_dataset.rglob("*")
            if x.is_file()
        ]
        random.Random(seed).shuffle(data)
        return data[: int(percentage * len(data) + 0.5)]

    @abstractmethod
    def download_dataset_fun(self):
        """
        Downloads the dataset to the root directory defined in the constructor.
        """
        raise NotImplementedError

    def save_dataset_checksum(self):
        """
        Writes dataset checksum to file.
        """
        checksum_file = self.root / "DATASET_CHECKSUM"

        checksum = hexlify(self._compute_dataset_checksum()).decode()
        timestamp = str(datetime.datetime.now())

        with open(checksum_file, "w") as file:
            file.write(f"{checksum} {timestamp}")

    def verify_dataset_checksum(self) -> bool:
        """
        Checks whether dataset is already downloaded in its directory.

        Returns
        -------
        bool
            True if dataset is downloaded.
        """
        checksum_file = self.root / "DATASET_CHECKSUM"
        if not checksum_file.exists():
            return False

        checksum = hexlify(self._compute_dataset_checksum()).decode()

        with open(checksum_file, "r") as file:
            try:
                valid_checksum = file.read().split()[0]
            except IndexError:
                return False

        return checksum == valid_checksum

    @abstractmethod
    def prepare(self):
        """
        Prepares dataX and dataY attributes based on the dataset contents.

        This can i.e. store file paths in dataX and classes in dataY that
        will be later loaded using prepare_input_samples and
        prepare_output_samples.
        """
        raise NotImplementedError

    def _reduce_dataset(self):
        """
        Reduces dataset to the given percentage.

        The order of remaining data is preserved.
        """
        full_size = len(self.dataX)
        np.random.seed(self.split_seed)
        _, _, _, _, reduced_indices, _ = self.train_test_split_representations(
            1 - self.dataset_percentage,
            append_index=True,
        )
        # Preserve order of dataset
        reduced_indices = np.sort(reduced_indices)
        if isinstance(self.dataX, np.ndarray):
            self.dataX = self.dataX[reduced_indices]
        else:
            self.dataX = [self.dataX[idx] for idx in reduced_indices]
        if isinstance(self.dataY, np.ndarray):
            self.dataY = self.dataY[reduced_indices]
        else:
            self.dataY = [self.dataY[idx] for idx in reduced_indices]
        reduced_size = len(self.dataX)
        KLogger.info(
            f"Dataset reduced from {full_size} to {reduced_size} entries"
            f" ({reduced_size / full_size * 100:.2f}%)"
        )

    @abstractmethod
    def evaluate(self, predictions: List, truth: List) -> "Measurements":
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
            The list of predictions from the model.
        truth : List
            The ground truth for given batch.

        Returns
        -------
        Measurements
            The dictionary containing the evaluation results.
        """
        ...

    def _evaluate(
        self,
        predictions: List,
        truth: List,
        specification: Optional[Dict] = None,
    ) -> Measurements:
        if specification:
            IOInterface.assert_data_format(
                predictions,
                specification,
            )
        return self.evaluate(predictions, truth)

    @abstractmethod
    def get_input_mean_std(self) -> Tuple[Any, Any]:
        """
        Returns mean and std values for input tensors.

        The mean and std values returned here should be computed using
        ``compute_input_mean_std`` method.

        Returns
        -------
        Tuple[Any, Any]
            Tuple of two variables describing mean and
            standardization values for a given train dataset.
        """
        ...

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Returns list of class names in order of their IDs.

        Returns
        -------
        List[str]
            List of class names.
        """
        ...

    def _compute_dataset_checksum(self) -> bytes:
        """
        Computes checksum of dataset files.

        Returns
        -------
        bytes
            Dataset checksum.
        """
        checksum_file = self.root / "DATASET_CHECKSUM"

        dataset_files = list(self.root.rglob("*"))
        if checksum_file in dataset_files:
            dataset_files.remove(checksum_file)
        dataset_files.sort()

        sha = hashlib.sha256()

        for file in dataset_files:
            sha.update(file.name.encode())
            sha.update(struct.pack("f", file.stat().st_mtime))

        return sha.digest()
