# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with Dataset generating random data for benchmark purposes.
"""

import random
from pathlib import Path
from string import ascii_letters
from typing import Any, List, Optional, Tuple, Type

import cv2
import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.datasets.helpers.detection_and_segmentation import (
    DetectObject,
    ObjectDetectionSegmentationDataset,
)


class RandomizedClassificationDataset(Dataset):
    """
    Creates a sample randomized classification dataset and
    stores it in `root` directory.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    arguments_structure = {
        "samplescount": {
            "argparse_name": "--num-samples",
            "description": "Number of samples to process",
            "type": int,
            "default": 100,
        },
        "numclasses": {
            "argparse_name": "--num-classes",
            "description": "Number of classes in inputs",
            "type": int,
            "default": 3,
        },
        "inputdims": {
            "argparse_name": "--input-dims",
            "description": "Dimensionality of the inputs",
            "type": int,
            "default": [224, 224, 3],
            "is_list": True,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        samplescount: int = 100,
        numclasses: int = 3,
        integer_classes: bool = False,
        inputdims: List = [224, 224, 3],
        dtype: Type = np.float32,
        seed: int = 1234,
        **kwargs: Any,
    ):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Directory for the randomized dataset - no data will be store there.
        batch_size : int
            The size of batches of data delivered during inference.
        download_dataset : bool
            Downloads the dataset before taking any action. If the dataset
            files are already downloaded then they are not downloaded again.
        force_download_dataset : bool
            Forces dataset download.
        samplescount : int
            The number of samples in the dataset.
        numclasses : int
            The number of classes in the dataset.
        integer_classes : bool
            Indicates if classes should be represented by single integer
            instead of one-hot encoding.
        inputdims : List
            The dimensionality of the inputs.
        dtype : Type
            Type of the data.
        seed : int
            Seed used for data generation.
        **kwargs : Any
            Optional keyword arguments.
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.numclasses = numclasses
        self.integer_classes = integer_classes
        self.dtype = dtype
        self.classnames = self.get_class_names()
        self.seed = seed

        super().__init__(
            root,
            batch_size,
            force_download_dataset,
            download_dataset,
            dataset_percentage=1,
        )

    def get_class_names(self):
        return [str(i) for i in range(self.numclasses)]

    def get_input_mean_std(self):
        return (0.0, 1.0)

    def prepare(self):
        np.random.seed(self.seed)
        self.dataX = [
            f"{self.root}/images/{i}.jpg" for i in range(self.samplescount)
        ]
        self.dataY = [j % self.numclasses for j in range(self.samplescount)]
        random.shuffle(self.dataY)
        if not self.integer_classes:
            self.dataY = [np.eye(self.numclasses)[y] for y in self.dataY]

        (self.root / "images").mkdir(parents=True, exist_ok=True)
        samples = self.prepare_input_samples(self.dataX)[0]
        for img_path, img_data in zip(self.dataX, samples):
            cv2.imwrite(img_path, img_data)

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples: List[str]) -> List[np.ndarray]:
        result = []
        for _ in samples:
            result.append(np.random.randn(*self.inputdims).astype(self.dtype))
        return [np.array(result)]

    def prepare_output_samples(self, samples: List[Any]) -> List[np.ndarray]:
        return [np.array(samples)]

    def evaluate(self, predictions, truth):
        return Measurements()

    def calibration_dataset_generator(
        self, percentage: float = 0.25, seed: int = 12345
    ):
        for _ in range(int(self.samplescount * percentage)):
            yield [np.random.randint(0, 255, size=self.inputdims)]


class RandomizedDetectionSegmentationDataset(
    ObjectDetectionSegmentationDataset
):
    """
    Creates a sample randomized detection dataset.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    arguments_structure = {
        "samplescount": {
            "argparse_name": "--num-samples",
            "description": "Number of samples to process",
            "type": int,
            "default": 100,
        },
        "numclasses": {
            "argparse_name": "--num-classes",
            "description": "Number of classes in inputs",
            "type": int,
            "default": 3,
        },
        "inputdims": {
            "argparse_name": "--input-dims",
            "description": "Dimensionality of the inputs",
            "type": int,
            "default": [224, 224, 3],
            "is_list": True,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        samplescount: int = 100,
        numclasses: int = 3,
        inputdims: List = [224, 224, 3],
        dtype: Type = np.float32,
        **kwargs: Any,
    ):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Deprecated argument, not used in this dataset.
        batch_size : int
            The size of batches of data delivered during inference.
        download_dataset : bool
            True if dataset should be downloaded first.
        samplescount : int
            The number of samples in the dataset.
        numclasses : int
            The number of classes in the dataset.
        inputdims : List
            The dimensionality of the inputs.
        dtype : Type
            Type of the data.
        **kwargs : Any
            Optional keyword arguments.
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.numclasses = numclasses
        self.dtype = dtype
        self.classnames = self.get_class_names()
        super().__init__(root, batch_size, download_dataset)

    def get_class_names(self):
        return [str(i) for i in range(self.numclasses)]

    def prepare(self):
        self.dataX = list(range(self.samplescount))
        self.dataY = []

        classes = [i % self.numclasses for i in range(self.samplescount)]
        random.shuffle(classes)
        for i in range(self.samplescount):
            x_rand = np.random.random((2,))
            y_rand = np.random.random((2,))
            self.dataY.append(
                [
                    DetectObject(
                        clsname=str(classes[i]),
                        xmin=x_rand.min(),
                        ymin=y_rand.min(),
                        xmax=x_rand.min(),
                        ymax=y_rand.max(),
                        score=1.0,
                        iscrowd=(np.random.randint(0, 1) == 1),
                    )
                ]
            )

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples: List[int]) -> List[np.ndarray]:
        result = []
        for sample in samples:
            np.random.seed(sample)
            result.append(np.random.randn(*self.inputdims).astype(self.dtype))
        return [np.array(result)]

    def prepare_output_samples(
        self, samples: List[List[DetectObject]]
    ) -> List[List[List[DetectObject]]]:
        return [samples]

    def evaluate(self, predictions, truth):
        return Measurements()

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotImplementedError


class RandomizedTextDataset(Dataset):
    """
    Creates randomized dataset consisting of text sentences and
    stores it in `root` directory.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    arguments_structure = {
        "samplescount": {
            "argparse_name": "--num-samples",
            "description": "Number of samples to process",
            "type": int,
            "default": 32,
        },
        "text_length": {
            "argparse_name": "--text-length",
            "description": "Length of sample sentences",
            "type": int,
            "default": 255,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        samplescount: int = 32,
        text_length: int = 32,
    ):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Directory, where the randomized dataset is stored.
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
        samplescount : int
            The number of samples in the dataset.
        text_length : int
            Length of sample sentences
        """
        self.samplescount = samplescount
        self.text_length = text_length

        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
        )

    def get_class_names(self) -> List[str]:
        raise NotImplementedError

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def train_test_split_representations(
        self,
        test_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        append_index: bool = False,
    ) -> Tuple[List, ...]:
        return super().train_test_split_representations(
            test_fraction,
            val_fraction,
            seed,
            False,
            append_index,
        )

    def prepare(self):
        self.dataX = [
            f"{self.root}/sentences/{i}.txt" for i in range(self.samplescount)
        ]

        samples = []
        for _ in range(self.samplescount):
            samples.append(
                "".join(
                    random.choices(ascii_letters + " ", k=self.text_length)
                )
            )

        (self.root / "sentences").mkdir(parents=True, exist_ok=True)
        for txt_path, txt_data in zip(self.dataX, samples):
            with open(txt_path, "w") as f:
                f.write(txt_data)

        self.dataY = [i for i in range(self.samplescount)]

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples: List[str]) -> List[List[str]]:
        result = []
        for txt_path in samples:
            with open(txt_path, "r") as f:
                result.append(f.read())

        return [result]

    def prepare_output_samples(self, samples: List[int]) -> List[List[int]]:
        return [samples]

    def evaluate(self, predictions, truth):
        return Measurements()


class RandomizedAnomalyDetectionDataset(RandomizedClassificationDataset):
    """
    Randomized dataset for Anomaly Detection, implementing classification
    dataset with adjustable window size and number of features per timestamp.
    """

    arguments_structure = {
        "window_size": {
            "argparse_name": "--window-size",
            "description": "The number of consecutive timestamps included in one entry",  # noqa: E501
            "type": int,
            "default": 5,
        },
        "features": {
            "argparse_name": "--features",
            "description": "The number of features per one timestamps",
            "type": int,
            "default": 10,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        samplescount: int = 100,
        numclasses: int = 3,
        integer_classes: bool = False,
        dtype: Type = np.float32,
        seed: int = 1234,
        window_size: int = 5,
        num_features: int = 10,
        **kwargs: Any,
    ):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Directory for the randomized dataset - no data will be store there.
        batch_size : int
            The size of batches of data delivered during inference.
        download_dataset : bool
            Downloads the dataset before taking any action. If the dataset
            files are already downloaded then they are not downloaded again.
        force_download_dataset : bool
            Forces dataset download.
        samplescount : int
            The number of samples in the dataset.
        numclasses : int
            The number of classes in the dataset.
        integer_classes : bool
            Indicates if classes should be represented by single integer
            instead of one-hot encoding.
        dtype : Type
            Type of the data.
        seed : int
            Seed used for data generation.
        window_size : int
            The number of consecutive timestamps included in one entry.
        num_features : int
            The number of features per one timestamps.
        **kwargs : Any
            Optional keyword arguments.
        """
        self.num_features = num_features
        self.window_size = window_size
        super().__init__(
            root=root,
            batch_size=batch_size,
            download_dataset=download_dataset,
            force_download_dataset=force_download_dataset,
            samplescount=samplescount,
            numclasses=numclasses,
            integer_classes=integer_classes,
            inputdims=(window_size, num_features),
            dtype=dtype,
            seed=seed,
            **kwargs,
        )

    def prepare_output_samples(self, samples: List[Any]) -> List[Any]:
        return [samples]
