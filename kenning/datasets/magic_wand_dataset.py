# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Tensorflow Magic Wand dataset.
"""

import glob
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import Resources, extract_tar


class MagicWandDataset(Dataset):
    """
    The Tensorflow Magic Wand dataset.

    It is a classification dataset with 4 classes representing different
    gestures captured by accelerometer and gyroscope.
    """

    resources = Resources(
        {
            "data": "http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz",
        }
    )
    arguments_structure = {
        "window_size": {
            "argparse_name": "--window-size",
            "description": "Determines the size of single sample window",
            "default": 128,
            "type": int,
        },
        "window_shift": {
            "argparse_name": "--window-shift",
            "description": "Determines the shift of single sample window",
            "default": 128,
            "type": int,
        },
        "noise_level": {
            "argparse_name": "--noise-level",
            "description": "Determines the level of noise added as padding",
            "default": 20,
            "type": int,
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
        window_size: int = 128,
        window_shift: int = 128,
        noise_level: int = 20,
    ):
        """
        Prepares all structures and data required for providing data samples.

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
        window_size : int
            Size of single sample window.
        window_shift : int
            Shift of single sample window.
        noise_level : int
            Noise level of padding added to sample.
        """
        self.window_size = window_size
        self.window_shift = window_shift
        self.noise_level = noise_level
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
            dataset_percentage,
        )

    def rev_class_id(self, classname: str) -> int:
        """
        Returns an integer representing a class based on a class name.

        It generates a reversed dictionary from the `self.classnames` and
        gets the ID that is assigned to that name.

        Parameters
        ----------
        classname : str
            The name of the class for which the ID will be returned.

        Returns
        -------
        int
            The class id.
        """
        return {v: k for k, v in self.classnames.items()}[classname]

    def prepare(self):
        self.classnames = {0: "wing", 1: "ring", 2: "slope", 3: "negative"}
        self.numclasses = 4
        tmp_dataX = []
        tmp_dataY = []
        for class_name in self.classnames.values():
            path = self.root / class_name
            if not path.is_dir():
                raise FileNotFoundError
            class_id = self.rev_class_id(class_name)
            for file in glob.glob(str(path / "*.txt")):
                data_frame = []
                with open(file) as f:
                    for line in f:
                        line_split = line.strip().split(",")
                        if len(line_split) != 3:
                            continue
                        try:
                            values = [float(i) for i in line_split]
                            data_frame.append(values)
                        except ValueError:
                            if data_frame:
                                tmp_dataX.append(data_frame)
                                tmp_dataY.append(class_id)
                                data_frame = []

        self.dataX = []
        self.dataY = []
        for data, label in zip(tmp_dataX, tmp_dataY):
            padded_data = np.array(
                self.split_sample_to_windows(self.generate_padding(data)),
                dtype="float32",
            )
            for sample in padded_data:
                self.dataX.append(sample)
                self.dataY.append(np.eye(self.numclasses)[label])

        assert len(self.dataX) == len(self.dataY)

    def prepare_input_samples(
        self, samples: List[np.ndarray]
    ) -> List[np.ndarray]:
        return [np.array(samples)]

    def prepare_output_samples(
        self, samples: List[np.ndarray]
    ) -> List[np.ndarray]:
        return [np.array(samples)]

    def download_dataset_fun(self):
        extract_tar(self.root, self.resources["data"])

        # cleanup MacOS-related hidden metadata files present in the dataset
        for macos_dotfile in glob.glob(str(self.root) + "/**/._*") + glob.glob(
            str(self.root) + "/._*"
        ):
            os.remove(macos_dotfile)

    def _generate_padding(
        self, noise_level: int, amount: int, neighbor: List
    ) -> List:
        """
        Generates noise padding of given length.

        Parameters
        ----------
        noise_level : int
            Level of generated noise.
        amount : int
            Length of generated noise.
        neighbor : List
            Neighbor data.

        Returns
        -------
        List
            Neighbor data with noise padding.
        """
        padding = (
            np.round((np.random.rand(amount, 3) - 0.5) * noise_level, 1)
            + neighbor
        )
        return [list(i) for i in padding]

    def generate_padding(self, data_frame: List) -> List:
        """
        Generates neighbor-based padding around a given data frame.

        Parameters
        ----------
        data_frame : List
            A frame of data to be padded.

        Returns
        -------
        List
            The padded data frame.
        """
        pre_padding = self._generate_padding(
            self.noise_level,
            abs(self.window_size - len(data_frame)) % self.window_size,
            data_frame[0],
        )
        unpadded_len = len(pre_padding) + len(data_frame)
        post_len = (-unpadded_len) % self.window_shift

        post_padding = self._generate_padding(
            self.noise_level, post_len, data_frame[-1]
        )
        return pre_padding + data_frame + post_padding

    def get_class_names(self) -> List[str]:
        return list(self.classnames.values())

    def evaluate(self, predictions: List, truth: Optional[List] = None):
        measurements = Measurements()
        while hasattr(predictions, "__len__"):
            if len(predictions) == 1:
                predictions = predictions[0]
            else:
                break
        shape = []
        lastlist = predictions
        while hasattr(lastlist, "__len__"):
            shape.append(len(lastlist))
            lastlist = lastlist[0]
        if len(shape) == 1:
            assert self.batch_size == 1
            assert shape[0] == self.numclasses
            predictions = [predictions]
        elif len(shape) > 1:
            assert shape[0] == self.batch_size
        if truth is not None:
            confusion_matrix = np.zeros((self.numclasses, self.numclasses))
            for prediction, label in zip(predictions, truth):
                while len(prediction) != self.numclasses:
                    assert len(prediction) == 1
                    prediction = prediction[0]
                confusion_matrix[np.argmax(label), np.argmax(prediction)] += 1
            measurements.accumulate(
                "eval_confusion_matrix",
                confusion_matrix,
                lambda: np.zeros((self.numclasses, self.numclasses)),
            )
        else:
            predictions_vector = np.zeros(self.numclasses)
            for prediction in predictions:
                while len(prediction) != self.numclasses:
                    assert len(prediction) == 1
                    prediction = prediction[0]
                predictions_vector[np.argmax(prediction)] += 1
            measurements.accumulate(
                "predictions",
                predictions_vector,
                lambda: np.zeros(self.numclasses),
            )
        measurements.accumulate("total", len(predictions), lambda: 0)
        return measurements

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        return (
            np.array([-219.346, 198.207, 854.390]),
            np.array([430.269, 326.288, 447.666]),
        )

    def split_sample_to_windows(self, data_frame: List) -> np.ndarray:
        """
        Splits given data sample into windows.

        Parameters
        ----------
        data_frame : List
            Data sample to be split.

        Returns
        -------
        np.ndarray
            Data sample split into windows.
        """
        return np.array(
            np.array_split(
                data_frame, len(data_frame) // self.window_size, axis=0
            )
        )
