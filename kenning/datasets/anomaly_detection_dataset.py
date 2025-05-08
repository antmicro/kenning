# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Dataset wrapper for anomaly detection in time series.
"""

import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import ResourceURI


class AnomalyDetectionDataset(Dataset):
    """
    Generic dataset for anomaly detection in time series problem.

    It reads data from provided CSV file and prepares sequences of data.

    CSV file has to follow the schema:

    +------------------+------------------+------------------+-----+------------------+----------------+
    | Timestamp column | Param 1 name     | Param 2 name     | ... | Param N name     | Label          |
    +==================+==================+==================+=====+==================+================+
    | timestamps       | Numerical values | Numerical values | ... | Numerical values | Integer values |
    +------------------+------------------+------------------+-----+------------------+----------------+

    Kenning automatically discards the timestamp column,
    as well as the header row.

    The numerical values of parameters are used as signals
    or data from sensors, whereas the labels specify
    anomaly occurrence (values greater than 0).

    Each label describes whether an anomaly has been observed within `window_size`
    previous samples.

    This results with final version of dataset where one entry looks like:

    +------------------------------------------+-----+------------------------------------------+-------------------------------+
    | X                                        |     |                                          | Y                             |
    +==========================================+=====+==========================================+===============================+
    | Param 1 value from `t - window_size + 1` | ... | Param N value from `t - window_size + 1` |                               |
    +------------------------------------------+-----+------------------------------------------+-------------------------------+
    | ...                                      | ... | ...                                      |                               |
    +------------------------------------------+-----+------------------------------------------+-------------------------------+
    | Param 1 value from `t - 1`               | ... | Param N value from `t - 1`               |                               |
    +------------------------------------------+-----+------------------------------------------+-------------------------------+
    | Param 1 value from `t`                   | ... | Param N value from `t`                   | 0 (no anomaly) or 1 (anomaly) |
    +------------------------------------------+-----+------------------------------------------+-------------------------------+
    """  # noqa: E501

    arguments_structure = {
        "csv_file": {
            "argparse_name": "--csv-file",
            "description": "Path or URL to the CSV file with dataset",
            "type": ResourceURI,
            "required": True,
        },
        "window_size": {
            "argparse_name": "--window-size",
            "description": "The number of consecutive timestamps included in one entry",  # noqa: E501
            "type": int,
            "default": 5,
        },
        "gather_predictions": {
            "argparse_name": "--gather-predictions",
            "description": "Determines whether returned measurements should include target and predicted anomalies",  # noqa: E501
            "type": bool,
            "default": True,
        },
    }

    def __init__(
        self,
        root: Path,
        csv_file: Union[str, ResourceURI],
        batch_size: int = 1,
        download_dataset: bool = True,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        dataset_percentage: float = 1,
        window_size: int = 5,
        gather_predictions: bool = True,
    ):
        """
        Representation of dataset for anomaly detection.

        Parameters
        ----------
        root : Path
            The path to the dataset data.
        csv_file : Union[str, ResourceURI]
            Path or URL to the CSV file with dataset.
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
            The number of consecutive timestamps included in one entry.
        gather_predictions : bool
            Whether returned measurements should include target
            and predicted sentences.
        """
        self.csv_file = csv_file
        if not isinstance(self.csv_file, ResourceURI):
            self.csv_file = ResourceURI(self.csv_file)
        self.num_features = None
        self.window_size = window_size
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
            dataset_percentage=dataset_percentage,
        )
        self.gather_predictions = gather_predictions
        self.classnames = self.get_class_names()

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        data_path = self.root / self.csv_file.stem
        if not data_path.exists():
            shutil.copy(self.csv_file, data_path)

    def prepare(self):
        data_path = self.root / self.csv_file.stem
        data = pd.read_csv(data_path)
        # Remove timestamps (first column)
        data = data.drop(columns=[data.columns[0]])
        data = data.to_numpy(np.float32)
        self.dataX = data[:, :-1]
        self.dataY = data[:, -1].astype(np.int_)

        # Simplify anomaly categories into one class
        self.dataY[self.dataY > 0] = 1
        self.num_features = self.dataX.shape[1]

        # Prepare data in format (window_size, num_features)
        self.dataX = [
            self.dataX[i : i + self.window_size]
            for i in range(0, self.dataX.shape[0] - self.window_size)
        ]
        self.dataX = np.asarray(self.dataX)
        self.dataY = self.dataY[self.window_size :]

    def evaluate(self, predictions, truth) -> Measurements:
        confusion_matrix = metrics.confusion_matrix(
            truth[0], predictions[0][: len(truth[0])], labels=[0, 1]
        )
        measurements = Measurements()
        measurements.accumulate(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((2, 2)),
        )
        measurements.accumulate("total", len(predictions), lambda: 0)
        if self.gather_predictions:
            measurements.add_measurement(
                "predictions",
                [
                    {
                        "target": t,
                        "prediction": p,
                    }
                    for t, p in zip(truth[0], predictions[0])
                ],
            )
        return measurements

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        return np.mean(self.dataX, axis=0), np.std(self.dataX, axis=0)

    def get_class_names(self) -> List[str]:
        return ["normal", "anomaly"]
