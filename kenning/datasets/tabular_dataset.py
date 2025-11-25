# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with generic datasets generated from CSV files.
"""

import hashlib

import numpy as np
import polars as pl
from sklearn import metrics

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements


class TabularDataset(Dataset):
    """
    Wrapper for generic classification dataset generated from CSV files.
    """

    arguments_structure = {
        "dataset_path": {
            "argparse_name": "--dataset-path",
            "description": "Path to the dataset",
            "type": str,
            "required": True,
        },
        "colsX": {
            "argparse_name": "--cols-x",
            "description": "",
            "type": list[str],
            "required": True,
        },
        "colY": {
            "argparse_name": "--col-y",
            "description": "",
            "type": str,
            "required": True,
        },
        "window_size": {
            "argparse_name": "--window-size",
            "description": "",
            "type": int,
            "nullable": True,
            "default": None,
        },
        "shuffle_data": {
            "argparse_name": "--shuffle-data",
            "type": bool,
            "default": True,
        },
    }

    def __init__(
        self,
        root,
        batch_size=1,
        download_dataset=True,
        colsX: list[str] | None = None,
        colY: str | None = None,
        dataset_path: str | None = None,
        window_size: int | None = None,
        force_download_dataset=False,
        external_calibration_dataset=None,
        split_fraction_test=0.2,
        split_fraction_val=None,
        split_seed=1234,
        dataset_percentage=1,
        shuffle_data=True,
    ):
        self.colsX = colsX
        self.colY = colY
        self.dataset_path = dataset_path
        assert dataset_path is not None
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
            dataset_percentage,
            shuffle_data,
        )

    def _get_csv_path(self):
        sha256 = hashlib.sha256(
            str(self.dataset_path).encode(), usedforsecurity=False
        ).hexdigest()
        return self.root / f"{sha256}.csv"

    def download_dataset_fun(self):
        self.root.mkdir(exist_ok=True, parents=True)
        csv_path = self._get_csv_path()
        df = pl.read_csv(self.dataset_path)
        df.write_csv(csv_path, include_header=True)

    def prepare(self):
        csv_path = self._get_csv_path()
        df = pl.read_csv(csv_path)

        dfX: pl.DataFrame = df[self.colsX]
        dfY: pl.Series = df[self.colY]

        rowsX = dfX.rows()
        rowsY = dfY.rank("dense") - 1
        if self.window_size is None:
            self.dataX = rowsX
            self.dataY = rowsY
        else:
            self.dataX = [
                rowsX[i : i + self.window_size]
                for i in range(max(0, len(rowsX) - self.window_size))
            ]
            self.dataY = rowsY[self.window_size :]

        self.mean = list(dfX.mean())
        self.std = list(dfX.std())
        self.class_names = list(map(str, dfY.unique().sort()))
        self.num_features = dfX.shape[1]

    def prepare_output_samples(self, samples):
        return [np.eye(len(self.class_names))[samples]]

    def evaluate(self, predictions, truth):
        measurements = Measurements()
        ts = np.argmax(truth[0], axis=-1)
        ps = predictions[0]
        confusion_matrix = metrics.confusion_matrix(
            ts, ps, labels=range(len(self.class_names))
        )
        measurements.accumulate(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((len(self.class_names), len(self.class_names))),
        )
        measurements.add_measurement(
            "predictions",
            [
                {
                    "target": t,
                    "prediction": p,
                }
                for t, p in zip(ts, ps)
            ],
        )
        return measurements

    def get_input_mean_std(self):
        return self.mean, self.std

    def get_class_names(self):
        return self.class_names
