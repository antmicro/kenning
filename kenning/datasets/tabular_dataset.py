# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with generic datasets generated from CSV files.
"""

import hashlib
from typing import Any

import numpy as np
import polars as pl
from sklearn import metrics

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import ResourceManager

DATA_TYPE = "float64"


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
        "expand_classes": {
            "argparse_name": "--expand-classes",
            "description": "If enabled, 'prepare_output_samples' function will"
            " transform outputs into vector of length N, where N is number of"
            " classes. Example: [1,1,0,2,0] will be changed to [[0,1,0],"
            " [0,1,0],[1,0,0],[0,0,1],[1,0,0]]. This is to prepare the dataset"
            " to be used for a typical classifier. It should be disabled when"
            " using AutoML, because AutoPyTorch does it on it's own.",
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
        expand_classes=True,
        **kwargs: Any,
    ):
        self.colsX = colsX
        self.colY = colY
        self.dataset_path = dataset_path
        assert dataset_path is not None
        self.window_size = window_size
        self.expand_classes = expand_classes

        if window_size is not None:
            shuffle_data = False

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
            **kwargs,
        )

    def _get_csv_path(self):
        sha256 = hashlib.sha256(
            str(self.dataset_path).encode(), usedforsecurity=False
        ).hexdigest()
        return self.root / f"{sha256}.csv"

    def download_dataset_fun(self):
        self.root.mkdir(exist_ok=True, parents=True)
        ResourceManager().get_resource(self.dataset_path, self._get_csv_path())

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
                for i in range(max(0, len(rowsX) - self.window_size + 1))
            ]
            self.dataY = rowsY[self.window_size - 1 :]
        # Making sure shapes of mean and std match shape of inputs (so that
        # it can be easily subtracted/added to inputs for standardization).
        # We also want normal numpy arrays, not whatever `polars` lib returns.
        self.mean = dfX.mean().to_numpy()
        self.std = dfX.std().to_numpy()
        self.median = dfX.median().to_numpy()
        percentile_25 = dfX.quantile(0.25)
        percentile_75 = dfX.quantile(0.75)
        self.iqr = (percentile_75 - percentile_25).to_numpy()
        self.classnames = list(map(str, dfY.unique().sort()))

        self.min = np.asarray(list(dfX.min())).transpose()
        self.max = np.asarray(list(dfX.max())).transpose()
        self.num_features = dfX.shape[1]

    def train_test_split_representations(
        self,
        test_fraction=None,
        val_fraction=None,
        seed=None,
        stratify=True,
        append_index=False,
    ):
        if self.window_size is None:
            return super().train_test_split_representations(
                test_fraction, val_fraction, seed, stratify, append_index
            )

        split = list(
            super().train_test_split_representations(
                test_fraction,
                val_fraction,
                seed,
                stratify=False,
                append_index=True,
            )
        )
        gap = self.window_size - 1
        if gap and split[0] and split[1]:
            for i in (0, 2, 6 if len(split) == 9 else 4):
                split[i] = split[i][:-gap]
            self.dataXtrain, self.dataYtrain = split[0], split[2]
            if len(split) == 9 and split[4]:
                for i in (4, 5, 8):
                    split[i] = split[i][:-gap]
                self.dataXval, self.dataYval = split[4], split[5]
        return tuple(
            split if append_index else split[: 6 if len(split) == 9 else 4]
        )

    def prepare_output_samples(self, samples):
        if self.expand_classes:
            return [np.eye(len(self.classnames))[samples]]
        return [samples]

    def evaluate(self, predictions, truth):
        measurements = Measurements()
        ts = np.asarray(truth[0])
        ps = np.asarray(predictions[0])
        if ts.ndim > 1:
            ts = np.argmax(ts, axis=-1)
        if ps.ndim > 1:
            ps = np.argmax(ps, axis=-1)
        ts = ts.reshape(-1).astype(np.int32)
        ps = ps.reshape(-1).astype(np.int32)
        confusion_matrix = metrics.confusion_matrix(
            ts, ps, labels=range(len(self.classnames))
        )
        measurements.add_measurement(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((len(self.classnames), len(self.classnames))),
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

    def get_robust_scaler_params(self):
        return self.iqr, self.median

    def get_min_max_scaler_params(self):
        return self.max - self.min, self.min

    def get_class_names(self):
        return self.classnames
