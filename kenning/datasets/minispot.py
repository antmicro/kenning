# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Dataset wrapper for the minispot dataset.
"""

from pathlib import Path
from typing import List, Optional, Union

from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.utils.resource_manager import ResourceURI

MINISPOT_URI = (
    "https://dl.antmicro.com/kenning/"
    "datasets/anomaly_detection/minispot.csv"
)


class MinispotDataset(AnomalyDetectionDataset):
    """
    The Minispot Dataset.
    """

    arguments_structure = {
        "csv_file": {
            "argparse_name": "--csv-file",
            "description": "Location of the minispot CSV file",
            "type": ResourceURI,
            "required": False,
        },
    }

    def __init__(
        self,
        root: Path,
        csv_file: Union[str, ResourceURI] = MINISPOT_URI,
        batch_size: int = 1,
        download_dataset: bool = True,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        dataset_percentage: float = 1,
        window_size: int = 8,
        gather_predictions: bool = True,
        label_type: str = "last_timestep",
        drop_columns: List[str] = [],
        timestamp_column=None,
    ):
        super().__init__(
            root=root,
            csv_file=csv_file,
            batch_size=batch_size,
            download_dataset=download_dataset,
            force_download_dataset=force_download_dataset,
            external_calibration_dataset=external_calibration_dataset,
            split_fraction_test=split_fraction_test,
            split_fraction_val=split_fraction_val,
            split_seed=split_seed,
            dataset_percentage=dataset_percentage,
            window_size=window_size,
            gather_predictions=gather_predictions,
            label_type=label_type,
            drop_columns=drop_columns,
            timestamp_column=timestamp_column,
        )
