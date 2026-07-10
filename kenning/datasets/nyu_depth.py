# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The NYU Depth Dataset V2 wrapper.
"""
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.measurements import Measurements
from kenning.datasets.helpers.depth_estimation import calculate_partial_metrics
from kenning.utils.resource_manager import Resources


class NYUDepthDatasetV2(Dataset):
    """
    Dataset wrapper for NYU Depth Dataset V2.
    It utilizes the "Labeled" variant of the dataset.

    https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

    It is a dataset of 1449 images that for each pixel in an image provides RGB
    values as features and depth in meters as labels.

    The original dataset also provides segmentation labels, but they are
    not exposed by this class.

    Citation:

    @inproceedings{Silberman:ECCV12,
        author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
        title     = {Indoor Segmentation and Support Inference from RGBD Images},
        booktitle = {ECCV},
        year      = {2012}
    }
    """  # noqa: E501

    resources = Resources(
        {
            "dataset": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
        }
    )

    arguments_structure = {
        "image_memory_layout": {
            "description": "Determines if images should be delivered in NHWC or NCHW format",  # noqa: E501
            "default": "NCHW",
            "enum": ["NHWC", "NCHW"],
        },
        "image_width": {
            "description": "Width of the input images",
            "type": int,
            "default": 640,
        },
        "image_height": {
            "description": "Height of the input images",
            "type": int,
            "default": 480,
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
        image_memory_layout: str = "NCHW",
        image_width: int = 640,
        image_height: int = 480,
    ):
        assert image_memory_layout in ["NHWC", "NCHW"]

        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height

        self.dataset_path = root / "dataset.mat"

        super().__init__(
            root=root,
            batch_size=batch_size,
            download_dataset=download_dataset,
            force_download_dataset=force_download_dataset,
            external_calibration_dataset=external_calibration_dataset,
            split_fraction_test=split_fraction_test,
            split_fraction_val=split_fraction_val,
            split_seed=split_seed,
            dataset_percentage=dataset_percentage,
            shuffle_data=shuffle_data,
        )

    def prepare_input_samples(self, samples: List) -> List:
        import cv2

        original_images = self._images[samples]

        result = np.empty(
            (
                original_images.shape[0],
                3,
                self.image_height,
                self.image_width,
            ),
            dtype=np.float32,
        )

        for idx, original_image in enumerate(original_images):
            for channel_idx in range(3):
                cv2.resize(
                    src=original_image[channel_idx],
                    dsize=(
                        self.image_width,
                        self.image_height,
                    ),
                    dst=result[idx, channel_idx],
                )

        if self.image_memory_layout == "NHWC":
            result = result.transpose((0, 2, 3, 1))

        return [result]

    def prepare_output_samples(self, samples: List) -> List:
        import cv2

        original_depths = self._depths[samples]

        result = np.empty(
            (
                original_depths.shape[0],
                self.image_height,
                self.image_width,
            ),
            dtype=np.float32,
        )

        for idx, original_depth in enumerate(original_depths):
            cv2.resize(
                src=original_depth,
                dsize=(
                    self.image_width,
                    self.image_height,
                ),
                dst=result[idx],
            )

        return [result]

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)

        dataset_path = self.root / "dataset.mat"

        shutil.copy(self.resources["dataset"], dataset_path)

    def train_test_split_representations(
        self,
        test_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        append_index: bool = False,
    ) -> Tuple[List, ...]:
        # This is not a classification dataset, so stratify=True
        # will throw errors as this doesn't make sense

        return super().train_test_split_representations(
            test_fraction, val_fraction, seed, False, append_index
        )

    def prepare(self):
        import h5py

        with h5py.File(self.dataset_path) as f:
            self._images = np.asarray(f["images"], dtype=np.float32)
            self._depths = np.asarray(f["depths"], dtype=np.float32)

        # Dataset uses NCWH instead of NCHW
        self._images = self._images.transpose((0, 1, 3, 2))
        self._depths = self._depths.transpose((0, 2, 1))

        self._images /= 255

        self.dataX = np.arange(self._images.shape[0])
        self.dataY = np.arange(self._depths.shape[0])

    def evaluate(self, predictions: List, truth: List) -> Measurements:
        # If the images resolution or dataset count are too big, storing
        # raw predictions can easily fill the whole memory, so we must
        # calculate the partial metrics here

        unpacked_preds = predictions[0]

        if not isinstance(unpacked_preds, np.ndarray):
            unpacked_preds = np.asarray(unpacked_preds)

        unpacked_truth = truth[0]

        return calculate_partial_metrics(unpacked_preds, unpacked_truth)

    def get_class_names(self):
        return NotSupportedError(
            "This dataset is not a classification dataset,"
            " it does not support class names."
        )

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        return (
            np.array([0.481, 0.411, 0.392]),
            np.array([0.289, 0.296, 0.309]),
        )
