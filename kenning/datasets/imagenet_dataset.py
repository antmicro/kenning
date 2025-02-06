# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The ImageNet 2012 wrapper.
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from kenning.core.dataset import CannotDownloadDatasetError, Dataset
from kenning.core.measurements import Measurements


class ImageNetDataset(Dataset):
    """
    The ImageNet Largest Scale Visual Recognition Challenge 2012.

    This is a classification dataset used in ImageNet competition.
    The purpose of the competition is to estimate the content of photographs
    for such tasks as automatic annotation of objects in images.

    It has 1000 classes, representing various real-life objects (animals,
    electronics, vehicles, plants, ...).

    The training dataset consists of 1.2 million images, the validation dataset
    consists of 50000 images, and the test dataset consists of 100000 images.

    *License*:
        `Download terms of access <https://image-net.org/download.php>`_.

    *Page*:
        `ImageNet site <https://image-net.org/index.php>`_.
    """

    arguments_structure = {
        "image_memory_layout": {
            "argparse_name": "--image-memory-layout",
            "description": "Determines if images should be delivered in NHWC or NCHW format",  # noqa: E501
            "default": "NHWC",
            "enum": ["NHWC", "NCHW"],
        },
        "preprocess_type": {
            "argparse_name": "--preprocess_type",
            "description": "Determines the preprocessing type. See ImageNetDataset documentation for more details",  # noqa: E501
            "default": "caffe",
            "enum": ["caffe", "tf", "torch", "none"],
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
        dataset_percentage: float = 1,
        image_memory_layout: str = "NHWC",
        preprocess_type: str = "caffe",
    ):
        """
        Prepares all structures and data required for providing data samples.

        The object of classification can be either breeds (37 classes) or
        species (2 classes).

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
        image_memory_layout : str
            Tells if the images should be delivered in NCHW or NHWC format.
            The default format is NHWC.
        preprocess_type : str
            Tells how data should be preprocessed.
            There are three modes:

                * tf - will convert values to range -1..1
                * torch - will apply torch standardization
                * caffe - will convert RGB to BGR and apply standardization
                * none - data is passed as is from file, without conversions
        """
        assert image_memory_layout in ["NHWC", "NCHW"]
        assert preprocess_type in ["caffe", "torch", "tf", "none"]
        self.numclasses = None
        self.classnames = dict()
        self.image_memory_layout = image_memory_layout
        self.preprocess_type = preprocess_type
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

    def download_dataset_fun(self):
        raise CannotDownloadDatasetError(
            "ImageNet dataset needs to be downloaded manually.\n"
            "The images need to be stored in <root>/images directory.\n"
            "The labels need to be provided in the JSON format.\n"
            "The JSON file should be saved in <root>/labels.json.\n"
            "JSON should be a dictionary mapping names to labels, i.e.\n"
            "{\n"
            '    "file1.png": 1,\n'
            '    "file2.png": 853,\n'
            "    ...\n"
            "}\n"
        )

    def prepare(self):
        with open(self.root / "labels.json", "r") as groundtruthdesc:
            groundtruth = json.load(groundtruthdesc)
            for imagename, label in groundtruth.items():
                self.dataX.append(str(self.root / "images" / imagename))
                self.dataY.append(int(label))
            self.classnames = [f"{i}" for i in range(1000)]
            self.numclasses = 1000

    def prepare_input_samples(self, samples: List[str]) -> List[np.ndarray]:
        result = []
        for sample in samples:
            img = Image.open(sample)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            npimg = np.array(img).astype(np.float32)
            if self.preprocess_type == "caffe":
                # convert to BGR
                npimg = npimg[:, :, ::-1]
            if self.preprocess_type == "tf":
                npimg /= 127.5
                npimg -= 1.0
            elif self.preprocess_type == "torch":
                npimg /= 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                npimg = (npimg - mean) / std
            elif self.preprocess_type == "caffe":
                mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
                npimg -= mean
            if self.image_memory_layout == "NCHW":
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return [np.array(result)]

    def prepare_output_samples(self, samples: List[int]) -> List[np.ndarray]:
        return [np.eye(self.numclasses)[samples]]

    def evaluate(self, predictions, truth):
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))
        top_5_count = 0
        currindex = self._dataindex - len(predictions)
        top_5_results = []
        for prediction, label in zip(predictions, truth):
            confusion_matrix[np.argmax(label), np.argmax(prediction)] += 1
            top_5 = np.argsort(prediction)[::-1][:5]
            top_5_count += 1 if np.argmax(label) in top_5 else 0
            top_5_results.append(
                {
                    Path(
                        self.dataX[self._indices[currindex]]
                    ).name: top_5.tolist()
                }
            )
            currindex += 1
        measurements = Measurements()
        measurements.accumulate(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((self.numclasses, self.numclasses)),
        )
        measurements.add_measurement("top_5", top_5_results)
        measurements.accumulate("top_5_count", top_5_count, lambda: 0)
        measurements.accumulate("total", len(predictions), lambda: 0)
        return measurements

    def get_class_names(self):
        return self.classnames

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotImplementedError
