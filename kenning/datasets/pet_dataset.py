# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Oxford-IIIT Pet Dataset wrapper.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import Resources, extract_tar


class PetDataset(Dataset):
    """
    The Oxford-IIIT Pet Dataset.

    Omkar M Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar

    It is a classification dataset with 37 classes, where 12 classes represent
    cat breeds, and the remaining 25 classes represent dog breeds.

    It is a seemingly balanced dataset breed-wise, with around 200 images
    examples per class.

    There are 7349 images in total, where 2371 images are cat images, and the
    4978 images are dog images.

    *License*: Creative Commons Attribution-ShareAlike 4.0 International
    License.

    *Page*: `Pet Dataset site <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    The images can be either classified by species (2 classes)
    or breeds (37 classes).

    The affinity of images to classes is taken from annotations, but the class
    IDs are starting from 0 instead of 1, as in the annotations.
    """

    classification_types = ["species", "breeds"]

    resources = Resources(
        {
            "images": "kenning:///datasets/pet_dataset/images.tar.gz",
            "annotations": "kenning:///datasets/pet_dataset/annotations.tar.gz",
        }
    )

    arguments_structure = {
        "classify_by": {
            "argparse_name": "--classify-by",
            "description": "Determines if classification should be performed by species or by breeds",  # noqa: E501
            "default": "breeds",
            "enum": classification_types,
        },
        "image_memory_layout": {
            "argparse_name": "--image-memory-layout",
            "description": "Determines if images should be delivered in NHWC or NCHW format",  # noqa: E501
            "default": "NHWC",
            "enum": ["NHWC", "NCHW"],
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
        classify_by: str = "breeds",
        image_memory_layout: str = "NHWC",
        standardize: bool = True,
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
        classify_by : str
            Determines what should be the object of classification.
            The valid values are "species" and "breeds".
        image_memory_layout : str
            Tells if the images should be delivered in NCHW or NHWC format.
            The default format is NHWC.
        standardize : bool
            Standardize the given input samples.
            Should be set to False when using `compute_input_mean_std`.
        """
        assert (
            classify_by in self.classification_types
        ), f"Invalid {classify_by}, should be {self.classification_types}"
        assert image_memory_layout in [
            "NHWC",
            "NCHW",
        ], f"Unsupported layout {image_memory_layout}"
        self.classify_by = classify_by
        self.numclasses = None
        self.classnames = dict()
        self.standardize = standardize
        if standardize:
            self.mean, self.std = self.get_input_mean_std()
        self.image_memory_layout = image_memory_layout
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
        self.root.mkdir(parents=True, exist_ok=True)
        extract_tar(self.root, self.resources["images"])
        extract_tar(self.root, self.resources["annotations"])

    def prepare(self):
        with open(self.root / "annotations" / "list.txt", "r") as datadesc:
            for line in datadesc:
                if line.startswith("#"):
                    continue
                fields = line.split(" ")
                self.dataX.append(
                    str(self.root / "images" / (fields[0] + ".jpg"))
                )
                if self.classify_by == "species":
                    self.dataY.append(int(fields[2]) - 1)
                else:
                    self.dataY.append(int(fields[1]) - 1)
                    clsname = fields[0].rsplit("_", 1)[0]
                    if self.dataY[-1] not in self.classnames:
                        self.classnames[self.dataY[-1]] = clsname
                    assert self.classnames[self.dataY[-1]] == clsname
            if self.classify_by == "species":
                self.numclasses = 2
                assert min(self.dataY) == 0
                assert max(self.dataX) == self.numclasses - 1
                self.classnames = {0: "cat", 1: "dog"}
            else:
                self.numclasses = len(self.classnames)

    def prepare_input_samples(self, samples: List[str]) -> List[np.ndarray]:
        result = []
        for sample in samples:
            img = Image.open(sample)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            npimg = np.array(img).astype(np.float32) / 255.0
            if self.standardize:
                npimg = (npimg - self.mean) / self.std
            if self.image_memory_layout == "NCHW":
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return [np.array(result)]

    def prepare_output_samples(self, samples: List[int]) -> List[np.ndarray]:
        return [np.eye(self.numclasses)[samples]]

    def evaluate(self, predictions, truth):
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))
        top_5_count = 0
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
        for prediction, label in zip(predictions, truth):
            # some compilers/frameworks wrap the output in an
            # additional dimension
            while len(prediction) != self.numclasses:
                assert len(prediction) == 1
                prediction = prediction[0]
            confusion_matrix[np.argmax(label), np.argmax(prediction)] += 1
            top_5_count += (
                1
                if np.argmax(label) in np.argsort(prediction)[::-1][:5]
                else 0
            )
        measurements = Measurements()
        measurements.accumulate(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((self.numclasses, self.numclasses)),
        )
        measurements.accumulate("top_5_count", top_5_count, lambda: 0)
        measurements.accumulate("total", len(predictions), lambda: 0)
        return measurements

    def compute_input_mean_std(self) -> Tuple[Any, Any]:
        """
        Computes mean and std values for a given dataset.

        The input standardization values for a given model are computed based
        on a train dataset.

        Returns
        -------
        Tuple[Any, Any]
            The standardization values for a given train dataset. Tuple of two
            variables describing mean and std values.
        """
        count = 0
        mean = np.zeros((3))
        std = np.zeros((3))
        for X, _ in iter(self):
            for img in X:
                mean += np.mean(img, axis=(0, 1))
                std += np.std(img, axis=(0, 1))
                count += 1
        mean /= count
        std /= count
        return mean, std

    def get_input_mean_std(self):
        # The first mean/std are computed based on Pet Dataset
        # However, it is recommended to use Imagenet-based mean and std values
        # return np.array([0.48136492, 0.44937421, 0.39576963]), np.array([0.22781384, 0.22496867, 0.22693157])  # noqa: E501
        return np.array([0.485, 0.456, 0.406], dtype="float32"), np.array(
            [0.229, 0.224, 0.225], dtype="float32"
        )

    def get_class_names(self):
        return [val for val in self.classnames.values()]
