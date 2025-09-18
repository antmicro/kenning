# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Visual Wake Words dataset.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.measurements import Measurements
from kenning.utils.resource_manager import Resources, extract_zip


class VisualWakeWordsDataset(Dataset):
    """
    The Visual Wake Words Dataset.

    It is a classification dataset for predicting whether some object is
    present or not in the image. There are 2 classes - 0 for images without
    selected object and 1 for images with it.

    This dataset is derived from COCO Dataset and the classes are determined
    based on annotations. If there is bounding box for selected object and its
    area is above selected threshold then such image class is set to 1. In
    other case it is 0.

    The selected object can be any of the object from COCO Dataset's
    categories.

    *Page*: `Visual Wake Words Dataset site <https://arxiv.org/abs/1906.05721>`
    """

    resources = Resources(
        {
            "train2017": {
                "images": "http://images.cocodataset.org/zips/train2017.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            },
            "val2017": {
                "images": "http://images.cocodataset.org/zips/val2017.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            },
        }
    )

    arguments_structure = {
        "dataset_type": {
            "argparse_name": "--dataset-type",
            "description": "Type of dataset to download and use",
            "default": "val2017",
            "enum": list(set([key[0] for key in resources.keys()])),
        },
        "objects_class": {
            "argparse_name": "--objects_class",
            "description": "Name of objects class to be filtered",
            "default": "person",
            "type": str,
        },
        "area_threshold": {
            "argparse_name": "--area-threshold",
            "description": "Threshold of fraction of image area below which "
            "objects are filtered",
            "default": 0.005,
            "type": float,
        },
        "image_memory_layout": {
            "argparse_name": "--image-memory-layout",
            "description": "Determines if images should be delivered in NHWC "
            "or NCHW format",
            "default": "NHWC",
            "enum": ["NHWC", "NCHW"],
        },
        "image_width": {
            "description": "Width of the input images",
            "type": int,
            "default": 416,
        },
        "image_height": {
            "description": "Height of the input images",
            "type": int,
            "default": 416,
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
        dataset_type: str = "val2017",
        objects_class: str = "person",
        area_threshold: float = 0.005,
        image_memory_layout: str = "NHWC",
        image_width: int = 416,
        image_height: int = 416,
    ):
        assert image_memory_layout in ["NHWC", "NCHW"]
        self.dataset_type = dataset_type
        self.objects_class = objects_class
        self.area_threshold = area_threshold
        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height
        self.numclasses = 2
        self.classnames = ["not-person", "person"]
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

    def get_class_names(self):
        return self.classnames

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        extract_zip(self.root, self.resources[self.dataset_type, "images"])
        extract_zip(
            self.root, self.resources[self.dataset_type, "annotations"]
        )

    def prepare(self):
        annotationspath = (
            self.root / f"annotations/instances_{self.dataset_type}.json"
        )
        self.coco = COCO(annotationspath)
        self.classmap = {}

        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]["name"]

        cocokeys = list(self.coco.imgs.keys())
        keystoimgs = dict()
        filepathtoarea = dict()

        for key, imgdata in zip(cocokeys, self.coco.loadImgs(cocokeys)):
            filepath = str(
                self.root / self.dataset_type / imgdata["file_name"]
            )
            self.dataX.append(filepath)
            keystoimgs[key] = filepath
            filepathtoarea[filepath] = imgdata["height"] * imgdata["width"]

        # compute sum of bounding boxes areas
        bbox_areas = defaultdict(lambda: 0)
        for _, anndata in self.coco.anns.items():
            img_file_path = keystoimgs[anndata["image_id"]]
            bbox = anndata["bbox"]
            bbox_area = bbox[2] * bbox[3]
            if self.objects_class == self.classmap[anndata["category_id"]]:
                bbox_areas[img_file_path] += bbox_area

        # compare sum of all bounding boxes areas to the threshold
        for img_file_path in self.dataX:
            img_area = filepathtoarea[img_file_path]
            self.dataY.append(
                int(bbox_areas[img_file_path] > self.area_threshold * img_area)
            )

    def prepare_input_samples(self, samples):
        result = []
        for imgpath in samples:
            img = cv2.imread(str(imgpath))
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.astype(np.float32) / 255.0
            if self.image_memory_layout == "NCHW":
                img = np.transpose(img, (2, 0, 1))
            result.append(img)

        return [np.array(result)]

    def evaluate(self, predictions, truth):
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))

        for prediction, label in zip(predictions, truth):
            confusion_matrix[label, np.argmax(prediction)] += 1

        measurements = Measurements()
        measurements.accumulate(
            "eval_confusion_matrix",
            confusion_matrix,
            lambda: np.zeros((self.numclasses, self.numclasses)),
        )
        measurements.accumulate("total", len(predictions), lambda: 0)
        return measurements

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotSupportedError
