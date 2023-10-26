# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

from kenning.utils.resource_manager import Resources, extract_zip

from pycocotools.coco import COCO

from kenning.datasets.helpers.detection_and_segmentation import (
    DetectObject,
    ObjectDetectionSegmentationDataset,
)


class COCODataset2017(ObjectDetectionSegmentationDataset):
    """
    The COCO Dataset 2017.

    https://cocodataset.org

    COCO is a large-scale object detection, segmentation and captioning
    dataset.

    For object detection, it provides 80 categories of objects.

    *License*: Creative Commons Attribution 4.0 License.

    *NOTE*: the license does not include the images, only annotations.

    *Page*: `COCO Dataset site <https://cocodataset.org>`_.
    """

    resources = Resources(
        {
            "train2017": {
                "images": "http://images.cocodataset.org/zips/train2017.zip",
                "object_detection": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # noqa: E501
            },
            "val2017": {
                "images": "http://images.cocodataset.org/zips/val2017.zip",
                "object_detection": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # noqa: E501
            },
        }
    )

    arguments_structure = {
        "task": {
            "argparse_name": "--task",
            "description": "The task type",
            "default": "object_detection",
            "enum": list(set([key[1] for key in resources.keys()])),
        },
        "dataset_type": {
            "argparse_name": "--dataset-type",
            "description": "Type of dataset to download and use",
            "default": "val2017",
            "enum": list(set([key[0] for key in resources.keys()])),
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
        task: str = "object_detection",
        dataset_type: str = "val2017",
        image_memory_layout: str = "NCHW",
        show_on_eval: bool = False,
        image_width: int = 416,
        image_height: int = 416,
    ):
        assert image_memory_layout in ["NHWC", "NCHW"]
        self.numclasses = 80
        self.dataset_type = dataset_type
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
            task,
            image_memory_layout,
            show_on_eval,
            image_width,
            image_height,
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        extract_zip(self.root, self.resources[self.dataset_type, "images"])
        extract_zip(self.root, self.resources[self.dataset_type, self.task])

    def prepare(self):
        annotationspath = (
            self.root / f"annotations/instances_{self.dataset_type}.json"
        )  # noqa: E501
        self.coco = COCO(annotationspath)
        self.classmap = {}
        self.classnames = []

        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]["name"]
            self.classnames.append(self.coco.cats[classid]["name"])

        cocokeys = list(self.coco.imgs.keys())
        self.keystoimgs = dict()
        self.imgstokeys = dict()

        for key, imgdata in zip(cocokeys, self.coco.loadImgs(cocokeys)):
            filepath = str(
                self.root / self.dataset_type / imgdata["file_name"]
            )
            self.dataX.append(filepath)
            self.keystoimgs[key] = filepath
            self.imgstokeys[filepath] = key

        annotations = defaultdict(list)
        for annkey, anndata in self.coco.anns.items():
            bbox = anndata["bbox"]
            width = self.coco.imgs[anndata["image_id"]]["width"]
            height = self.coco.imgs[anndata["image_id"]]["height"]
            annotations[self.keystoimgs[anndata["image_id"]]].append(
                DetectObject(
                    clsname=self.classmap[anndata["category_id"]],
                    xmin=bbox[0] / width,
                    ymin=bbox[1] / height,
                    xmax=(bbox[0] + bbox[2]) / width,
                    ymax=(bbox[1] + bbox[3]) / height,
                    score=1.0,
                    iscrowd=anndata["iscrowd"] == 1,
                )
            )

        for inputid in self.dataX:
            self.dataY.append(annotations[inputid])

    def prepare_input_samples(self, samples):
        result = []
        for imgpath in samples:
            img = cv2.imread(str(imgpath))
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.image_memory_layout == "NCHW":
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return result

    def get_class_names(self):
        return self.classnames

    def evaluate(self, predictions, truth):
        measurements = super().evaluate(predictions, truth)
        currindex = self._dataindex - len(predictions)
        for pred, groundtruth in zip(predictions, truth):
            for p in pred:
                cocoid = self.imgstokeys[self.dataX[currindex]]
                width = self.coco.imgs[cocoid]["width"]
                height = self.coco.imgs[cocoid]["height"]
                xmin = max(min(p.xmin * width, width), 0)
                xmax = max(min(p.xmax * width, width), 0)
                ymin = max(min(p.ymin * height, height), 0)
                ymax = max(min(p.ymax * height, height), 0)
                w = xmax - xmin
                h = ymax - ymin
                measurements.add_measurement(
                    "predictions",
                    [
                        {
                            "image_name": self.imgstokeys[
                                self.dataX[currindex]
                            ],
                            "category": p.clsname,
                            "bbox": [xmin, ymin, w, h],
                            "score": p.score,
                        }
                    ],
                )
        return measurements
