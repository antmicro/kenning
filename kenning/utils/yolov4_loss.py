# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for YOLOv4 loss function.
"""

from typing import List

import torch

from kenning.core.dataset import Dataset
from kenning.datasets.open_images_dataset import OpenImagesDatasetV6
from kenning.modelwrappers.object_detection.yolov4 import ONNXYOLOV4
from kenning.utils.resource_manager import ResourceURI


class YOLOv4Loss(object):
    """
    A wrapper for YOLOv4 loss function.
    """

    def __init__(self, dataset: Dataset = None):
        classes = "coco"
        if dataset and isinstance(dataset, OpenImagesDatasetV6):
            classes = dataset.classes_path
        self.model_wrapper = ONNXYOLOV4(
            ResourceURI("kenning:///models/detection/yolov4.onnx"),
            dataset,
            class_names=classes,
        )

    def __call__(self, output: List[torch.Tensor], target: List):
        return self.model_wrapper.loss_torch(output, target)
