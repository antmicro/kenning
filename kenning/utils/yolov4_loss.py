# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for YOLOv4 loss function.
"""

import torch
from typing import List

from kenning.modelwrappers.object_detection.yolov4 import ONNXYOLOV4
from kenning.utils.resource_manager import ResourceURI


class YOLOv4Loss(object):
    """
    A wrapper for YOLOv4 loss function.
    """

    def __init__(self):
        self.model_wrapper = ONNXYOLOV4(
            ResourceURI("kenning:///models/object_detection/yolov4.onnx"), None
        )

    def __call__(self, output: List[torch.Tensor], target: List):
        return self.model_wrapper.loss_torch(output, target)
