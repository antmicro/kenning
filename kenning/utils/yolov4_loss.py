import torch
from typing import List

from kenning.modelwrappers.detectors.yolov4 import ONNXYOLOV4


class YOLOv4Loss(object):
    def __init__(self):
        self.model_wrapper = ONNXYOLOV4(
            'kenning/resources/models/detection/yolov4.onnx', None)

    def __call__(self, output: List[torch.Tensor], target: List):
        return self.model_wrapper.loss_torch(output, target)
