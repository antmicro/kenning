# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A wrapper for the TVM runtime of the YOLOv3 algorithm.

This ModelWrapper handles specific outputs to the YOLOv3
model compiled directly using TVM framework.
Except for the actual model output, there is
additional metadata from the CFG model definition stored in the outputs
from TVM-compiled model.
"""
import sys
if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files

from kenning.modelwrappers.detectors.yolo_wrapper import YOLOWrapper
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.resources.models import detection


class TVMDarknetCOCOYOLOV3(YOLOWrapper):

    pretrained_modelpath = files(detection) / 'yolov3.cfg'
    default_dataset = COCODataset2017
    arguments_structure = {}

    @classmethod
    def _get_io_specification(cls, keyparams):
        return {
            'input': [{'name': 'data', 'shape': (1, 3, keyparams['width'], keyparams['height']), 'dtype': 'float32'}],  # noqa: E501
            'output': []
        }

    def get_output_formats(self):
        return ['darknet']

    def get_framework_and_version(self):
        return ('darknet', 'alexeyab')
