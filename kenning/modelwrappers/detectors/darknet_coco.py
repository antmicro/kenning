"""
A wrapper for the TVM runtime of the YOLOv3 algorithm.

This ModelWrapper handles specific outputs to the YOLOv3
model compiled directly using TVM framework.
Except for the actual model output, there is
additional metadata from the CFG model definition stored in the outputs
from TVM-compiled model.
"""

from kenning.modelwrappers.detectors.yolo_wrapper import YOLOWrapper


class TVMDarknetCOCOYOLOV3(YOLOWrapper):

    arguments_structure = {}

    # TODO: Fill the output, probably move it from yolov4 to YOLOWrapper
    def get_io_specification_from_model(self):
        return {
            'input': [{'name': 'data', 'shape': (1, 3, self.keyparams['width'], self.keyparams['height']), 'dtype': 'float32'}],  # noqa: E501
            'output': []
        }

    def get_output_formats(self):
        return ['darknet']

    def get_framework_and_version(self):
        return ('darknet', 'alexeyab')
