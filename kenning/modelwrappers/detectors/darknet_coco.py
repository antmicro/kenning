"""
A wrapper for the TVM runtime of the YOLOv3 algorithm.

This ModelWrapper handles specific outputs to the YOLOv3
model compiled directly using TVM framework.
Except for the actual model output, there is
additional metadata from the CFG model definition stored in the outputs
from TVM-compiled model.
"""

import numpy as np

from kenning.modelwrappers.detectors.yolo_wrapper import YOLOWrapper


class TVMDarknetCOCOYOLOV3(YOLOWrapper):

    def postprocess_outputs(self, y):
        # YOLOv3 has three stages of outputs
        # each one contains:
        # - real output
        # - masks
        # - biases

        # TVM-based model output provides 12 arrays
        # Those are subdivided into three groups containing
        # - actual YOLOv3 output
        # - masks IDs
        # - anchors
        # - 6 integers holding number of dects per cluster, actual output
        #   number of channels, actual output height and width, number of
        #   classes and unused parameter

        # iterate over each group
        lastid = 0
        outputs = []
        for i in range(3):
            # first extract the actual output
            # each output layer shape follows formula:
            # (BS, B * (4 + 1 + C), w / (8 * (i + 1)), h / (8 * (i + 1)))
            # BS is the batch size
            # w, h are width and height of the input image
            # the resolution is reduced over the network, and is 8 times
            # smaller in each dimension for each output
            # the "pixels" in the outputs are responsible for the chunks of
            # image - in the first output each pixel is responsible for 8x8
            # squares of input image, the second output covers objects from
            # 16x16 chunks etc.
            # Each "pixel" can predict up to B bounding boxes.
            # Each bounding box is described by its 4 coordinates,
            # objectness prediction and per-class predictions
            outshape = (
                self.batch_size,
                len(self.perlayerparams['mask'][i]),
                4 + 1 + self.numclasses,
                self.keyparams['width'] // (8 * 2 ** i),
                self.keyparams['height'] // (8 * 2 ** i)
            )

            outputs.append(
                y[lastid:(lastid + np.prod(outshape))].reshape(outshape)
            )

            # drop additional info provided in the TVM output
            # since it's all 4-bytes values, ignore the insides
            lastid += (
                np.prod(outshape)
                + len(self.perlayerparams['mask'][i])
                + len(self.perlayerparams['anchors'][i])
                + 6  # layer parameters
            )

        # change the dimensions so the output format is
        # batches layerouts dets params width height
        perbatchoutputs = []
        for i in range(outputs[0].shape[0]):
            perbatchoutputs.append([
                outputs[0][i],
                outputs[1][i],
                outputs[2][i]
            ])
        result = []
        # parse the combined outputs for each image in batch, and return result
        for out in perbatchoutputs:
            result.append(self.parse_outputs(out))

        return result

    def get_io_specs(self):
        return {
            'input': [{'name': 'data', 'shape': (1, 3, self.keyparams['width'], self.keyparams['height']), 'dtype': 'float32'}],  # noqa: E501
            'output': []
        }

    def get_output_formats(self):
        return ['darknet']

    def get_framework_and_version(self):
        return ('darknet', 'alexeyab')
