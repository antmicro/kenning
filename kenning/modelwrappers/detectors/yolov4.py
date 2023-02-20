# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ModelWrapper for the YOLOv4 model generated from darknet repository using:

https://github.com/Tianxiaomo/pytorch-YOLOv4

To create an ONNX model from darknet yolov4.cfg and yolov4.weights files
(check https://github.com/AlexeyAB/darknet for those files), follow
repositories' README (Darknet2ONNX section).

After this, to remove the embedded processing of outputs, run in Python shell::

    from kenning.modelwrappers.detectors.yolov4 import \
            yolov4_remove_postprocessing


    yolov4_remove_postprocessing('<input_onnx_path>', '<output_onnx_path>')
"""

import onnx
import numpy as np
from pathlib import Path
import shutil
from typing import List
import sys
if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files

from kenning.modelwrappers.detectors.yolo_wrapper import YOLOWrapper
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.resources.models import detection


def yolov4_remove_postprocessing(
        inputpath: Path,
        outputpath: Path,
        input_names: List[str] = ['input'],
        output_names: List[str] = ['output', 'output.3', 'output.7']):
    """
    Extracts the actual model from the Darknet2ONNX output.

    Darknet2ONNX tool (https://github.com/Tianxiaomo/pytorch-YOLOv4) creates
    an ONNX file that contains a YOLOv4 model and postprocessing steps to
    extract bounding boxes and scores.

    To keep the model simple, this method extracts the actual model
    and removes the postprocessing.

    Parameters
    ----------
    inputpath: Path
        Path to the ONNX file containing model with postprocessing
    outputpath: Path
        Path to the ONNX output file containing pure model
    """
    onnx.utils.extract_model(
        str(inputpath),
        str(outputpath),
        input_names,
        output_names
    )


class ONNXYOLOV4(YOLOWrapper):

    pretrained_modelpath = files(detection) / 'yolov4.onnx'
    default_dataset = COCODataset2017
    arguments_structure = {}

    def postprocess_outputs(self, y):
        # YOLOv4, as YOLOv3, has three outputs for three stages of computing.
        # Each output layer has information about bounding boxes, scores and
        # classes in a grid.

        # iterate over each output
        lastid = 0
        outputs = []
        for i in range(3):
            # each output layer shape follows formula:
            # (BS, B, 4 + 1 + C, w / (8 * (i + 1)), h / (8 * (i + 1)))
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

            # Extract the output and reshape it to match actual form
            outarr = \
                y[lastid:(lastid + np.prod(outshape))].reshape(outshape).copy()
            # x and y offsets need to be passed through sigmoid function
            # NOTE: w and h are NOT passed through sigmoid function - they are
            # later computed in parse_outputs methods using anchors and mask
            # parameters.
            outarr[:, :, :2, :, :] = \
                1 / (1 + np.exp(-outarr[:, :, :2, :, :]))
            # objectness and classes are also passed through sigmoid function
            outarr[:, :, 4:, :, :] = \
                1 / (1 + np.exp(-outarr[:, :, 4:, :, :]))
            outputs.append(
                outarr
            )

            lastid += np.prod(outshape)

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

    # NOTE: In postprocess_outputs function the second output layer `output.3`
    # of size 255 is split into two layers of size (4 + 1 + C) and B,
    # where C is a class vector and B is the number of detectable object
    # in a pixel.
    def get_io_specification_from_model(self):
        return {
            'input': [{'name': 'input', 'shape': (1, 3, self.keyparams['width'], self.keyparams['height']), 'dtype': 'float32'}],  # noqa: E501
            'output': [
                {'name': 'output', 'shape': (1, 255, self.keyparams['width'] // (8 * 2 ** 0), self.keyparams['height'] // (8 * 2 ** 0)), 'dtype': 'float32'},  # noqa: E501
                {'name': 'output.3', 'shape': (1, 255, self.keyparams['width'] // (8 * 2 ** 1), self.keyparams['height'] // (8 * 2 ** 1)), 'dtype': 'float32'},  # noqa: E501
                {'name': 'output.7', 'shape': (1, 255, self.keyparams['width'] // (8 * 2 ** 2), self.keyparams['height'] // (8 * 2 ** 2)), 'dtype': 'float32'}  # noqa: E501
            ],
            'processed_output': [
                {'name': 'detection_output', 'type': 'List[DectObject]'}
            ]
        }

    def get_framework_and_version(self):
        return ('onnx', str(onnx.__version__))

    def get_output_formats(self):
        return ['onnx']

    def save_to_onnx(self, modelpath: Path):
        shutil.copy(self.modelpath, modelpath)
