# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Sources for YOLOv4 ModelWrapper.

ModelWrapper for the YOLOv4 model generated from darknet repository using:

https://github.com/Tianxiaomo/pytorch-YOLOv4

To create an ONNX model from darknet yolov4.cfg and yolov4.weights files
(check https://github.com/AlexeyAB/darknet for those files), follow
repositories' README (Darknet2ONNX section).

After this, to remove the embedded processing of outputs, run in Python shell::

    from kenning.modelwrappers.object_detection.yolov4 import \
            yolov4_remove_postprocessing


    yolov4_remove_postprocessing('<input_onnx_path>', '<output_onnx_path>')

Parts of loss function implementation were taken from https://d2l.ai/chapter_computer-vision/anchor.html
"""

import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx

from kenning.core.exceptions import NotSupportedError
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.helpers.detection_and_segmentation import DetectObject
from kenning.modelwrappers.object_detection.yolo_loss import YoloLoss
from kenning.modelwrappers.object_detection.yolo_wrapper import YOLOWrapper
from kenning.utils.resource_manager import PathOrURI


def yolov4_remove_postprocessing(
    inputpath: Path,
    outputpath: Path,
    input_names: List[str] = ["input"],
    output_names: List[str] = ["output", "output.3", "output.7"],
):
    """
    Extracts the actual model from the Darknet2ONNX output.

    Darknet2ONNX tool (https://github.com/Tianxiaomo/pytorch-YOLOv4) creates
    an ONNX file that contains a YOLOv4 model and postprocessing steps to
    extract bounding boxes and scores.

    To keep the model simple, this method extracts the actual model
    and removes the postprocessing.

    Parameters
    ----------
    inputpath : Path
        Path to the ONNX file containing model with postprocessing.
    outputpath : Path
        Path to the ONNX output file containing pure model.
    input_names : List[str]
        List of model inputs names.
    output_names : List[str]
        List of model outputs names.
    """
    onnx.utils.extract_model(
        str(inputpath), str(outputpath), input_names, output_names
    )


class ONNXYOLOV4(YOLOWrapper):
    """
    Model wrapper for YOLOv4 model in ONNX format.
    """

    pretrained_model_uri = "kenning:///models/detection/yolov4.onnx"
    default_dataset = COCODataset2017
    arguments_structure = {}

    def postprocess_outputs(
        self, y: List[np.ndarray]
    ) -> List[List[List[DetectObject]]]:
        # YOLOv4, as YOLOv3, has three outputs for three stages of computing.
        # Each output layer has information about bounding boxes, scores and
        # classes in a grid.

        outputs = []
        for i in range(3):
            outshape = (
                self.batch_size,
                len(self.perlayerparams["mask"][i]),
                4 + 1 + self.numclasses,
                self.keyparams["width"] // (8 * 2**i),
                self.keyparams["height"] // (8 * 2**i),
            )
            # Extract the output and reshape it to match actual form
            outarr = y[i].reshape(outshape).copy()

            # x and y offsets need to be passed through sigmoid function
            # NOTE: w and h are NOT passed through sigmoid function - they are
            # later computed in parse_outputs methods using anchors and mask
            # parameters.
            outarr[:, :, :2, :, :] = 1 / (1 + np.exp(-outarr[:, :, :2, :, :]))
            # objectness and classes are also passed through sigmoid function
            outarr[:, :, 4:, :, :] = 1 / (1 + np.exp(-outarr[:, :, 4:, :, :]))
            outputs.append(outarr)

        return [self.parse_batches(outputs)]

    def loss_torch(
        self,
        outputs: List,
        target: List[List[DetectObject]],
        eps: float = 1e-7,
    ) -> float:
        """
        Loss function for YOLOv4, implemented to work one batch in form
        of torch.Tensors. YOLOv4 use sum of few losses - CIoU, binary
        cross-entropy of objectness and classification scores.

        Parameters
        ----------
        outputs : List
            One batch of YOLO network output of type torch.Tensor
        target : List[List[DetectObject]]
            True bounding boxes of object on precessed image
        eps : float
            Epsilon to prevent dividing by 0

        Returns
        -------
        float
            Value of loss
        """
        device = outputs[0].device
        dtype = outputs[0].dtype

        criterion = YoloLoss(
            self.perlayerparams,
            self.keyparams,
            self.numclasses,
            self.batch_size,
            device,
            dtype,
            self.dataset,
        )
        return criterion(outputs, target)

    # NOTE: In postprocess_outputs function the second output layer `output.3`
    # of size 255 is split into two layers of size (4 + 1 + C) and B,
    # where C is a class vector and B is the number of detectable object
    # in a pixel.
    @classmethod
    def _get_io_specification(cls, keyparams, batch_size):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": (
                        batch_size,
                        3,
                        keyparams["width"],
                        keyparams["height"],
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "output",
                    "shape": (
                        batch_size,
                        255,
                        keyparams["width"] // (8 * 2**0),
                        keyparams["height"] // (8 * 2**0),
                    ),
                    "dtype": "float32",
                },
                {
                    "name": "output.3",
                    "shape": (
                        batch_size,
                        255,
                        keyparams["width"] // (8 * 2**1),
                        keyparams["height"] // (8 * 2**1),
                    ),
                    "dtype": "float32",
                },
                {
                    "name": "output.7",
                    "shape": (
                        batch_size,
                        255,
                        keyparams["width"] // (8 * 2**2),
                        keyparams["height"] // (8 * 2**2),
                    ),
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "detection_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject",  # noqa: E501
                    },
                }
            ],
        }

    @classmethod
    def get_framework(cls) -> str:
        return "onnx"

    @classmethod
    def get_framework_version(cls) -> str:
        return str(onnx.__version__)

    @classmethod
    def get_output_formats(cls):
        return ["onnx"]

    def save_to_onnx(self, model_path: PathOrURI):
        self.save_model(model_path)

    def run_inference(self, X: List) -> Any:
        raise NotSupportedError

    def save_model(self, model_path: PathOrURI):
        shutil.copy(self.model_path, model_path)

    def train_model(self):
        raise NotSupportedError
