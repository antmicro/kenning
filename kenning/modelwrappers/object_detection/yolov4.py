# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
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
"""

import math
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx

from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.helpers.detection_and_segmentation import DetectObject
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

    pretrained_model_uri = "kenning:///models/object_detection/yolov4.onnx"
    default_dataset = COCODataset2017
    arguments_structure = {}

    def postprocess_outputs(self, y):
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

        return self.parse_batches(outputs)

    def loss_torch(
        self,
        outputs: List,
        target: List[List[DetectObject]],
        scale_noobj: float = 0.5,
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
        scale_noobj : float
            Scaling factor of bounding boxes without object error
        eps : float
            Epsilon to prevent dividing by 0

        Returns
        -------
        float
            Value of loss
        """
        import torch

        device = outputs[0].device
        dtype = outputs[0].dtype
        loss = torch.zeros(outputs[0].shape[0], device=device)

        # Preprocessing outputs
        # detect_output = []
        result = []
        for id, out in enumerate(outputs):
            # Reshaping to (BS, BB, 4+1+C, W', H')
            # BS - batch_size, BB - bounding boxes per one 'pixel'/chunk
            # 4 parameters responsible for x, y, w, h
            # 1 parameter - objectness logit, C logits of classification
            out = out.view(
                out.shape[0],
                len(self.perlayerparams["mask"][id]),
                4 + 1 + self.numclasses,
                out.shape[-2],
                out.shape[-1],
            )
            # Calculating centers (x, y) of bounding boxes
            x_y = torch.sigmoid(out[:, :, :2])
            x_y_ids = torch.arange(out.shape[-1], device=device).expand_as(x_y)
            x_y = x_y + x_y_ids
            x_y[:, :, 0] /= out.shape[-2]
            x_y[:, :, 1] /= out.shape[-1]

            # Calculating width, height of bounding boxes
            anchors = self.perlayerparams["anchors"][id]
            mask = self.perlayerparams["mask"][id]
            w_h_anchor = torch.tensor(
                [
                    [
                        anchors[2 * mask[bounding_box]]
                        / self.keyparams["width"],
                        anchors[2 * mask[bounding_box] + 1]
                        / self.keyparams["height"],
                    ]
                    for bounding_box in range(out.shape[1])
                ],
                device=device,
                dtype=dtype,
            )
            w_h_anchor = (
                w_h_anchor.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(out[:, :, 2:4])
            )
            w_h = w_h_anchor * torch.exp(out[:, :, 2:4])

            # result shape (BS, all BB, (x, y, w, h, o, clf))
            result.append(
                torch.cat((x_y, w_h, out[:, :, 4:]), dim=2).reshape(
                    out.shape[0], -1, out.shape[2]
                )
            )
        result = torch.cat(result, dim=1)

        for id, (detect_batch, target_batch) in enumerate(zip(result, target)):
            # Converting DectObject to tensor
            target_batch_torch = torch.tensor(
                [
                    [
                        (_target.xmin - _target.xmin) / 2,
                        (_target.ymin - _target.ymin) / 2,
                        _target.xmax - _target.xmin,
                        _target.ymax - _target.ymin,
                        *[
                            1.0 if name == _target.clsname else 0.0
                            for name in self.classnames
                        ],
                    ]
                    for _target in target_batch
                ],
                device=device,
            )
            loss[id] = self._loss_one_batch_torch(
                detect_batch, target_batch_torch, scale_noobj, eps
            )

        return torch.mean(loss)

    def _loss_one_batch_torch(
        self, detect_batch, target_batch, scale_noobj=0.5, eps=1e-7
    ):
        import torch
        import torch.nn.functional as F

        def ten(x):
            return torch.tensor(x, device=detect_batch[0].device)

        best_detect_ids = set()
        # Preparing losses
        ciou = ten(0.0)
        obj = ten(0.0)
        clf = ten(0.0)
        # Placements and sizes of detected bounding boxes
        detect_box_size = detect_batch[:, 2] * detect_batch[:, 3]
        detect_min_max = torch.stack(
            (
                detect_batch[:, 0] - detect_batch[:, 2] / 2,
                detect_batch[:, 1] - detect_batch[:, 3] / 2,
                detect_batch[:, 0] + detect_batch[:, 2] / 2,
                detect_batch[:, 1] + detect_batch[:, 3] / 2,
            ),
            dim=-1,
        )
        for target_obj in target_batch:
            # IoU
            target_box_size = target_obj[2] * target_obj[3]
            target_min_max = [
                target_obj[0] - target_obj[2] / 2,
                target_obj[1] - target_obj[3] / 2,
                target_obj[0] + target_obj[2] / 2,
                target_obj[1] + target_obj[3] / 2,
            ]
            intersection_w = torch.maximum(
                torch.minimum(detect_min_max[:, 2], target_min_max[2])
                - torch.maximum(detect_min_max[:, 0], target_min_max[0]),
                ten(0.0),
            )
            intersection_h = torch.maximum(
                torch.minimum(detect_min_max[:, 3], target_min_max[3])
                - torch.maximum(detect_min_max[:, 1], target_min_max[1]),
                ten(0.0),
            )
            intersection = intersection_w * intersection_h
            iou_detect = intersection / (
                detect_box_size + target_box_size - intersection + eps
            )

            # CIoU
            best_detect_id = torch.argmax(iou_detect)
            best_detect_ids.add(best_detect_id)
            best_detect = detect_batch[best_detect_id]
            iou = iou_detect[best_detect_id]
            distance_centers_2 = (best_detect[0] - target_obj[0]) ** 2 + (
                best_detect[1] - target_obj[1]
            ) ** 2
            best_min_max = detect_min_max[best_detect_id]
            spanning_box_diagonal_2 = (
                (
                    torch.min(best_min_max[0], target_min_max[0])
                    - torch.max(best_min_max[2], target_min_max[2])
                )
                ** 2
                + (
                    torch.min(best_min_max[1], target_min_max[1])
                    - torch.max(best_min_max[3], target_min_max[3])
                )
                ** 2
            ) + eps
            ratio_detect = best_detect[2] / (best_detect[3] + eps)
            ratio_target = target_obj[2] / (target_obj[3] + eps)
            ratio_consistency = (
                4
                / math.pi**2
                * torch.abs(
                    torch.atan(ratio_target) - torch.atan(ratio_detect)
                )
                ** 2
            )
            alpha = ratio_consistency / (1 - iou + ratio_consistency + eps)

            ciou += (
                1.0
                - iou
                + distance_centers_2 / spanning_box_diagonal_2
                + alpha * ratio_consistency
            )

            # Classification
            clf += F.binary_cross_entropy_with_logits(
                best_detect[5:], target_obj[4:]
            )

        # Objectness + No Objectness
        obj = F.binary_cross_entropy_with_logits(
            detect_batch[:, 4],
            ten(
                [
                    1.0 if id in best_detect_ids else 0.0
                    for id in range(detect_batch.shape[0])
                ]
            ),
        )

        return ciou + obj + clf

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
                {"name": "detection_output", "type": "List[DetectObject]"}
            ],
        }

    def get_framework_and_version(self):
        return ("onnx", str(onnx.__version__))

    def get_output_formats(self):
        return ["onnx"]

    def save_to_onnx(self, model_path: PathOrURI):
        shutil.copy(self.model_path, model_path)

    def run_inference(self, X: List) -> Any:
        raise NotImplementedError

    def save_model(self, model_path: PathOrURI):
        raise NotImplementedError

    def train_model(
        self, batch_size: int, learning_rate: float, epochs: int, logdir: Path
    ):
        raise NotImplementedError
