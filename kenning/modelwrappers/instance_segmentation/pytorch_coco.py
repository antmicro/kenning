# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Load a pre-trained PyTorch Mask R-CNN model.

Pretrained on COCO dataset.
"""

import numpy as np
from functools import reduce
from typing import Optional, Dict, Tuple
import operator

from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.resources import coco_instance_segmentation
from kenning.utils.resource_manager import PathOrURI
import sys

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class PyTorchCOCOMaskRCNN(PyTorchWrapper):
    """
    Model wrapper for Mask-RCNN model implemented in PyTorch.
    """

    default_dataset = COCODataset2017

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.threshold = 0.7
        if dataset is not None:
            self.numclasses = dataset.numclasses

    def create_model_structure(self):
        from torchvision import models

        self.model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,  # downloads mask r-cnn model to torchhub dir
            progress=True,
            num_classes=91,
            pretrained_backbone=True,  # downloads backbone to torchhub dir
        )

    def prepare_model(self):
        if self.model_prepared:
            return None
        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.create_model_structure()
            self.model_prepared = True
            self.save_model(self.model_path)

        self.model.to(self.device)
        self.model.eval()
        self.custom_classnames = []
        with path(coco_instance_segmentation, "pytorch_classnames.txt") as p:
            with open(p, "r") as f:
                for line in f:
                    self.custom_classnames.append(line.strip())

    def postprocess_outputs(self, out_all: list) -> list:
        ret = []
        for out in out_all:
            ret.append([])
            if not isinstance(out, dict):
                continue
            for i in range(len(out["labels"])):
                masks_np = out["masks"][i].cpu().detach().numpy()
                ret[-1].append(
                    SegmObject(
                        clsname=self.custom_classnames[int(out["labels"][i])],
                        maskpath=None,
                        xmin=float(out["boxes"][i][0]),
                        ymin=float(out["boxes"][i][1]),
                        xmax=float(out["boxes"][i][2]),
                        ymax=float(out["boxes"][i][3]),
                        mask=np.multiply(
                            masks_np.transpose(1, 2, 0), 255
                        ).astype("uint8"),
                        score=float(out["scores"][i]),
                        iscrowd=False,
                    )
                )
        return ret

    def convert_input_to_bytes(self, input_data):
        data = bytes()
        for i in input_data.detach().cpu().numpy():
            data += i.tobytes()
        return data

    def convert_output_from_bytes(self, output_data):
        # The unknown size in the output specification is the
        # number of detected object. It can be calculated
        # manually using the size of the output
        S = len(output_data)
        f = np.dtype(np.float32).itemsize
        i = np.dtype(np.int64).itemsize
        num_dets = S // (416 * 416 * f + i)

        output_specification = self.get_io_specification()["        "]

        result = {}
        for spec in output_specification:
            name = spec["name"]
            shape = list(
                num_dets if val == -1 else val for val in spec["shape"]
            )
            dtype = np.dtype(spec["dtype"])
            tensorsize = reduce(operator.mul, shape) * dtype.itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(
                np.frombuffer(output_data[:tensorsize], dtype=dtype)
            ).reshape(shape)
            result[name] = outputtensor
            output_data = output_data[tensorsize:]

        return [result]

    @classmethod
    def _get_io_specification(cls):
        return {
            "input": [
                {
                    "name": "input.1",
                    "shape": (1, 3, 416, 416),
                    "dtype": "float32",
                }
            ],  # noqa: E501
            "output": [
                {"name": "boxes", "shape": (-1, 4), "dtype": "float32"},
                {"name": "labels", "shape": (-1,), "dtype": "int64"},
                {"name": "scores", "shape": (-1,), "dtype": "float32"},
                {
                    "name": "masks",
                    "shape": (-1, 1, 416, 416),
                    "dtype": "float32",
                },  # noqa: E501
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification()

    def get_io_specification_from_model(self):
        return self._get_io_specification()


def dict_to_tuple(out_dict: Dict) -> Tuple:
    """
    Converter of instance segmentation predictions into tuple.

    Parameters
    ----------
    out_dict: Dict
        Dictionary with boxes, labels, scores and masks

    Returns
    -------
    Tuple:
        Tuple holding the data from dictionary
    """
    return (
        out_dict["boxes"],
        out_dict["labels"],
        out_dict["scores"],
        out_dict["masks"],
    )
