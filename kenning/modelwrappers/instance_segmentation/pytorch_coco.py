"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on COCO dataset
"""

import numpy as np
from pathlib import Path
from torchvision import models
from functools import reduce
import operator

from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper

from kenning.resources import coco_instance_segmentation
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class PyTorchCOCOMaskRCNN(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.threshold = 0.7
        self.numclasses = dataset.numclasses
        super().__init__(modelpath, dataset, from_file)

    def create_model_structure(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,  # downloads mask r-cnn model to torchhub dir
            progress=True,
            num_classes=91,
            pretrained_backbone=True  # downloads backbone to torchhub dir
        )

    def prepare_model(self):
        if self.from_file:
            self.load_model(self.modelpath)
        else:
            self.create_model_structure()
            self.save_model(self.modelpath)

        self.model.to(self.device)
        self.model.eval()
        self.custom_classnames = []
        with path(coco_instance_segmentation, 'pytorch_classnames.txt') as p:
            with open(p, 'r') as f:
                for line in f:
                    self.custom_classnames.append(line.strip())

    def postprocess_outputs(self, out_all: list) -> list:
        ret = []
        for out in out_all:
            ret.append([])
            if isinstance(out, dict):
                for i in range(len(out['labels'])):
                    ret[-1].append(SegmObject(
                        clsname=self.custom_classnames[
                            int(out['labels'][i])
                        ],
                        maskpath=None,
                        xmin=float(out['boxes'][i][0]),
                        ymin=float(out['boxes'][i][1]),
                        xmax=float(out['boxes'][i][2]),
                        ymax=float(out['boxes'][i][3]),
                        mask=np.multiply(
                            out['masks'][i].transpose(1, 2, 0),  # noqa: E501
                            255
                        ).astype('uint8'),
                        score=float(out['scores'][i])
                    ))
                return ret

    def convert_input_to_bytes(self, input_data):
        data = bytes()
        for i in input_data.detach().cpu().numpy():
            data += i.tobytes()
        return data

    def convert_output_from_bytes(self, output_data):
        S = len(output_data)
        f = np.dtype(np.float32).itemsize
        i = np.dtype(np.int64).itemsize
        num_dets = S // (416 * 416 * f + i)

        output_parameters = [
            ((num_dets, 4), np.float32, 'boxes'),
            ((num_dets,), np.int64, 'labels'),
            ((num_dets,), np.float32, 'scores'),
            ((num_dets, 1, 416, 416), np.float32, 'masks')
        ]

        result = {}
        for shape, dtype, name in output_parameters:
            tensorsize = reduce(operator.mul, shape) * np.dtype(dtype).itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(np.frombuffer(
                output_data[:tensorsize],
                dtype=dtype
            )).reshape(shape)
            result[name] = outputtensor
            output_data = output_data[tensorsize:]

        return [result]

    def get_io_specs(self):
        return {
            'input': [{'name': 'input.1', 'shape': (1, 3, 416, 416), 'dtype': 'float32'}],  # noqa: E501
            'output': [
                {'name': 'boxes', 'shape': (-1, 4), 'dtype': 'float32'},
                {'name': 'labels', 'shape': (-1,), 'dtype': 'int64'},
                {'name': 'scores', 'shape': (-1,), 'dtype': 'float32'},
                {'name': 'masks', 'shape': (-1, 1, 416, 416), 'dtype': 'float32'}  # noqa: E501
            ]
        }


def dict_to_tuple(out_dict):
    return \
        out_dict["boxes"],\
        out_dict["labels"],\
        out_dict["scores"],\
        out_dict["masks"]
