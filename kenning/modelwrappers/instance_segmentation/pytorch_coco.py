"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on COCO dataset
"""

import numpy as np
from pathlib import Path

import torch
from torchvision import models

from kenning.core.dataset import Dataset
from kenning.datasets.open_images_dataset import SegmObject
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper

from kenning.resources import coco_instance_segmentation
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class PyTorchCOCOMaskRCNN(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=False):
        self.threshold = 0.7
        self.numclasses = dataset.numclasses
        super().__init__(modelpath, dataset, from_file)

    def prepare_model(self):
        if self.from_file:
            self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=False,
                progress=True,
                num_classes=91,
                pretrained_backbone=True  # downloads backbone to torchhub dir
            )
            self.load_model(self.modelpath)
        else:
            self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,  # downloads mask r-cnn model to torchhub dir
                progress=True,
                num_classes=91,
                pretrained_backbone=True  # downloads backbone to torchhub dir
            )
        self.model.to(self.device)
        self.model.eval()
        self.custom_classnames = []
        with path(coco_instance_segmentation, 'pytorch_classnames.txt') as p:
            with open(p, 'r') as f:
                for line in f:
                    self.custom_classnames.append(line.strip())

    def postprocess_outputs(self, out_all: list[dict]) -> list[SegmObject]:
        ret = []
        for i in range(len(out_all)):
            ret.append([])
            out = out_all[i]
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
                            out['masks'][i].detach().cpu().numpy().transpose(1, 2, 0),  # noqa: E501
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
        return torch.load(output_data)  # this may not work, in fact it probably will not
