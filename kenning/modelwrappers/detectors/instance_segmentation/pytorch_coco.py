"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on Coco classes
"""

import numpy as np
from pathlib import Path

import torch
from torchvision import models

from kenning.core.dataset import Dataset
from kenning.datasets.open_images_dataset import SegmObject
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper


class PyTorchCOCOMaskRCNN(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.numclasses = dataset.numclasses
        print(self.numclasses)
        super().__init__(modelpath, dataset, from_file)

    def prepare_input_data(self, data: np.array):
        print(np.shape(data))
        return torch.Tensor(data)

    def prepare_model(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            progress=True,
            num_classes=91,
            pretrained_backbone=True
        )

    def run_model(self, torch_data: torch.Tensor):
        self.model.eval()
        return self.model([torch_data])

    def parse_output(self,out: dict) -> list[SegmObject]:
        ret = []
        for i in range(len(out['labels'])):
            ret.append(SegmObject(
                clsname=self.dataset.classnames[int(out['labels'][i])],
                maskpath=None,
                xmin=None,
                ymin=None,
                xmax=None,
                ymax=None,
                mask=out['masks'][i].detach().numpy().transpose(1,2,0),
                score=1.0
            ))
        return ret
