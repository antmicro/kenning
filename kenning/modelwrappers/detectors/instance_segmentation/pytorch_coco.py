"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on Coco classes
"""

import numpy as np
from pathlib import Path

import torch
from torchvision import models, transforms
from torch.utils.data import Dataset

from kenning.core.dataset import Dataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper


class PyTorchCOCOMaskRCNN(PyTorchWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.numclasses = dataset.numclasses
        print(self.numclasses)
        super().__init__(modelpath, dataset, from_file)

    def prepare_input_data(self,data : np.array):
        print(np.shape(data))
        return torch.Tensor(data)

    def prepare_model(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, 
            progress=True, 
            num_classes=91, # TODO: check why it crashes upon changing it to something else
            pretrained_backbone=True
        )
    def run_model(self, torch_data : torch.Tensor):
        self.model.eval()
        return self.model([torch_data])