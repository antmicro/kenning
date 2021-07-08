"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on COCO dataset
"""

import numpy as np
from pathlib import Path

from torchvision import models

from kenning.core.dataset import Dataset
from kenning.datasets.open_images_dataset import SegmObject
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper


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
        self.custom_classnames = [
            '__background__', 'person', 'bicycle',
            'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
            'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

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
