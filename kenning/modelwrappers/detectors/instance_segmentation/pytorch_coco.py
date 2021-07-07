"""
Load a pre-trained PyTorch Mask R-CNN model

Prerained on Coco classes
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
        print(self.numclasses)
        super().__init__(modelpath, dataset, from_file)

    def prepare_model(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            progress=True,
            num_classes=91,
            pretrained_backbone=True
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

    def parse_output(self, out: dict) -> list[SegmObject]:
        ret = []
        for i in range(len(out['labels'])):
            if float(out['scores'][i]) > self.threshold:
                ret.append(SegmObject(
                    clsname=self.custom_classnames[int(out['labels'][i])],
                    maskpath=None,
                    xmin=float(out['boxes'][i][0]),
                    ymin=float(out['boxes'][i][1]),
                    xmax=float(out['boxes'][i][2]),
                    ymax=float(out['boxes'][i][3]),
                    mask=np.multiply(
                        self.postprocess_outputs(
                            out['masks'][i]
                        ).transpose(1, 2, 0),
                        255
                    ).astype('uint8'),
                    score=1.0
                ))
        return ret
