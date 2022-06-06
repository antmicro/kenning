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
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        self.threshold = 0.7
        self.numclasses = dataset.numclasses
        super().__init__(modelpath, dataset, from_file)

    def prepare_model(self):
        if self.from_file:
            self.model = torch.load(self.modelpath)
            self.model.eval()
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

    def postprocess_outputs(self, out_all: list) -> list:
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

    def save_to_onnx(self, modelpath):
        x = torch.randn(1, 3, 416, 416).to(device='cpu')
        torch.onnx.export(
            self.model.to(device='cpu'),
            x,
            modelpath,
            opset_version=11,
            input_names=["input.1"]
        )

    def convert_input_to_bytes(self, input_data):
        data = bytes()
        for i in input_data.detach().cpu().numpy():
            data += i.tobytes()
        return data

    def convert_output_from_bytes(self, output_data):
        import json
        from base64 import b64decode
        out = json.loads(output_data.decode("ascii"))
        all_masks = np.frombuffer(
            b64decode(bytes(out['3'], 'ascii')),
            dtype='float32'
        )
        all_boxes = np.frombuffer(
            b64decode(bytes(out['0'], 'ascii')),
            dtype='float32'
        )
        all_scores = np.frombuffer(
            b64decode(bytes(out['1'], 'ascii')),
            dtype='float32'
        )
        all_labels = np.frombuffer(
            b64decode(bytes(out['2'], 'ascii')),
            dtype='int64'
        )
        return [{
            "boxes":
                np.reshape(
                    all_boxes, (int(np.size(all_boxes)/4), 4)
                ),
            "scores":
                all_scores,
            "labels":
                all_labels,
            "masks":
                torch.from_numpy(
                    np.reshape(
                        all_masks,
                        (int(np.size(all_masks)/(416**2)), 1, 416, 416)
                    )
                )
        }]

    def get_input_spec(self):
        return {'input.1': (1, 3, 416, 416)}, 'float32'


def dict_to_tuple(out_dict):
    return \
        out_dict["boxes"],\
        out_dict["scores"],\
        out_dict["labels"],\
        out_dict["masks"]
