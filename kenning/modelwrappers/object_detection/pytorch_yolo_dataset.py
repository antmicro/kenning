# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains implementation of a pytorch dataset used for Yolov4 fine tuning.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from torch.utils.data import Dataset

from kenning.datasets.helpers.detection_and_segmentation import DetectObject

if TYPE_CHECKING:
    from torchvision.transforms.v2._container import Compose

    from kenning.core.dataset import Dataset as KenningDataset
    from kenning.core.model import ModelWrapper


class YoloDataset(Dataset):
    """
    Dataset for Yolov4 fine tuning.

    Allows for applying torchvision transforms to the data.
    """

    def __init__(
        self,
        inputs: List[Any],
        labels: List[List[DetectObject]],
        dataset: KenningDataset,
        wrapper: ModelWrapper,
        transforms: Optional[Compose] = None,
    ):
        self.inputs = inputs
        self.labels = labels
        self.dataset = dataset
        self.wrapper = wrapper
        self.device = wrapper.device
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        import torch
        from torchvision import tv_tensors

        batch_x = [self.inputs[idx]]
        data = self.dataset.prepare_input_samples(batch_x)
        data = self.wrapper._preprocess_input(data)
        data = torch.as_tensor(data, device=self.device)

        batch_y = [self.labels[idx]]
        label = self.dataset.prepare_output_samples(batch_y)

        # labels and data are nested inside lists, we only want one image and
        # labels for detect objects inside it
        label = label[0][0]
        img = data[0][0]
        w, h = img.shape[1], img.shape[2]

        tensor_boxes = torch.tensor(
            [
                [
                    d.xmin * w,
                    d.ymin * h,
                    d.xmax * w,
                    d.ymax * h,
                ]
                for d in label
            ],
            dtype=torch.float32,
        )

        boxes = tv_tensors.BoundingBoxes(
            tensor_boxes,
            format="XYXY",
            canvas_size=img.shape[-2:],
        )

        target = {
            "boxes": boxes,
            "labels": torch.tensor(
                [
                    self.wrapper.classnames_to_index[det_obj.clsname] + 1
                    for det_obj in label
                ]
            ),
            "iscrowd": torch.tensor([det_obj.iscrowd for det_obj in label]),
            "scores": torch.tensor([det_obj.score for det_obj in label]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        boxes = target["boxes"].as_subclass(torch.Tensor)
        labels = (target["labels"] - 1).unsqueeze(1)
        label = torch.cat([boxes, labels], dim=1)

        return img, label
