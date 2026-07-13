# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains implementation of a pytorch dataset used for Yolov4 fine tuning.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from kenning.datasets.helpers.detection_and_segmentation import DetectObject

if TYPE_CHECKING:
    import torch
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
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
        batch_y = [self.labels[idx]]
        label = self.dataset.prepare_output_samples(batch_y)

        data = torch.Tensor(data).to(self.device)
        try:
            label = [
                torch.from_numpy(np.asarray(_l)).to(self.device)
                for _l in label
            ]
        except (ValueError, TypeError):
            pass

        data = data[0].squeeze(0)
        label = label[0][0]
        w, h = data.shape[1], data.shape[2]

        def dobject_to_tensor(dobj: DetectObject) -> torch.Tensor:
            return torch.tensor(
                [
                    dobj.xmin * w,
                    dobj.ymin * h,
                    dobj.xmax * w,
                    dobj.ymax * h,
                ]
            )

        boxes = tv_tensors.BoundingBoxes(
            torch.stack([dobject_to_tensor(dobj) for dobj in label]),
            format="XYXY",
            canvas_size=data.shape[-2:],
        )

        target = {
            "boxes": boxes,
            "labels": torch.tensor(
                list(
                    map(
                        lambda do: (
                            self.wrapper.classnames_to_index[do.clsname] + 1
                        ),
                        label,
                    )
                )
            ),
            "iscrowd": torch.tensor(list(map(lambda do: do.iscrowd, label))),
            "scores": torch.tensor(list(map(lambda do: do.score, label))),
        }

        if self.transforms:
            data, target = self.transforms(data, target)

        label = [
            DetectObject(
                self.wrapper.index_to_classnames[
                    target["labels"][i].item() - 1
                ],
                target["boxes"][i][0].item() / w,
                target["boxes"][i][1].item() / h,
                target["boxes"][i][2].item() / w,
                target["boxes"][i][3].item() / h,
                target["scores"][i].item(),
                target["iscrowd"][i].item(),
            )
            for i in range(len(target["labels"]))
        ]
        return data, [label]
