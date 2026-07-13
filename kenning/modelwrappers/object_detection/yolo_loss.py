# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains implementation for loss function for training Yolov4.
https://github.com/Tianxiaomo/pytorch-YOLOv4.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    import torch

from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import DetectObject


def box_center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts boxes format from (center, width, height) to
    (upper-left, lower-right).

    Parameters
    ----------
    boxes : torch.Tensor
        List of bounding boxes with shape (N, 4) in the (center, width, height)
        format.

    Returns
    -------
    torch.Tensor
        List of bounding boxes with the same shape but in the
        (upper-left, lower-right) format.
    """
    import torch

    return torch.stack(
        (
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ),
        dim=-1,
    )


def box_iou(
    boxes1: torch.tensor,
    boxes2: torch.tensor,
    eps: float = 1e-7,
    CIoU: bool = False,
    conv2xyxy: bool = False,
) -> torch.tensor:
    """
    Computes IoU between target and detect boxes.

    Parameters
    ----------
    boxes1 : torch.tensor
        shape (BS_1, 4 + ...).
    boxes2 : torch.tensor
        shape: (BS_2, 4 + ...).
    eps : float
        Epsilon to prevent division by 0.
    CIoU : bool
        Calculate CIoU.
    conv2xyxy : bool
        Convert boxes to (upper-left, bottom-right) format.

    Returns
    -------
    iou : torch.tensor
        Matrix with shape (BS_1, BS_2) with IoU between all
        boxes from boxes1 and boxes2.
    """
    import torch

    # Convert boxes to corner notation
    if conv2xyxy:
        boxes1 = box_center_to_corner(boxes1)
        boxes2 = box_center_to_corner(boxes2)

    def box_area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    boxes1_area = box_area(boxes1)
    boxes2_area = box_area(boxes2)

    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:4], boxes2[:, 2:4])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = boxes1_area[:, None] + boxes2_area - inter_areas + eps
    iou = inter_areas / union_areas
    if not CIoU:
        return iou

    # centerpoint distance squared
    rho2 = (
        (boxes1[:, None, 0] + boxes1[:, None, 2])
        - (boxes2[:, 0] + boxes2[:, 2])
    ) ** 2 / 4 + (
        (boxes1[:, None, 1] + boxes1[:, None, 3])
        - (boxes2[:, 1] + boxes2[:, 3])
    ) ** 2 / 4

    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    # convex top left and bottom right
    con_tl = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    con_br = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
    v = (4 / math.pi**2) * torch.pow(
        torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v)
    return iou - (rho2 / c2 + v * alpha)  # CIoU


class YoloLoss:
    """
    Class for calculating loss for Yolov4 using PyTorch.

    Currently assumes image is a square.

    Yolo learns predictions at three scales to detect small, medium and large
    objects. Each detection layer forms a grid (feature map) over the input
    image. Every grid cell in a layer has `k` predefined anchor boxes, the
    network learns offsets to adjust these anchor's position's and size to
    produce the final bounding box.

    For example here are shapes three output layers for dataset with 608x608
    images and 80 classes:
    - (batch_size, 255, 76, 76)
    - (batch_size, 255, 38, 38)
    - (batch_size, 255, 19, 19)

    255 comes from this `(num_classes + 5) * 3`. `5` stands for offsets for
    positions and size and objectness. `3` is for the 3 predefined anchor boxes
    that are for every grid cell. Last two dimensions are the dimensions of the
    feature map.

    When calculating loss we first find the best anchor for the given
    ground-truth (in the `build_target` method).

    Loss consists of three parts:
    - IoU, how well the predicted bbox matches gt
    - Objectness, does this anchor contain an object
    - Class, which class does this anchor contain

    TODO: maybe add more detailed description and links to papers
    """

    def __init__(
        self,
        perlayerparams: Dict[str, List[np.array]],
        keyparams: Dict[str, int],
        numclasses: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        dataset: Dataset,
    ):
        import torch

        self.classnames = dataset.get_class_names()
        self.width = keyparams["width"]
        self.height = keyparams["height"]
        self.mask = perlayerparams["mask"]
        # Number of bounding boxes per grid cell per layer
        self.n_anchors = len(self.mask[0])
        # calculating better way to store anchors
        _anchors = perlayerparams["anchors"][0]
        self.anchors = [
            [
                _anchors[2 * i],
                _anchors[2 * i + 1],
            ]
            for mask in self.mask
            for i in mask
        ]
        self.ignore_threshold = 0.5
        # TODO: maybe find a way to get these automatically
        self.strides = [8, 16, 32]

        self.n_classes = numclasses
        self.device = device

        # Anchor sizes scaled for the grid
        self.anchor_w, self.anchor_h = [], []
        # Anchor references that are centered in (0, 0)
        self.ref_anchors = []
        # Anchors divided by stride and grouped by layers
        self.masked_anchors = []
        # Used to translate network predicted offsets into the feature map
        # (tx, ty) - predicted offsets
        # x = sigmoid(tx) + grid_x
        # y = sigmoid(ty) + grid_y
        # (x, y) - coordinates in the feature map
        self.grid_x, self.grid_y = [], []

        image_size = self.width
        for i in range(3):
            # scale anchors from image size to grid size
            all_anchors_grid = [
                (w / self.strides[i], h / self.strides[i])
                for w, h in self.anchors
            ]
            # get anchors for this layer
            masked_anchors = np.array(
                [all_anchors_grid[j] for j in self.mask[i]],
                dtype=np.float32,
            )
            ref_anchors = np.zeros(
                (len(all_anchors_grid), 4), dtype=np.float32
            )
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)

            fsize = image_size // self.strides[i]
            grid_x = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(1, 3, fsize, 1)
                .to(device)
            )
            grid_y = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(1, 3, fsize, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(1, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(1, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batch_size, fsize, n_ch, output_id):
        """Builds targets for computing Yolo loss. Assigns an anchor for each
        ground truth box.
        """
        import torch

        # target assignment
        tgt_mask = torch.zeros(
            batch_size, self.n_anchors, fsize, fsize, 4 + self.n_classes
        ).to(device=self.device)
        obj_mask = torch.ones(batch_size, self.n_anchors, fsize, fsize).to(
            device=self.device
        )
        tgt_scale = torch.zeros(
            batch_size, self.n_anchors, fsize, fsize, 2
        ).to(self.device)
        target = torch.zeros(
            batch_size, self.n_anchors, fsize, fsize, n_ch
        ).to(self.device)

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (
            self.strides[output_id] * 2
        )
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (
            self.strides[output_id] * 2
        )
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[
            output_id
        ]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[
            output_id
        ]
        target_xywh = torch.zeros_like(target[..., :4])

        for b in range(batch_size):
            valid = (labels[b, :, 2] > labels[b, :, 0]) & (
                labels[b, :, 3] > labels[b, :, 1]
            )
            truth_x = truth_x_all[b][valid].to(torch.float)
            truth_y = truth_y_all[b][valid].to(torch.float)
            truth_w = truth_w_all[b][valid].to(torch.float)
            truth_h = truth_h_all[b][valid].to(torch.float)
            valid_labels = labels[b][valid]

            n = truth_x.shape[0]
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w[:n]
            truth_box[:n, 3] = truth_h[:n]
            truth_i = truth_x.to(torch.int64)
            truth_j = truth_y.to(torch.int64)

            # calculate iou between truth and reference anchors
            anchor_ious = box_iou(
                truth_box.cpu(),
                self.ref_anchors[output_id],
                conv2xyxy=False,
            )

            best_n_all = anchor_ious.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.mask[output_id][0])
                | (best_n_all == self.mask[output_id][1])
                | (best_n_all == self.mask[output_id][2])
            )
            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x[:n]
            truth_box[:n, 1] = truth_y[:n]

            pred_ious = box_iou(pred[b].view(-1, 4), truth_box, conv2xyxy=True)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_threshold
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x[ti] - truth_x[ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 1] = truth_y[ti] - truth_y[ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w[ti]
                        / torch.Tensor(self.masked_anchors[output_id])[
                            best_n[ti], 0
                        ]
                        + 1e-16
                    )
                    target[b, a, j, i, 3] = torch.log(
                        truth_h[ti]
                        / torch.Tensor(self.masked_anchors[output_id])[
                            best_n[ti], 1
                        ]
                        + 1e-16
                    )
                    target[b, a, j, i, 4] = 1
                    cls_index = (
                        valid_labels[ti, 4].to(torch.int16).cpu().numpy()
                    )
                    target[b, a, j, i, 5 + cls_index] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w[ti] * truth_h[ti] / fsize / fsize
                    )
                    target_xywh[b, a, j, i, 0] = truth_x[ti]
                    target_xywh[b, a, j, i, 1] = truth_y[ti]
                    target_xywh[b, a, j, i, 2] = truth_w[ti]
                    target_xywh[b, a, j, i, 3] = truth_h[ti]
        return obj_mask, tgt_mask, tgt_scale, target, target_xywh

    def _preprocess_labels(
        self, targets: List[List[DetectObject]], batch_size: int
    ):
        import torch

        labels = []
        for id, target_batch in enumerate(targets):
            if not target_batch:
                continue

            # Converting DectObject to tensor
            # Shape: (N, center_x, center_y, width, height, cls)
            target_batch_torch = torch.tensor(
                [
                    [
                        _target.xmin * self.width,
                        _target.ymin * self.height,
                        _target.xmax * self.width,
                        _target.ymax * self.height,
                        self.classnames.index(_target.clsname),
                    ]
                    for _target in target_batch
                ],
                device=self.device,
            )
            labels.append(target_batch_torch)

        # This is mainly for tests where we get no target bounding boxes
        if not labels:
            return torch.zeros((batch_size, 1, 5), device=self.device)

        max_objects = max([label.shape[0] for label in labels])
        padded_labels = []
        for label in labels:
            n_objects = label.shape[0]
            padding = torch.zeros(
                (max_objects - n_objects, label.shape[1]), device=self.device
            )
            padded_labels.append(torch.cat([label, padding], dim=0))
        labels = torch.stack(padded_labels)
        return labels

    def __call__(self, outputs: List, targets: List[List[DetectObject]]):
        import torch
        import torch.nn.functional as F

        batch_size = outputs[0].shape[0]
        labels = self._preprocess_labels(targets, batch_size)
        n_ch = 5 + self.n_classes
        loss_iou, loss_cls, loss_obj = 0, 0, 0
        total_pos = 0

        for out_id, out in enumerate(outputs):
            fsize = out.shape[2]
            # Reshaping to (BS, BB, 4+1+C, W', H')
            # BS - batch_size, BB - bounding boxes per one 'pixel'/chunk
            # 4 parameters responsible for x, y, w, h
            # 1 parameter - objectness logit, C logits of classification
            out = out.view(
                out.shape[0],
                len(self.mask[out_id]),
                n_ch,
                out.shape[-2],
                out.shape[-1],
            )
            # Reshaping to (BS, BB, W', H', 4+1+C)
            out = out.permute(0, 1, 3, 4, 2).contiguous()

            # Sigmoid activation for x and y coordinates
            out[..., :2] = torch.sigmoid(out[..., :2])

            pred = out[..., :4].clone()
            # Translating
            pred[..., 0] += self.grid_x[out_id]
            pred[..., 1] += self.grid_y[out_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[out_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[out_id]

            # build targets for predictions
            (
                obj_mask,
                tgt_mask,
                tgt_scale,
                target,
                target_xywh,
            ) = self.build_target(
                pred, labels, batch_size, fsize, n_ch, out_id
            )

            # calculate loss
            out[..., 4] *= obj_mask
            out[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            out[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            pos_mask = tgt_mask[..., 0] > 0
            total_pos += pos_mask.sum()
            if pos_mask.any():
                target_pos = target_xywh[pos_mask]
                pred_pos = pred[pos_mask]
                ious = box_iou(pred_pos, target_pos, CIoU=True).diagonal()
                loss_iou += (1 - ious).sum()

                loss_cls += F.binary_cross_entropy_with_logits(
                    out[..., 5:][pos_mask],
                    target[..., 5:][pos_mask],
                    reduction="sum",
                )

            loss_obj += F.binary_cross_entropy_with_logits(
                out[..., 4], target[..., 4], reduction="sum"
            )

        num_pos = total_pos.clamp(min=1)
        loss_iou = loss_iou / num_pos
        loss_cls = loss_cls / num_pos
        loss_obj = loss_obj / batch_size

        loss = loss_iou + loss_cls + loss_obj
        return loss
