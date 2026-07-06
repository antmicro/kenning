# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Loss function for the YOLACT model implemented in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kenning.modelwrappers.instance_segmentation.model_yolact.utils import (
    _center_size_form,
    _crop,
    _log_sum_exp,
    _match,
)


class _YolactLoss(nn.Module):
    """
    Sum the dict values to get the scalar for backward().

    Loss terms:
        B - box localization
        C - classification with OHEM 3:1 (softmax)
        M - lincomb masks (BCE, crop + ROI-pool normalization)
        S - semantic segmentation (BCE-with-logits)
    """

    def __init__(
        self,
        num_classes: int = 81,
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.4,
        negpos_ratio: int = 3,
        masks_to_train: int = 100,
        bbox_alpha: float = 1.5,
        conf_alpha: float = 1.0,
        mask_alpha: float = 6.125,
        semantic_alpha: float = 1.0,
        crowd_iou_threshold: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Total number of classes including background.
            Defaults to 81 (COCO).
        pos_threshold : float, optional
            IoU threshold above which a prior is treated as a positive match
            during anchor assignment. Defaults to 0.5.
        neg_threshold : float, optional
            IoU threshold below which a prior is a clear negative. Priors
            between neg_threshold and pos_threshold are neutral and excluded
            from the loss. Defaults to 0.4.
        negpos_ratio : int, optional
            Maximum ratio of hard negatives to positives kept by OHEM for the
            classification loss. Defaults to 3.
        masks_to_train : int, optional
            Maximum number of positive priors used per image when computing the
            mask loss; extras are randomly subsampled. Defaults to 100.
        bbox_alpha : float, optional
            Loss weight for the box localization term B. Defaults to 1.5.
        conf_alpha : float, optional
            Loss weight for the classification term C. Defaults to 1.0.
        mask_alpha : float, optional
            Loss weight for the lincomb mask term M. Defaults to 6.125.
        semantic_alpha : float, optional
            Loss weight for the semantic segmentation term S. Defaults to 1.0.
        crowd_iou_threshold : float, optional
            Priors overlapping crowd regions above this IoA are marked neutral
            during anchor matching. Defaults to 0.7.
        """
        super().__init__()
        self.num_classes = num_classes
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.masks_to_train = masks_to_train
        self.bbox_alpha = bbox_alpha
        self.conf_alpha = conf_alpha
        self.mask_alpha = mask_alpha
        self.semantic_alpha = semantic_alpha
        self.crowd_iou_threshold = crowd_iou_threshold

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: list[torch.Tensor],
        masks: list[torch.Tensor],
        num_crowds: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Compute all four loss terms for a batch.

        Matches priors to ground truth, then evaluates the box, classification,
        mask and semantic-segmentation losses and normalizes them.

        Parameters
        ----------
        predictions : dict[str, torch.Tensor]
            Dictionary of model outputs with keys:

            - box: raw box offsets, shape (B, P, 4).
            - class: class logits, shape (B, P, num_classes).
            - coef: mask coefficients, shape (B, P, coef_dim).
            - proto: prototype maps, shape (B, mask_h, mask_w, coef_dim).
            - seg: semantic segmentation logits,
              shape (B, num_classes-1, mask_h, mask_w).
            - priors: anchor priors in center form, shape (P, 4).

        targets : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape (M_i, 5)
            containing [xmin, ymin, xmax, ymax, class] for the M_i
            ground-truth objects in i-th image.
        masks : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape
            (M_i, H, W) with binary ground-truth instance masks.
        num_crowds : list[int]
            List of length B indicating how many of the last entries in
            targets[i] / masks[i] are crowd annotations.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with scalar loss tensors for each term:

            - B: weighted box localization loss.
            - C: weighted OHEM classification loss.
            - M: weighted lincomb mask loss.
            - S: weighted semantic segmentation loss.
        """
        loc_data = predictions["box"]
        conf_data = predictions["class"]
        mask_data = predictions["coef"]
        proto_data = predictions["proto"]
        segm_data = predictions["seg"]
        priors = predictions["priors"]

        batch_size = loc_data.size(0)
        num_priors = priors.size(0)

        # Pre-allocate target buffers that _match will fill in place
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        labels = [None] * batch_size

        for idx in range(batch_size):
            truths = targets[idx][:, :-1].detach()
            labels[idx] = targets[idx][:, -1].detach().long()

            # Split crowd annotations off the end of truths/labels/masks so
            # they are passed separately to _match for IoA-based neutralization
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                crowd_boxes, truths = (
                    truths[-cur_crowds:],
                    truths[:-cur_crowds],
                )
                labels[idx] = labels[idx][:-cur_crowds]
                masks[idx] = masks[idx][:-cur_crowds]
            else:
                crowd_boxes = None

            # Assign each prior to a GT box and write encoded targets in place
            _match(
                self.pos_threshold,
                self.neg_threshold,
                truths,
                priors,
                labels[idx],
                crowd_boxes,
                loc_t,
                conf_t,
                idx_t,
                idx,
                self.crowd_iou_threshold,
            )

            # Store the actual GT box (in corner form) for each prior match,
            # needed by the mask loss to crop assembled masks
            gt_box_t[idx] = truths[idx_t[idx]]

        # Targets are constants; detach from the autograd graph
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        idx_t.requires_grad = False

        # Build positive mask and expand it to index into loc_data
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        # Box loss: smooth-L1 between predicted and encoded GT offsets,
        # computed only over positive priors
        losses = {}
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t_p = loc_t[pos_idx].view(-1, 4)
        losses["B"] = (
            F.smooth_l1_loss(loc_p, loc_t_p, reduction="sum") * self.bbox_alpha
        )

        # Mask, classification and semantic losses delegated to sub-methods
        losses["M"] = self._lincomb_mask_loss(
            pos, idx_t, mask_data, proto_data, masks, gt_box_t
        )
        losses["C"] = self._ohem_conf_loss(conf_data, conf_t, pos, batch_size)
        losses["S"] = self._semantic_segmentation_loss(
            segm_data, masks, labels
        )

        # Normalize each term: S by batch size, the rest by total
        # positive count
        total_num_pos = num_pos.data.sum().float().clamp(min=1)
        for k in losses:
            losses[k] = losses[k] / (batch_size if k == "S" else total_num_pos)
        return losses

    def _ohem_conf_loss(
        self,
        conf_data: torch.Tensor,
        conf_t: torch.Tensor,
        pos: torch.Tensor,
        num: int,
    ) -> torch.Tensor:
        """
        Classification loss with Online Hard Example Mining.

        Keeps all positive priors and the hardest negative examples at the
        configured negative-to-positive ratio, then applies cross-entropy over
        them.

        Parameters
        ----------
        conf_data : torch.Tensor
            Class logits of shape (B, P, num_classes) from the prediction head.
        conf_t : torch.Tensor
            Long tensor of shape (B, P) with per-prior class targets
            (0 = background, -1 = neutral, >0 = class index).
        pos : torch.Tensor
            Boolean tensor of shape (B, P) indicating positive priors
            (conf_t > 0).
        num : int
            Batch size B, used to reshape the flat logits.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the conf_alpha-weighted cross-entropy loss
            summed over selected positives and hard negatives.
        """
        # Compute softmax loss proxy for each prior
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = _log_sum_exp(batch_conf) - batch_conf[:, 0]
        loss_c = loss_c.view(num, -1)

        # Zero out positive examples and neutral examples
        loss_c[pos] = 0
        loss_c[conf_t < 0] = 0

        # Rank priors by loss descending to find the hardest negative examples
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # Keep at most negpos_ratio negatives per positive
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Remove positives and neutrals that leaked into the negative mask
        neg[pos] = 0
        neg[conf_t < 0] = 0

        # Gather logits and targets for selected positives and hard negatives
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
            -1, self.num_classes
        )
        targets_weighted = conf_t[(pos + neg).gt(0)]

        loss = F.cross_entropy(conf_p, targets_weighted, reduction="sum")

        return self.conf_alpha * loss

    def _lincomb_mask_loss(
        self,
        pos: torch.Tensor,
        idx_t: torch.Tensor,
        mask_data: torch.Tensor,
        proto_data: torch.Tensor,
        masks: list[torch.Tensor],
        gt_box_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear combination mask loss (YOLACT's mask assembly + BCE).

        For each image, assembles per-instance masks as sigmoid(proto @ coef),
        crops them to the matched GT box, and applies box-area-normalized
        binary cross-entropy against downsampled GT masks. Positives beyond
        masks_to_train are randomly subsampled.

        Parameters
        ----------
        pos : torch.Tensor
            Boolean tensor of shape (B, P) marking positive priors.
        idx_t : torch.Tensor
            Long tensor of shape (B, P) with the GT box index matched to each
            prior, as produced by _match.
        mask_data : torch.Tensor
            Mask coefficients of shape (B, P, coef_dim) from the prediction
            head.
        proto_data : torch.Tensor
            Prototype feature maps of shape (B, mask_h, mask_w, coef_dim)
            from the protonet.
        masks : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape
            (M_i, H, W) with binary ground-truth instance masks for i-th image.
        gt_box_t : torch.Tensor
            Corner-form GT boxes of shape (B, P, 4) matched to each prior,
            used to crop the assembled masks.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the mask_alpha-weighted, box-area-normalized
            binary cross-entropy summed over all positive priors in the batch.
        """
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)
        loss_m = 0
        for idx in range(mask_data.size(0)):
            # Downsample GT masks to prototype resolution and binarize at 0.5;
            # transpose to (mask_h, mask_w, M_i) for spatial indexing later
            with torch.no_grad():
                dsm = F.interpolate(
                    masks[idx].unsqueeze(0),
                    (mask_h, mask_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                dsm = dsm.permute(1, 2, 0).contiguous().gt(0.5).float()

            # Select only positive priors for this image. Skip if none matched
            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            if pos_idx_t.size(0) == 0:
                continue
            pos_gt_box_t = gt_box_t[idx, cur_pos]
            proto_masks = proto_data[idx]
            proto_coef = mask_data[idx, cur_pos, :]

            # Randomly subsample positives to masks_to_train to cap memory use;
            # remember old count for loss rescaling below
            old_num_pos = proto_coef.size(0)
            if old_num_pos > self.masks_to_train:
                perm = torch.randperm(
                    proto_coef.size(0), device=proto_coef.device
                )
                select = perm[: self.masks_to_train]
                proto_coef = proto_coef[select, :]
                pos_idx_t = pos_idx_t[select]
                pos_gt_box_t = pos_gt_box_t[select, :]
            num_pos = proto_coef.size(0)

            # Gather GT mask slices for the matched instances:
            # (mask_h, mask_w, num_pos)
            mask_t = dsm[:, :, pos_idx_t]

            # Assemble predicted masks via linear combination of prototypes,
            # then zero out pixels outside each detection box
            pred_masks = torch.sigmoid(proto_masks @ proto_coef.t())
            pred_masks = _crop(pred_masks, pos_gt_box_t)

            # Compute per-pixel BCE without reduction:
            # (mask_h, mask_w, num_pos)
            pre_loss = F.binary_cross_entropy(
                torch.clamp(pred_masks, 0, 1), mask_t, reduction="none"
            )

            # Normalize by GT box area so large and small objects
            # contribute equally. The global weight (mask_h * mask_w)
            # counteracts the /mask_h/mask_w applied after the loop
            weight = mask_h * mask_w
            csize = _center_size_form(pos_gt_box_t)
            gt_w = csize[:, 2] * mask_w
            gt_h = csize[:, 3] * mask_h
            pre_loss = (
                pre_loss.sum(dim=(0, 1))
                / gt_w.clamp(min=1e-4)
                / gt_h.clamp(min=1e-4)
                * weight
            )

            # Rescale to account for the subsampled positives
            # so the loss magnitude stays consistent regardless
            # of how many were dropped
            if old_num_pos > num_pos:
                pre_loss = pre_loss * (old_num_pos / num_pos)
            loss_m += torch.sum(pre_loss)

        return (loss_m * self.mask_alpha) / (mask_h * mask_w)

    def _semantic_segmentation_loss(
        self,
        segment_data: torch.Tensor,
        mask_t: list[torch.Tensor],
        class_t: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Auxiliary semantic segmentation loss (per-class BCE-with-logits).

        Builds a per-class semantic target by max-pooling each instance's
        downsampled GT mask into its class channel, then compares against the
        predicted semantic map.

        Parameters
        ----------
        segment_data : torch.Tensor
            Predicted semantic segmentation logits of shape
            (B, num_classes-1, mask_h, mask_w). Background is excluded — the
            semantic head predicts only foreground classes.
        mask_t : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape
            (M_i, H, W) with binary ground-truth instance masks for i-th image.
        class_t : list[torch.Tensor]
            List of length B. Each element is a long tensor of shape (M_i,)
            with the foreground class index (0-based, background excluded) for
            each instance.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the semantic_alpha-weighted BCE-with-logits loss
            averaged over the spatial dimensions and batch size.
        """
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]
            with torch.no_grad():
                # Downsample GT instance masks to the semantic head resolution
                # and binarize at 0.5: (M_i, mask_h, mask_w)
                dsm = (
                    F.interpolate(
                        mask_t[idx].unsqueeze(0),
                        (mask_h, mask_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .gt(0.5)
                    .float()
                )

                # Build the per-class semantic target by max-pooling instance
                # masks into their class channel. Pixels covered by any
                # instance of class c are set to 1
                segment_t = torch.zeros_like(cur_segment)
                for obj_idx in range(dsm.size(0)):
                    c = cur_class_t[obj_idx]
                    segment_t[c] = torch.max(segment_t[c], dsm[obj_idx])

            # Sum BCE-with-logits over all class channels and spatial positions
            loss_s += F.binary_cross_entropy_with_logits(
                cur_segment, segment_t, reduction="sum"
            )

        # Normalize by spatial size and apply the semantic loss weight
        return (loss_s * self.semantic_alpha) / (mask_h * mask_w)
