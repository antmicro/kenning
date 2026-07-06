# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Utils for the YOLACT model implemented in PyTorch.
"""

import math
from functools import lru_cache
from itertools import product
from typing import Optional

import torch
import torchvision


def _point_form(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center form to corner form.

    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of shape (N, 4) with columns [cx, cy, w, h]
        in normalized coordinates.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 4) with columns [xmin, ymin, xmax, ymax]
        in normalized coordinates.
    """
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1
    )


def _center_size_form(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from corner form to center form.

    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of shape (N, 4) with columns [xmin, ymin, xmax, ymax]
        in normalized coordinates.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 4) with columns [cx, cy, w, h]
        in normalized coordinates.
    """
    return torch.cat(
        ((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1
    )


def _intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the intersection area between every pair of boxes.
    Boxes are given in corner form and batched.

    Parameters
    ----------
    box_a : torch.Tensor
        First set of boxes in corner form, shape (N, A, 4).
    box_b : torch.Tensor
        Second set of boxes in corner form, shape (N, B, 4).

    Returns
    -------
    torch.Tensor
        Pairwise intersection areas, shape (N, A, B).
    """
    n, A, B = box_a.size(0), box_a.size(1), box_b.size(1)
    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2),
    )

    return torch.clamp(max_xy - min_xy, min=0).prod(3)


def _jaccard(
    box_a: torch.Tensor, box_b: torch.Tensor, iscrowd: bool = False
) -> torch.Tensor:
    """
    Compute the IoU between two sets of boxes.

    Accepts either unbatched (2D) or batched (3D) inputs in corner form. A 2D
    input is treated as a single batch and the batch dimension is squeezed out
    of the result.

    Parameters
    ----------
    box_a : torch.Tensor
        Boxes in corner form, either (A, 4) or (N, A, 4).
    box_b : torch.Tensor
        Boxes in corner form, either (B, 4) or (N, B, 4).
    iscrowd : bool, optional
        If True, compute intersection-over-area-of-box_a instead of the
        standard intersection-over-union. Used for crowd annotations where
        the crowd box can contain many objects. Defaults to False.

    Returns
    -------
    torch.Tensor
        IoU (or IoA when iscrowd is True) matrix of shape (A, B) for
        unbatched inputs or (N, A, B) for batched inputs.
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = _intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(2)
        .expand_as(inter)
    )
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union

    return out if use_batch else out.squeeze(0)


def _encode(matched: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    """
    Encode matched ground-truth boxes into localization regression targets.

    Produces the offsets (relative to each prior) that the box head is trained
    to predict, scaled by the standard YOLACT variances [0.1, 0.2].

    Parameters
    ----------
    matched : torch.Tensor
        Ground-truth boxes in corner form, shape (N, 4), already matched to
        the corresponding priors, with columns [xmin, ymin, xmax, ymax].
    priors : torch.Tensor
        Prior anchor boxes in center form, shape (N, 4), with columns
        [cx, cy, w, h] in normalized coordinates.

    Returns
    -------
    torch.Tensor
        Encoded regression targets of shape (N, 4) with columns
        [delta_cx, delta_cy, delta_w, delta_h] scaled by the YOLACT variances.
    """
    variances = [0.1, 0.2]
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)


def _log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log-sum-exp over the class dimension.

    Used by the OHEM classification loss to compute the log denominator of the
    softmax without overflow.

    Parameters
    ----------
    x : torch.Tensor
        Logit tensor of shape (N, C) where C is the number of classes.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N,) with the log-sum-exp computed along the class
        dimension for each of the N priors.
    """
    x_max = x.data.max()

    return torch.log(torch.sum(torch.exp(x - x_max), 1)) + x_max


def _sanitize_coordinates(
    _x1: torch.Tensor,
    _x2: torch.Tensor,
    img_size: int,
    padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale a normalized coordinate pair to pixels, order it and clamp it.

    Multiplies the coordinates by img_size, guarantees x1 <= x2 (so
    reversed boxes are handled), applies optional padding and clamps to
    [0, img_size].

    Parameters
    ----------
    _x1 : torch.Tensor
        First normalized coordinate, shape (N,). May be larger than _x2 for
        degenerate boxes — the function swaps them if needed.
    _x2 : torch.Tensor
        Second normalized coordinate, shape (N,).
    img_size : int
        Pixel dimension used for denormalization (width or height).
    padding : int, optional
        Integer pixel margin subtracted from x1 and added to x2 before
        clamping. Defaults to 0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pair (x1, x2) of pixel-space tensors both of shape (N,),
        guaranteed to satisfy 0 <= x1 <= x2 <= img_size.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def _crop(
    masks: torch.Tensor, boxes: torch.Tensor, padding: int = 1
) -> torch.Tensor:
    """
    Zero out mask pixels lying outside their corresponding box.

    Used by both the training loss and inference post-processing so predicted
    masks are only kept inside the detection box.

    Parameters
    ----------
    masks : torch.Tensor
        Float tensor of shape (H, W, N) containing assembled prototype masks
        for N detections.
    boxes : torch.Tensor
        Normalized corner-form boxes of shape (N, 4) with columns
        [xmin, ymin, xmax, ymax], one per mask.
    padding : int, optional
        Extra pixel margin kept on each side of the box boundary.
        Defaults to 1.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (H, W, N) with the same dtype as masks,
        where pixels outside the padded box region are set to zero.
    """
    h, w, n = masks.size()
    x1, x2 = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = (
        torch.arange(w, device=masks.device, dtype=x1.dtype)
        .view(1, -1, 1)
        .expand(h, w, n)
    )
    cols = (
        torch.arange(h, device=masks.device, dtype=x1.dtype)
        .view(-1, 1, 1)
        .expand(h, w, n)
    )
    crop_mask = (
        (rows >= x1.view(1, 1, -1))
        & (rows < x2.view(1, 1, -1))
        & (cols >= y1.view(1, 1, -1))
        & (cols < y2.view(1, 1, -1))
    )

    return masks * crop_mask.float()


@torch.no_grad()
def _match(
    pos_thresh: float,
    neg_thresh: float,
    truths: torch.Tensor,
    priors: torch.Tensor,
    labels: torch.Tensor,
    crowd_boxes: Optional[torch.Tensor],
    loc_t: torch.Tensor,
    conf_t: torch.Tensor,
    idx_t: torch.Tensor,
    idx: int,
    crowd_iou_threshold: float,
) -> None:
    """
    Match priors to ground-truth boxes and write per-prior training targets.

    For one image, assigns each prior to the best-overlapping GT box, marks
    positives/negatives/neutrals by IoU thresholds, optionally neutralizes
    priors overlapping crowd regions, and writes the encoded localization
    targets, class labels and matched GT indices in place.

    Parameters
    ----------
    pos_thresh : float
        IoU threshold above which a prior is considered a positive match
        (typically 0.5).
    neg_thresh : float
        IoU threshold below which a prior is considered a clear negative.
        Priors between neg_thresh and pos_thresh are treated as neutral and
        ignored during training.
    truths : torch.Tensor
        Ground-truth boxes for one image in corner form, shape (M, 4).
    priors : torch.Tensor
        All anchor priors in center form, shape (P, 4).
    labels : torch.Tensor
        Integer class labels for the M ground-truth objects, shape (M,).
    crowd_boxes : Optional[torch.Tensor]
        Crowd-region boxes in corner form, shape (C, 4), or None if the image
        has no crowd annotations.
    loc_t : torch.Tensor
        Pre-allocated tensor of shape (B, P, 4) that receives the encoded
        localization targets for image idx in place.
    conf_t : torch.Tensor
        Pre-allocated long tensor of shape (B, P) that receives the class
        label for each prior (0 = background, -1 = neutral).
    idx_t : torch.Tensor
        Pre-allocated long tensor of shape (B, P) that receives the index
        of the matched GT box for each prior.
    idx : int
        Batch index of the current image; used to index into loc_t, conf_t
        and idx_t.
    crowd_iou_threshold : float
        Priors whose IoA with any crowd box exceeds this value are neutralized
        (set to -1 in conf_t).

    Returns
    -------
    None
        Results are written into loc_t, conf_t, and idx_t at position idx.
    """
    # Convert center priors to corner form for IoU computation
    decoded_priors = _point_form(priors)

    # Compute Jaccard overlap (IoU) between GT boxes and priors
    overlaps = _jaccard(truths, decoded_priors)
    best_truth_overlap, best_truth_idx = overlaps.max(0)

    # Bipartite matching: ensure each GT box gets
    #  at least one best-matching prior
    for _ in range(overlaps.size(0)):
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        j = best_prior_overlap.max(0)[1]
        i = best_prior_idx[j]

        overlaps[:, i] = -1
        overlaps[j, :] = -1
        best_truth_overlap[i] = 2
        best_truth_idx[i] = j

    # Assign labels and matched GT boxes to all priors (background class is 0)
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1

    # Filter by thresholds: set neutrals to -1 and background to 0
    conf[best_truth_overlap < pos_thresh] = -1
    conf[best_truth_overlap < neg_thresh] = 0

    # Ignore background priors that overlap heavily with crowd regions
    if crowd_boxes is not None and crowd_iou_threshold < 1:
        crowd_overlaps = _jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        best_crowd_overlap, _ = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > crowd_iou_threshold)] = -1

    # Encode bounding box offsets and save results in-place
    loc_t[idx] = _encode(matches, priors)
    conf_t[idx] = conf
    idx_t[idx] = best_truth_idx


@lru_cache(maxsize=1)
def _generate_priors() -> torch.Tensor:
    """
    Build the fixed set of anchor priors for a 550x550 input.

    Iterates the five FPN feature-map levels (sizes 69, 35, 18, 9, 5) and,
    for each cell, emits one prior per aspect ratio. The result only depends
    on the fixed input size, so it is cached and reused across calls.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (P, 4) with columns [cx, cy, w, h] in normalized
        coordinates, where P is the total number of priors across all
        feature-map levels and aspect ratios (sum of size**2 * len(aspects)
        for each level).
    """
    scales = [24, 48, 96, 192, 384]
    aspects = [1, 0.5, 2]
    sizes = [69, 35, 18, 9, 5]
    max_size = 550.0

    priors = []
    for idx, size in enumerate(sizes):
        scale = scales[idx]
        # Iterate over all grid cells in the feature map
        for i, j in product(range(size), range(size)):
            cx = (j + 0.5) / size
            cy = (i + 0.5) / size

            # Generate priors for all aspect ratios at this position
            for ar in aspects:
                ar = math.sqrt(ar)
                # Correct calculation for width and
                # height based on aspect ratio
                w = (scale * ar) / max_size

                # IMPORTANT: Keep h = w for backward compatibility with
                # the official YOLACT pretrained weights, which were trained
                # with a bug that accidentally made all anchors square
                h = (scale / ar) / max_size

                priors.append([cx, cy, w, h])

    return torch.tensor(priors, dtype=torch.float32)


def _decode_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted box offsets into absolute normalized boxes.

    Inverts encode applies the predicted offsets to the priors using
    the YOLACT variances (0.1, 0.2) and converts the result to corner form.

    Parameters
    ----------
    boxes : torch.Tensor
        Raw box-head outputs of shape (P, 4) with columns
        [delta_cx, delta_cy, delta_w, delta_h] encoded with the YOLACT
        variances, where P is the total number of priors.

    Returns
    -------
    torch.Tensor
        Decoded boxes of shape (P, 4) in corner form [xmin, ymin, xmax, ymax]
        in normalized coordinates.
    """
    priors = _generate_priors().to(boxes.device)

    cx = boxes[:, 0] * 0.1 * priors[:, 2] + priors[:, 0]
    cy = boxes[:, 1] * 0.1 * priors[:, 3] + priors[:, 1]
    w = torch.exp(boxes[:, 2] * 0.2) * priors[:, 2]
    h = torch.exp(boxes[:, 3] * 0.2) * priors[:, 3]

    # From [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )


def _fast_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5,
    top_k: int = 200,
) -> torch.Tensor:
    """
    Run per-class non-maximum suppression and return the kept indices.

    Suppression is done independently within each class: boxes are sorted by
    score, capped at top_k, and any box whose IoU with a higher-scoring
    box of the same class exceeds iou_threshold is dropped.

    Parameters
    ----------
    boxes : torch.Tensor
        Decoded boxes in corner form, shape (N, 4).
    scores : torch.Tensor
        Confidence scores for each detection, shape (N,).
    classes : torch.Tensor
        Integer class index for each detection, shape (N,).
    iou_threshold : float, optional
        Boxes with IoU above this value with a higher-scoring box of the same
        class are suppressed. Defaults to 0.5.
    top_k : int, optional
        Maximum number of boxes considered per class before NMS.
        Defaults to 200.

    Returns
    -------
    torch.Tensor
        Long tensor of kept indices into the original N detections, sorted by
        descending score. May be empty if all detections are suppressed.
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    kept_idx = torchvision.ops.batched_nms(
        boxes=boxes, scores=scores, idxs=classes, iou_threshold=iou_threshold
    )

    return kept_idx[:top_k]


def _select_detections(
    boxes: torch.Tensor,
    confs: torch.Tensor,
    coefs: torch.Tensor,
    score_threshold: float = 0.05,
    max_detections: int = 200,
    iou_threshold: float = 0.5,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Decode predictions, drop background, score-threshold, top-k and NMS.

    Pure post-processing step shared by any dataset wrapper: turns one
    image's raw head outputs into the surviving detections.

    Parameters
    ----------
    boxes : torch.Tensor
        Raw box-head outputs for one image, shape (P, 4).
    confs : torch.Tensor
        Raw class logits for one image, shape (P, num_classes), including
        the background class at index 0.
    coefs : torch.Tensor
        Mask coefficients for one image, shape (P, coef_dim).
    score_threshold : float, optional
        Detections whose max foreground score falls below this value are
        discarded before NMS. Defaults to 0.05.
    max_detections : int, optional
        Maximum number of detections kept after top-k filtering, applied both
        before and as the NMS top_k cap. Defaults to 200.
    iou_threshold : float, optional
        IoU threshold passed to _fast_nms. Defaults to 0.5.

    Returns
    -------
    Optional[torch.Tensor]
        Decoded boxes of shape (K, 4), or None.
    Optional[torch.Tensor]
        Scores of shape (K,), or None.
    Optional[torch.Tensor]
        Class indices of shape (K,), or None.
    Optional[torch.Tensor]
        Mask coefficients of shape (K, coef_dim), or None.
    """
    import torch.nn.functional as F

    # Decode bounding box offsets into
    # absolute corner coordinates (xmin, ymin, xmax, ymax)
    dets = _decode_boxes(boxes)

    # Convert raw confidences to probabilities via softmax,
    # drop the background class (index 0),
    # and extract the maximum score and corresponding class ID for each box
    scores, classes = F.softmax(confs, dim=-1)[:, 1:].max(dim=-1)

    # Filter out low-confidence detections based on the defined score threshold
    keep = scores > score_threshold
    if keep.sum() == 0:
        return None, None, None, None

    scores, classes, dets, coef = (
        scores[keep],
        classes[keep],
        dets[keep],
        coefs[keep],
    )

    # Keep only the top K highest-scoring
    # detections to speed up NMS and save memory
    if scores.shape[0] > max_detections:
        top = scores.topk(max_detections).indices
        scores, classes, dets, coef = (
            scores[top],
            classes[top],
            dets[top],
            coef[top],
        )

    # Apply Non-Maximum Suppression (NMS)
    #  to remove overlapping, redundant boxes
    keep_nms = _fast_nms(
        dets,
        scores,
        classes,
        iou_threshold=iou_threshold,
        top_k=max_detections,
    )

    # Return None if no detections survived the NMS filtering
    if keep_nms.shape[0] == 0:
        return None, None, None, None

    return dets[keep_nms], scores[keep_nms], classes[keep_nms], coef[keep_nms]


def _assemble_masks(
    proto: torch.Tensor,
    coef: torch.Tensor,
    boxes: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Assemble instance masks from prototypes and upsample to target size.

    Computes sigmoid(proto @ coef) per detection, crops each mask to its
    box and resizes to the output resolution.

    Parameters
    ----------
    proto : torch.Tensor
        Prototype feature map of shape (mask_h, mask_w, coef_dim) produced by
        the protonet for a single image.
    coef : torch.Tensor
        Mask coefficients of shape (N, coef_dim) for N surviving detections.
    boxes : torch.Tensor
        Normalized corner-form boxes of shape (N, 4) used to crop each
        assembled mask.
    target_h : int
        Output mask height in pixels (typically the original image height).
    target_w : int
        Output mask width in pixels (typically the original image width).

    Returns
    -------
    torch.Tensor
        Float tensor of shape (N, target_h, target_w) with assembled, cropped
        and upsampled instance masks. Values are in [0, 1] (sigmoid outputs
        before thresholding).
    """
    import torch
    import torch.nn.functional as F

    # Combine mask prototypes and instance
    # coefficients using matrix multiplication,
    # then apply sigmoid to squash the results into [0, 1] probabilities
    masks = torch.sigmoid(
        torch.einsum("hwc,nc->hwn", proto, coef)
    )  # (mask_h, mask_w, N)

    # Zero out any mask pixels that fall
    # outside of their predicted bounding boxes
    masks = _crop(masks, boxes)

    # Rearrange dimensions to standard PyTorch
    # image format (N, 1, H, W) for interpolation
    masks = masks.permute(2, 0, 1).unsqueeze(1)  # (N,1, mask_h, mask_w)

    # Upsample the masks from the feature map
    # resolution to the final target image size
    masks = F.interpolate(
        masks, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(1)

    return masks  # (N, H, W)


def _make_cosine_warmup_lambda(
    warmup_iters: int, total_iters: int
) -> callable:
    """
    Build a LambdaLR multiplier: linear warmup then cosine decay to 0.1.

    Parameters
    ----------
    warmup_iters : int
        Number of iterations over which the learning rate ramps linearly from
        0 to the base LR.
    total_iters : int
        Total number of training iterations, including the warmup phase. Used
        to compute the cosine decay progress.

    Returns
    -------
    callable
        Function lr_lambda(it: int) -> float compatible with
        torch.optim.lr_scheduler.LambdaLR. Returns a multiplier in [0, 1]
        that rises linearly during warmup and then decays via cosine annealing
        down to 0.1 of the base LR.
    """
    import math

    def lr_lambda(it):
        if it < warmup_iters:
            return it / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda
