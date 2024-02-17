# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Collection of methods, classes and structures for
object detection models and datasets.
"""

from abc import ABC
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.logger import KLogger

DetectObject = namedtuple(
    "DetectObject",
    ["clsname", "xmin", "ymin", "xmax", "ymax", "score", "iscrowd"],
)
DetectObject.__doc__ = """
Represents single detectable object in an image.

Attributes
----------
clsname : str
    Class of the object.
xmin, ymin, xmax, ymax : float
    Coordinates of the bounding box.
score : float
    The probability of correctness of the detected object.
iscrowd : Optional[bool]
    Tells if the bounding box is a crowd of objects.
    None or False if not crowd, True if bounding box represents
    crowd.
"""

SegmObject = namedtuple(
    "SegmObject",
    [
        "clsname",
        "maskpath",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "mask",
        "score",
        "iscrowd",
    ],
)
SegmObject.__doc__ = """
Represents single segmentable mask in an image.

Attributes
----------
clsname : str
    Class of the object.
maskpath : Optional[Path]
    Path to mask file.
xmin,ymin,xmax,ymax : float
    Coordinates of the bounding box.
mask : np.array
    Loaded mask image.
score : float
    The probability of correctness of the detected object.
iscrowd : Optional[bool]
    Tells if the bounding box is a crowd of objects.
    None or False if not crowd, True if bounding box represents
    crowd.
"""


def compute_ap(
    recall: List[float], precision: List[float], points: int = 101
) -> float:
    """
    Computes N-point Average Precision for a single class.

    Parameters
    ----------
    recall : List[float]
        List of recall values.
    precision : List[float]
        List of precision values.
    points : int
        Number of points for Average Precision estimation.

    Returns
    -------
    float
        N-point interpolated average precision value.
    """
    return np.mean(
        np.interp(np.linspace(0, 1.0, num=points), recall, precision)
    )


def get_recall_precision(
    measurementsdata: Dict, scorethresh: float = 0.5, recallpoints: int = 101
) -> List[Tuple[List[float], List[float]]]:
    """
    Computes recall and precision values at a given objectness threshold.

    Parameters
    ----------
    measurementsdata : Dict
        Data from Measurements object with eval_gcount/{cls} and eval_det/{cls}
        fields containing number of ground truths and per-class detections.
    scorethresh : float
        Minimal objectness score threshold for detections.
    recallpoints : int
        Number of points to use for recall-precision curves, default 101
        (as in COCO dataset evaluation).

    Returns
    -------
    List[Tuple[List[float], List[float]]]
        List with per-class lists of recall and precision values.
    """
    lines = -np.ones(
        [len(measurementsdata["class_names"]), 2, recallpoints],
        dtype=np.float32,
    )
    for clsid, cls in enumerate(measurementsdata["class_names"]):
        gt_count = (
            measurementsdata[f"eval_gtcount/{cls}"]
            if f"eval_gtcount/{cls}" in measurementsdata
            else 0
        )
        if gt_count == 0:
            continue
        dets = (
            measurementsdata[f"eval_det/{cls}"]
            if f"eval_det/{cls}" in measurementsdata
            else []
        )
        dets = [d for d in dets if d[0] >= scorethresh]
        dets.sort(key=lambda d: -d[0])
        tps = np.array([entry[1] != 0.0 for entry in dets])
        fps = np.array([entry[1] == 0.0 for entry in dets])
        tpacc = np.cumsum(tps).astype(dtype=np.float)
        fpacc = np.cumsum(fps).astype(dtype=np.float)

        recalls = tpacc / gt_count
        precisions = tpacc / (fpacc + tpacc + np.spacing(1))

        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        recallthresholds = np.linspace(0.0, 1.0, num=recallpoints)
        inds = np.searchsorted(recalls, recallthresholds, side="left")
        newprecisions = np.zeros(recallthresholds.shape, dtype=np.float32)
        try:
            for oldid, newid in enumerate(inds):
                newprecisions[oldid] = precisions[newid]
        except IndexError:
            pass

        lines[clsid, 0] = recallthresholds
        lines[clsid, 1] = newprecisions
    return lines


def compute_map_per_threshold(
    measurementsdata: Dict, scorethresholds: List[float]
) -> List[float]:
    """
    Computes mAP values depending on the objectness threshold.

    Parameters
    ----------
    measurementsdata : Dict
        Data from Measurements object with eval_gcount/{cls} and eval_det/{cls}
        fields containing number of ground truths and per-class detections.
    scorethresholds : List[float]
        List of threshold values to verify the mAP for.

    Returns
    -------
    List[float]
        Mean Average Precision values per thresholds
    """
    maps = []
    for thresh in scorethresholds:
        recallprecisions = get_recall_precision(measurementsdata, thresh)
        precisions = recallprecisions[:, 1, :]
        maps.append(np.mean(precisions[precisions > -1]))

    return np.array(maps, dtype=np.float32)


def compute_detect_iou(b1: DetectObject, b2: DetectObject) -> float:
    """
    Computes the IoU between two bounding boxes.

    In evaluation, b1 is considered prediction, while
    b2 is the ground truth.

    If the ground truth has iscrowd parameter set to true,
    then the intersect area is divided by the area of the
    prediction bounding box instead of full area of the
    prediction and the ground truth.

    Parameters
    ----------
    b1 : DetectObject
        First bounding box.
    b2 : DetectObject
        Second bounding box.

    Returns
    -------
    float
        IoU value.
    """
    xmn = max(b1.xmin, b2.xmin)
    ymn = max(b1.ymin, b2.ymin)
    xmx = min(b1.xmax, b2.xmax)
    ymx = min(b1.ymax, b2.ymax)

    intersectarea = max(0, xmx - xmn) * max(0, ymx - ymn)

    b1area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin)
    b2area = (b2.xmax - b2.xmin) * (b2.ymax - b2.ymin)

    if b2.iscrowd:
        iou = intersectarea / b1area
    else:
        iou = intersectarea / (b1area + b2area - intersectarea)

    return iou


def compute_segm_iou(segm_pred: SegmObject, segm_true: SegmObject) -> float:
    """
    Computes IoU between two segmentation objects.

    Parameters
    ----------
    segm_pred : SegmObject
        Predicted segmentation.
    segm_true : SegmObject
        True segmentation.

    Returns
    -------
    float
        IoU value.
    """
    # Mask shapes must be equal
    if segm_pred.mask.shape != segm_true.mask.shape:
        KLogger.warn(
            "Segmentation masks have different shapes "
            f"({segm_pred.mask.shape} != {segm_true.mask.shape}). "
            "Returning 0.0"
        )
        return 0.0

    mask_i = np.logical_and(segm_pred.mask, segm_true.mask)
    mask_u = np.logical_or(segm_pred.mask, segm_true.mask)

    mask_u_nonzero = np.count_nonzero(mask_u)
    # in empty masks union is zero
    if mask_u_nonzero != 0:
        align = np.count_nonzero(mask_i) / mask_u_nonzero
    else:
        align = 0.0
    return align


class ObjectDetectionSegmentationDataset(Dataset, ABC):
    """
    Base for object detection datasets.
    """

    arguments_structure = {
        "task": {
            "argparse_name": "--task",
            "description": "The task type",
            "default": "object_detection",
            "enum": ["object_detection", "instance_segmentation"],
        },
        "image_memory_layout": {
            "argparse_name": "--image-memory-layout",
            "description": "Determines if images should be delivered in NHWC or NCHW format",  # noqa: E501
            "default": "NCHW",
            "enum": ["NHWC", "NCHW"],
        },
        "show_on_eval": {
            "argparse_name": "--show-predictions-on-eval",
            "description": "Show predictions during evaluation",
            "type": bool,
            "default": False,
        },
        "image_width": {
            "description": "Width of the input images",
            "type": int,
            "default": 416,
        },
        "image_height": {
            "description": "Height of the input images",
            "type": int,
            "default": 416,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        task: str = "object_detection",
        image_memory_layout: str = "NCHW",
        show_on_eval: bool = False,
        image_width: Optional[int] = 416,
        image_height: Optional[int] = 416,
    ):
        assert image_memory_layout in ["NHWC", "NCHW"]
        self.task = task
        self.classmap = {}
        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height
        self.classnames = []
        self.show_on_eval = show_on_eval
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
        )

    def train_test_split_representations(
        self,
        *args,
        stratify: bool = True,
        **kwargs,
    ) -> Tuple[List, ...]:
        return super().train_test_split_representations(
            *args,
            stratify=False,
            **kwargs,
        )

    def get_hashable(
        self, unhashable: Union["DetectObject", "SegmObject"]
    ) -> Union["DetectObject", "SegmObject"]:
        """
        Returns hashable versions of objects depending on `self.task`.

        Parameters
        ----------
        unhashable : Union['DetectObject', 'SegmObject']
            Object to be made hashable.

        Returns
        -------
        Union['DetectObject', 'SegmObject']
            The hashable object.
        """
        if self.task == "object_detection":
            hashable = unhashable
        elif self.task == "instance_segmentation":
            hashable = SegmObject(
                clsname=unhashable.clsname,
                maskpath=unhashable.maskpath,
                xmin=unhashable.xmin,
                ymin=unhashable.ymax,
                xmax=unhashable.xmax,
                ymax=unhashable.ymax,
                mask=None,
                score=1.0,
                iscrowd=False,
            )
        return hashable

    def compute_iou(
        self,
        b1: Union[DetectObject, SegmObject],
        b2: Union[DetectObject, SegmObject],
    ) -> float:
        """
        Computes the IoU between two bounding boxes.

        Parameters
        ----------
        b1 : Union[DetectObject, SegmObject]
            First bounding box.
        b2 : Union[DetectObject, SegmObject]
            Second bounding box.

        Returns
        -------
        float
            IoU value.
        """
        if self.task == "object_detection":
            return compute_detect_iou(b1, b2)
        elif self.task == "instance_segmentation":
            return compute_segm_iou(b1, b2)

    def show_eval_images(self, predictions: List, truth: List):
        """
        Shows the predictions on screen compared to ground truth.

        It uses proper method based on task parameter.

        The method runs a preview of inference results during
        evaluation process.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model.
        truth : List
            The ground truth for given batch.
        """
        KLogger.debug(f"\ntruth\n{truth}")
        KLogger.debug(f"\npredictions\n{predictions}")
        for idx, (pred, gt) in enumerate(zip(predictions, truth)):
            img_idx = self._dataindices[self._dataindex - len(truth) + idx]
            img = self.prepare_input_samples([self.dataX[img_idx]])[0]
            if self.image_memory_layout == "NCHW":
                img = img.transpose(1, 2, 0)
            int_img = np.multiply(img, 255).astype("uint8")
            int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
            int_img = cv2.cvtColor(int_img, cv2.COLOR_GRAY2RGB)
            int_img = self.apply_predictions(int_img, pred, gt)

            cv2.imshow("evaluatedimage", int_img)
            while True:
                c = cv2.waitKey(100)
                is_alive = cv2.getWindowProperty(
                    "evaluated image", cv2.WND_PROP_VISIBLE
                )
                if is_alive < 1 or c != -1:
                    break

    def apply_predictions(
        self, image: np.ndarray, predictions: List, ground_truth: List
    ) -> np.ndarray:
        """
        Applies predictions on the image.

        Parameters
        ----------
        image : np.ndarray
            The image to apply predictions on.
        predictions : List
            The list of predictions from the model.
        ground_truth : List
            The ground truth for given batch.

        Returns
        -------
        np.ndarray
            The image with predictions applied.
        """
        height, width = image.shape[0], image.shape[1]
        # Apply bounding boxes
        for pred_bb in predictions:
            cv2.rectangle(
                image,
                (int(pred_bb.xmin * width), int(pred_bb.ymin * height)),
                (int(pred_bb.xmax * width), int(pred_bb.ymax * height)),
                (0, 0, 255),
                2,
            )
        for truth_bb in ground_truth:
            cv2.rectangle(
                image,
                (int(truth_bb.xmin * width), int(truth_bb.ymin * height)),
                (int(truth_bb.xmax * width), int(truth_bb.ymax * height)),
                (0, 255, 0),
                2,
            )

        # Apply segmentation masks
        if self.task == "instance_segmentation":

            def apply_mask(
                image: np.array, mask: np.array, color: np.array
            ) -> np.array:
                """
                Applies the mask to the image.

                Parameters
                ----------
                image : np.array
                    The image to which the mask should be applied.
                mask : np.array
                    The mask to be applied.
                color : np.array
                    The color of the mask.

                Returns
                -------
                np.array
                    The image with the mask applied.
                """
                mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                mask_img = mask_img.astype("float32") / 255.0
                mask_img *= np.array(color)
                mask_img = np.multiply(mask_img, 255).astype("uint8")
                image = cv2.addWeighted(image, 1, mask_img, 0.7, 0)
                return image

            # Red
            for pred_mask in predictions:
                image = apply_mask(
                    image, pred_mask.mask, np.array([0.1, 0.1, 0.5])
                )
            # Green
            for truth_mask in ground_truth:
                image = apply_mask(
                    image, truth_mask.mask, np.array([0.1, 0.5, 0.1])
                )
        return image

    def evaluate(self, predictions, truth):
        MIN_IOU = 0.5
        MAX_DETS = 100
        measurements = Measurements()

        # TODO add support for specifying ground truth area ranges
        # TODO add support for specifying IoU ranges

        for preds, groundtruths in zip(predictions, truth):
            # operate on a single image
            # first, let's sort predictions by score
            preds.sort(key=lambda x: -x.score)

            preds = preds[:MAX_DETS]

            # store array of matched ground truth bounding boxes
            matchedgt = np.zeros([len(groundtruths)], dtype=np.int32)

            # for each prediction
            for predid, pred in enumerate(preds):
                # store index of best-matching ground truth
                bestiou = 0.0
                bestgt = -1
                # iterate over ground truth
                for gtid, gt in enumerate(groundtruths):
                    # skip mismatching classes
                    if pred.clsname != gt.clsname:
                        continue
                    # skip if ground truth is already matched and is not a
                    # crowd
                    if matchedgt[gtid] > 0 and not gt.iscrowd:
                        continue
                    iou = self.compute_iou(pred, gt)
                    if iou < bestiou:
                        continue
                    bestiou = iou
                    bestgt = gtid
                if bestgt == -1 or bestiou < MIN_IOU:
                    measurements.add_measurement(
                        f"eval_det/{pred.clsname}",
                        [[float(pred.score), float(0), float(bestiou)]],
                        lambda: list(),
                    )
                    continue
                measurements.add_measurement(
                    f"eval_det/{pred.clsname}",
                    [[float(pred.score), float(1), float(bestiou)]],
                    lambda: list(),
                )
                matchedgt[bestgt] = 1

            for gt in groundtruths:
                measurements.accumulate(
                    f"eval_gtcount/{gt.clsname}", 1, lambda: 0
                )

        if self.show_on_eval:
            self.show_eval_images(predictions, truth)

        return measurements
