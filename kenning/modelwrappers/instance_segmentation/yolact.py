# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for YOLACT model for instance segmentation.

Pretrained on COCO dataset.
"""

import operator
import shutil
from abc import ABC
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnx
import pyximport

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.interfaces.io_interface import IOInterface
from kenning.utils.resource_manager import PathOrURI

pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True
)
from kenning.modelwrappers.instance_segmentation.cython_nms import (  # noqa: E402
    nms,
)


def crop(masks: np.ndarray, boxes: np.ndarray, padding: int = 1) -> np.ndarray:
    """
    "Crop" predicted masks by zeroing out everything not in
    the predicted bbox.

    Parameters
    ----------
    masks : np.ndarray
        Array of (H, W, N) elements, (H, W) being the dimension of an image,
        N being number of detected objects. Masks should contain elements
        from [0, 1] range, whether the pixel is a part of detected object
        or not.
    boxes : np.ndarray
        Boxes should be of (N, 4) shape, each box is defined by four numbers
        (x1, y1, x2, y2), where (x1, y1) are coordinates of northwestern point
        and (x2, y2) is coordinate for southeastern point. The coordinates are
        given in a relative form, i.e. each number is from [0, 1] interval,
        0 and 1 means point is on the margin of an image.
    padding : int
        Padding used for sanitize_coordinates function.

    Returns
    -------
    np.ndarray
        Masks for detected objects, each mask is cropped to the bounding box
        (there are no non-zero pixels outside the bbox).
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(
        boxes[:, 1],
        boxes[:, 3],
        h,
        padding,
    )

    rows = (
        np.arange(w, dtype=x1.dtype)
        .reshape(1, -1, 1)
        .repeat(h, axis=0)
        .repeat(n, axis=2)
    )
    cols = (
        np.arange(h, dtype=x1.dtype)
        .reshape(-1, 1, 1)
        .repeat(w, axis=1)
        .repeat(n, axis=2)
    )

    masks_left = rows >= x1.reshape(1, 1, -1)
    masks_right = rows < x2.reshape(1, 1, -1)
    masks_up = cols >= y1.reshape(1, 1, -1)
    masks_down = cols < y2.reshape(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.astype(np.float32)


def sanitize_coordinates(
    _x1: np.ndarray, _x2: np.ndarray, img_size: int, padding: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size. Also converts from relative to absolute coordinates.

    Parameters
    ----------
    _x1 : np.ndarray
        Array of (N,) elements.
    _x2 : np.ndarray
        Array of (N,) elements.
    img_size : int
        Upper bound for elements in the resulting array. Conversion from
        relative to absolute coordinates is done according to this number.
    padding : int
        Margin how close the number can be to the margin before it is
        cropped. Smaller number is cropped to the max(x - padding, 0),
        higher number is min(x + padding, img_size).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Result is (x1, x2), each array has a (N,) shape, elementwise
        each element from both arrays satisfy: 0 <= x1 <= x2 <= img_size.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, 0, None)
    x2 = np.clip(x2 + padding, None, img_size)

    return x1, x2


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Result of element wise sigmoid function.
    """
    return np.where(
        x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))
    )


MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32)[:, None, None]
STD = np.array([57.38, 57.12, 58.40], dtype=np.float32)[:, None, None]


class YOLACTWrapper(ModelWrapper, ABC):
    """
    Abstract wrapper for YOLACT-based models.
    """

    default_dataset = COCODataset2017
    arguments_structure = {
        "top_k": {
            "argparse_name": "--top-k",
            "description": "Maximum number of returned detected objects",
            "type": int,
            "default": None,
            "nullable": True,
        },
        "score_threshold": {
            "argparse_name": "--score-threshold",
            "description": "Option to filter out detected objects with score lower than the threshold",  # noqa: E501
            "type": float,
            "default": 0.05,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file=True,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: float = 0.05,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.model = None
        if dataset is not None:
            self.class_names = dataset.get_class_names()
        else:
            io_spec = self.load_io_specification(self.model_path)
            segmentation_output = IOInterface.find_spec(
                io_spec, "processed_output", "segmentation_output"
            )
            self.class_names = segmentation_output["class_names"]

        self.top_k = top_k
        self.score_threshold = score_threshold
        self.original_model_path = self.model_path

    def prepare_model(self):
        if self.model_prepared:
            return None
        if not self.from_file:
            raise NotSupportedError(
                "Yolact ModelWrapper only supports loading model from a file."
            )
        self.load_model(self.model_path)
        self.model_prepared = True

    def load_model(self, model_path: PathOrURI):
        if self.model is not None:
            del self.model
        self.model = onnx.load_model(str(model_path))

    def save_model(self, model_path: PathOrURI):
        shutil.copy(self.original_model_path, model_path)

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocesses input image to be compatible with the model.

        Uses Z-score normalization with the mean and standard deviation
        of the MS COCO dataset.

        Assumes that the input is in RGB format with NCHW layout.

        Parameters
        ----------
        X : List[np.ndarray]
            A single image in a batch.

        Returns
        -------
        List[np.ndarray]
            Preprocessed image.

        Raises
        ------
        RuntimeError
            If the batch size is not 1.
        """
        if len(X) != 1:
            raise RuntimeError(
                "YOLACT model expects only single image in a batch."
            )
        X = X[0]
        _, self.w, self.h = X[0].shape
        X = np.transpose(X[0], (1, 2, 0))
        if X.max() > 1:
            X = X / 255.0
        X = cv2.resize(X, (550, 550))
        X = np.transpose(X, (2, 0, 1))
        X = (X * 255.0 - MEANS) / STD
        return [X[None, [2, 1, 0], ...].astype(np.float32)]

    def get_framework_and_version(self):
        return ("onnx", onnx.__version__)

    @classmethod
    def get_output_formats(cls):
        return ["onnx"]

    def save_to_onnx(self, model_path: PathOrURI):
        self.save_model(model_path)

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        return inputdata[0].tobytes()

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification()

    def get_io_specification_from_model(self):
        io_spec = self._get_io_specification()
        io_spec["processed_output"][0]["class_names"] = self.class_names
        return io_spec


class YOLACTWithPostprocessing(YOLACTWrapper):
    """
    Variant of YOLACT implementation with built-in postprocessing in the model.
    """

    pretrained_model_uri = "kenning:///models/instance_segmentation/yolact_with_postprocessing.onnx"

    def postprocess_outputs(
        self, y: List[np.ndarray]
    ) -> List[List[List[SegmObject]]]:
        # The signature of the y input
        # 0 - BOX
        # 1 - MASK
        # 2 - CLASS
        # 3 - SCORE
        # 4 - PROTO

        masks = y[4] @ y[1].T
        masks = sigmoid(masks)
        masks = crop(masks, y[0])
        # Resize masks to original image size in batches of 512 to avoid OOM
        masks = [
            cv2.resize(
                masks[:, :, i : i + 512],
                (self.h, self.w),
                interpolation=cv2.INTER_LINEAR,
            )
            for i in range(0, masks.shape[2], 512)
        ]
        masks = [
            np.expand_dims(m, axis=2) if len(m.shape) == 2 else m
            for m in masks
        ]
        masks = np.concatenate(masks, axis=2).transpose(2, 0, 1)
        y[1] = (masks >= 0.5).astype(np.float32) * 255.0

        boxes = y[0]
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], 550
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], 550
        )
        y[0] = boxes / 550

        if self.top_k is not None:
            idx = np.argsort(y[3], 0)[: -(self.top_k + 1) : -1]
            for k in range(len(y)):
                if k != 4:
                    y[k] = y[k][idx]

        keep = y[3] >= self.score_threshold
        for k in range(len(y)):
            if k != 4:
                y[k] = y[k][keep]

        Y = []
        for i in range(len(y[3])):
            x1, y1, x2, y2 = y[0][i, :]
            Y.append(
                SegmObject(
                    clsname=self.class_names[y[2][i]],
                    maskpath=None,
                    xmin=x1,
                    ymin=y1,
                    xmax=x2,
                    ymax=y2,
                    mask=y[1][i],
                    score=y[3][i],
                    iscrowd=False,
                )
            )
        return [[Y]]

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        # Signatures of outputs of the model:
        # BOX:   size=(num_dets, 4)  dtype=float32
        # MASK:  size=(num_dets, 32) dtype=float32
        # CLASS: size=(num_dets, )   dtype=int64
        # SCORE: size=(num_dets, )   dtype=float32
        # PROTO: size=(138, 138, 32) dtype=float32
        # Where num_dets is a number of detected objects.
        # Because it is a variable dependent on model input,
        # some maths is required to retrieve it.

        S = len(outputdata)
        f = np.dtype(np.float32).itemsize
        i = np.dtype(np.int64).itemsize
        num_dets = (S - 138 * 138 * 32 * f) // (37 * f + i)

        output_specification = self.get_io_specification()["output"]

        result = []
        for spec in output_specification:
            shape = list(
                num_dets if val == -1 else val for val in spec["shape"]
            )
            dtype = np.dtype(spec["dtype"])
            tensorsize = reduce(operator.mul, shape) * dtype.itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(
                np.frombuffer(outputdata[:tensorsize], dtype=dtype)
            ).reshape(shape)
            result.append(outputtensor)
            outputdata = outputdata[tensorsize:]

        return result

    @classmethod
    def _get_io_specification(cls):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": (1, 3, -1, -1),
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input",
                    "shape": (1, 3, 550, 550),
                    "dtype": "float32",
                }
            ],
            "output": [
                {"name": "output_0", "shape": (-1, 4), "dtype": "float32"},
                {"name": "output_1", "shape": (-1, 32), "dtype": "float32"},
                {"name": "output_2", "shape": (-1,), "dtype": "int64"},
                {"name": "output_3", "shape": (-1,), "dtype": "float32"},
                {
                    "name": "output_4",
                    "shape": (138, 138, 32),
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "segmentation_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.SegmObject",  # noqa: E501
                    },
                }
            ],
        }

    def run_inference(self, X: List) -> Any:
        raise NotSupportedError

    def train_model(self):
        raise NotSupportedError("This model does not support training.")


class YOLACT(YOLACTWrapper):
    """
    Model wrapper for YOLACT model provided in ONNX format.
    """

    pretrained_model_uri = (
        # YOLACT with ResNet50 backbone
        "kenning:///models/instance_segmentation/yolact.onnx"
    )

    def postprocess_outputs(
        self, y: List[np.ndarray]
    ) -> List[List[List[SegmObject]]]:
        y = self._detect(y)

        if not y:
            return [[] for _ in self.io_specification["processed_output"]]

        if self.top_k is not None:
            for k in y:
                if k != "proto":
                    y[k] = y[k][: self.top_k]

        masks = sigmoid(y["proto"] @ y["mask"].T)
        masks = crop(masks, y["box"])
        # Resize masks to original image size in batches of 512 to avoid OOM
        masks = [
            cv2.resize(
                masks[:, :, i : i + 512],
                (self.h, self.w),
                interpolation=cv2.INTER_LINEAR,
            )
            for i in range(0, masks.shape[2], 512)
        ]
        masks = [
            np.expand_dims(m, axis=2) if len(m.shape) == 2 else m
            for m in masks
        ]
        masks = np.concatenate(masks, axis=2)
        y["mask"] = ((masks >= 0.5).astype(np.uint8) * 255).transpose(2, 0, 1)

        boxes = y["box"]
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], 550
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], 550
        )
        y["box"] = boxes / 550

        Y = []
        for i in range(len(y["score"])):
            x1, y1, x2, y2 = y["box"][i, :]
            Y.append(
                SegmObject(
                    clsname=self.class_names[y["class"][i]],
                    maskpath=None,
                    xmin=x1,
                    ymin=y1,
                    xmax=x2,
                    ymax=y2,
                    mask=y["mask"][i],
                    score=y["score"][i],
                    iscrowd=False,
                )
            )
        return [[Y]]

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        # Signatures of outputs of the model:
        # LOC:    size=(1, num_dets, 4)     dtype=float32
        # CONF:   size=(1, num_dets, 81)    dtype=float32
        # MASK:   size=(1, num_dets, 32)    dtype=float32
        # PRIORS: size=(num_dets, 4)        dtype=float32
        # PROTO:  size=(1, 138, 138, 32)    dtype=float32
        # Where num_dets is a number of detected objects.
        # Because it is a variable dependent on model input,
        # some maths is required to retrieve it.

        S = len(outputdata)
        f = np.dtype(np.float32).itemsize
        num_dets = (S - 138 * 138 * 32 * f) // (121 * f)

        output_specification = self.get_io_specification()["output"]

        result = []
        for spec in output_specification:
            shape = list(
                num_dets if val == -1 else val for val in spec["shape"]
            )
            dtype = np.dtype(spec["dtype"])
            tensorsize = reduce(operator.mul, shape) * dtype.itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(
                np.frombuffer(outputdata[:tensorsize], dtype=dtype)
            ).reshape(shape)
            result.append(outputtensor)
            outputdata = outputdata[tensorsize:]

        return result

    def _decode(self, loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """
        Decodes bounding boxes from the model outputs.

        Parameters
        ----------
        loc : np.ndarray
            Array of locations.
        priors : np.ndarray
            Array of priors.

        Returns
        -------
        np.ndarray
            Array of bounding boxes.
        """
        variances = [0.1, 0.2]

        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _filter_detections(
        self,
        conf_preds: np.ndarray,
        decode_boxes: np.ndarray,
        mask_data: np.ndarray,
        nms_thresh: Optional[float] = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Filters detections using confidence threshold and NMS.

        Parameters
        ----------
        conf_preds : np.ndarray
            Array of confidence predictions.
        decode_boxes : np.ndarray
            Array of decoded bounding boxes.
        mask_data : np.ndarray
            Array of mask data.
        nms_thresh : Optional[float]
            NMS threshold.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of detected objects.
        """
        # Remove predictions with the background label
        cur_scores = conf_preds[1:, :]

        conf_scores = np.max(cur_scores, axis=0)
        keep = conf_scores > self.score_threshold
        scores = cur_scores[:, keep]
        boxes = decode_boxes[keep, :]
        masks = mask_data[keep, :]

        if scores.shape[1] == 0:
            return None

        # Apply NMS to boxes for each class separately
        idx_lst, cls_lst, scr_lst = [], [], []

        boxes = boxes * 550
        for _cls in range(scores.shape[0]):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > self.score_threshold
            idx = np.arange(cls_scores.size)

            cls_scores = cls_scores[conf_mask]
            cls_boxes = boxes[conf_mask]

            keep = nms(cls_boxes, cls_scores, nms_thresh)

            idx_lst.append((idx[conf_mask])[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])

        idx = np.concatenate(idx_lst, axis=0)
        classes = np.concatenate(cls_lst, axis=0)
        scores = np.concatenate(scr_lst, axis=0)

        scores, idx2 = np.sort(scores)[::-1], np.argsort(scores)[::-1]
        idx = idx[idx2]
        return {
            "box": boxes[idx] / 550,
            "mask": masks[idx],
            "class": classes[idx2],
            "score": scores,
        }

    def _detect(self, y: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detects objects from the model outputs.

        The signature of the y output by index:
        * 0 - LOC
        * 1 - CONF
        * 2 - MASK
        * 3 - PRIORS
        * 4 - PROTO

        Parameters
        ----------
        y : List[ np.ndarray]
            Dictionary of model outputs.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of model outputs with detected objects.
        """
        decode_boxes = self._decode(y[0].squeeze(0), y[3])
        result = self._filter_detections(
            y[1].squeeze(0).T, decode_boxes, y[2].squeeze(0)
        )
        if result is not None:
            result["proto"] = y[4].squeeze(0)
        return result

    @classmethod
    def _get_io_specification(cls):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": (1, 3, -1, -1),
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input",
                    "shape": (1, 3, 550, 550),
                    "dtype": "float32",
                }
            ],
            "output": [
                {"name": "output_0", "shape": (1, -1, 4), "dtype": "float32"},
                {"name": "output_1", "shape": (1, -1, 81), "dtype": "float32"},
                {"name": "output_2", "shape": (1, -1, 32), "dtype": "float32"},
                {"name": "output_3", "shape": (-1, 4), "dtype": "float32"},
                {
                    "name": "output_4",
                    "shape": (1, 138, 138, 32),
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "segmentation_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.SegmObject",  # noqa: E501
                    },
                }
            ],
        }

    def run_inference(self, X: List) -> Any:
        raise NotSupportedError

    def train_model(self):
        raise NotSupportedError("This model does not support training.")
