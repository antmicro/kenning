# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for YOLACT model for instance segmentation.

Pretrained on COCO dataset.
"""

from pathlib import Path

import onnx
import cv2
import numpy as np
from functools import reduce
from typing import Tuple, Dict, Optional, List
import operator
import pyximport
import shutil
import sys
if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files

from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import SegmObject  # noqa: E501
from kenning.core.model import ModelWrapper
from kenning.interfaces.io_interface import IOInterface
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.resources.models import instance_segmentation

pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from kenning.modelwrappers.instance_segmentation.cython_nms import nms  # noqa: E402, E501


def crop(
        masks: np.ndarray,
        boxes: np.ndarray,
        padding: int = 1
) -> np.ndarray:
    """
    "Crop" predicted masks by zeroing out everything not in
    the predicted bbox.

    Parameters
    ----------
    masks : numpy.ndarray
        Array of (H, W, N) elements, (H, W) being the dimension of an image,
        N being number of detected objects. Masks should contain elements
        from [0, 1] range, whether the pixel is a part of detected object
        or not.
    boxes : numpy.ndarray
        Boxes should be of (N, 4) shape, each box is defined by four numbers
        (x1, y1, x2, y2), where (x1, y1) are coordinates of northwestern point
        and (x2, y2) is coordinate for southeastern point. The coordinates are
        given in a relative form, i.e. each number is from [0, 1] interval,
        0 and 1 means point is on the margin of an image.
    padding : int
        Padding used for sanitize_coordinates function.

    Returns
    -------
    numpy.ndarray :
        Masks for detected objects, each mask is cropped to the bounding box
        (there are no non-zero pixels outside the bbox).
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(
        boxes[:, 0],
        boxes[:, 2],
        w, padding
    )
    y1, y2 = sanitize_coordinates(
        boxes[:, 1],
        boxes[:, 3],
        h, padding,
    )

    rows = np.arange(
        w, dtype=x1.dtype
    ).reshape(1, -1, 1).repeat(h, axis=0).repeat(n, axis=2)
    cols = np.arange(
        h, dtype=x1.dtype
    ).reshape(-1, 1, 1).repeat(w, axis=1).repeat(n, axis=2)

    masks_left = rows >= x1.reshape(1, 1, -1)
    masks_right = rows < x2.reshape(1, 1, -1)
    masks_up = cols >= y1.reshape(1, 1, -1)
    masks_down = cols < y2.reshape(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.astype(np.float32)


def sanitize_coordinates(
        _x1: np.ndarray,
        _x2: np.ndarray,
        img_size: int,
        padding: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size. Also converts from relative to absolute coordinates.

    Parameters
    ----------
    _x1 : numpy.ndarray
        Array of (N,) elements.
    _x2 : numpy.ndarray
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
    Tuple[numpy.ndarray, numpy.ndarray] :
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
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray :
        Result of element wise sigmoid function.
    """
    return np.where(
        x >= 0,
        1. / (1. + np.exp(-x)),
        np.exp(x) / (1. + np.exp(x))
    )


MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32).reshape(-1, 1, 1)
STD = np.array([57.38, 57.12, 58.40], dtype=np.float32).reshape(-1, 1, 1)
FACTOR = 255.0 / STD
RATIO = MEANS / STD


class YOLACTWrapper(ModelWrapper):

    default_dataset = COCODataset2017
    arguments_structure = {
        'top_k': {
            'argparse_name': '--top-k',
            'description': 'Maximum number of returned detected objects',
            'type': int,
            'default': None,
            'nullable': True
        },
        'score_threshold': {
            'argparse_name': '--score-threshold',
            'description': 'Option to filter out detected objects with score lower than the threshold',  # noqa: E501
            'type': float,
            'default': 0.05
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file=True,
            top_k: int = None,
            score_threshold: float = 0.05,
    ):
        self.model = None
        if dataset is not None:
            self.class_names = dataset.get_class_names()
        else:
            io_spec = self.load_io_specification(modelpath)
            segmentation_output = IOInterface.find_spec(
                io_spec,
                'processed_output',
                'segmentation_output'
            )
            self.class_names = segmentation_output['class_names']

        self.top_k = top_k
        self.score_threshold = score_threshold
        self.original_model_path = modelpath
        super().__init__(modelpath, dataset, from_file)

    def prepare_model(self):
        if self.model_prepared:
            return None
        if not self.from_file:
            raise NotImplementedError(
                "Yolact ModelWrapper only supports loading model from a file."
            )
        self.load_model(self.modelpath)
        self.model_prepared = True

    def load_model(self, modelpath):
        if self.model is not None:
            del self.model
        self.model = onnx.load_model(modelpath)

    def save_model(self, modelpath):
        shutil.copy(self.original_model_path, modelpath)

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)

    def get_output_formats(self):
        return ["onnx"]

    def save_to_onnx(self, modelpath):
        self.save_model(modelpath)

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification()

    def get_io_specification_from_model(self):
        io_spec = self._get_io_specification()
        io_spec['processed_output'][0]['class_names'] = self.class_names
        return io_spec


class YOLACT(YOLACTWrapper):

    pretrained_modelpath = files(instance_segmentation) / 'yolact.onnx'

    def preprocess_input(self, X):
        if len(X) > 1:
            raise RuntimeError(
                "YOLACT model expects only single image in a batch."
            )
        _, self.w, self.h = X[0].shape

        X = np.transpose(X[0], (1, 2, 0))
        X = cv2.resize(X, (550, 550))
        X = np.transpose(X, (2, 0, 1))
        X = (X * 255. - MEANS) / STD
        return X[None, ...].astype(np.float32)

    def postprocess_outputs(self, y):
        # The signature of the y input
        # output_0 - BOX
        # output_1 - MASK
        # output_2 - CLASS
        # output_3 - SCORE
        # output_4 - PROTO

        masks = y['output_4'] @ y['output_1'].T
        masks = sigmoid(masks)
        masks = crop(masks, y['output_0'])
        masks = cv2.resize(
            masks, (self.w, self.h), interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        y['output_1'] = (masks >= 0.5).astype(np.float32) * 255.

        boxes = y['output_0']
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0],
            boxes[:, 2],
            550
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
            boxes[:, 1],
            boxes[:, 3],
            550
        )
        y['output_0'] = (boxes / 550)

        if self.top_k is not None:
            idx = np.argsort(y['output_3'], 0)[:-(self.top_k + 1):-1]
            for k in y:
                if k != 'output_4':
                    y[k] = y[k][idx]

        keep = y['output_3'] >= self.score_threshold
        for k in y:
            if k != 'output_4':
                y[k] = y[k][keep]

        Y = []
        for i in range(len(y['output_3'])):
            x1, y1, x2, y2 = y['output_0'][i, :]
            Y.append(SegmObject(
                clsname=self.class_names[y['output_2'][i]],
                maskpath=None,
                xmin=x1,
                ymin=y1,
                xmax=x2,
                ymax=y2,
                mask=y['output_1'][i],
                score=y['output_3'][i],
                iscrowd=False
            ))
        return [Y]

    def convert_output_from_bytes(self, outputdata):
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

        output_specification = self.get_io_specification()['output']

        result = {}
        for spec in output_specification:
            name = spec['name']
            shape = list(
                num_dets if val == -1 else val for val in spec['shape']
            )
            dtype = np.dtype(spec['dtype'])
            tensorsize = reduce(operator.mul, shape) * dtype.itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(np.frombuffer(
                outputdata[:tensorsize],
                dtype=dtype
            )).reshape(shape)
            result[name] = outputtensor
            outputdata = outputdata[tensorsize:]

        return result

    @classmethod
    def _get_io_specification(cls):
        return {
            'input': [{'name': 'input', 'shape': (1, 3, 550, 550), 'dtype': 'float32'}],  # noqa: E501
            'output': [
                {'name': 'output_0', 'shape': (-1, 4), 'dtype': 'float32'},
                {'name': 'output_1', 'shape': (-1, 32), 'dtype': 'float32'},
                {'name': 'output_2', 'shape': (-1,), 'dtype': 'int64'},
                {'name': 'output_3', 'shape': (-1,), 'dtype': 'float32'},
                {'name': 'output_4', 'shape': (138, 138, 32), 'dtype': 'float32'}  # noqa: E501
            ],
            'processed_output': [{
                'name': 'segmentation_output',
                'type': 'List[SegmObject]'
            }]
        }


class YOLACTCore(YOLACTWrapper):

    pretrained_modelpath = (files(instance_segmentation) /
                            'yolact_core.onnx')

    def preprocess_input(self, X):
        if len(X) > 1:
            raise RuntimeError(
                "YOLACT model expects only single image in a batch."
            )
        _, self.h, self.w = X[0].shape
        X = np.resize(X[0].astype(np.float32), (1, 3, 550, 550))
        return X * FACTOR - RATIO

    def postprocess_outputs(self, y):
        if not y:
            return []

        y = self._detect(y)

        if not y:
            return []

        if self.top_k is not None:
            for k in y:
                if k != 'proto':
                    y[k] = y[k][:self.top_k]

        masks = sigmoid(y['proto'] @ y['mask'].T)
        masks = crop(masks, y['box'])
        masks = cv2.resize(
            masks, (self.w, self.h), interpolation=cv2.INTER_LINEAR
        )
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        y['mask'] = ((masks >= 0.5).astype(np.uint8) * 255
                     ).transpose(2, 0, 1)

        boxes = y['box']
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0],
            boxes[:, 2],
            550
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
            boxes[:, 1],
            boxes[:, 3],
            550
        )
        y['box'] = boxes / 550

        Y = []
        for i in range(len(y['score'])):
            x1, y1, x2, y2 = y['box'][i, :]
            Y.append(SegmObject(
                clsname=self.class_names[y['class'][i]],
                maskpath=None,
                xmin=x1,
                ymin=y1,
                xmax=x2,
                ymax=y2,
                mask=y['mask'][i],
                score=y['score'][i],
                iscrowd=False
            ))
        return [Y]

    def convert_output_from_bytes(self, outputdata):
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

        output_specification = self.get_io_specification()['output']

        result = []
        for spec in output_specification:
            shape = list(
                num_dets if val == -1 else val for val in spec['shape']
            )
            dtype = np.dtype(spec['dtype'])
            tensorsize = reduce(operator.mul, shape) * dtype.itemsize

            # Copy of numpy array is needed because the result of np.frombuffer
            # is not writeable, which breaks output postprocessing.
            outputtensor = np.array(np.frombuffer(
                outputdata[:tensorsize],
                dtype=dtype
            )).reshape(shape)
            result.append(outputtensor)
            outputdata = outputdata[tensorsize:]

        return result

    def _decode(self, loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """
        Decodes bounding boxes from the model outputs.

        Parameters
        ----------
        loc : numpy.ndarray
            Array of locations.
        priors : numpy.ndarray
            Array of priors.

        Returns
        -------
        numpy.ndarray :
            Array of bounding boxes.
        """
        variances = [0.1, 0.2]

        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _filter_detections(self,
                           conf_preds: np.ndarray,
                           decode_boxes: np.ndarray,
                           mask_data: np.ndarray,
                           nms_thresh: Optional[float] = 0.5
                           ) -> Dict[str, np.ndarray]:
        """
        Filters detections using confidence threshold and NMS.

        Parameters
        ----------
        conf_preds : numpy.ndarray
            Array of confidence predictions.
        decode_boxes: numpy.ndarray
            Array of decoded bounding boxes.
        mask_data : numpy.ndarray
            Array of mask data.
        nms_thresh : Optional[float]
            NMS threshold.

        Returns
        -------
        Dict[str, numpy.ndarray] :
            Dictionary of detected objects.
        """

        # Remove predictions with the background label
        cur_scores = conf_preds[1:, :]

        conf_scores = np.max(cur_scores, axis=0)
        keep = (conf_scores > self.score_threshold)
        scores = cur_scores[:, keep]
        boxes = decode_boxes[keep, :]
        masks = mask_data[keep, :]

        if scores.shape[1] == 0:
            return None

        # Apply NMS to boxes for each class separately
        idx_lst, cls_lst, scr_lst = [], [], []

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
        return {'box': boxes[idx], 'mask': masks[idx], 'class': classes[idx2],
                'score': scores}

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
        y : Dict[str, np.ndarray]
            Dictionary of model outputs.

        Returns
        -------
        Dict[str, np.ndarray] :
            Dictionary of model outputs with detected objects.
        """
        decode_boxes = self._decode(y[0].squeeze(0), y[3])
        result = self._filter_detections(y[1].squeeze(0).T,
                                         decode_boxes,
                                         y[2].squeeze(0))
        if result is not None:
            result['proto'] = y[4].squeeze(0)
        return result

    @classmethod
    def _get_io_specification(cls):
        return {
            'input': [{'name': 'input', 'shape': (1, 3, 550, 550), 'dtype': 'float32'}],    # noqa: E501
            'output': [
                {'name': 'output_0', 'shape': (1, -1, 4), 'dtype': 'float32'},
                {'name': 'output_1', 'shape': (1, -1, 81), 'dtype': 'float32'},
                {'name': 'output_2', 'shape': (1, -1, 32), 'dtype': 'float32'},
                {'name': 'output_3', 'shape': (-1, 4), 'dtype': 'float32'},
                {'name': 'output_4', 'shape': (1, 138, 138, 32), 'dtype': 'float32'}    # noqa: E501
            ],
            'processed_output': [{
                'name': 'segmentation_output',
                'type': 'List[SegmObject]'
            }]
        }
