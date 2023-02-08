# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for YOLACT model for instance segmentation

Pretrained on COCO dataset
"""

from pathlib import Path

import onnx
import cv2
import numpy as np
from functools import reduce
from typing import Tuple
import operator
import shutil

from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.core.model import ModelWrapper
from kenning.interfaces.io_interface import IOInterface
from kenning.datasets.coco_dataset import COCODataset2017


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
        Padding used for sanitize_coordinates function

    Returns
    -------
    numpy.ndarray :
        Masks for detected objects, each mask is cropped to the bounding box
        (there are no non-zero pixels outside the bbox)
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
            Array of (N,) elements
        _x2 : numpy.ndarray
            Array of (N,) elements
        img_size : int
            Upper bound for elements in the resulting array. Conversion from
            relative to absolute coordinates is done according to this number
        padding : int
            Margin how close the number can be to the margin before it is
            cropped. Smaller number is cropped to the max(x - padding, 0),
            higher number is min(x + padding, img_size).

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray] :
        Result is (x1, x2), each array has a (N,) shape, elementwise
        each element from both arrays satisfy: 0 <= x1 <= x2 <= img_size
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
        Input array

    Returns
    -------
    numpy.ndarray :
        Result of elementwise sigmoid function
    """
    return np.where(
        x >= 0,
        1. / (1. + np.exp(-x)),
        np.exp(x) / (1. + np.exp(x))
    )


MEANS = np.array([103.94, 116.78, 123.68]).reshape(-1, 1, 1)
STD = np.array([57.38, 57.12, 58.40]).reshape(-1, 1, 1)


class YOLACT(ModelWrapper):

    pretrained_modelpath = r'kenning/resources/models/instance_segmentation/yolact.onnx'    # noqa: 501
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
            'default': 0.
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file=True,
            top_k: int = None,
            score_threshold: float = 0.,
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

    @classmethod
    def from_argparse(cls, dataset, args, from_file=True):
        return cls(
            args.model_path,
            dataset,
            from_file,
            args.top_k,
            args.score_threshold
        )

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

        self.save_model_metadata(modelpath, {'class_names': self.class_names})

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

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)

    def get_output_formats(self):
        return ["onnx"]

    def save_to_onnx(self, modelpath):
        self.save_model(modelpath)

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata):
        # Signatures of outputs of the model:
        # BOX:   size=(num_dets, 4)  dtype=float32
        # MASK:  size=(num_dets, 32) dtype=float32
        # CLASS: size=(num_dets,)    dtype=int64
        # SCORE: size=(num_dets,)    dtype=float32
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

    def get_io_specification_from_model(self):
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
                'type': 'List[SegmObject]',
                'class_names': self.class_names
            }]
        }
