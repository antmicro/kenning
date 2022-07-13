"""
Wrapper for YOLACT model for instance segmentation

Pretrained on COCO dataset
"""

from pathlib import Path

import onnx
import cv2
import numpy as np
from functools import reduce
import operator

from kenning.core.dataset import Dataset
from kenning.datasets.open_images_dataset import SegmObject
from kenning.core.model import ModelWrapper

import torch
import torch.nn.functional as F


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in
    the predicted bbox. Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords
            in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(
        boxes[:, 0],
        boxes[:, 2],
        w, padding, cast=False
    )
    y1, y2 = sanitize_coordinates(
        boxes[:, 1],
        boxes[:, 3],
        h, padding, cast=False
    )

    rows = torch.arange(
        w, device=masks.device, dtype=x1.dtype
    ).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(
        h, device=masks.device, dtype=x1.dtype
    ).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def sanitize_coordinates(_x1, _x2, img_size: int,
                         padding: int = 0, cast: bool = True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size. Also converts from relative to absolute coordinates
    and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


MEANS = np.array([103.94, 116.78, 123.68]).reshape(-1, 1, 1)
STD = np.array([57.38, 57.12, 58.40]).reshape(-1, 1, 1)


class YOLACT(ModelWrapper):
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
            score_threshold: float = 0.2,
            interpolation_mode: str = 'bilinear'
    ):
        self.model = None
        self.top_k = top_k
        self.interpolation_mode = interpolation_mode
        self.score_threshold = score_threshold
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

    def get_input_spec(self):
        return {'input_1': (1, 3, 550, 550)}, 'float32'

    def prepare_model(self):
        if not self.from_file:
            raise NotImplementedError(
                "Yolact ModelWrapper only supports loading model from a file."
            )
        self.load_model(self.modelpath)

    def load_model(self, modelpath):
        if self.model is not None:
            del self.model
        self.model = onnx.load_model(modelpath)

    def save_model(self, modelpath):
        raise NotImplementedError

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
        masks = torch.tensor(y['proto'] @ y['mask'].T)
        masks = torch.sigmoid(masks)
        masks = crop(masks, torch.tensor(y['box'])).permute(2, 0, 1)

        masks = F.interpolate(
            masks[None, ...],
            (self.w, self.h),
            mode=self.interpolation_mode,
            align_corners=False
        )[0]
        masks.gt_(0.5)
        masks = masks.numpy()

        boxes = torch.tensor(y['box'])
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0],
            boxes[:, 2],
            550, cast=False
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
            boxes[:, 1],
            boxes[:, 3],
            550, cast=False
        )
        y['box'] = (boxes / 550).numpy()

        if self.top_k is not None:
            idx = torch.argsort(torch.tensor(y['score']), 0, descending=True)
            idx = idx[:self.top_k]
            for k in y:
                if k != 'proto':
                    y[k] = y[k][idx]

        keep = y['score'] >= self.score_threshold
        for k in y:
            if k != "proto":
                y[k] = y[k][keep]

        Y = []
        for i in range(len(y['score'])):
            x1, y1, x2, y2 = y['box'][i, :]
            Y.append(SegmObject(
                clsname=self.dataset.get_class_names()[y['class'][i]],
                maskpath=None,
                xmin=x1,
                ymin=y1,
                xmax=x2,
                ymax=y2,
                mask=masks[i] * 255.,
                score=y['score'][i]
            ))
        return [Y]

    def run_inference(self, X):
        raise NotImplementedError

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)

    def get_output_formats(self):
        raise NotImplementedError

    def save_to_onnx(self, modelpath):
        raise NotImplementedError

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
        # Because it is a variable dependant on model input,
        # some maths is required to retrieve it.

        S = len(outputdata)
        f = np.dtype(np.float32).itemsize
        i = np.dtype(np.int64).itemsize
        num_dets = (S - 138 * 138 * 32 * f) // (37 * f + i)

        output_parameters = [
            ((num_dets, 4), np.float32, 'box'),
            ((num_dets, 32), np.float32, 'mask'),
            ((num_dets,), np.int64, 'class'),
            ((num_dets,), np.float32, 'score'),
            ((138, 138, 32), np.float32, 'proto')
        ]

        result = {}
        for shape, dtype, name in output_parameters:
            tensorsize = reduce(operator.mul, shape) * np.dtype(dtype).itemsize
            outputtensor = np.frombuffer(
                outputdata[:tensorsize],
                dtype=dtype
            ).reshape(shape)
            result[name] = outputtensor
            outputdata = outputdata[tensorsize:]

        return result
