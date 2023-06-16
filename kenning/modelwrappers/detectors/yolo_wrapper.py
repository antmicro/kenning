# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains methods for YOLO models for object detection.
"""

import re
import sys
import numpy as np
from collections import defaultdict

from kenning.resources import coco_detection
from kenning.core.model import ModelWrapper
from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import DetectObject, compute_dect_iou  # noqa: E501

from pathlib import Path
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class YOLOWrapper(ModelWrapper):

    arguments_structure = {
        'class_names': {
            'argparse_name': '--classes',
            'description': 'File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type',  # noqa: E501
            'default': 'coco',
            'type': str
        }
    }

    maxscore = 1.0
    thresh = 0.2
    iouthresh = 0.5
    finthresh = 0.2

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool = True,
            class_names: str = "coco"):
        self.class_names = class_names
        # for work with dataproviders, this is handling dataset-less operation
        super().__init__(modelpath, dataset, from_file)
        self.classnames = []
        if class_names == 'coco':
            with path(coco_detection, 'coco.names') as p:
                with open(p, 'r') as f:
                    for line in f:
                        self.classnames.append(line.strip())
        else:
            with Path(class_names) as p:
                with open(p, 'r') as f:
                    for line in f:
                        self.classnames.append(line.strip())
        self.numclasses = len(self.classnames)
        self.batch_size = 1
        if dataset:
            self.batch_size = dataset.batch_size
            assert self.numclasses == len(dataset.get_class_names())
        self.prepare_model()
        self.model_prepared = True

    @classmethod
    def from_argparse(
            cls,
            dataset: Dataset,
            args,
            from_file: bool = True):
        return cls(args.model_path, dataset, from_file, args.classes)

    @classmethod
    def load_config_file(cls, config_path):
        keyparamsrgx = re.compile(r'(width|height|classes)=(\d+)')
        perlayerrgx = re.compile(r'(mask|anchors|num)=((\d+,?)+)')
        keyparams = {}
        perlayerparams = defaultdict(list)
        with open(config_path.with_suffix(".cfg"), 'r') as config:
            for line in config:
                line = line.replace(' ', '')
                res = keyparamsrgx.match(line)
                if res:
                    keyparams[res.group(1)] = int(res.group(2))
                res = perlayerrgx.match(line)
                if res:
                    perlayerparams[res.group(1)].append(res.group(2))
        perlayerparams = {
            k: [np.array([int(x) for x in s.split(',')]) for s in v]
            for k, v in perlayerparams.items()
        }
        return keyparams, perlayerparams

    def load_model(self, modelpath):
        self.keyparams, self.perlayerparams = self.load_config_file(
            self.modelpath.with_suffix('.cfg')
        )
        self.save_io_specification(self.modelpath)

    def prepare_model(self):
        if self.model_prepared:
            return None
        self.load_model(self.modelpath)
        self.model_prepared = True

    def preprocess_input(self, X):
        return np.array(X)

    def convert_to_dectobject(self, entry):
        # array x, y, w, h, classid, score
        x1 = entry[0] - entry[2] / 2
        x2 = entry[0] + entry[2] / 2
        y1 = entry[1] - entry[3] / 2
        y2 = entry[1] + entry[3] / 2
        return DetectObject(
            self.classnames[entry[4]],
            x1, y1, x2, y2,
            entry[5] / self.maxscore,
            False
        )

    def parse_outputs(self, data):
        # get all bounding boxes with objectness score over given threshold
        boxdata = []
        for i in range(len(data)):
            ids = np.asarray(np.where(data[i][:, 4, :, :] > self.thresh))
            ids = np.transpose(ids)
            if ids.shape[0] > 0:
                ids = np.append([[i]] * ids.shape[0], ids, axis=1)
                boxdata.append(ids)

        if len(boxdata) > 0:
            boxdata = np.concatenate(boxdata)

        # each entry in boxdata contains:
        # - layer id
        # - det id
        # - y id
        # - x id

        bboxes = []
        for box in boxdata:
            # x and y values from network are coordinates in a chunk
            # to get the actual coordinates, we need to compute
            # new_coords = (chunk_coords + out_coords) / out_resolution
            x = (box[3] + data[box[0]][box[1], 0, box[2], box[3]]) / data[box[0]].shape[2]  # noqa: E501
            y = (box[2] + data[box[0]][box[1], 1, box[2], box[3]]) / data[box[0]].shape[3]  # noqa: E501

            # width and height are computed using following formula:
            # w = anchor_w * exp(out_w) / input_w
            # h = anchor_h * exp(out_h) / input_h
            # anchors are computed based on dataset analysis
            maskid = self.perlayerparams['mask'][box[0]][box[1]]
            anchors = self.perlayerparams['anchors'][box[0]][2 * maskid:2 * maskid + 2]  # noqa: E501
            w = anchors[0] * np.exp(data[box[0]][box[1], 2, box[2], box[3]]) / self.keyparams['width']  # noqa: E501
            h = anchors[1] * np.exp(data[box[0]][box[1], 3, box[2], box[3]]) / self.keyparams['height']  # noqa: E501

            # get objectness score
            objectness = data[box[0]][box[1], 4, box[2], box[3]]

            # get class with highest probability
            classid = np.argmax(data[box[0]][box[1], 5:, box[2], box[3]])

            # compute final class score (objectness * class probability)
            score = objectness * data[box[0]][box[1], classid + 5, box[2], box[3]]  # noqa: E501

            bboxes.append([x, y, w, h, classid, score])

        # sort the bboxes by score descending
        bboxes.sort(key=lambda x: x[5], reverse=True)

        bboxes = [self.convert_to_dectobject(b) for b in bboxes if b[5] / self.maxscore > self.finthresh]  # noqa: E501

        # group bboxes by class to perform NMS sorting
        grouped_bboxes = defaultdict(list)
        for item in bboxes:
            grouped_bboxes[item.clsname].append(item)

        # perform NMS sort to drop overlapping predictions for the same class
        cleaned_bboxes = []
        for clsbboxes in grouped_bboxes.values():
            for i in range(len(clsbboxes)):
                # if score equals 0, the bbox is dropped
                if clsbboxes[i].score == 0:
                    continue
                # add current bbox to final results
                cleaned_bboxes.append(clsbboxes[i])

                # look for overlapping bounding boxes with lower probability
                # and IoU exceeding specified threshold
                for j in range(i + 1, len(clsbboxes)):
                    if compute_dect_iou(clsbboxes[i], clsbboxes[j]) > self.iouthresh:  # noqa: E501
                        clsbboxes[j] = clsbboxes[j]._replace(score=0)
        return cleaned_bboxes

    def postprocess_outputs(self, y):
        # YOLOv3 has three stages of outputs
        # each one contains:
        # - real output
        # - masks
        # - biases

        # TVM-based model output provides 12 arrays
        # Those are subdivided into three groups containing
        # - actual YOLOv3 output
        # - masks IDs
        # - anchors
        # - 6 integers holding number of dects per cluster, actual output
        #   number of channels, actual output height and width, number of
        #   classes and unused parameter

        # iterate over each group
        lastid = 0
        outputs = []
        for i in range(3):
            # first extract the actual output
            # each output layer shape follows formula:
            # (BS, B * (4 + 1 + C), w / (8 * (i + 1)), h / (8 * (i + 1)))
            # BS is the batch size
            # w, h are width and height of the input image
            # the resolution is reduced over the network, and is 8 times
            # smaller in each dimension for each output
            # the "pixels" in the outputs are responsible for the chunks of
            # image - in the first output each pixel is responsible for 8x8
            # squares of input image, the second output covers objects from
            # 16x16 chunks etc.
            # Each "pixel" can predict up to B bounding boxes.
            # Each bounding box is described by its 4 coordinates,
            # objectness prediction and per-class predictions
            outshape = (
                self.batch_size,
                len(self.perlayerparams['mask'][i]),
                4 + 1 + self.numclasses,
                self.keyparams['width'] // (8 * 2 ** i),
                self.keyparams['height'] // (8 * 2 ** i)
            )

            outputs.append(
                y[lastid:(lastid + np.prod(outshape))].reshape(outshape)
            )

            # drop additional info provided in the TVM output
            # since it's all 4-bytes values, ignore the insides
            lastid += (
                np.prod(outshape)
                + len(self.perlayerparams['mask'][i])
                + len(self.perlayerparams['anchors'][i])
                + 6  # layer parameters
            )

        # change the dimensions so the output format is
        # batches layerouts dets params width height
        perbatchoutputs = []
        for i in range(outputs[0].shape[0]):
            perbatchoutputs.append([
                outputs[0][i],
                outputs[1][i],
                outputs[2][i]
            ])
        result = []
        # parse the combined outputs for each image in batch, and return result
        for out in perbatchoutputs:
            result.append(self.parse_outputs(out))

        return result

    def convert_input_to_bytes(self, inputdata):
        return inputdata.tobytes()

    def convert_output_from_bytes(self, outputdata):
        return np.frombuffer(outputdata, dtype='float32')

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        keyparams, _ = cls.load_config_file(json_dict['modelpath'])
        return cls._get_io_specification(keyparams)

    def get_io_specification_from_model(self):
        return self._get_io_specification(self.keyparams)
