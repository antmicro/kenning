# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A wrapper for the MMPose model used for pose estimation.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnx

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.datasets.helpers.detection_and_segmentation import (
    SegmObject,
)
from kenning.datasets.helpers.pose_estimation import Keypoint2D, Pose
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class MMPoseRTMOONNX(ModelWrapper):
    """
    Wrapper used tu run inference with MMPose model
    using ONNXRuntime.
    """

    pretrained_model_uri = "kenning:///models/pose_estimation/mmpose_rtmo.onnx"
    arguments_structure = {
        "keypoint_threshold": {
            "description": "Minimal score for " "keypoint to pass.",
            "type": float,
            "default": 0.7,
        },
        "box_threshold": {
            "description": "Minimal score for " "keypoint to pass.",
            "type": float,
            "default": 0.45,
        },
        "input_size": {
            "description": "Model input size.",
            "type": List[int],
            "default": [640, 640],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        keypoint_threshold: float = 0.7,
        box_threshold: float = 0.45,
        input_size: List[int] = [640, 640],
    ):
        """
        Creates model wrapper for MMPose
        model in ONNX format.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Optional[Dataset]
            The dataset to verify the inference.
        from_file : bool
            True if model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report
        keypoint_threshold : float
            Minimal score for keypoint to
            pass.
        box_threshold : float
            Minimal score for bounding
            box to pass.
        input_size : List[int]
            Model input size.
        """
        super().__init__(
            model_path=model_path,
            dataset=dataset,
            from_file=from_file,
            model_name=model_name,
        )

        self.original_model_path = model_path

        self.model_prepared = False

        self.keypoint_threshold = keypoint_threshold
        self.box_threshold = box_threshold

        self.input_size = input_size

    def prepare_model(self):
        if self.model_prepared:
            return
        self.model_prepared = True

    def load_model(self, model_path: PathOrURI):
        pass

    def save_model(self, model_path: PathOrURI):
        raise NotSupportedError

    def save_to_onnx(self, model_path: PathOrURI):
        self.save_model(model_path)

    def run_inference(self, X: List[np.ndarray]) -> List[Any]:
        raise NotSupportedError

    def get_framework_and_version(self):
        return ("onnx", onnx.__version__)

    @classmethod
    def get_output_formats(cls) -> List[str]:
        return ["onnx"]

    def _get_io_specification(cls, input_size: List[int]):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": ([1, 3, -1, -1], [1, -1, -1, 3]),
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": [1, 3, input_size[1], input_size[0]],
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "det_outputs",
                    "shape": [1, -1, 5],
                    "dtype": "float32",
                },
                {
                    "name": "pose_outputs",
                    "shape": [1, -1, 17, 3],
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "pose_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.pose_estimation.Pose",  # noqa: E501
                    },
                },
            ],
        }

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        X = X[0][0]

        if X.shape[0] == 3:
            X = np.transpose(X, (1, 2, 0))

        if len(X.shape) == 3:
            padded_img = np.ones(
                (self.input_size[0], self.input_size[1], 3), dtype=np.float32
            ) * (114.0 / 255.0)
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.float32) * (
                114.0 / 255.0
            )

        ratio = min(
            self.input_size[0] / X.shape[0], self.input_size[1] / X.shape[1]
        )

        KLogger.debug(f"Ratio: {ratio}")

        resized_img = cv2.resize(
            X,
            (int(X.shape[1] * ratio), int(X.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        # X = cv2.resize(X, self.input_size)

        padded_shape = (int(X.shape[0] * ratio), int(X.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img

        # transpose from (c,w,h) to (w,h,c)
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # padded_img = (padded_img - mean) / std

        padded_img = padded_img[None, :]

        x = np.transpose(padded_img, (0, 3, 1, 2))

        x *= 255.0

        return [x]

    def postprocess_outputs(
        self, y: List[np.ndarray]
    ) -> List[List[List[Pose]]]:
        poses = []

        from rtmlib.tools.object_detection.post_processings import (
            multiclass_nms,
        )

        from kenning.utils.logger import KLogger

        det_outputs, pose_outputs = y

        boxes, bbox_scores = (det_outputs[0, :, :4], det_outputs[0, :, 4])

        keypoints, keypoints_scores = (
            pose_outputs[0, :, :, :2],
            pose_outputs[0, :, :, 2],
        )

        dets, keep = multiclass_nms(
            boxes,
            bbox_scores[:, np.newaxis],
            nms_thr=self.box_threshold,
            score_thr=self.keypoint_threshold,
        )

        if keep is not None:
            keypoints = keypoints[keep]
            scores = keypoints_scores[keep]
        else:
            keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
            scores = np.expand_dims(np.zeros_like(keypoints_scores[0]), axis=0)
            dets = [[0, 0, self.input_size[0], self.input_size[1], 1.0, 0.0]]

        for det, _keypoints, _scores in zip(
            list(dets), list(keypoints), list(scores)
        ):
            class_id = det[5]

            if class_id != 0:
                continue

            keypoints_id = np.arange(start=0, stop=17, step=1, dtype=np.uint32)

            box = det[0:4]

            xmin = box[0] / self.input_size[0]
            ymin = box[1] / self.input_size[1]

            xmax = box[2] / self.input_size[0]
            ymax = box[3] / self.input_size[1]

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            score = det[4]

            score_mask = _scores >= self.keypoint_threshold

            keypoints_id = keypoints_id[score_mask]

            passed_keypoints = _keypoints[score_mask] / self.input_size

            keypoints = []

            for id, keypoint in zip(keypoints_id, passed_keypoints):
                _x = (keypoint[0] - xmin) / bbox_width
                _y = (keypoint[1] - ymin) / bbox_height

                KLogger.debug(f"Points: {_x}, {_y}")

                keypoints.append(Keypoint2D(x=_x, y=_y, id=int(id)))

            segm = SegmObject(
                clsname="person",
                maskpath=None,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                mask=None,
                score=score,
                iscrowd=False,
            )

            pose = Pose(keypoints=keypoints, segm=segm)

            poses.append(pose)

        return [[poses]]

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.input_size)

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        raise NotSupportedError

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        return inputdata[0].tobytes()

    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        data = np.frombuffer(outputdata, dtype=np.float32)

        simcc_x = data[: 1 * 17 * 384].reshape(1, 17, 384)

        data = data[1 * 17 * 384 :]

        simcc_y = data.reshape(1 * 17 * 512)

        return [simcc_x, simcc_y]

    def train_model(self):
        raise NotSupportedError
