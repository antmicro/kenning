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
    FrameSegmObject,
    SegmObject,
)
from kenning.datasets.helpers.pose_estimation import Keypoint2D, Pose
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class MMPoseONNX(ModelWrapper):
    """
    Wrapper used tu run inference with MMPose model
    using ONNXRuntime.
    """

    pretrained_model_uri = (
        "kenning:///models/pose_estimation/mmpose_simcc.onnx"
    )
    arguments_structure = {
        "keypoint_threshold": {
            "description": "Keypoints threshold.",
            "type": float,
            "default": 0.3,
        },
        "input_size": {
            "description": "Model input size.",
            "type": List[int],
            "default": [192, 256],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        keypoint_threshold: float = 0.3,
        input_size: List[int] = [192, 256],
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

        self.input_size = input_size

        self._segm = []

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
                    "name": "simcc_x",
                    "shape": [1, 17, 384],
                    "dtype": "float32",
                },
                {
                    "name": "simcc_y",
                    "shape": [1, 17, 512],
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
        self._segm.append(
            SegmObject(
                score=1.0,
                xmin=0.0,
                ymin=0.0,
                xmax=1.0,
                ymax=1.0,
                clsname="",
                maskpath=None,
                mask=None,
                iscrowd=False,
            )
        )

        X = X[0][0]

        if X.shape[0] == 3:
            X = np.transpose(X, (1, 2, 0))

        X = cv2.resize(X, self.input_size)

        # transpose from (c,w,h) to (w,h,c)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        x = (X - mean) / std

        x = x[None, :]

        x = np.transpose(x, (0, 3, 1, 2))

        return [x]

    def postprocess_outputs(
        self, y: List[np.ndarray]
    ) -> List[List[List[Pose]]]:
        from rtmlib.tools.pose_estimation.post_processings import (
            get_simcc_maximum,
        )

        poses = []

        from kenning.utils.logger import KLogger

        locs, scores = get_simcc_maximum(y[0], y[1])

        for segm, _locs, _scores in zip(self._segm, locs, scores):
            raw_keypoints = _locs / 2

            raw_keypoints = raw_keypoints / self.input_size

            keypoints_id = np.arange(start=0, stop=17, step=1, dtype=np.uint32)

            score_mask = _scores >= self.keypoint_threshold

            passed_keypoints = raw_keypoints[score_mask]

            keypoints_id = keypoints_id[score_mask]

            keypoints = []

            for id, keypoint in zip(keypoints_id, passed_keypoints):
                _x = keypoint[0]
                _y = keypoint[1]

                KLogger.debug(f"Points: {_x}, {_y}")

                keypoints.append(Keypoint2D(x=_x, y=_y, id=int(id)))

            pose = Pose(keypoints=keypoints, segm=segm)

            poses.append(pose)

        self._segm.clear()

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


class MMPoseDetectionInput(MMPoseONNX):
    """
    ONNX MMPose model wrapper that takes
    detection output into account.
    """

    pretrained_model_uri = (
        "kenning:///models/pose_estimation/mmpose_simcc.onnx"
    )
    arguments_structure = {
        "class_name": {
            "description": "A class name to inference",
            "type": str,
            "default": "person",
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        class_name: str = "person",
        model_name: Optional[str] = None,
        keypoint_threshold: float = 0.3,
        input_size: List[int] = [192, 256],
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
        class_name : str
            Class name for detected object for
            which inference will be performed.
        model_name : Optional[str]
            Name of the model used for the report
        keypoint_threshold : float
            Minimal score for keypoint to
            pass.
        input_size : List[int]
            Model input size.
        """
        super().__init__(
            model_path=model_path,
            dataset=dataset,
            from_file=from_file,
            model_name=model_name,
            keypoint_threshold=keypoint_threshold,
            input_size=input_size,
        )

        self._class_name = class_name

    def _get_io_specification(cls, input_size: List[int]):
        return {
            "input": [
                {
                    "name": "segmentations",
                    "type": "List",
                    "dtype": "kenning.datasets.helpers."
                    "detection_and_segmentation.FrameSegmObject",
                },
            ],
            "processed_input": [
                {
                    "name": "input",
                    "shape": [-1, 3, input_size[1], input_size[0]],
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "simcc_x",
                    "shape": [-1, 17, 384],
                    "dtype": "float32",
                },
                {
                    "name": "simcc_y",
                    "shape": [-1, 17, 512],
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

    def preprocess_input(
        self, X: List[List[FrameSegmObject]]
    ) -> List[np.ndarray]:
        X = X[0][0]

        frame = X.frame

        KLogger.debug(f"Frame shape: {frame.shape}")

        frame = np.transpose(frame, (1, 2, 0))

        w, h, _ = frame.shape

        segments = X.segments

        images = []

        for segment in segments:
            if segment.clsname != self._class_name:
                KLogger.debug(f"Dropping class: {segment.clsname}")
                continue

            x = int(segment.xmin * w)
            y = int(segment.ymin * h)

            cw = int((segment.xmax) * w)
            ch = int((segment.ymax) * h)

            self._segm.append(
                SegmObject(
                    score=segment.score,
                    xmin=segment.xmin,
                    ymin=segment.ymin,
                    xmax=segment.xmax,
                    ymax=segment.ymax,
                    clsname=self._class_name,
                    iscrowd=False,
                    maskpath=None,
                    mask=segment.mask,
                )
            )

            KLogger.debug(f"Cropped {x},{y} size: {cw}, {ch}")

            X = frame[y:ch, x:cw]

            KLogger.debug(f"Cropped: {X.shape}")

            X = cv2.resize(X, self.input_size)

            # try to apply mask

            mask_obj = segment.mask.astype(np.uint8)

            mask_obj = cv2.resize(mask_obj, self.input_size)

            # transpose from (c,w,h) to (w,h,c)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            X = (X - mean) / std

            _x = np.transpose(X, (2, 0, 1))

            images.append(_x)

        if len(images) == 0:
            images.append(
                np.zeros(
                    shape=(
                        3,
                        self.input_size[1],
                        self.input_size[0],
                    ),
                    dtype=np.float32,
                )
            )
            self._segm.append(
                SegmObject(
                    score=1.0,
                    xmin=0.0,
                    ymin=0.0,
                    xmax=1.0,
                    ymax=1.0,
                    clsname="",
                    maskpath=None,
                    mask=None,
                    iscrowd=False,
                )
            )

        KLogger.debug(f"Images to parse: {len(images)}")

        output = np.stack(images, axis=0)

        KLogger.debug(f"Output shape: {output.shape}")

        return [output]
