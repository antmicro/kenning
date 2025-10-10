# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A wrapper for the MMPose model used for pose estimation.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.datasets.helpers.pose_estimation import Keypoint2D, Pose
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI
from kenning.utils.visualization_tools import draw_pose, generate_color


class MMPoseModelWrapper(ModelWrapper):
    """
    Wrapper used tu run inference with MMPose model,
    mainly RTMPose with optional Yolact detection and RTMO.
    """

    pretrained_model_uri = "kenning:///models/pose_estimation/mmpose_rtmo.onnx"
    default_dataset = ImageNetDataset
    arguments_structure = {
        "device": {"description": "Device used for inference.", "type": str},
        "keypoint_threshold": {
            "description": "Keypoints threshold.",
            "type": float,
            "default": 0.3,
        },
        "pose_input_size": {
            "description": "Input size of main pose model",
            "type": List[int],
            "default": [640, 640],
        },
        "model_type": {
            "description": "Model type to use rtmpose or rtmo",
            "type": str,
            "default": "rtmo",
            "enum": ["rtmo", "rtmpose"],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_type: str = "rtmo",
        pose_input_size: List[int] = [640, 640],
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        keypoint_threshold: float = 0.3,
    ):
        """
        Creates model wrapper for MMPose
        model.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Optional[Dataset]
            The dataset to verify the inference.
        from_file : bool
            True if model should be loaded from file.
        model_type : str
            Model type to use rtmpose or rtmo.
        pose_input_size : List[int]
            Input size of main pose model.
        model_name : Optional[str]
            Name of the model used for the report
        device : Optional[str]
            Name of the device in which
            model is going to be
            executed.
        keypoint_threshold : float
            Minimal score for keypoints to
            pass.
        """
        super().__init__(
            model_path=model_path,
            dataset=dataset,
            from_file=from_file,
            model_name=model_name,
        )

        self.model_prepared = False
        self.device = device

        self.pose_input_size = pose_input_size
        self.model_type = model_type

        self.keypoint_threshold = keypoint_threshold

        self._last_image = None
        self.color_dict = defaultdict(generate_color)

        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_model(self):
        if self.model_prepared:
            return
        self.model_prepared = True

        self.load_model(self.model_path)

    def load_model(self, model_path: PathOrURI):
        from rtmlib import RTMO, RTMPose

        if self.model_type == "rtmo":
            self.model = RTMO(
                str(self.model_path),
                model_input_size=self.pose_input_size,
                to_openpose=False,
                backend="onnxruntime",
                device=self.device,
            )
        else:
            self.model = RTMPose(
                str(self.model_path),
                model_input_size=self.pose_input_size,
                to_openpose=False,
                backend="onnxruntime",
                device=self.device,
            )

    def save_model(self, model_path: PathOrURI):
        raise NotSupportedError

    def save_to_onnx(self, model_path: PathOrURI):
        raise NotSupportedError

    def run_inference(self, X: List[np.ndarray]) -> List[Any]:
        self.prepare_model()

        if isinstance(X, list):
            X = X[0]

        keypoints, scores = self.model(X)

        return [keypoints, scores]

    def get_framework_and_version(self):
        import onnx

        return ("onnx", onnx.__version__)

    @classmethod
    def get_output_formats(cls) -> List[str]:
        return ["onnx"]

    @classmethod
    def _get_io_specification(cls, input_size: Tuple[int, int]):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": ([1, 3, -1, -1], [1, -1, -1, 3]),
                    "dtype": "float32",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": [input_size[0], input_size[1], 3],
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "keypoints",
                    "shape": [-1, 17, 2],
                    "dtype": "float32",
                },
                {
                    "name": "scores",
                    "shape": [-1, 17],
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {"name": "poses", "shape": [-1, -1, 3], "dtype": "float32"},
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
        # transpose from (c,w,h) to (w,h,c)
        x = X[0][0] * 255

        if x.shape[0] == 3:
            x = np.transpose(X, (1, 2, 0))

        self._last_image = np.ascontiguousarray(x)

        x = cv2.resize(x, (self.pose_input_size[1], self.pose_input_size[0]))

        return [x]

    def postprocess_outputs(
        self, y: List[List[np.ndarray]]
    ) -> Tuple[np.ndarray, List[Pose]] | List[List[Pose]]:
        obj_keypoints = y[0]
        obj_scores = y[1]

        poses = []

        output_img = self._last_image

        for keypoints, scores in zip(obj_keypoints, obj_scores):
            score_mask = scores >= self.keypoint_threshold

            filtered_keypoints = keypoints[score_mask] / self.pose_input_size

            if len(filtered_keypoints) == 0:
                return [output_img, []]

            keypoints_id = np.arange(start=0, stop=17, step=1)

            keypoints_id = keypoints_id[score_mask]

            xmin = np.min(filtered_keypoints[:, 0])
            xmax = np.max(filtered_keypoints[:, 0])

            ymin = np.min(filtered_keypoints[:, 1])
            ymax = np.max(filtered_keypoints[:, 1])

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            KLogger.debug(f"Keypoints: {filtered_keypoints}")

            output_keypoints = []

            for keypoint, id in zip(filtered_keypoints, keypoints_id):
                output_keypoints.append(
                    Keypoint2D(
                        x=(keypoint[0] - xmin) / bbox_width,
                        y=(keypoint[1] - ymin) / bbox_height,
                        id=id,
                    )
                )

            if len(filtered_keypoints) > 0:
                pose = Pose(
                    keypoints=output_keypoints,
                    segm=SegmObject(
                        score=1.0,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        iscrowd=False,
                        maskpath=None,
                        clsname="person",
                        mask=None,
                    ),
                )

                output_img = draw_pose(output_img, pose)

                poses.append(pose)

        return [output_img, [poses]]

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.pose_input_size)

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        raise NotSupportedError

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        return inputdata[0].tobytes()

    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        data = np.frombuffer(outputdata, dtype=np.float32)

        number_of_samples = data[0]

        data = data[1:]

        keypoints = data[: number_of_samples * 17 * 2].reshape(
            (number_of_samples, 17, 2)
        )

        data = data[number_of_samples * 17 * 2 :]

        scores = data.reshape((number_of_samples, 17))

        return [keypoints, scores]

    def train_model(self):
        raise NotSupportedError
