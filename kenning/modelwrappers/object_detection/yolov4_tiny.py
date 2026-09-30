# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Sources for the YOLOv4-tiny ModelWrapper.

ModelWrapper for a YOLOv4-tiny model exported to ONNX with the darknet2onnx
tool from https://github.com/Tianxiaomo/pytorch-YOLOv4.

The original Darknet .cfg file must be placed next to the .onnx file, it is
used to recover the input size, the class count and the per-head anchor counts.
A class-name file with one name per line must be supplied through class_names.
"""

import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import onnx

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.helpers.detection_and_segmentation import (
    DetectObject,
    compute_detect_iou,
)
from kenning.modelwrappers.object_detection.yolo_wrapper import YOLOWrapper
from kenning.utils.resource_manager import PathOrURI


class ONNXYOLOV4Tiny(YOLOWrapper):
    """
    Model wrapper for a YOLOv4-tiny model exported to ONNX by the
    darknet2onnx tool, which emits decoded boxes and confs outputs.
    """

    pretrained_model_uri = "kenning:///models/detection/hand_detector.onnx"
    default_dataset = COCODataset2017
    arguments_structure = {
        "temporal_smoothing": {
            "description": "Smooth the top-scoring detection across frames "
            "with an exponential moving average and persist the last box for "
            "a few frames when a frame has no detection.",
            "type": bool,
            "default": False,
        },
        "smoothing_alpha": {
            "description": "Weight of the newest detection in the exponential "
            "moving average (0-1). Lower values smooth more but add lag.",
            "type": float,
            "default": 0.4,
        },
        "max_dropout_frames": {
            "description": "Maximum number of consecutive detection-less "
            "frames for which the last smoothed box is still emitted.",
            "type": int,
            "default": 5,
        },
    }

    thresh = 0.4

    # Downsampling factor of each detection head, in the order the heads are
    # emitted by the network (largest feature map first). YOLOv4-tiny emits
    # two heads at strides 16 and 32.
    STRIDES: List[int] = [16, 32]

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        class_names: str = "coco",
        temporal_smoothing: bool = False,
        smoothing_alpha: float = 0.4,
        max_dropout_frames: int = 5,
    ):
        super().__init__(
            model_path, dataset, from_file, model_name, class_names
        )
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.max_dropout_frames = max_dropout_frames
        # Per-batch-item temporal state for the top-scoring detection.
        self._smoothed: List[Optional[DetectObject]] = []
        self._dropouts: List[int] = []

    def _temporally_smooth(
        self, frame_dets: List[DetectObject], b: int
    ) -> List[DetectObject]:
        """
        Return the temporally smoothed detections for batch item ``b``.

        The top-scoring detection is tracked with an exponential moving
        average to reduce jitter, and the last smoothed box is re-emitted for
        up to ``max_dropout_frames`` consecutive frames without a detection so
        a stationary hand does not flicker out of existence.
        """
        if not self.temporal_smoothing:
            return frame_dets

        while len(self._smoothed) <= b:
            self._smoothed.append(None)
            self._dropouts.append(0)

        if not frame_dets:
            if (
                self._smoothed[b] is not None
                and self._dropouts[b] < self.max_dropout_frames
            ):
                self._dropouts[b] += 1
                return [self._smoothed[b]]
            self._smoothed[b] = None
            self._dropouts[b] = 0
            return []

        best = max(frame_dets, key=lambda d: d.score)
        prev = self._smoothed[b]
        if prev is None:
            smoothed = best
        else:
            a = self.smoothing_alpha
            smoothed = DetectObject(
                clsname=best.clsname,
                xmin=a * best.xmin + (1 - a) * prev.xmin,
                ymin=a * best.ymin + (1 - a) * prev.ymin,
                xmax=a * best.xmax + (1 - a) * prev.xmax,
                ymax=a * best.ymax + (1 - a) * prev.ymax,
                score=best.score,
                iscrowd=False,
            )
        self._smoothed[b] = smoothed
        self._dropouts[b] = 0

        result = [smoothed]
        result.extend(d for d in frame_dets if d is not best)
        return result

    @classmethod
    def load_config_file(cls, config_path: PathOrURI):
        """
        Parse the .cfg file and record the per-head anchor counts.
        """
        keyparams, perlayerparams = super().load_config_file(config_path)
        keyparams["anchors_per_head"] = [
            len(mask) for mask in perlayerparams.get("mask", [])
        ]
        return keyparams, perlayerparams

    def _num_boxes_for(self, keyparams: Dict[str, Any]) -> int:
        """
        Total number of candidate boxes across both detection heads.
        """
        anchors_per_head = keyparams.get("anchors_per_head") or [
            1 for _ in self.STRIDES
        ]
        total = 0
        for i, stride in enumerate(self.STRIDES):
            anchors = anchors_per_head[i] if i < len(anchors_per_head) else 1
            grid_h = keyparams["height"] // stride
            grid_w = keyparams["width"] // stride
            total += anchors * grid_h * grid_w
        return total

    def load_model(self, model_path: PathOrURI) -> None:
        """
        Load the .cfg file and cache the candidate-box count.
        """
        super().load_model(model_path)
        self.num_boxes = self._num_boxes_for(self.keyparams)

    def postprocess_outputs(self, y: List[np.ndarray]) -> Any:
        """
        Turn the decoded boxes and confs outputs into a list of
        DetectObject instances.
        """
        boxes = np.asarray(y[0])
        confs = np.asarray(y[1])

        # (batch, num_boxes, 1, 4) -> (batch, num_boxes, 4)
        if boxes.ndim == 4 and boxes.shape[2] == 1:
            boxes = boxes[:, :, 0, :]

        num_classes = confs.shape[-1]
        max_conf = confs.max(axis=2)
        max_id = confs.argmax(axis=2)

        per_batch: List[List[DetectObject]] = []
        for b in range(boxes.shape[0]):
            keep = max_conf[b] > self.thresh
            box_b = boxes[b][keep]
            conf_b = max_conf[b][keep]
            id_b = max_id[b][keep]

            # group candidate boxes by their best class
            by_class: Dict[int, List[DetectObject]] = {}
            for (x1, y1, x2, y2), score, cls in zip(box_b, conf_b, id_b):
                by_class.setdefault(int(cls), []).append(
                    DetectObject(
                        self.classnames[int(cls)],
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(score),
                        False,
                    )
                )

            frame_dets: List[DetectObject] = []
            for cls_id in range(num_classes):
                cls_dets = by_class.get(cls_id, [])
                if not cls_dets:
                    continue
                cls_dets.sort(key=lambda d: d.score, reverse=True)
                # NMS: drop overlapping lower-scored boxes
                for i in range(len(cls_dets)):
                    if cls_dets[i].score <= self.finthresh:
                        continue
                    for j in range(i + 1, len(cls_dets)):
                        if (
                            cls_dets[j].score > 0
                            and compute_detect_iou(cls_dets[i], cls_dets[j])
                            > self.iouthresh
                        ):
                            cls_dets[j] = cls_dets[j]._replace(score=0.0)
                frame_dets.extend(
                    [d for d in cls_dets if d.score > self.finthresh]
                )

            per_batch.append(self._temporally_smooth(frame_dets, b))

        return [per_batch]

    @classmethod
    def _get_io_specification(
        cls, keyparams: Dict[str, Any], batch_size: int
    ) -> Dict[str, List[Dict]]:
        """
        Build the input/output specification for the decoded model.
        """
        anchors_per_head = keyparams.get("anchors_per_head") or [
            1 for _ in cls.STRIDES
        ]
        classes = keyparams.get("classes", 1)
        num_boxes = 0
        for i, stride in enumerate(cls.STRIDES):
            anchors = anchors_per_head[i] if i < len(anchors_per_head) else 1
            num_boxes += (
                anchors
                * (keyparams["height"] // stride)
                * (keyparams["width"] // stride)
            )

        return {
            "input": [
                {
                    "name": "input",
                    "shape": (
                        batch_size,
                        3,
                        keyparams["width"],
                        keyparams["height"],
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "boxes",
                    "shape": (batch_size, num_boxes, 1, 4),
                    "dtype": "float32",
                },
                {
                    "name": "confs",
                    "shape": (batch_size, num_boxes, classes),
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "detection_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers."
                        "detection_and_segmentation.DetectObject",
                    },
                }
            ],
        }

    def convert_output_from_bytes(self, outputdata: bytes) -> List[np.ndarray]:
        """
        Rebuild the decoded boxes and confs tensors from a flat buffer.
        """
        y = np.frombuffer(outputdata, dtype="float32")
        boxes_shape = (self.batch_size, self.num_boxes, 1, 4)
        confs_shape = (self.batch_size, self.num_boxes, self.numclasses)
        n_boxes = int(np.prod(boxes_shape))
        boxes = y[:n_boxes].reshape(boxes_shape)
        confs = y[n_boxes : n_boxes + int(np.prod(confs_shape))].reshape(
            confs_shape
        )
        return [boxes, confs]

    @classmethod
    def get_framework(cls) -> str:
        return "onnx"

    @classmethod
    def get_framework_version(cls) -> str:
        return str(onnx.__version__)

    @classmethod
    def get_output_formats(cls) -> List[str]:
        return ["onnx"]

    def save_to_onnx(self, model_path: PathOrURI) -> None:
        self.save_model(model_path)

    def run_inference(self, X: List) -> Any:
        raise NotSupportedError

    def save_model(self, model_path: PathOrURI) -> None:
        shutil.copy(self.model_path, model_path)

    def train_model(self) -> None:
        raise NotSupportedError
