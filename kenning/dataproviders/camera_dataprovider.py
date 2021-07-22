"""
A DataProvider-derived class used to interface with a
camera or a dummy video device
(consult ffmpeg and v4l2loopback for configuration for dummy video devices)
"""

from kenning.core.dataprovider import DataProvider
from kenning.datasets.open_images_dataset import DectObject
from kenning.resources import coco_detection
import cv2
import numpy as np
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from pathlib import Path


class CameraDataProvider(DataProvider):
    def __init__(
            self,
            camera_device_id: int = -1,
            memory_layout: str = "NCHW",
            class_names: str = 'coco'):

        self.device_id = camera_device_id
        self.classnames = []
        if class_names == 'coco':
            with path(coco_detection, 'cocov6.classes') as p:
                with open(p, 'r') as f:
                    for line in f:
                        self.classnames.append(line.split(',')[1].strip())
        else:
            with Path(class_names) as p:
                with open(p, 'r') as f:
                    for line in f:
                        self.classnames.append(line.split(',')[1].strip())

        self.numclasses = len(self.classnames)
        self.image_width = 416
        self.image_height = 416
        self.memory_layout = memory_layout
        self.batch_size = 1

        self.device = None

        super().__init__()

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--video-file-path',
            help='Video file path (for cameras, use /dev/videoX where X is the device ID eg. /dev/video0)',  # noqa: E501
            type=Path,
            required=True
        )
        group.add_argument(
            '--image-memory-layout',
            help='Determines if images should be delivered in NHWC or NCHW format',  # noqa: E501
            choices=['NHWC', 'NCHW'],
            default='NCHW'
        )
        group.add_argument(
            '--classes',
            help='File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type',  # noqa: E501
            type=str,
            default='coco'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.camera_device_id,
            args.image_memory_layout,
            args.classes
        )

    def prepare(self):
        self.device = cv2.VideoCapture(self.device_id)

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(
            data,
            (self.image_width, self.image_height)
        )
        if self.memory_layout == "NCHW":
            img = np.transpose(data, (2, 0, 1))
            return np.array(img, dtype=np.float32) / 255.0
        else:
            return np.array(data, dtype=np.float32) / 255.0

    def fetch_input(self):
        ret, frame = self.device.read()
        if ret:
            return frame
        else:
            raise VideoCaptureDeviceException(self.device_id)
            return None

    def detach_from_source(self):
        if self.device:
            self.device.release()

    def compute_iou(self, b1: DectObject, b2: DectObject) -> float:
        """
        Computes the IoU between two bounding boxes.

        Parameters
        ----------
        b1 : DectObject
            First bounding box
        b2 : DectObject
            Second bounding box

        Returns
        -------
        float : IoU value
        """
        xmn = max(b1.xmin, b2.xmin)
        ymn = max(b1.ymin, b2.ymin)
        xmx = min(b1.xmax, b2.xmax)
        ymx = min(b1.ymax, b2.ymax)

        intersectarea = max(0, xmx - xmn) * max(0, ymx - ymn)

        b1area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin)
        b2area = (b2.xmax - b2.xmin) * (b2.ymax - b2.ymin)

        iou = intersectarea / (b1area + b2area - intersectarea)

        return iou


class VideoCaptureDeviceException(Exception):
    """
    Exception to be raised when VideoCaptureDevice malfunctions
    during frame capture
    """
    def __init__(self, device_id, message="Video device {} read error"):
        super().__init__(message.format(device_id))
