"""
A DataProvider-derived class used to interface with a
camera or a dummy video device
(consult ffmpeg and v4l2loopback for configuration for dummy video devices)
"""

from kenning.core.dataprovider import DataProvider
import cv2
import numpy as np
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


class VideoCaptureDeviceException(Exception):
    """
    Exception to be raised when VideoCaptureDevice malfunctions
    during frame capture
    """
    def __init__(self, device_id, message="Video device {} read error"):
        super().__init__(message.format(device_id))
