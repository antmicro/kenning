"""
A Dataprovider-derived class used to interface with a
camera or a dummy video device
(consult ffmpeg and v4l2loopback for configuration for dummy video devices)
"""

from kenning.core.dataprovider import Dataprovider
import cv2
import numpy as np


class CameraDataprovider(Dataprovider):
    def __init__(
            self,
            camera_device_id: int = -1,
            memory_layout: str = "NCHW"):

        self.device_id = camera_device_id

        self.image_width = 416
        self.image_height = 416
        self.memory_layout = memory_layout

        super().__init__()

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--camera-device-id',
            help='Numeric ID of the camera device to be used',
            type=int,
            default=-1
        )
        group.add_argument(
            '--image-memory-layout',
            help='Determines if images should be delivered in NHWC or NCHW format',  # noqa: E501
            choices=['NHWC', 'NCHW'],
            default='NCHW'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.camera_device_id,
            args.image_memory_layout
        )

    def prepare(self):
        self.device = cv2.VideoCapture(self.device_id)

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(
            data,
            (self.image_width, self.image_height)
        )
        if self.memory_layout == "NCHW":
            return np.transpose(data, (2, 0, 1))
        else:
            return data

    def get_input(self):
        ret, frame = self.device.read()
        if ret:
            return self.preprocess_input(frame)
        else:
            return None
