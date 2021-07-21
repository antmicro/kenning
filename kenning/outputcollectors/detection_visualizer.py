"""
A OutputCollector-derived class used to visualize bounding box
data on input images and display/save them.
"""

from kenning.core.outputcollector import Outputcollector
from kenning.datasets.open_images_dataset import DectObject
import random
import cv2
import numpy as np
from typing import Tuple, List
from pathlib import Path
import sys


class DetectionVisualizer(Outputcollector):
    def __init__(
            self,
            output_width: int = int(1920/2),
            output_height: int = int(1080/2),
            image_layout: str = 'NCHW',
            save_to_file: bool = False,
            save_path: Path = './'):

        self.window_name = "Test-window"
        self.output_width = output_width
        self.output_height = output_height
        self.save_to_file = save_to_file
        self.save_path = Path(save_path)
        self.out = None
        if save_to_file:
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(
                str(self.save_path),
                codec,
                25,
                (self.output_width, self.output_height)
            )
        else:
            try:
                cv2.namedWindow(
                    self.window_name,
                    cv2.WINDOW_OPENGL+cv2.WINDOW_GUI_NORMAL+cv2.WINDOW_AUTOSIZE
                )
            except cv2.error:
                cv2.namedWindow(
                    self.window_name,
                    cv2.WINDOW_GUI_NORMAL+cv2.WINDOW_AUTOSIZE
                )
        self.font_scale = 0.7
        self.font_thickness = 2
        self.layout = image_layout

        super().__init__()

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--output-width',
            help='Width of the output window or file',
            type=int,
            default=int(1920/2)
        )
        group.add_argument(
            '--output-height',
            help='Height of the output window or file',
            type=int,
            default=int(1080/2)
        )
        group.add_argument(
            '--save-to-file',
            help='Save visualized output to file',
            action='store_true'
        )
        group.add_argument(
            '--save-path',
            help='Path to save the output images',
            required='--save-to-file' in sys.argv,  # TODO: test this StackOverflow solution  # noqa: E501
            type=Path,
            default='./'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.output_width,
            args.output_height,
            args.image_memory_layout,
            args.save_to_file,
            args.save_path
        )

    def get_class_color(self, clsname: str) -> Tuple[int, int, int]:
        """
        Generates a random RGB color seeded by the class name

        Parameters
        ----------
        clsname : str
            the class name

        Returns
        -------
        Tuple[int, int, int] : color in (r,g,b) format
        """
        random.seed(hash(clsname), version=2)
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    def compute_coordinates(
            self,
            coord: Tuple[float, float]) -> Tuple[int, int]:
        """
        Computes coordinates in pixel-position form from 0-1 floats

        Parameters
        ----------
        coord : Tuple[float, float]
            0-1 ranged coordinates

        Returns
        -------
        Tuple[int, int] : size-based coordinates
        """
        return (
            int(coord[0]*self.output_width),
            int(coord[1]*self.output_height)
        )

    def visualize_data(
            self,
            input_data: np.ndarray,
            output_data: List[DectObject]) -> np.ndarray:
        out_img = input_data
        for i in output_data:
            low_pair = self.compute_coordinates((i.xmin, i.ymin))
            high_pair = self.compute_coordinates((i.xmax, i.ymax))
            out_img = cv2.rectangle(
                out_img,
                low_pair,
                high_pair,
                self.get_class_color(i.clsname),
                2
            )
            out_img = cv2.putText(
                out_img,
                i.clsname,
                (low_pair[0], low_pair[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.get_class_color(i.clsname),
                self.font_thickness
            )
            out_img = cv2.putText(
                out_img,
                "score: {:.2f}".format(i.score),
                (low_pair[0]+5, high_pair[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.get_class_color(i.clsname),
                self.font_thickness
            )
        return out_img

    def return_output(self, input_data, output_data):
        # assume that output_data is provided as DectObjects,
        # TBD by actually running everything later
        output_data = output_data[0]
        if self.layout == 'NCHW':
            input_data = np.transpose(input_data, (1, 2, 0))
        input_data = np.multiply(input_data, 255).astype('uint8')
        input_data = cv2.resize(
            input_data,
            (self.output_width, self.output_height)
        )
        if self.save_to_file:
            self.out.write(
                self.visualize_data(input_data, output_data)
            )
        else:
            cv2.imshow(
                self.window_name,
                self.visualize_data(input_data, output_data)
            )

    def detach_from_output(self):
        if self.out:
            self.out.release()
        else:
            cv2.destroyWindow(self.window_name)
