"""
A OutputCollector-derived class used to visualize bounding box
data on input images and display/save them.
"""

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.open_images_dataset import DectObject
import random
import cv2
import numpy as np
from typing import Tuple, List
from pathlib import Path
import sys
from collections import defaultdict


def generate_color() -> Tuple[int, int, int]:
    """
    Generates a random RGB color
    Returns
    -------
    Tuple[int, int, int] : color in (r,g,b) format
    """
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


class DetectionVisualizer(OutputCollector):
    def __init__(
            self,
            output_width: int = int(1920/2),
            output_height: int = int(1080/2),
            save_to_file: bool = False,
            save_path: Path = './',
            save_fps: int = 25,
            window_name: str = "stream"):

        self.window_name = window_name
        self.output_width = output_width
        self.output_height = output_height
        self.save_to_file = save_to_file
        self.video_fps = save_fps
        if save_to_file:
            self.save_path = Path(save_path)
            codec = cv2.VideoWriter_fourcc(*'avc1')
            self.out = cv2.VideoWriter(
                str(self.save_path),
                codec,
                self.video_fps,
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
        self.color_dict = defaultdict(generate_color)

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
        """
        Method used to add visualizations of the models output
        It draws bounding boxes, class names and score onto
        the original image

        Parameters
        ----------
        input_data : np.ndarray
            the original image
        output_data : List[DectObject]
            list of found objects represented as DectObjects

        Returns
        -------
        np.ndarray : the modified image with visualizations drawn
        """

        out_img = input_data
        for i in output_data:
            low_pair = self.compute_coordinates((i.xmin, i.ymin))
            high_pair = self.compute_coordinates((i.xmax, i.ymax))
            out_img = cv2.rectangle(
                out_img,
                low_pair,
                high_pair,
                self.color_dict[i.clsname],
                2
            )
            out_img = cv2.putText(
                out_img,
                i.clsname,
                (low_pair[0], low_pair[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color_dict[i.clsname],
                self.font_thickness
            )
            out_img = cv2.putText(
                out_img,
                "score: {:.2f}".format(i.score),
                (low_pair[0]+5, high_pair[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color_dict[i.clsname],
                self.font_thickness
            )
        return out_img

    def return_output(
            self,
            input_data: np.ndarray,  # since the original frames are passed in, this should always be HWC, uint8  # noqa: E501
            output_data: List[List[DectObject]]):
        output_data = output_data[0]
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

    def check_exit_condition(self):
        return cv2.waitKey(1) != 27

    def detach_from_output(self):
        if self.out:
            self.out.release()
        else:
            cv2.destroyWindow(self.window_name)
