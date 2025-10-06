# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script with function useful for visualization data.
"""

import colorsys
from enum import IntEnum
from typing import Tuple

import cv2
import numpy as np

from kenning.datasets.helpers.pose_estimation import Pose


class Limbs(IntEnum):
    """
    Enum with limbs names
    used in pose visualization.
    """

    Nose = (0,)
    REye = (1,)
    LEye = (2,)
    REar = (3,)
    LEar = (4,)
    RShoulder = (5,)
    LShoulder = (6,)
    RElbow = (7,)
    LElbow = (8,)
    RHand = (9,)
    LHand = (10,)
    RHip = (11,)
    LHip = (12,)
    RKnee = (13,)
    LKnee = (14,)
    RFoot = (15,)
    LFoot = 16


FACE_COLOR = (0, 0, 255)
TORSO_COLOR = (255, 255, 0)
RIGHT_ARM_COLOR = (255, 0, 0)
LEFT_ARM_COLOR = (0, 255, 255)
RIGHT_LEG_COLOR = (0, 255, 0)
LEFT_LEG_COLOR = (255, 0, 255)

POSE_LIMBS = {
    Limbs.REye: {Limbs.LEye: FACE_COLOR, Limbs.REar: FACE_COLOR},
    Limbs.LEye: {Limbs.LEar: FACE_COLOR},
    Limbs.LEar: {Limbs.Nose: FACE_COLOR},
    Limbs.REar: {Limbs.Nose: FACE_COLOR},
    Limbs.Nose: {Limbs.RShoulder: TORSO_COLOR, Limbs.LShoulder: TORSO_COLOR},
    Limbs.RShoulder: {
        Limbs.LShoulder: TORSO_COLOR,
        Limbs.RElbow: RIGHT_ARM_COLOR,
        Limbs.RHip: TORSO_COLOR,
    },
    Limbs.RElbow: {
        Limbs.RHand: RIGHT_ARM_COLOR,
    },
    Limbs.LShoulder: {Limbs.LHip: TORSO_COLOR, Limbs.LElbow: LEFT_ARM_COLOR},
    Limbs.RHip: {Limbs.LHip: TORSO_COLOR, Limbs.RKnee: RIGHT_LEG_COLOR},
    Limbs.RKnee: {Limbs.RFoot: RIGHT_LEG_COLOR},
    Limbs.LHip: {Limbs.LKnee: LEFT_LEG_COLOR},
    Limbs.LKnee: {Limbs.LFoot: LEFT_LEG_COLOR},
}


def generate_color() -> Tuple[float, float, float]:
    """
    Generates a random RGB color.

    Returns
    -------
    Tuple[float, float, float]
        Color in (r,g,b) format.
    """
    return colorsys.hsv_to_rgb(np.random.rand(), 1, 1)


def draw_pose(
    input_img: np.ndarray,
    pose: Pose,
    keypoint_size: int = 3,
    bbox_thickness: int = 2,
    line_thickness: int = 2,
    keypoint_color: Tuple[float, float, float] = generate_color(),
    bbox_color: Tuple[float, float, float] = generate_color(),
) -> np.ndarray:
    """
    Method used to draw pose onto input image.

    Parameters
    ----------
    input_img : np.ndarray
        The original image.
    pose : Pose
        Estimated poses represented as Pose.
    keypoint_size : int
        Size of the key points.
    bbox_thickness : int
        Width of the bbox lines.
    line_thickness : int
            A thickness of line in pose.
    keypoint_color : Tuple[float,float,float]
        Color of the key points.
    bbox_color : Tuple[float,float,float]
        Color of the bounding boxes.

    Returns
    -------
    np.ndarray
        The modified image with poses drawn.
    """
    w, h, _ = input_img.shape

    out_img = input_img

    bbox = pose.segm

    pt1 = (int(bbox.xmin * w), int(bbox.ymin * h))
    pt2 = (int(bbox.xmax * w), int(bbox.ymax * h))

    out_img = cv2.rectangle(
        img=out_img,
        pt1=pt1,
        pt2=pt2,
        color=bbox_color,
        thickness=bbox_thickness,
    )

    points = pose.keypoints

    points_mapped = {}

    for point in points:
        points_mapped[Limbs(point.id)] = (
            int(point.x * w) + pt1[0],
            int(point.y * h) + pt1[1],
        )

    for limb1 in POSE_LIMBS.keys():
        if limb1 not in points_mapped.keys():
            continue

        limbs = POSE_LIMBS[limb1]

        for limb2 in limbs.keys():
            if limb2 in points_mapped.keys():
                p1 = points_mapped[limb1]
                p2 = points_mapped[limb2]

                color = limbs[limb2]

                out_img = cv2.line(
                    img=out_img,
                    pt1=p1,
                    pt2=p2,
                    color=color,
                    thickness=line_thickness,
                )

    for point in points:
        out_img = cv2.circle(
            img=out_img,
            center=(int(point.x * w) + pt1[0], int(point.y * h) + pt1[1]),
            radius=keypoint_size,
            thickness=-1,
            color=keypoint_color,
        )

    return out_img
