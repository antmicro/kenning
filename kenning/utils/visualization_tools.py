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


FACE_COLOR = (1.0, 0.1, 0.2)
RIGHT_ARM_COLOR = (0.0, 0.5, 1.0)
LEFT_ARM_COLOR = (0, 1.0, 0)
RIGHT_LEG_COLOR = (0.19, 0.1, 1.0)
LEFT_LEG_COLOR = (0.91, 1.0, 0.17)

KEYPOINT_COLORS = {
    Limbs.Nose: FACE_COLOR,
    Limbs.LEye: FACE_COLOR,
    Limbs.REye: FACE_COLOR,
    Limbs.LEar: FACE_COLOR,
    Limbs.REar: FACE_COLOR,
    Limbs.LShoulder: LEFT_ARM_COLOR,
    Limbs.RShoulder: RIGHT_ARM_COLOR,
    Limbs.LElbow: LEFT_ARM_COLOR,
    Limbs.RElbow: RIGHT_ARM_COLOR,
    Limbs.LHand: LEFT_ARM_COLOR,
    Limbs.RHand: RIGHT_ARM_COLOR,
    Limbs.LHip: LEFT_LEG_COLOR,
    Limbs.RHip: RIGHT_LEG_COLOR,
    Limbs.LKnee: LEFT_LEG_COLOR,
    Limbs.RKnee: RIGHT_LEG_COLOR,
    Limbs.LFoot: LEFT_LEG_COLOR,
    Limbs.RFoot: RIGHT_LEG_COLOR,
}


POSE_LIMBS = {
    Limbs.REye: [Limbs.LEye, Limbs.REar],
    Limbs.LEye: [Limbs.LEar],
    Limbs.LEar: [Limbs.Nose],
    Limbs.REar: [Limbs.Nose],
    Limbs.Nose: [
        Limbs.RShoulder,
        Limbs.LShoulder,
    ],
    Limbs.RShoulder: [
        Limbs.LShoulder,
        Limbs.RElbow,
        Limbs.RHip,
    ],
    Limbs.RElbow: [
        Limbs.RHand,
    ],
    Limbs.LShoulder: [Limbs.LHip, Limbs.LElbow],
    Limbs.RHip: [Limbs.LHip, Limbs.RKnee],
    Limbs.RKnee: [Limbs.RFoot],
    Limbs.LHip: [Limbs.LKnee],
    Limbs.LKnee: [Limbs.LFoot],
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

    GRADIENT_STEPS = 40

    pt1 = (int(bbox.xmin * w), int(bbox.ymin * h))
    pt2 = (int(bbox.xmax * w), int(bbox.ymax * h))

    out_img = cv2.rectangle(
        img=out_img,
        pt1=pt1,
        pt2=pt2,
        color=bbox_color,
        thickness=bbox_thickness,
    )

    w, h = (pt2[0] - pt1[0]), (pt2[1] - pt1[1])

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

        for limb2 in limbs:
            if limb2 in points_mapped.keys():
                p1 = np.array(points_mapped[limb1]).astype(np.float32)
                p2 = np.array(points_mapped[limb2]).astype(np.float32)

                color1 = np.array(KEYPOINT_COLORS[limb1]).astype(np.float32)
                color2 = np.array(KEYPOINT_COLORS[limb2]).astype(np.float32)

                dp = (p2 - p1) / GRADIENT_STEPS
                dcolor = (color2 - color1) / GRADIENT_STEPS

                p2 = p1 + dp

                for i in range(GRADIENT_STEPS):
                    out_img = cv2.line(
                        img=out_img,
                        pt1=p1.astype(np.uint32),
                        pt2=p2.astype(np.uint32),
                        color=(
                            float(color1[0]),
                            float(color1[1]),
                            float(color1[2]),
                        ),
                        thickness=line_thickness,
                    )

                    p1 += dp
                    p2 += dp
                    color1 += dcolor

    for point in points:
        out_img = cv2.circle(
            img=out_img,
            center=(int(point.x * w) + pt1[0], int(point.y * h) + pt1[1]),
            radius=keypoint_size,
            thickness=-1,
            color=KEYPOINT_COLORS[point.id],
        )

    return out_img
