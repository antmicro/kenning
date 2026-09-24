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


class HandJoints(IntEnum):
    """
    Enum with hand (21-keypoint) joints names
    used in pose visualization.
    """

    Wrist = 0
    Thumb1 = 1
    Thumb2 = 2
    Thumb3 = 3
    Thumb4 = 4
    Index1 = 5
    Index2 = 6
    Index3 = 7
    Index4 = 8
    Middle1 = 9
    Middle2 = 10
    Middle3 = 11
    Middle4 = 12
    Ring1 = 13
    Ring2 = 14
    Ring3 = 15
    Ring4 = 16
    Pinky1 = 17
    Pinky2 = 18
    Pinky3 = 19
    Pinky4 = 20


HAND_COLOR = (0.1, 0.9, 0.9)
THUMB_COLOR = (1.0, 0.6, 0.0)
INDEX_COLOR = (0.2, 0.8, 0.3)
MIDDLE_COLOR = (0.3, 0.4, 0.9)
RING_COLOR = (0.9, 0.3, 0.7)
PINKY_COLOR = (0.95, 0.95, 0.2)


HAND_KEYPOINT_COLORS = {
    HandJoints.Wrist: HAND_COLOR,
    HandJoints.Thumb1: THUMB_COLOR,
    HandJoints.Thumb2: THUMB_COLOR,
    HandJoints.Thumb3: THUMB_COLOR,
    HandJoints.Thumb4: THUMB_COLOR,
    HandJoints.Index1: INDEX_COLOR,
    HandJoints.Index2: INDEX_COLOR,
    HandJoints.Index3: INDEX_COLOR,
    HandJoints.Index4: INDEX_COLOR,
    HandJoints.Middle1: MIDDLE_COLOR,
    HandJoints.Middle2: MIDDLE_COLOR,
    HandJoints.Middle3: MIDDLE_COLOR,
    HandJoints.Middle4: MIDDLE_COLOR,
    HandJoints.Ring1: RING_COLOR,
    HandJoints.Ring2: RING_COLOR,
    HandJoints.Ring3: RING_COLOR,
    HandJoints.Ring4: RING_COLOR,
    HandJoints.Pinky1: PINKY_COLOR,
    HandJoints.Pinky2: PINKY_COLOR,
    HandJoints.Pinky3: PINKY_COLOR,
    HandJoints.Pinky4: PINKY_COLOR,
}


HAND_POSE_LIMBS = {
    HandJoints.Wrist: [
        HandJoints.Thumb1,
        HandJoints.Index1,
        HandJoints.Middle1,
        HandJoints.Ring1,
        HandJoints.Pinky1,
    ],
    HandJoints.Thumb1: [HandJoints.Thumb2],
    HandJoints.Thumb2: [HandJoints.Thumb3],
    HandJoints.Thumb3: [HandJoints.Thumb4],
    HandJoints.Index1: [HandJoints.Index2],
    HandJoints.Index2: [HandJoints.Index3],
    HandJoints.Index3: [HandJoints.Index4],
    HandJoints.Middle1: [HandJoints.Middle2],
    HandJoints.Middle2: [HandJoints.Middle3],
    HandJoints.Middle3: [HandJoints.Middle4],
    HandJoints.Ring1: [HandJoints.Ring2],
    HandJoints.Ring2: [HandJoints.Ring3],
    HandJoints.Ring3: [HandJoints.Ring4],
    HandJoints.Pinky1: [HandJoints.Pinky2],
    HandJoints.Pinky2: [HandJoints.Pinky3],
    HandJoints.Pinky3: [HandJoints.Pinky4],
}


POSE_SKELETONS = {
    "body": {
        "limbs": Limbs,
        "colors": KEYPOINT_COLORS,
        "limbs_map": POSE_LIMBS,
    },
    "hand": {
        "limbs": HandJoints,
        "colors": HAND_KEYPOINT_COLORS,
        "limbs_map": HAND_POSE_LIMBS,
    },
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
    pose_type: str = "body",
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
    pose_type : str
        Type of the pose skeleton to draw,
        one of "body" or "hand".

    Returns
    -------
    np.ndarray
        The modified image with poses drawn.
    """
    skeleton = POSE_SKELETONS[pose_type]
    limbs_enum = skeleton["limbs"]
    keypoint_colors = skeleton["colors"]
    pose_limbs = skeleton["limbs_map"]

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
        points_mapped[limbs_enum(point.id)] = (
            int(point.x * w) + pt1[0],
            int(point.y * h) + pt1[1],
        )

    for limb1 in pose_limbs.keys():
        if limb1 not in points_mapped.keys():
            continue

        limbs = pose_limbs[limb1]

        for limb2 in limbs:
            if limb2 in points_mapped.keys():
                p1 = np.array(points_mapped[limb1]).astype(np.float32)
                p2 = np.array(points_mapped[limb2]).astype(np.float32)

                color1 = np.array(keypoint_colors[limb1]).astype(np.float32)
                color2 = np.array(keypoint_colors[limb2]).astype(np.float32)

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
            color=keypoint_colors[point.id],
        )

    return out_img
