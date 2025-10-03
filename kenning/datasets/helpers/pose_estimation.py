# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Structure that holds pose estimation results.
"""


from typing import List, NamedTuple

from kenning.datasets.helpers.detection_and_segmentation import SegmObject


class Keypoint2D(NamedTuple):
    """
    Represents a single point of
    the pose.

    Attributes
    ----------
    x : float
        An x coordinate of the point.
    y : float
        An y coordinate of the point.
    id : int
        An id of estimated point.
    """

    x: float
    y: float
    id: int


class Pose(NamedTuple):
    """
    Represents single pose acquired
    from pose estimation.

    Attributes
    ----------
    keypoints : List[Keypoint2D]
        A list of keypoints that pose is
        made of.
    segm : SegmObject
        Segmentation associated with pose.
    """

    keypoints: List[Keypoint2D]
    segm: SegmObject
