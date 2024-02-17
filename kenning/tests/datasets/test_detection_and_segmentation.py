# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from kenning.datasets.helpers.detection_and_segmentation import (
    SegmObject,
    compute_segm_iou,
)


class TestComputeIou:
    @pytest.mark.fast
    def test_compute_segm_iou_no_overlap(self):
        """
        Tests the compute_segm_iou function with no overlap.

        List of methods that are being tested
        --------------------------------
        compute_segm_iou()
        """
        segm1 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )

        segm2 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=2.0,
            ymin=2.0,
            xmax=3.0,
            ymax=3.0,
            mask=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )

        assert compute_segm_iou(segm1, segm2) == 0.0

    @pytest.mark.fast
    def test_compute_segm_iou_diff_shapes(self):
        """
        Tests the compute_segm_iou function with different shapes.

        List of methods that are being tested
        --------------------------------
        compute_segm_iou()
        """
        segm1 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )

        segm2 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )
        assert compute_segm_iou(segm1, segm2) == 0.0

    @pytest.mark.fast
    def test_compute_segm_iou(self):
        """
        Tests the compute_segm_iou function.

        List of methods that are being tested
        --------------------------------
        compute_segm_iou()
        """
        segm1 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )
        segm2 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )

        assert compute_segm_iou(segm1, segm2) == 1.0

        segm2 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )
        assert compute_segm_iou(segm1, segm2) == 0.8

        segm2 = SegmObject(
            clsname="class1",
            maskpath=None,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            mask=np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8),
            score=1.0,
            iscrowd=False,
        )
        assert compute_segm_iou(segm1, segm2) == 0.6
