# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from kenning.core.metrics import (
    hausdorff_distance_metric,
    mean_signed_difference,
    nab_metric,
    prob_auc_metric,
    z_score_detection,
)

PARAMETERS_FOR_MISMATCH_TEST = (
    [0] * 50,
    [0] * 25,
)
PARAMETERS_FOR_MISMATCH_AND_DIMENSION_TEST = (
    [[0, 0]] * 10,
    [0] * 4,
)
PARAMETERS_FOR_MISMATCH_TEST_WITH_SCORE = (
    [0] * 50,
    [0] * 25,
    [0] * 25,
)

NAB_INVALID_PARAMETERS_LIST = [
    (
        -1.0,
        0.5,
        0.25,
        -0.25,
    ),
    (
        2.0,
        2.0,
        0.25,
        -0.25,
    ),
    (
        2.0,
        0.5,
        -1.5,
        -0.25,
    ),
    (
        2.0,
        0.5,
        0.25,
        1,
    ),
]


def convert_to_numpy(func):
    def inner(self, metric, inputs):
        _inputs = []
        for _input in inputs:
            _inputs.append(np.array(_input, dtype=np.float32))

        return func(self, metric, _inputs)

    return inner


class TestMetrics:
    @pytest.mark.parametrize(
        "metric,inputs",
        [
            (
                nab_metric,
                PARAMETERS_FOR_MISMATCH_TEST,
            ),
            (
                hausdorff_distance_metric,
                PARAMETERS_FOR_MISMATCH_AND_DIMENSION_TEST,
            ),
            (mean_signed_difference, PARAMETERS_FOR_MISMATCH_TEST),
            (prob_auc_metric, PARAMETERS_FOR_MISMATCH_TEST_WITH_SCORE),
            (z_score_detection, PARAMETERS_FOR_MISMATCH_TEST),
        ],
    )
    @convert_to_numpy
    def test_metric_mismatched_shape(self, metric, inputs):
        """
        Test for detecting mismatched shapes between inputs.
        """
        with pytest.raises(ValueError):
            metric(*inputs)

    @pytest.mark.parametrize("params", NAB_INVALID_PARAMETERS_LIST)
    def tes_nab_metric_invalid_parameters(self, params):
        """
        Test for detecting invalid parameters for NAB metric.
        """
        x = [0] * 25
        y = [0] * 25

        with pytest.raises(ValueError):
            nab_metric(x, y, *params)

    def test_z_score_detection(self):
        """
        Test for testing Z-Score metric.
        It should detect three anomalies in input.
        """
        x = np.array([0.1] * 100, dtype=np.float32)
        y = np.array([0.1] * 100, dtype=np.float32)

        x[10] = 5.0
        x[55] = 1.0
        x[70] = 10.0

        assert z_score_detection(x, y) == 3

    def test_nab_metric(self):
        """
        Test for testing NAB metric with few simple sets of data and checking
        returning score with expected score.
        """
        x = np.zeros(100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)

        # Score for no anomlay datasets
        score = nab_metric(x, y)

        assert score == 100

        x = np.zeros(100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)

        y[10] = 1
        # Score for lack of any detections
        score = nab_metric(x, y)

        assert score == 0

        y[10] = 1
        x[10] = 1

        y[30] = 1
        x[30] = 1
        # Score for perfect detector
        score = nab_metric(x, y)

        assert score == 100

        x = np.zeros(100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)

        y[10] = 1
        x[1] = 1

        # Score for detector with early detections
        score = nab_metric(x, y)

        assert score == 100

        x = np.zeros(100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)

        y[10] = 1
        x[19] = 1

        # Score for detector with late detections
        score = nab_metric(x, y)

        assert score < 99

        x = np.zeros(100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)

        y[10] = 1
        x[10] = 1

        y[40] = 1

        # Score for detector with one missed window
        score = nab_metric(x, y)

        assert score == 50

    def test_mean_signed_difference(self):
        """
        Test for Mean Signed Difference, it checks few simple scenarios.
        """
        # 1. Two identical sets
        x = np.random.randint(-100, 100, 64)
        y = np.copy(x)

        assert mean_signed_difference(x, y) == 0.0

        e = np.random.randint(-100, 100)

        assert mean_signed_difference(x, y + e) == -e

        e = np.random.randint(-100, 100)

        assert mean_signed_difference(x + e, y) == e

    def test_p_auc_metric(self):
        """
        Test pAUC metric against examples from https://dmip.webs.upv.es/ROCML2005/papers/ferriCRC.pdf.
        """
        x = np.array([1, 1, 1, 1], dtype=np.float32)
        y = np.array([1, 1, 0, 0], dtype=np.float32)
        probs = np.array([0.65, 0.55, 0.45, 0.35], dtype=np.float32)

        score = prob_auc_metric(y, x, probs)

        assert score >= 0.599 and score <= 0.601

        x = np.array([1, 1, 1], dtype=np.float32)
        y = np.array([1, 0, 1], dtype=np.float32)
        probs = np.array([1, 0.1, 0.0], dtype=np.float32)

        score = prob_auc_metric(y, x, probs)

        assert score >= 0.699 and score <= 0.701

        x = np.array([1, 1, 1, 1], dtype=np.float32)
        y = np.array([1, 0, 1, 0], dtype=np.float32)
        probs = np.array([1, 0.51, 0.49, 0.0], dtype=np.float32)

        score = prob_auc_metric(y, x, probs)

        assert score >= 0.744 and score <= 0.746

    def test_hausdorff_distance_metric(self):
        """
        Test Hausdorff metric against a few sets of data.
        """
        # 1. Two identical sets
        x = np.array(
            [
                [55, 34],
                [17, 34],
                [-55, 54],
                [-72, -36],
                [85, -6],
                [-48, 14],
                [12, 26],
                [-2, -74],
            ],
            dtype=np.int32,
        )
        y = np.copy(x)

        assert hausdorff_distance_metric(x, y) == 0

        y[0][1] += 10

        assert hausdorff_distance_metric(x, y) == 10

        y = np.copy(x)
        y[0][1] -= 10

        assert hausdorff_distance_metric(x, y) == 10
        # Hausdorff should return the bigger distance
        # between two sets
        y = np.copy(x)
        x[4][0] += 100
        y[5][0] -= 10

        assert hausdorff_distance_metric(x, y) == 100
