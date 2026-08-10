# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with tests for auto_pytorch.
"""

from pathlib import Path

import pytest

from kenning.automl.auto_pytorch import AutoPyTorchML
from kenning.datasets.anomaly_detection_dataset import (
    AnomalyDetectionDataset,
)
from kenning.platforms.local import LocalPlatform
from kenning.tests.core.conftest import (
    get_dataset_random_mock,
)


@pytest.fixture
def model():
    return AutoPyTorchML(
        get_dataset_random_mock(AnomalyDetectionDataset),
        LocalPlatform,
        Path("./build/_autoPyTorch_tmp/"),
        time_limit=1,
    )


def test_search_no_prepare(model: AutoPyTorchML):
    with pytest.raises(AssertionError):
        model.search()


def test_autopytorch(
    model: AutoPyTorchML,
):
    model.prepare_framework()
    model.search()
    assert model.get_statistics() != ""
    list(model.get_best_configs())
