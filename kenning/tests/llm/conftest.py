# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Conftest for llm compatibility tests.
"""

from typing import Tuple, Type

import pytest

from kenning.core.optimizer import Optimizer
from kenning.modelwrappers.llm.llm import LLM
from kenning.tests.conftest import get_tmp_path
from kenning.utils.resource_manager import ResourceURI


@pytest.fixture
def prepare_objects():
    def _prepare_objects(
        modelwrapper_cls: Type[LLM], optimizer_cls: Type[Optimizer]
    ) -> Tuple[LLM, Optimizer]:
        optimizer_kwargs = {
            "batch_size": 1,
            "seqlen": 16,
            "calibration_samples": 4,
        }

        if modelwrapper_cls.__name__ == "SmolLM2":
            optimizer_kwargs["group_size"] = 64

        raw_uri = modelwrapper_cls.pretrained_model_uri
        model_path = ResourceURI(raw_uri)
        model = modelwrapper_cls(model_path, None)
        optimizer = optimizer_cls(
            None,
            get_tmp_path(),
            model_wrapper=model,
            **optimizer_kwargs,
        )

        return model, optimizer

    return _prepare_objects
