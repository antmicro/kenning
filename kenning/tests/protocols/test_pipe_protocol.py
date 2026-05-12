# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import uuid
from random import choices, randint
from string import ascii_lowercase
from typing import Any, Dict

import numpy as np
import pytest

from kenning.core.model import ModelWrapper
from kenning.protocols.pipe_protocol import PipeProtocol
from kenning.protocols.uart import (
    RUNTIME_STAT_NAME_MAX_LEN,
)
from kenning.tests.protocols.test_core_protocol import (
    TestCoreProtocol,
)
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.resource_manager import ResourceURI

MODEL_WRAPPER_SUBCLASSES = get_all_subclasses(
    "kenning.modelwrappers", ModelWrapper, raise_exception=True
)
MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC = [
    modelwrapper_cls
    for modelwrapper_cls in MODEL_WRAPPER_SUBCLASSES
    if hasattr(modelwrapper_cls, "pretrained_model_uri")
    and modelwrapper_cls.pretrained_model_uri is not None
    and not modelwrapper_cls.pretrained_model_uri.startswith("hf://")
]


@pytest.fixture
def valid_io_spec() -> Dict[str, Any]:
    modelwrapper_cls = MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC[0]
    valid_io_spec_path = ResourceURI(
        f"{modelwrapper_cls.pretrained_model_uri}.json"
    )
    with open(valid_io_spec_path, "r") as io_spec_f:
        io_spec = json.load(io_spec_f)

    return io_spec


@pytest.fixture
def valid_iree_stats() -> bytes:
    stats = np.random.randint(
        np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, 6, np.uint32
    )
    return stats.tobytes()


@pytest.fixture
def valid_generic_stats() -> bytes:
    stats_names = [
        "".join(
            choices(
                ascii_lowercase + "_",
                k=randint(1, RUNTIME_STAT_NAME_MAX_LEN - 1),
            )
        )
        for _ in range(4)
    ]
    stats_values = np.random.randint(
        np.iinfo(np.uint64).min, np.iinfo(np.uint64).max, 4, np.uint64
    )

    stats = b""
    for name, value in zip(stats_names, stats_values):
        struct.pack(f"{RUNTIME_STAT_NAME_MAX_LEN}sQQ", name.encode(), 0, value)

    return stats


def random_path_name():
    return uuid.uuid4().hex


class TestPipeProtocol(TestCoreProtocol):
    pipe_path = random_path_name()

    def init_protocol(self):
        return PipeProtocol(pipe_path=self.pipe_path)

    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.init_protocol()
        assert server.initialize_server()
        second_server = self.init_protocol()
        assert not second_server.initialize_server()
        server.disconnect()

    def test_initialize_client(self):
        """
        Tests the `initialize_client()` method.
        """
        client = self.init_protocol()
        server = self.init_protocol()
        assert server.initialize_server()
        assert client.initialize_client()
        client.disconnect()
        server.disconnect()
