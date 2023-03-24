# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import pytest


class HandlerTests(ABC):

    @pytest.fixture
    def dataflow_json(self):
        return {
            'panning': {'x': 0, 'y': 0},
            'scaling': 1,
            'nodes': self.dataflow_nodes,
            'connections': self.dataflow_connections
        }

    @pytest.fixture(scope="class")
    @abstractmethod
    def handler(self):
        raise NotImplementedError

    def test_parse_dataflow(self, dataflow_json, handler):
        status, _ = handler.parse_dataflow(dataflow_json)
        assert status

    def test_validate_dataflow(self, dataflow_json, handler):
        _, pipeline_json = handler.parse_dataflow(dataflow_json)
        handler.parse_json(pipeline_json)
