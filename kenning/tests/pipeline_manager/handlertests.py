# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
import pytest
import json


def load_json_files(path_to_json_files):
    path_to_json_files = Path(path_to_json_files)
    assert path_to_json_files.exists()

    pipeline_jsons = []
    for json_file in Path(path_to_json_files).iterdir():
        with open(json_file) as f:
            pipeline_jsons.append(json.load(f))
    return pipeline_jsons


def factory_test_create_dataflow(path_to_json_files):
    pipeline_jsons = load_json_files(path_to_json_files)

    @pytest.mark.parametrize(
        "pipeline_json",
        pipeline_jsons
    )
    def test_create_dataflow(self, pipeline_json, handler):
        _ = handler.create_dataflow(pipeline_json)

    return test_create_dataflow


def factory_test_equivalence(path_to_json_files):
    pipeline_jsons = load_json_files(path_to_json_files)

    @pytest.mark.parametrize(
        'pipeline_json',
        pipeline_jsons
    )
    def test_equivalence(self, pipeline_json, handler):
        dataflow = handler.create_dataflow(pipeline_json)
        status, result_json = handler.parse_dataflow(dataflow)
        if not status:
            pytest.xfail('JSON file is incompatible with Pipeline Manager')
        assert self.equivalence_check(result_json, pipeline_json)

    return test_equivalence


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

    @abstractmethod
    def equivalence_check(self, dataflow1, dataflow2):
        raise NotImplementedError

    def test_parse_dataflow(self, dataflow_json, handler):
        status, _ = handler.parse_dataflow(dataflow_json)
        assert status

    def test_validate_dataflow(self, dataflow_json, handler):
        _, pipeline_json = handler.parse_dataflow(dataflow_json)
        pipeline = handler.parse_json(pipeline_json)
        handler.destroy_dataflow(pipeline)
