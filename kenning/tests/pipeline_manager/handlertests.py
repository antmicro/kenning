# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Union
import pytest
import json

from kenning.pipeline_manager.core import BaseDataflowHandler


def load_json_files(path_to_json_files: Union[str, Path]) -> List[Dict]:
    """
    Loads JSON files from given directory.

    Parameters
    ----------
    path_to_json_files : Union[str, Path]
        Directory containing purely JSON configurations

    Returns
    -------
    List[Dict] :
        List of loaded JSON files
    """
    path_to_json_files = Path(path_to_json_files)
    assert path_to_json_files.exists()

    pipeline_jsons = []
    for json_file in Path(path_to_json_files).iterdir():
        with open(json_file) as f:
            pipeline_jsons.append(json.load(f))
    return pipeline_jsons


def factory_test_create_dataflow(
        path_to_json_files: Union[str, Path]) -> Callable:
    """
    Creates test for `create_dataflow` method of dataflow handlers. The test
    does not check the validity of output, only if the parsing ended
    successfully

    Parameters
    ----------
    path_to_json_files : Union[str, Path]
        Directory containing JSONs defining dataflow configuration

    Returns
    -------
    Callable :
        Test for `create_dataflow` method.
    """
    pipeline_jsons = load_json_files(path_to_json_files)

    @pytest.mark.parametrize(
        "pipeline_json",
        pipeline_jsons
    )
    def test_create_dataflow(self, pipeline_json, handler):
        _ = handler.create_dataflow(pipeline_json)

    return test_create_dataflow


def factory_test_equivalence(path_to_json_files: Union[str, Path]) -> Callable:
    """
    Creates `test_equivalence`, that runs `create_dataflow`, then
    `parse_dataflow` on the output, and checks whether the results is
    equivalent to the input JSON configuration. The test utilizes the
    `equivalence_check` method that should be defined in `HandlerTests`
    subclasses to check whether two JSONs define the same dataflow.

    Parameters
    ----------
    path_to_json_files : Union[str, Path]
        Directory containing JSONs defining dataflow configuration

    Returns
    -------
    Callable :
        Test whether parsing JSON to Pipeline Manager dataflow and back does
        not change underlying pipeline.
    """
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
    def dataflow_json(self) -> Dict:
        """
        Example of dataflow in Pipeline Manager Format
        """
        return {
            'panning': {'x': 0, 'y': 0},
            'scaling': 1,
            'nodes': self.dataflow_nodes,
            'connections': self.dataflow_connections
        }

    @pytest.fixture(scope="class")
    @abstractmethod
    def handler(self) -> BaseDataflowHandler:
        """
        Creates subclass of BaseDataflowHandler
        """
        raise NotImplementedError

    @abstractmethod
    def equivalence_check(self, dataflow1, dataflow2):
        """
        Method that checks whether two JSON defining dataflow in a specific
        Kenning format are equivalent (define the same dataflow)
        """
        raise NotImplementedError

    def test_parse_dataflow(self, dataflow_json, handler):
        """
        Test for `parse_dataflow`. Does not check the validity of output,
        only if the parsing ended successfully
        """
        status, _ = handler.parse_dataflow(dataflow_json)
        assert status

    def test_validate_dataflow(self, dataflow_json, handler):
        """
        Test whether the output of `parse_dataflow` can be successfully parsed
        using `parse_json` method.
        """
        _, pipeline_json = handler.parse_dataflow(dataflow_json)
        pipeline = handler.parse_json(pipeline_json)
        handler.destroy_dataflow(pipeline)
