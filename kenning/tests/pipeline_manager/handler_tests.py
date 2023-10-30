# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pytest

from kenning.pipeline_manager.core import BaseDataflowHandler


def load_json_files(
    path_to_json_files: Union[str, Path]
) -> Tuple[List[Dict], List[str]]:
    """
    Loads JSON files from given directory.

    Parameters
    ----------
    path_to_json_files : Union[str, Path]
        Directory containing purely JSON configurations

    Returns
    -------
    Tuple[List[Dict], List[str]]
        Tuple containing list of loaded JSON files and list of their names
    """
    path_to_json_files = Path(path_to_json_files)
    assert path_to_json_files.exists()

    pipeline_jsons = []
    pipeline_jsons_names = []
    for json_file in Path(path_to_json_files).iterdir():
        if json_file.suffix == ".json":
            with open(json_file) as f:
                pipeline_jsons.append(json.load(f))
                pipeline_jsons_names.append(json_file.stem)
    return pipeline_jsons, pipeline_jsons_names


def factory_test_create_dataflow(
    path_to_json_files: Union[str, Path]
) -> Callable:
    """
    Creates test for `create_dataflow` method of dataflow handlers. The test
    does not check the validity of output, only if the parsing ended
    successfully.

    Parameters
    ----------
    path_to_json_files : Union[str, Path]
        Directory containing JSONs defining dataflow configuration

    Returns
    -------
    Callable
        Test for `create_dataflow` method.
    """
    pipeline_jsons, pipeline_jsons_names = load_json_files(path_to_json_files)

    @pytest.mark.parametrize(
        "pipeline_json", pipeline_jsons, ids=pipeline_jsons_names
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
    Callable
        Test whether parsing JSON to Pipeline Manager dataflow and back does
        not change underlying pipeline.
    """
    pipeline_jsons, pipeline_jsons_names = load_json_files(path_to_json_files)

    @pytest.mark.parametrize(
        "pipeline_json", pipeline_jsons, ids=pipeline_jsons_names
    )
    def test_equivalence(self, pipeline_json, handler):
        dataflow = handler.create_dataflow(pipeline_json)
        status, result_json = handler.parse_dataflow(dataflow)
        if not status:
            pytest.xfail(
                "JSON file is incompatible with Pipeline Manager\n\n"
                f"Source scenario:\n{json.dumps(pipeline_json, indent=4)}\n\n"
                f"Status:  {status}\n\n"
            )
        assert self.equivalence_check(result_json, pipeline_json), (
            "Equivalence test failed.\n\n"
            f"Source JSON:\n{json.dumps(pipeline_json, indent=4)}\n\n"
            f"Result JSON:\n{json.dumps(result_json, indent=4)}\n\n"
        )

    return test_equivalence


@pytest.mark.usefixtures("mock_environment")
class HandlerTests(ABC):
    @pytest.fixture
    def dataflow_json(self) -> Dict:
        """
        Example of dataflow in Pipeline Manager Format.
        """
        return {
            "graph": {
                "nodes": self.dataflow_nodes,
                "connections": self.dataflow_connections,
                "inputs": {},
                "outputs": {},
            },
            "graphTemplates": {},
        }

    @pytest.fixture(scope="class")
    @abstractmethod
    def handler(self) -> BaseDataflowHandler:
        """
        Creates subclass of BaseDataflowHandler.
        """
        raise NotImplementedError

    @abstractmethod
    def equivalence_check(self, dataflow1, dataflow2):
        """
        Method that checks whether two JSON defining dataflow in a specific
        Kenning format are equivalent (define the same dataflow).
        """
        raise NotImplementedError

    def test_parse_dataflow(self, dataflow_json, handler):
        """
        Test for `parse_dataflow`. Does not check the validity of output,
        only if the parsing ended successfully.
        """
        status, _ = handler.parse_dataflow(dataflow_json)
        assert status

    @pytest.mark.xdist_group(name="use_resources")
    def test_validate_dataflow(self, dataflow_json, handler):
        """
        Test whether the output of `parse_dataflow` can be successfully parsed
        using `parse_json` method.
        """
        _, pipeline_json = handler.parse_dataflow(dataflow_json)
        pipeline = handler.parse_json(pipeline_json)
        handler.destroy_dataflow(pipeline)
