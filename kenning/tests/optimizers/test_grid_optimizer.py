# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List

import pytest

from kenning.scenarios.optimization_runner import (
    filter_invalid_pipelines,
    get_block_product,
    grid_search,
    ordered_powerset,
    replace_paths,
)
from kenning.utils.pipeline_runner import PipelineRunner

OPTIMIZATION_CONFIGS_PATHS = list(
    Path("./scripts/optimizationconfigs").glob("*.json")
)

EXAMPLE_PIPELINE = json.loads(
    open("./scripts/jsonconfigs/sample-tflite-pipeline.json").read()
)
EXAMPLE_PIPELINE["dataset"]["parameters"]["download_dataset"] = False


class TestGetBlockProduct:
    @pytest.mark.parametrize(
        "optimization_levels_count,dtypes_count", product([1, 2, 3, 4], [2, 3])
    )
    def test_get_block_product(
        self, optimization_levels_count: int, dtypes_count: int
    ):
        """
        Test get_block_product function.
        """
        optimization_levels = list(range(1, optimization_levels_count + 1))
        dtypes = ["int8", "float16", "float32"][:dtypes_count]
        block = {
            "type": "example",
            "parameters": {
                "optimization_level": optimization_levels,
                "dtype": dtypes,
            },
        }

        block_product = get_block_product(block)

        assert len(block_product) == dtypes_count * optimization_levels_count
        for optimization_level in optimization_levels:
            for dtype in dtypes:
                assert {
                    "type": "example",
                    "parameters": {
                        "optimization_level": optimization_level,
                        "dtype": dtype,
                    },
                } in block_product


class TestOrderedPowerset:
    @pytest.mark.parametrize(
        "min_elements,expected_powerset",
        [
            (0, [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]),
            (1, [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]),
            (2, [[1, 2], [1, 3], [2, 3], [1, 2, 3]]),
            (3, [[1, 2, 3]]),
            (4, []),
            (5, []),
        ],
    )
    def test_ordered_powerset(
        self, min_elements: int, expected_powerset: List[List[int]]
    ):
        """
        Test ordered_powerset function.
        """
        powerset = ordered_powerset([1, 2, 3], min_elements)
        assert powerset == expected_powerset


class TestGridSearch:
    @pytest.mark.parametrize(
        "optimization_config_path",
        OPTIMIZATION_CONFIGS_PATHS,
        ids=[path.stem for path in OPTIMIZATION_CONFIGS_PATHS],
    )
    def test_grid_search(self, optimization_config_path: Path):
        """
        Test if grid_search function returns valid pipelines.
        """
        with open(optimization_config_path, "r") as opt_f:
            optimization_config = json.load(opt_f)

        pipelines = grid_search(optimization_config)

        for pipeline in pipelines:
            # check if pipeline is valid
            PipelineRunner.from_json_cfg(pipeline, assert_integrity=True)


class TestReplacePaths:
    @pytest.mark.parametrize("pipeline_id", [0, 1, 2, 3, 100, 999])
    def test_replace_path(self, pipeline_id: int):
        """
        Test if replace_path function return pipeline with updated paths and
        that new path contains provided pipeline id.
        """
        base_pipeline = deepcopy(EXAMPLE_PIPELINE)
        pipeline = replace_paths(base_pipeline, pipeline_id)

        assert (
            base_pipeline["optimizers"][0]["parameters"]["compiled_model_path"]
            != pipeline["optimizers"][0]["parameters"]["compiled_model_path"]
        )
        assert (
            str(pipeline_id)
            in pipeline["optimizers"][0]["parameters"]["compiled_model_path"]
        )

        assert (
            base_pipeline["runtime"]["parameters"]["save_model_path"]
            != pipeline["runtime"]["parameters"]["save_model_path"]
        )
        assert (
            str(pipeline_id)
            in pipeline["runtime"]["parameters"]["save_model_path"]
        )

    def test_replace_path_injectivity(self):
        """
        Test if replace_path function return distinct paths for distinct ids.
        """
        pipeline_0 = replace_paths(EXAMPLE_PIPELINE, 0)
        pipeline_1 = replace_paths(EXAMPLE_PIPELINE, 1)

        assert (
            pipeline_0["optimizers"][0]["parameters"]["compiled_model_path"]
            != pipeline_1["optimizers"][0]["parameters"]["compiled_model_path"]
        )

        assert (
            pipeline_0["runtime"]["parameters"]["save_model_path"]
            != pipeline_1["runtime"]["parameters"]["save_model_path"]
        )


class TestFilterInvalidPipelines:
    def test_filter_invalid_pipelines_does_not_filter_valid_pipeline(self):
        """
        Test if filter_invalid_pipelines function does not filter valid
        pipelines.
        """
        pipelines = [EXAMPLE_PIPELINE]

        filtered_pipelines = filter_invalid_pipelines(pipelines)
        assert len(filtered_pipelines) == 1
        assert EXAMPLE_PIPELINE in filtered_pipelines

    def test_filter_invalid_pipelines_filters_invalid_pipeline(self):
        """
        Test if filter_invalid_pipelines function does filters invalid
        pipelines.
        """
        invalid_pipeline = deepcopy(EXAMPLE_PIPELINE)
        invalid_pipeline["optimizers"].insert(
            0,
            {
                "type": "kenning.optimizers.tvm.TVMCompiler",
                "parameters": {
                    "target": "llvm -mcpu=core-avx2",
                    "compiled_model_path": "./build/compiled_tvm.tar",
                    "opt_level": 3,
                },
            },
        )
        pipelines = [EXAMPLE_PIPELINE, invalid_pipeline]

        filtered_pipelines = filter_invalid_pipelines(pipelines)
        assert len(filtered_pipelines) == 1
        assert EXAMPLE_PIPELINE in filtered_pipelines
