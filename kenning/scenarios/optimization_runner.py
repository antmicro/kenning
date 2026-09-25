# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script for running automated optimizations for pipelines. It performs
a search on given set of blocks based on JSON configuration passed.

The `optimization_parameters` specifies the parameters of the
optimization search. Currently supported strategy is `grid_search`, which
performs a grid search to find optimal parameters for each block specified
in `optimizable` parameter. Every block that is to be optimized should have
list of parameters instead of a singular value specified.
"""
import argparse
import copy
import json
import sys
from collections import defaultdict
from itertools import chain, combinations, permutations, product
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import yaml
from argcomplete.completers import DirectoriesCompleter, FilesCompleter
from jsonschema.exceptions import ValidationError

from kenning.cli.command_template import (
    FINE_TUNE,
    GROUP_SCHEMA,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.core.exceptions import ConfigurationError
from kenning.core.measurements import MeasurementsCollector
from kenning.core.metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
    compute_performance_metrics,
)
from kenning.dispatcher.block_config import (
    BLOCK_CONFIG_PARAMETERS_KEY,
    BLOCK_CONFIGURATIONS_KEY,
    BLOCK_DIRECT_ARGUMENTS_KEY,
    ConfigKey,
    KenningBlockConfigDict,
    set_block_direct_argument,
    yaml_or_json_to_config_dict,
)
from kenning.utils.class_loader import objs_from_full_dict_config
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import ResourceURI


def get_block_product(parameters: Dict[str, List[Any]]) -> List:
    """
    Gets a cartesian product of the parameter values.

    Parameters
    ----------
    parameters: Dict[str, List[Any]]
        Dictionary with parameters, where each parameter is a list of possible
        options.

    Returns
    -------
    List
        Cartesian product of input `block`.

    Examples
    --------
    For argument
    ```python
        {
            'optimization_level' : [1, 2],
            'dtype': ['int8', 'float32']
        }
    ```
    will return
    ```python
    [
        {
            'optimization_level' : 1,
            'dtype': 'int8'
        },
        {
            'optimization_level' : 1,
            'dtype': 'float32'
        },
        {
            'optimization_level' : 2,
            'dtype': 'int8'
        },
        {
            'optimization_level' : 2,
            'dtype': 'float32'
        },
    ]
    ```
    """
    return [
        dict(zip(parameters.keys(), p)) for p in product(*parameters.values())
    ]


def ordered_powerset(iterable: List, min_elements: int = 1) -> List[List]:
    """
    Generates a powerset of ordered elements of `iterable` argument.

    Parameters
    ----------
    iterable : List
        List of arguments.
    min_elements : int
        Minimal number of elements in the powerset.

    Returns
    -------
    List[List]
        Powerset of ordered values.

    Examples
    --------
    ```python
    >>> ordered_powerset([1, 2, 3], 1)
    [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    ```
    """
    res = []
    for i in range(min_elements, len(iterable) + 1):
        comb = [list(c) for c in list(combinations(iterable, r=i))]
        res.append(comb)
    return list(chain(*res))


def grid_search(
    config: KenningBlockConfigDict, blocks_to_optimize: List[ConfigKey]
) -> List[KenningBlockConfigDict]:
    """
    Creates all possible pipeline configurations based on the passed standard
    config dict. For every type of block it creates a list of parametrized
    blocks of this type that can be used to run a pipeline. Then for all of the
    generated blocks cartesian product is computed.

    Parameters
    ----------
    config: KenningBlockConfigDict
        Configuration for the grid search optimization.
    blocks_to_optimize: List[ConfigKey]
        List of the block types that are to be optimized.

    Returns
    -------
    List[KenningBlockConfigDict]
        List of pipeline configurations (in the form of standard config dicts).

    Examples
    --------
    An example of an optimizable runtime block
    ```python
    "runtime":
    {
        "TVMRuntime": {
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tar"]
            },
            "direct_extra_scenario_arguments": {},
        },
        "TFLiteRuntime": {
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tflite"],
                "num_threads": [2, 4]
            },
            "direct_extra_scenario_arguments": {},
        }
    }
    ```
    will yield a list of valid runtime blocks that can be used.
    Those are valid runtime blocks and every one of them can be used
    as a runtime.
    ```python
    "runtime":
    [
        {
            "TVMRuntime": {
                "parameters":
                {
                    "save_model_path": "./build/compiled_model.tar"
                },
                "direct_extra_scenario_arguments": {},
            },
        },
        {
            "TFLiteRuntime": {
                "parameters":
                {
                    "save_model_path": "./build/compiled_model.tflite",
                    "num_threads": 2
                },
                "direct_extra_scenario_arguments": {},
            }
        },
        {
            "TFLiteRuntime": {
                "parameters":
                {
                    "save_model_path": "./build/compiled_model.tflite",
                    "num_threads": 4
                },
                "direct_extra_scenario_arguments": {},
            }
        }
    ]
    ```
    This is done to every block type.
    Then a cartesian product is computed that returns all possible
    pipeline configurations.
    """
    block_configurations = config[BLOCK_CONFIGURATIONS_KEY]

    all_blocks = {
        ConfigKey.model_wrapper,
        ConfigKey.dataset,
        ConfigKey.optimizers,
        ConfigKey.runtime,
        ConfigKey.protocol,
    }
    remaining_blocks = all_blocks & (
        set(block_configurations.keys()) - set(blocks_to_optimize)
    )

    optimization_configuration = {}

    for block_type in remaining_blocks:
        optimization_configuration[block_type] = [
            block_configurations[block_type]
        ]

    # Grid search
    # Creating all possible block configuration for every block type
    for block_type in blocks_to_optimize:
        variants = []
        for name, block_config in block_configurations[block_type].items():
            parameter_variants = get_block_product(
                block_config[BLOCK_CONFIG_PARAMETERS_KEY]
            )
            for parameter_variant in parameter_variants:
                variants.append(
                    {
                        name: {
                            BLOCK_CONFIG_PARAMETERS_KEY: parameter_variant,
                            BLOCK_DIRECT_ARGUMENTS_KEY: block_config[
                                BLOCK_DIRECT_ARGUMENTS_KEY
                            ],
                        }
                    }
                )

        # We need to treat optimizers differently, as those can be chained.
        # For other blocks we have to pick only one.
        if block_type == ConfigKey.optimizers:
            blocks = defaultdict(list)
            for variant in variants:
                blocks[list(variant.keys())[0]].append(variant)

            subsets = ordered_powerset(list(blocks.keys()))
            chains = [
                permutation
                for subset in subsets
                for permutation in permutations(subset)
            ]

            final_variants = []
            for single_chain in chains:
                chain_variants = []
                for element in single_chain:
                    if len(chain_variants) == 0:
                        chain_variants = blocks[element]
                        continue
                    new_chain_variants = [
                        copy.deepcopy(chain_variant) | block
                        for block in blocks[element]
                        for chain_variant in chain_variants
                    ]
                    chain_variants = new_chain_variants
                final_variants.extend(chain_variants)
            variants = final_variants
        optimization_configuration[block_type] = variants

    # Create all possible pipelines from all possible blocks configurations
    # by taking a cartesian product.
    # TODO: For bigger optimizations problems consider using yield.
    pipelines = []
    for pipeline in product(*optimization_configuration.values()):
        pipeline_config = copy.deepcopy(config)
        pipeline_config[BLOCK_CONFIGURATIONS_KEY] = dict(
            zip(optimization_configuration.keys(), pipeline)
        )
        pipelines.append(pipeline_config)
    return pipelines


def replace_paths(
    pipeline: KenningBlockConfigDict, pipeline_id: int
) -> KenningBlockConfigDict:
    """
    Copies given `pipeline` and puts `pipeline_id`_ in front of
    `compiled_model_path` parameter in every optimizer and in front of
    `save_model_path` parameter in runtime.

    It is used when running pipelines so that every pipeline gets its own
    unique namespace. Thanks to that collision names are avoided.

    Parameters
    ----------
    pipeline : KenningBlockConfigDict
        Pipeline that gets copied and its parameters are replaced.
    pipeline_id : int
        Value that is used to create a prefix for the path.

    Returns
    -------
    KenningBlockConfigDict
        Pipeline with `compiled_model_path` and `save_model_path` parameters
        changed.
    """
    pipeline = copy.deepcopy(pipeline)
    block_configurations = pipeline[BLOCK_CONFIGURATIONS_KEY]
    for optimizer in block_configurations[ConfigKey.optimizers].values():
        path = Path(
            optimizer[BLOCK_CONFIG_PARAMETERS_KEY]["compiled_model_path"]
        )
        new_path = path.with_stem(f"{str(pipeline_id)}_{path.stem}")
        optimizer[BLOCK_CONFIG_PARAMETERS_KEY]["compiled_model_path"] = str(
            new_path
        )

    path = Path(
        list(block_configurations[ConfigKey.runtime].values())[0][
            BLOCK_CONFIG_PARAMETERS_KEY
        ]["save_model_path"]
    )
    new_path = path.with_stem(f"{str(pipeline_id)}_{path.stem}")
    list(block_configurations[ConfigKey.runtime].values())[0][
        BLOCK_CONFIG_PARAMETERS_KEY
    ]["save_model_path"] = str(new_path)
    return pipeline


def filter_invalid_pipelines(
    pipelines: List[Tuple[int, KenningBlockConfigDict, Dict[ConfigKey, Any]]]
) -> List[Tuple[int, KenningBlockConfigDict, Dict[ConfigKey, Any]]]:
    """
    Filter pipelines with incompatible blocks.

    Parameters
    ----------
    pipelines : List[Tuple[int, KenningBlockConfigDict, Dict[ConfigKey, Any]]]
        List of pipelines.

    Returns
    -------
    List[Tuple[int, KenningBlockConfigDict, Dict[ConfigKey, Any]]]
        Valid pipelines from provided pipelines.
    """
    filtered_pipelines = []

    for pipeline in pipelines:
        try:
            _, __, objs = pipeline
            PipelineRunner.from_objs_dict(objs, assert_integrity=True)
            filtered_pipelines.append(pipeline)
        except ConfigurationError:
            pass

    return filtered_pipelines


class OptimizationRunner(CommandTemplate):
    """
    Command template for scenario fine-tuning subcommand.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            OptimizationRunner, OptimizationRunner
        ).configure_parser(parser, command, types, groups)

        command_group = parser.add_argument_group(
            GROUP_SCHEMA.format(FINE_TUNE)
        )

        command_group.add_argument(
            "--json-cfg",
            "--cfg",
            help="The path to the input JSON file with configuration",
            type=ResourceURI,
            required=True,
        ).completer = FilesCompleter(allowednames=("yaml", "yml", "json"))
        command_group.add_argument(
            "--output",
            help="The path to the output JSON file with the best pipeline",
            type=Path,
            required=True,
        ).completer = FilesCompleter("*.json")
        command_group.add_argument(
            "--generate-scenarios",
            help="Generate JSON scenarios for the given JSON configuration without actually running them",  # noqa: E501
            type=Path,
            default=None,
        ).completer = DirectoriesCompleter()
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        with open(args.json_cfg, "r") as f:
            json_cfg = yaml.safe_load(f)

        optimization_parameters = json_cfg["optimization_parameters"]
        optimization_strategy = optimization_parameters["strategy"]
        policy = optimization_parameters["policy"]
        metric = optimization_parameters["metric"]
        blocks_to_optimize = [
            getattr(ConfigKey, block_type)
            for block_type in optimization_parameters["optimizable"]
        ]

        del json_cfg["optimization_parameters"]
        config = yaml_or_json_to_config_dict(json_cfg)

        set_block_direct_argument(
            "from_file", True, config, ConfigKey.model_wrapper
        )

        if optimization_strategy == "grid_search":
            pipeline_configs = grid_search(config, blocks_to_optimize)
        else:
            raise ValueError(
                f"Invalid optimization strategy: {optimization_strategy}"
            )

        pipelines = [
            (
                idx,
                config,
                objs_from_full_dict_config(replace_paths(config, idx)),
            )
            for idx, config in enumerate(pipeline_configs)
        ]

        KLogger.info(f"Constructed {len(pipelines)} pipelines.")

        KLogger.info("Filtering broken pipelines...")
        pipelines = filter_invalid_pipelines(pipelines)

        pipelines_num = len(pipelines)
        pipelines_scores = []
        KLogger.info(f"Testing {pipelines_num} pipelines.")

        if args.generate_scenarios is not None:
            Path(args.generate_scenarios).mkdir(parents=True, exist_ok=True)

            for pipeline_idx, pipeline_config, pipeline_objs in pipelines:
                with open(
                    args.generate_scenarios / f"scenario_{pipeline_idx}.json",
                    "w",
                ) as f:
                    json.dump(pipeline_config, f, indent=4)
            return

        KLogger.info(f"Finding {policy} for {metric}")
        for pipeline in pipelines:
            pipeline_idx, pipeline_config, pipeline_objs = pipeline
            module_error = None
            MeasurementsCollector.clear()
            try:
                KLogger.info(f"Running pipeline {pipeline_idx + 1}")
                KLogger.info(f"Configuration {pformat(pipeline_config)}")
                measurements_path = args.output.with_stem(
                    f"{args.output.stem}_{pipeline_idx}"
                )

                pipeline_runner = PipelineRunner.from_objs_dict(pipeline_objs)
                pipeline_runner.run(
                    output=Path(measurements_path), verbosity=args.verbosity
                )

                # Consider using MeasurementsCollector.measurements
                with open(measurements_path, "r") as measurements_file:
                    measurements = json.load(measurements_file)

                computed_metrics = {}

                computed_metrics |= compute_performance_metrics(measurements)
                computed_metrics |= compute_classification_metrics(
                    measurements
                )
                computed_metrics |= compute_detection_metrics(measurements)
                computed_metrics.pop("session_utilization_cpus_percent_avg")

                try:
                    pipelines_scores.append(
                        {
                            "pipeline": pipeline_config,
                            "metrics": computed_metrics,
                        }
                    )
                except KeyError:
                    KLogger.error(f"{metric} not found in the metrics")
                    raise
            except ValidationError as ex:
                KLogger.error("Incorrect parameters passed")
                KLogger.error(ex, stack_info=True)
                raise
            except ModuleNotFoundError as missing_module_error:
                module_error = missing_module_error
            except Exception as ex:
                KLogger.warning("Pipeline was invalid")
                KLogger.warning(ex)

            if module_error:
                raise module_error

        if pipelines_scores:
            policy_fun = min if policy == "min" else max
            best_pipeline = policy_fun(
                pipelines_scores,
                key=lambda pipeline: pipeline["metrics"][metric],
            )

            best_score = best_pipeline["metrics"][metric]
            KLogger.info(f"Best score for {metric} is {best_score}")
            with open(args.output, "w") as f:
                json.dump(best_pipeline, f, indent=4)
            KLogger.info(f"Pipeline stored in {args.output}")

            path_all_results = args.output.with_stem(
                f"{args.output.stem}_all_results"
            )
            with open(path_all_results, "w") as f:
                json.dump(pipelines_scores, f, indent=4)
            KLogger.info(f"All results stored in {path_all_results}")
        else:
            KLogger.info("No pipeline was found for the optimization problem")


if __name__ == "__main__":
    sys.exit(OptimizationRunner.scenario_run())
