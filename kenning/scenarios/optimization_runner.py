# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
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
from argcomplete.completers import FilesCompleter
from itertools import chain, combinations, product
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

from jsonschema.exceptions import ValidationError

import kenning.utils.logger as logger
from kenning.cli.command_template import (
    FINE_TUNE,
    GROUP_SCHEMA,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.core.measurements import MeasurementsCollector
from kenning.core.metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
    compute_performance_metrics,
)
from kenning.utils.pipeline_runner import PipelineRunner

log = logger.get_logger()


def get_block_product(block: Dict[str, Any]) -> List:
    """
    Gets a cartesian product of the parameter values.

    Parameters
    ----------
    block : Dict[str, List]
        Dictionary with parameters and type keys.
        For parameters, key is the name
        of a parameter and value defines range of values.

    Returns
    -------
    List :
        Cartesian product of input `block`.

    Examples
    --------
    For argument
    ```python
    block = {
        'type': 'example',
        'parameters': {
            'optimization_level' : [1, 2],
            'dtype': ['int8', 'float32']
        }
    }
    ```
    will return
    ```python
    [
        {
            'type': 'example',
            'parameters': {
                'optimization_level' : 1,
                'dtype': 'int8'
            }
        },
        {
            'type': 'example',
            'parameters': {
                'optimization_level' : 1,
                'dtype': 'float32'
            }
        },
        {
            'type': 'example',
            'parameters': {
                'optimization_level' : 2,
                'dtype': 'int8'
            }
        },
        {
            'type': 'example',
            'parameters': {
                'optimization_level' : 2,
                'dtype': 'float32'
            }
        }
    ]
    ```
    """
    return [
        {
            'type': block['type'],
            'parameters': dict(zip(block['parameters'].keys(), p)),
        }
        for p in product(*block['parameters'].values())
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
    List[List] :
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


def grid_search(json_cfg: Dict) -> List[Dict]:
    """
    Creates all possible pipeline configurations based on input `json_cfg`.
    For every type of block it creates a list of parametrized blocks
    of this type that can be used to run a pipeline.
    Then for all of the generated blocks cartesian product is computed.

    Parameters
    ----------
    json_cfg : Dict
        Configuration for the grid search optimization.

    Returns
    -------
    List[Dict] :
        List of pipeline configurations.

    Examples
    --------
    An example of an optimizable runtime block
    ```python
    "runtime":
    [
        {
            "type": "kenning.runtimes.tvm.TVMRuntime",
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tar"]
            }
        },
        {
            "type": "kenning.runtimes.tflite.TFLiteRuntime",
            "parameters":
            {
                "save_model_path": ["./build/compiled_model.tflite"],
                "num_threads": [2, 4]
            }
        }
    ]
    ```
    will yield a list of valid runtime blocks that can be used.
    Those are valid runtime blocks and every one of them can be used
    as a runtime.
    ```python
    "runtime":
    [
        {
            "type": "kenning.runtimes.tvm.TVMRuntime",
            "parameters":
            {
                "save_model_path": "./build/compiled_model.tar"
            }
        },
        {
            "type": "kenning.runtimes.tflite.TFLiteRuntime",
            "parameters":
            {
                "save_model_path": "./build/compiled_model.tflite",
                "num_threads": 2
            }
        },
        {
            "type": "kenning.runtimes.tflite.TFLiteRuntime",
            "parameters":
            {
                "save_model_path": "./build/compiled_model.tflite",
                "num_threads": 4
            }
        }
    ]
    ```
    This is done to every block type.
    Then a cartesian product is computed that returns all possible
    pipeline configurations.
    """
    optimization_parameters = json_cfg['optimization_parameters']
    blocks_to_optimize = set(optimization_parameters['optimizable'])
    all_blocks = {
        'model_wrapper',
        'dataset',
        'optimizers',
        'runtime',
        'protocol',
    }
    remaining_blocks = all_blocks & (set(json_cfg.keys()) - blocks_to_optimize)

    optimization_configuration = {}

    for block in remaining_blocks:
        optimization_configuration[block] = [json_cfg[block]]

    # Grid search
    # Creating all possible block configuration for every block type
    for block in blocks_to_optimize:
        block_parameters = [get_block_product(b) for b in json_cfg[block]]

        # We need to treat optimizers differently, as those can be chained.
        # For other blocks we have to pick only one.
        if block == 'optimizers':
            optimizers_blocks = [list(p) for p in product(*block_parameters)]

            # We also need to take every mask of the list of optimizers.
            # For example if we have optimizers A and B then we could use
            # only A, only B, both A and B or neither in the final pipeline.
            final_parameters = []
            for bp in optimizers_blocks:
                final_parameters.append(ordered_powerset(bp))
            final_parameters = list(chain(*final_parameters))

            # Get rid of the duplicates
            # For great numbers of pipelines this may be very expensive.
            block_parameters = []
            for p in final_parameters:
                if p not in block_parameters:
                    block_parameters.append(p)
        else:
            block_parameters = list(chain(*block_parameters))

        optimization_configuration[block] = block_parameters

    # Create all possible pipelines from all possible blocks configurations
    # by taking a cartesian product.
    # TODO: For bigger optimizations problems consider using yield.
    pipelines = [
        dict(zip(optimization_configuration.keys(), pipeline))
        for pipeline in product(*optimization_configuration.values())
    ]
    return pipelines


def replace_paths(pipeline: Dict, id: int) -> Dict:
    """
    Copies given `pipeline` and puts `id`_ in front of `compiled_model_path`
    parameter in every optimizer and in front of `save_model_path` parameter
    in runtime.

    It is used when running pipelines so that every pipeline gets its own
    unique namespace. Thanks to that collision names are avoided.

    Parameters
    ----------
    pipeline : Dict
        Pipeline that gets copied and its parameters are replaced.
    id : int
        Value that is used to create a prefix for the path.

    Returns
    -------
    Dict :
        Pipeline with `compiled_model_path` and `save_model_path` parameters
        changed.
    """
    pipeline = copy.deepcopy(pipeline)
    for optimizer in pipeline['optimizers']:
        path = Path(optimizer['parameters']['compiled_model_path'])
        new_path = path.with_stem(f'{str(id)}_{path.stem}')
        optimizer['parameters']['compiled_model_path'] = str(new_path)

    path = Path(pipeline['runtime']['parameters']['save_model_path'])
    new_path = path.with_stem(f'{str(id)}_{path.stem}')
    pipeline['runtime']['parameters']['save_model_path'] = str(new_path)
    return pipeline


def filter_invalid_pipelines(pipelines: List[Dict]) -> List[Dict]:
    """
    Filter pipelines with incompatible blocks.

    Parameters
    ----------
    pipelines : List[Dict]
        List of pipelines configs.

    Returns
    -------
    List[Dict] :
        Valid pipelines from provided pipelines.
    """
    filtered_pipelines = []

    for pipeline in pipelines:
        try:
            PipelineRunner.from_json_cfg(
                pipeline,
                assert_integrity=True
            )
            filtered_pipelines.append(pipeline)
        except ValueError:
            pass

    return filtered_pipelines


class OptimizationRunner(CommandTemplate):
    parse_all = True
    description = __doc__.split('\n\n')[0]

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
            '--json-cfg',
            help='The path to the input JSON file with configuration',
            type=Path,
            required=True,
        ).completer = FilesCompleter("*.json")
        command_group.add_argument(
            '--output',
            help='The path to the output JSON file with the best pipeline',
            type=Path,
            required=True,
        ).completer = FilesCompleter("*.json")

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        logger.set_verbosity(args.verbosity)

        with open(args.json_cfg, 'r') as f:
            json_cfg = json.load(f)

        optimization_parameters = json_cfg['optimization_parameters']
        optimization_strategy = optimization_parameters['strategy']
        policy = optimization_parameters['policy']
        metric = optimization_parameters['metric']

        if optimization_strategy == 'grid_search':
            pipelines = grid_search(json_cfg)
        else:
            raise ValueError(
                f'Invalid optimization strategy: {optimization_strategy}'
            )

        pipelines = filter_invalid_pipelines(pipelines)

        pipelines_num = len(pipelines)
        pipelines_scores = []

        log.info(f'Finding {policy} for {metric}')
        for pipeline_idx, pipeline in enumerate(pipelines):
            module_error = None
            pipeline = replace_paths(pipeline, pipeline_idx)
            MeasurementsCollector.clear()
            try:
                log.info(
                    f'Running pipeline {pipeline_idx + 1} / {pipelines_num}'
                )
                log.info(f'Configuration {pformat(pipeline)}')
                measurements_path = args.output.with_stem(
                    f'{args.output.stem}_{pipeline_idx}'
                )

                pipeline_runner = PipelineRunner.from_json_cfg(pipeline)
                pipeline_runner.run(
                    output=Path(measurements_path),
                    verbosity=args.verbosity
                )

                # Consider using MeasurementsCollector.measurements
                with open(measurements_path, 'r') as measurements_file:
                    measurements = json.load(measurements_file)

                computed_metrics = {}

                computed_metrics |= compute_performance_metrics(measurements)
                computed_metrics |= compute_classification_metrics(
                    measurements
                )
                computed_metrics |= compute_detection_metrics(measurements)
                computed_metrics.pop('session_utilization_cpus_percent_avg')

                try:
                    pipelines_scores.append(
                        {'pipeline': pipeline, 'metrics': computed_metrics}
                    )
                except KeyError:
                    log.error(f'{metric} not found in the metrics')
                    raise
            except ValidationError as ex:
                log.error('Incorrect parameters passed')
                log.error(ex)
                raise
            except ModuleNotFoundError as missing_module_error:
                module_error = missing_module_error
            except Exception as ex:
                log.warning('Pipeline was invalid')
                log.warning(ex)

            if module_error:
                raise module_error

        if pipelines_scores:
            policy_fun = min if policy == 'min' else max
            best_pipeline = policy_fun(
                pipelines_scores,
                key=lambda pipeline: pipeline['metrics'][metric],
            )

            best_score = best_pipeline['metrics'][metric]
            log.info(f'Best score for {metric} is {best_score}')
            with open(args.output, 'w') as f:
                json.dump(best_pipeline, f, indent=4)
            log.info(f'Pipeline stored in {args.output}')

            path_all_results = args.output.with_stem(
                f'{args.output.stem}_all_results'
            )
            with open(path_all_results, 'w') as f:
                json.dump(pipelines_scores, f, indent=4)
            log.info(f'All results stored in {path_all_results}')
        else:
            log.info('No pipeline was found for the optimization problem')


if __name__ == '__main__':
    sys.exit(OptimizationRunner.scenario_run())
