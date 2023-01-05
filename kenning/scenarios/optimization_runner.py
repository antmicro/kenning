"""
Script for running automated optimizations for pipelines. It performs
a search on given set of blocks based on JSON configuration passed.

The `optimization_parameters` specifies the parameters of the
optimization search. Currently supported strategy is `grid_search`, which
performs a grid search to find optimal parameters for each block specified
in `optimizable` parameter. Every block that is to be optimized should have
list of parameters instead of a singular value specified.
"""

import argparse
import json
import sys
from itertools import chain, product, combinations
from pathlib import Path
from typing import Dict, List
from pprint import pformat

from jsonschema.exceptions import ValidationError

from kenning.core.metrics import compute_classification_metrics, compute_performance_metrics, compute_detection_metrics  # noqa: E501
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector
from kenning.utils.pipeline_runner import run_pipeline_json

log = logger.get_logger()


def get_block_product(block: Dict[str, List]) -> List:
    """
    Gets a cartesian product of the parameter values

    Example:
    ```python
    block = {
        'type': 'example',
        'parameters': {
            'optimization_level' : [1, 2],
            'dtype': ['int8', 'float32']
        }
    }
    ```
    will yield
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
        {
            'type': 'example',
            'parameters': {
                'optimization_level' : 2,
                'dtype': 'float32'
            }
        }
    ]
    ```

    Parameters
    ----------
    block : dict
        Dictionary with parameters and type keys.
        For parameters, key is the name
        of a parameter and value defines range of values.

    Returns
    -------
    list :
        Cartesian product of input `block`
    """
    return [
        {
            'type': block['type'],
            'parameters': dict(zip(block['parameters'].keys(), p))
        }
        for p in product(*block['parameters'].values())
    ]


def ordered_powerset(iterable: List, min_elements: int = 1) -> List[List]:
    """
    Generates a powerset of ordered elements of `iterable` argument.

    Example
    ```python
    iterable = [1, 2, 3]
    min_elements = 1
    ```
    will return
    ```
    [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    ```

    Parameters
    ----------
    iterable : List
        List of arguments
    min_elements : int
        Minimal number of elements in the powerset

    Returns
    -------
    List[List]
        Powerset of ordered values
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

    Parameters
    ----------
    json_cfg : Dict
        Configuaration for the grid search optimization

    Returns
    -------
    List[Dict] :
        List of pipeline configurations.
    """
    optimization_parameters = json_cfg['optimization_parameters']
    blocks_to_optimize = [
        block
        for block in optimization_parameters['optimizable']
    ]
    remaining_blocks = [
        block
        for block in ['model_wrapper', 'dataset', 'optimizers', 'runtime', 'runtime_protocol']  # noqa: E501
        if block not in blocks_to_optimize and block in json_cfg
    ]

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


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'jsoncfg',
        help='The path to the input JSON file with configuration',
        type=Path
    )
    parser.add_argument(
        'output',
        help='The path to the output JSON file with the best pipeline',
        type=Path,
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    args, _ = parser.parse_known_args(argv[1:])
    logger.set_verbosity(args.verbosity)

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    optimization_parameters = json_cfg['optimization_parameters']
    optimization_strategy = optimization_parameters['strategy']
    policy = optimization_parameters['policy']
    metric = optimization_parameters['metric']

    pipelines = None
    if optimization_strategy == 'grid_search':
        pipelines = grid_search(json_cfg)

    pipelines_num = len(pipelines)
    pipelines_scores = []

    log.info(f'Finding {policy} for {metric}')
    for pipeline_count, pipeline in enumerate(pipelines):
        MeasurementsCollector.clear()
        try:
            log.info(f'Running pipeline {pipeline_count + 1} / {pipelines_num}')  # noqa: E501
            log.info(f'Configuration {pformat(pipeline)}')
            measurementspath = str(args.output.with_suffix('')) + \
                '_' + \
                str(pipeline_count) + \
                str(args.output.suffix)

            run_pipeline_json(
                pipeline,
                Path(measurementspath),
                args.verbosity
            )

            # Consider using MeasurementsCollector.measurements
            with open(measurementspath, 'r') as measurementsfile:
                measurements = json.load(measurementsfile)

            computed_metrics = {}

            computed_metrics |= compute_performance_metrics(measurements)
            computed_metrics |= compute_classification_metrics(measurements)
            computed_metrics |= compute_detection_metrics(measurements)
            computed_metrics.pop('session_utilization_cpus_percent_avg')

            try:
                pipelines_scores.append(
                    {
                        'pipeline': pipeline,
                        'metrics': computed_metrics
                    }
                )
            except KeyError:
                log.error(f'{metric} not found in the metrics')
                raise
        except ValidationError as ex:
            log.error('Incorrect parameters passed')
            log.error(ex)
            raise
        except Exception as ex:
            log.error('Pipeline was invalid')
            log.error(ex)

    if pipelines_scores:
        policy_fun = min if policy == 'min' else max
        best_pipeline = policy_fun(
            pipelines_scores,
            key=lambda pipeline: pipeline['metrics'][metric]
        )

        best_score = best_pipeline['metrics'][metric]
        log.info(f'Best score for {metric} is {best_score}')
        with open(args.output, 'w') as f:
            json.dump(best_pipeline, f)
        log.info(f'Pipeline stored in {args.output}')

        path_all_results = str(args.output.with_suffix('')) + \
            '_' + \
            'all_results' + \
            str(args.output.suffix)
        with open(path_all_results, 'w') as f:
            json.dump(pipelines_scores, f)
        log.info(f'All results stored in {path_all_results}')
    else:
        log.info('No pipeline was found for the optimization problem')


if __name__ == '__main__':
    main(sys.argv)
