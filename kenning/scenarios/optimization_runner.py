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
from itertools import chain, product
from pathlib import Path
from typing import Dict, List

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


def grid_search(json_cfg: Dict) -> Dict:
    """
    Creates a configuration for running the scenarios. For every type of block
    it creates a list of parametrized blocks of this type that can be used to
    run a scenario.

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

    Parameters
    ----------
    json_cfg : Dict
        Configuaration for the grid search optimization

    Returns
    -------
    Dict :
        Dictionary that for every type of block in the scenario
        creates a list of parametrized blocks of this type that can be
        directly used in the final scenario.
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
    for block in blocks_to_optimize:
        block_parameters = [get_block_product(b) for b in json_cfg[block]]

        # We need to treat optimizers differently, as those can be chained.
        # For other blocks we have to pick only one.
        if block == 'optimizers':
            block_parameters = [list(p) for p in product(*block_parameters)]
        else:
            block_parameters = list(chain(*block_parameters))

        optimization_configuration[block] = block_parameters
    return optimization_configuration


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'jsoncfg',
        help='The path to the input JSON file with configuration',
        type=Path
    )
    parser.add_argument(
        'output',
        help='The path to the output JSON file with measurements',
        type=str,
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    args, _ = parser.parse_known_args(argv[1:])

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    optimization_parameters = json_cfg['optimization_parameters']
    optimization_strategy = optimization_parameters['strategy']

    optimization_configuration = None
    if optimization_strategy == 'grid_search':
        optimization_configuration = grid_search(json_cfg)

    scenarios = [
        dict(zip(optimization_configuration.keys(), scenario))
        for scenario in product(*optimization_configuration.values())
    ]

    output_count = 0
    for scenario in scenarios:
        MeasurementsCollector.clear()
        try:
            run_pipeline_json(
                scenario,
                Path(str(output_count) + '_' + args.output),
                args.verbosity
            )
            output_count += 1
        except Exception as ex:
            log.error(f'Scenario: {scenario} was invalid.')
            log.error(ex)


if __name__ == '__main__':
    main(sys.argv)
