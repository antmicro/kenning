# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with pipelines running helper functions.
"""

import os
import tempfile
from pathlib import Path

from typing import Optional, Dict, Tuple, List, Union
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.runtime import Runtime

from kenning.utils.class_loader import load_class, get_all_subclasses
from kenning.utils.args_manager import serialize_inference
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector

UNOPTIMIZED_MEASUREMENTS = '__unoptimized__'


def assert_io_formats(
        model: Optional[ModelWrapper],
        optimizers: Union[List[Optimizer], Optimizer],
        runtime: Optional[Runtime]):
    """
    Asserts that given blocks can be put together in a pipeline.

    Parameters
    ----------
    model : Optional[ModelWrapper]
        ModelWrapper of the pipeline.
    optimizers : Union[List[Optimizer], Optimizer]
        Optimizers of the pipeline.
    runtime : Optional[Runtime]
        Runtime of the pipeline.

    Raises
    ------
    ValueError :
        Raised if blocks are incompatible.
    """
    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]

    chain = [model] + optimizers + [runtime]
    chain = [block for block in chain if block is not None]

    for previous_block, next_block in zip(chain, chain[1:]):
        check_model_type = getattr(next_block, 'consult_model_type', None)
        if callable(check_model_type):
            check_model_type(previous_block)
            continue
        elif (set(next_block.get_input_formats()) &
                set(previous_block.get_output_formats())):
            continue

        if next_block == runtime:
            log = logger.get_logger()
            log.warning(
                f'Runtime {next_block} has no matching format with the '
                f'previous block: {previous_block}\nModel may not run '
                'correctly'
            )
            continue

        output_formats_str = ', '.join(previous_block.get_output_formats())
        input_formats_str = ', '.join(previous_block.get_output_formats())
        raise ValueError(
            f'No matching formats between two objects: {previous_block} and '
            f'{next_block}\n'
            f'Output block supported formats: {output_formats_str}\n'
            f'Input block supported formats: {input_formats_str}'
        )


def parse_json_pipeline(
    json_cfg: Dict,
    assert_integrity: bool = True,
    skip_optimizers: bool = False,
    skip_runtime: bool = False,
) -> Tuple:
    """
    Method that parses a json configuration of an inference pipeline.

    It also checks whether the pipeline is correct in terms of connected
    blocks if `assert_integrity` is set to true.

    Can be used to check whether blocks' parameters
    and the order of the blocks are correct.

    Parameters
    ----------
    json_cfg : Dict
        Configuration of the inference pipeline.
    assert_integrity : bool
        States whether integrity of connected blocks should be checked.
    skip_optimizers : bool
        States whether optimizers should be created
    skip_runtime : bool
        States whether runtime should be created

    Returns
    -------
    Tuple :
        Tuple that consists of (Dataset, Model, List[Optimizer],
        Optional[Runtime], Optional[RuntimeProtocol], Optional[str])
        It can be used to run a pipeline using `run_pipeline` function.

    Raises
    ------
    ValueError :
        Raised if blocks are connected incorrectly.
    jsonschema.exceptions.ValidationError :
        Raised if parameters are incorrect.
    """
    modelwrappercfg = json_cfg['model_wrapper']
    datasetcfg = json_cfg['dataset'] if 'dataset' in json_cfg else None
    runtimecfg = (
        json_cfg['runtime']
        if 'runtime' in json_cfg and not skip_runtime else None
    )
    optimizerscfg = (
        json_cfg['optimizers']
        if 'optimizers' in json_cfg and not skip_optimizers else []
    )
    protocolcfg = (
        json_cfg['runtime_protocol']
        if 'runtime_protocol' in json_cfg else None
    )
    try:
        last_compiled_model_path = \
            json_cfg['optimizers'][-1]['parameters']['compiled_model_path']
    except (KeyError, IndexError):
        last_compiled_model_path = None

    modelwrappercls = load_class(modelwrappercfg['type'])
    datasetcls = load_class(datasetcfg['type']) if datasetcfg else None
    optimizerscls = [load_class(cfg['type']) for cfg in optimizerscfg]
    protocolcls = (
        load_class(protocolcfg['type'])
        if protocolcfg else None
    )
    runtimecls = (
        load_class(runtimecfg['type'])
        if runtimecfg else None
    )
    dataset = (
        datasetcls.from_json(datasetcfg['parameters'])
        if datasetcls else None
    )
    model = modelwrappercls.from_json(dataset, modelwrappercfg['parameters'])
    optimizers = [
        cls.from_json(dataset, cfg['parameters'])
        for cfg, cls in zip(optimizerscfg, optimizerscls)
    ]
    protocol = (
        protocolcls.from_json(protocolcfg['parameters'])
        if protocolcls else None
    )
    runtime = (
        runtimecls.from_json(protocol, runtimecfg['parameters'])
        if runtimecls else None
    )
    model_name = modelwrappercfg['parameters']['model_name'] \
        if 'model_name' in modelwrappercfg['parameters'] else None

    if assert_integrity:
        assert_io_formats(model, optimizers, runtime)
    return (
        dataset, model, optimizers, runtime, protocol,
        model_name, last_compiled_model_path
    )


def run_pipeline_json(
        json_cfg: Dict,
        output: Optional[Path],
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        command: List = ['Run in a different environment'],
        run_optimizations: bool = True,
        run_benchmarks: bool = True,
        evaluate_unoptimized: bool = False,
) -> int:
    """
    Simple wrapper for `run_pipeline` method that parses `json_cfg` argument,
    asserts its integrity and then runs the pipeline with given parameters.

    Parameters
    ----------
    json_cfg : dict
        Configuration of the inference pipeline.
    output : Optional[Path]
        Path to the output JSON file with measurements.
    verbosity : Optional[str]
        Verbosity level.
    convert_to_onnx : Optional[Path]
        Before compiling the model, convert it to ONNX and use in the inference
        (provide a path to save here).
    command : Optional[List]
        Command used to run this inference pipeline. It is put in
        the output JSON file.
    run_optimizations : bool
        If False, optimizations will not be executed.
    run_benchmarks : bool
        If False, model will not be tested.
    evaluate_unoptimized : bool
        Defines if unoptimized model should be tested.

    Returns
    -------
    int :
        The 0 value if the inference was successful, 1 otherwise.

    Raises
    ------
    ValueError :
        Raised if blocks are connected incorrectly.
    jsonschema.exceptions.ValidationError :
        Raised if parameters are incorrect.
    """
    (
        dataset, model, optimizers, runtime, protocol,
        model_name, last_compiled_model_path
    ) = parse_json_pipeline(
            json_cfg, skip_optimizers=not run_optimizations,
            skip_runtime=not run_benchmarks
        )
    return run_pipeline(
        dataset,
        model,
        optimizers,
        runtime,
        protocol,
        last_compiled_model_path,
        output,
        verbosity,
        convert_to_onnx,
        command,
        run_optimizations and optimizers,
        run_benchmarks and dataset,
        model_name,
        evaluate_unoptimized,
    )


def mark_measurements_unoptimized():
    """
    Marks current measurements as unoptimized
    """
    unoptimized = MeasurementsCollector.measurements.copy()
    MeasurementsCollector.clear()
    MeasurementsCollector.measurements += {
        UNOPTIMIZED_MEASUREMENTS: unoptimized
    }


def test_unoptimized(
    dataset,
    model,
    protocol,
    log: logger.logging.Logger,
    verbosity: str = 'INFO',
    model_name: Optional[str] = None,
):
    """
    Evaluates unoptimized model.

    If model's framework does not have dedicated runtime class
    default test_inference is used, otherwise runtime is created.

    Parameters
    ----------
    dataset : Dataset
        Dataset to use in inference.
    model : ModelWrapper
        ModelWrapper to use in inference.
    protocol : RuntimeProtocol
        RuntimeProtocol to use in inference.
    log : logging.Logger
        Logger for printing messages.
    verbosity : str
        Verbosity of the evaluation process.
    model_name: Optional[str]:
        Custom name of the model.
    """
    try:
        run_pipeline(
            dataset,
            model,
            [],
            None,
            protocol,
            last_compiled_model_path=None,
            output=os.devnull,
            verbosity=verbosity,
            convert_to_onnx=None,
            command=[],
            run_optimizations=False,
            run_benchmarks=True,
            model_name='unoptimized_' + model_name if model_name else None,
            evaluate_unoptimized=False,
        )
    except NotImplementedError:
        log.warn("Model's run_inference is not implemented")
        MeasurementsCollector.clear()
    else:
        mark_measurements_unoptimized()
        return True

    framework = model.get_framework_and_version()[0]
    runtime = None
    runtime_classes = get_all_subclasses(
        'kenning.runtimes',
        Runtime,
        raise_exception=False,
        import_classes=True,
        show_warnings=False
    )
    # Get first available Runtime with matching inputtype
    runtime_cls = next(
        filter(lambda _cls: framework in _cls.inputtypes, runtime_classes),
        None
    )
    # Initialize Runtime
    if runtime_cls:
        with tempfile.TemporaryFile() as tmpfile:
            runtime = runtime_cls(
                protocol,
                model.get_path() if not protocol else tmpfile,
                disable_performance_measurements=False,
            )
    if not runtime:
        log.error(f'Unoptimized {model.__class__.__name__} cannot be tested, there is no Runtime for used framework')  # noqa: E501
        return False
    if 0 != run_pipeline(
        dataset,
        model,
        [],
        runtime,
        protocol,
        last_compiled_model_path=None,
        output=os.devnull,
        verbosity=verbosity,
        convert_to_onnx=None,
        command=[],
        run_optimizations=False,
        run_benchmarks=True,
        model_name='unoptimized_' + model_name if model_name else None,
        evaluate_unoptimized=False,
    ):
        log.error('Testing unoptimized model failed')
    mark_measurements_unoptimized()
    return True


def run_pipeline(
        dataset,
        model,
        optimizers,
        runtime,
        protocol,
        last_compiled_model_path: Optional[Path] = None,
        output: Optional[Path] = None,
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        command: List = ['Run in a different environment'],
        run_optimizations: bool = True,
        run_benchmarks: bool = True,
        model_name: Optional[str] = None,
        evaluate_unoptimized: bool = False,
) -> int:
    """
    Wrapper function that runs a pipeline using given parameters.

    Parameters
    ----------
    dataset : Dataset
        Dataset to use in inference.
    model : ModelWrapper
        ModelWrapper to use in inference.
    optimizers : List[Optimizer]
        Optimizers to use in inference.
    runtime : Runtime
        Runtime to use in inference.
    protocol : RuntimeProtocol
        RuntimeProtocol to use in inference.
    last_compiled_model_path : Optional[Path]
        Model path from last optimizer
    output : Optional[Path]
        Path to the output JSON file with measurements.
    verbosity : Optional[str]
        Verbosity level.
    convert_to_onnx : Optional[Path]
        Before compiling the model, convert it to ONNX and use
        in the inference (provide a path to save here).
    command : Optional[List]
        Command used to run this inference pipeline. It is put in
        the output JSON file.
    run_optimizations : bool
        If False, optimizations will not be executed.
    run_benchmarks : bool
        If False, model will not be tested.
    model_name : Optional[str]
        Custom name of the model.
    evaluate_unoptimized : bool
        Defines if unoptimized model should be tested.

    Returns
    -------
    int :
        The 0 value if the inference was successful, 1 otherwise.

    Raises
    ------
    ValueError :
        Raised if blocks are connected incorrectly.
    jsonschema.exceptions.ValidationError :
        Raised if parameters are incorrect.
    """
    assert run_optimizations or run_benchmarks, (
        'If both optimizations and benchmarks are skipped, pipeline will not '
        'be executed'
    )
    logger.set_verbosity(verbosity)
    log = logger.get_logger()

    assert_io_formats(model, optimizers, runtime)

    if evaluate_unoptimized and optimizers:
        log.info("Evaluating unoptimized model")
        if not test_unoptimized(
            dataset,
            model,
            protocol,
            log,
            verbosity,
            model_name
        ):
            return 1

    modelframeworktuple = model.get_framework_and_version()

    if run_benchmarks and not output:
        log.warning(
            'Running benchmarks without defined output -- measurements will '
            'not be saved'
        )

    if output:
        MeasurementsCollector.measurements += {
            'model_framework': modelframeworktuple[0],
            'model_version': modelframeworktuple[1],
            'optimizers': [
                {
                    'compiler_framework':
                        optimizer.get_framework_and_version()[0],
                    'compiler_version':
                        optimizer.get_framework_and_version()[1],
                }
                for optimizer in optimizers
            ],
            'command': command,
            'build_cfg': serialize_inference(
                dataset,
                model,
                optimizers,
                protocol,
                runtime
            ),
        }
        if model_name is not None:
            MeasurementsCollector.measurements += {"model_name": model_name}

        # TODO add method for providing metadata to dataset
        if hasattr(dataset, 'classnames'):
            MeasurementsCollector.measurements += {
                'class_names': [val for val in dataset.get_class_names()]
            }

    model_path = model.get_path()

    if run_optimizations:
        prev_block = model
        if convert_to_onnx:
            log.warning(
                'Force conversion of the input model to the ONNX format'
            )
            model_path = convert_to_onnx
            prev_block.save_to_onnx(model_path)

        for i in range(len(optimizers)):
            next_block = optimizers[i]

            log.info(f'Processing block:  {type(next_block).__name__}')

            format = next_block.consult_model_type(
                prev_block,
                force_onnx=(convert_to_onnx and prev_block == model)
            )

            if (format == 'onnx' and prev_block == model) and \
                    not convert_to_onnx:
                model_path = Path(tempfile.NamedTemporaryFile().name)
                prev_block.save_to_onnx(model_path)

            prev_block.save_io_specification(model_path)
            next_block.set_input_type(format)
            next_block.compile(model_path)
            del prev_block

            prev_block = next_block
            model_path = prev_block.compiled_model_path

        del next_block
        if not optimizers:
            model.save_io_specification(model_path)
    else:
        if last_compiled_model_path:
            model_path = last_compiled_model_path
        elif runtime:
            model_path = runtime.model_path
            model.save_io_specification(model_path)

    ret = True
    if run_benchmarks and runtime:
        if not dataset:
            log.error('The benchmarks cannot be performed without a dataset')
            ret = False
        elif protocol:
            ret = runtime.run_client(dataset, model, model_path)
        else:
            ret = runtime.run_locally(dataset, model, model_path)
    elif run_benchmarks:
        model.model_path = model_path
        model.test_inference()
        ret = True

    if not ret:
        return 1

    if output:
        model_path = Path(model_path)
        # If model compressed in ZIP exists use its size
        # It is more accurate for Keras models
        if model_path.with_suffix('.zip').exists():
            model_path = model_path.with_suffix('.zip')

        MeasurementsCollector.measurements += {
            'compiled_model_size': model_path.stat().st_size
        }

        MeasurementsCollector.save_measurements(output)
    return 0
