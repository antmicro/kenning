"""
Module with scenarios running helper functions
"""

import tempfile
from pathlib import Path

from typing import Optional, Dict, Tuple, List

from kenning.utils.class_loader import load_class
from kenning.utils.args_manager import serialize_inference
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector


def assert_io_formats(model, optimizers, runtime) -> None:
    """
    Asserts that given blocks can be put together in a scenario.

    Parameters
    ----------
    model : ModelWrapper
        ModelWrapper of the scenario
    optimizers : Union[List[Optimizer], Optimizer]
        Optimizers of the scenario
    runtime : Runtime
        Runtime of the scenario

    Raises
    ------
    ValueError : raised if blocks are incompatible.
    """
    chain = [model] + optimizers + [runtime]

    for previous_block, next_block in zip(chain, chain[1:]):
        if (set(next_block.get_input_formats()) &
                set(previous_block.get_output_formats())):
            continue

        raise ValueError(
            f'No matching formats between two objects: {previous_block} and ' +  # noqa: E501
            f'{next_block}\n' +
            f'Output block supported formats: {", ".join(previous_block.get_output_formats())}\n' +  # noqa: E501
            f'Input block supported formats: {", ".join(next_block.get_input_formats())}'  # noqa: E501
        )


def parse_argparse_scenario():
    # TODO: Decide on how to implement that
    pass


def parse_json_scenario(
        json_cfg: Dict,
        assert_integrity: bool = True) -> Tuple:
    """
    Method that parses a json configuration of an inference scenario.

    It also checks whether the scenario is correct in terms of connected
    blocks if `assert_integrity` is set to true.

    Can be used to check whether blocks' parameters
    and the order of the scenario are correct.

    Parameters
    ----------
    json_cfg: Dict
        Configuration of the inference scenario
    assert_integrity: bool
        States whether integrity of connected blocks should be checked.

    Raises
    ------
    ValueError : raised if blocks are connected incorrectly

    Returns
    -------
    Tuple :
        Tuple that consists of (Dataset, Model, List[Optimizer], Optional[Runtime], Optional[RuntimeProtocol])  # noqa: E501
        It can be used to run a scenario using `run_scenario` function.
    """
    modelwrappercfg = json_cfg['model_wrapper']
    datasetcfg = json_cfg['dataset']
    runtimecfg = (
        json_cfg['runtime']
        if 'runtime' in json_cfg else None
    )
    optimizerscfg = (
        json_cfg['optimizers']
        if 'optimizers' in json_cfg else []
    )
    protocolcfg = (
        json_cfg['runtime_protocol']
        if 'runtime_protocol' in json_cfg else None
    )

    modelwrappercls = load_class(modelwrappercfg['type'])
    datasetcls = load_class(datasetcfg['type'])
    optimizerscls = [load_class(cfg['type']) for cfg in optimizerscfg]
    protocolcls = (
        load_class(protocolcfg['type'])
        if protocolcfg else None
    )
    runtimecls = load_class(runtimecfg['type'])

    dataset = datasetcls.from_json(datasetcfg['parameters'])
    model = modelwrappercls.from_json(dataset, modelwrappercfg['parameters'])
    optimizers = [
        cls.from_json(dataset, cfg['parameters'])
        for cfg, cls in zip(optimizerscfg, optimizerscls)
    ]
    protocol = (
        protocolcls.from_json(protocolcfg['parameters'])
        if protocolcls else None
    )
    runtime = runtimecls.from_json(protocol, runtimecfg['parameters'])

    if assert_integrity:
        assert_io_formats(model, optimizers, runtime)
    return (dataset, model, optimizers, runtime, protocol)


def run_scenario_json(
        json_cfg: Dict,
        output: Path,
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        command: List = ['Run in a different environment'],
        run_benchmarks_only: bool = False) -> int:
    """
    Simple wrapper for `run_scenario` method that parses `json_cfg` argument,
    asserts its integrity and then runs the scenario with given parameters.

    Parameters
    ----------
    json_cfg : dict
        Configuration of the inference scenario
    output : Path
        Path to the output JSON file with measurements
    verbosity : Optional[str]
        Verbosity level
    convert_to_onnx : Optional[Path]
        Before compiling the model, convert it to ONNX and use in the inference (provide a path to save here)  # noqa: E501
    command : Optional[List]
        Command used to run this inference scenario. It is put in
        the output JSON file
    run_benchmarks_only : bool
        Instead of running the full compilation and testing flow,
        only testing of the model is executed
    """
    run_scenario(
        *parse_json_scenario(json_cfg),
        output,
        verbosity,
        convert_to_onnx,
        command,
        run_benchmarks_only
    )


def run_scenario(
        dataset,
        model,
        optimizers,
        runtime,
        protocol,
        output: Path,
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        command: List = ['Run in a different environment'],
        run_benchmarks_only: bool = False):
    """
    Wrapper function that runs a scenario using given parameters

    Parameters
    ----------
    dataset : Dataset
        Dataset to use in inference
    model : ModelWrapper
        ModelWrapper to use in inference
    optimizers : List[Optimizer]
        Optimizers to use in inference
    runtime : Runtime
        Runtime to use in inference
    runtimeprotocol : RuntimeProtocol
        RuntimeProtocol to use in inference
    output : Path
        Path to the output JSON file with measurements
    verbosity : Optional[str]
        Verbosity level
    convert_to_onnx : Optional[Path]
        Before compiling the model, convert it to ONNX and use in the inference (provide a path to save here)  # noqa: E501
    command : Optional[List]
        Command used to run this inference scenario. It is put in
        the output JSON file
    run_benchmarks_only : bool
        Instead of running the full compilation and testing flow,
        only testing of the model is executed

    Returns
    -------
    int : 0 if the inference was successful, 1 otherwise
    """
    logger.set_verbosity(verbosity)
    log = logger.get_logger()

    assert_io_formats(model, optimizers, runtime)

    modelframeworktuple = model.get_framework_and_version()

    MeasurementsCollector.measurements += {
        'model_framework': modelframeworktuple[0],
        'model_version': modelframeworktuple[1],
        'compilers': [
            {
                'compiler_framework': optimizer.get_framework_and_version()[0],
                'compiler_version': optimizer.get_framework_and_version()[1]
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
        )
    }

    # TODO add method for providing metadata to dataset
    if hasattr(dataset, 'classnames'):
        MeasurementsCollector.measurements += {
            'class_names': [val for val in dataset.get_class_names()]
        }

    modelpath = model.get_path()

    if not run_benchmarks_only:
        prev_block = model
        if convert_to_onnx:
            log.warn(
                'Force conversion of the input model to the ONNX format'
            )
            modelpath = convert_to_onnx
            prev_block.save_to_onnx(modelpath)

        for i in range(len(optimizers)):
            next_block = optimizers[i]

            log.info(f'Processing block:  {type(next_block).__name__}')

            format = next_block.consult_model_type(
                prev_block,
                force_onnx=(convert_to_onnx and prev_block == model)
            )

            if (format == 'onnx' and prev_block == model) and \
                    not convert_to_onnx:
                modelpath = Path(tempfile.NamedTemporaryFile().name)
                prev_block.save_to_onnx(modelpath)

            prev_block.save_io_specification(modelpath)
            next_block.set_input_type(format)
            next_block.compile(modelpath)

            prev_block = next_block
            modelpath = prev_block.compiled_model_path

        if not optimizers:
            model.save_io_specification(modelpath)
    else:
        if len(optimizers) > 0:
            modelpath = optimizers[-1].compiled_model_path

    if runtime:
        if protocol:
            ret = runtime.run_client(dataset, model, modelpath)
        else:
            ret = runtime.run_locally(dataset, model, modelpath)
    else:
        model.test_inference()
        ret = True

    if not ret:
        return 1

    MeasurementsCollector.measurements += {
        'compiled_model_size': Path(modelpath).stat().st_size
    }

    MeasurementsCollector.save_measurements(output)
    return 0
