# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with pipelines running helper functions.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

import kenning.utils.logger as logger
from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.measurements import (Measurements, MeasurementsCollector,
                                       systemstatsmeasurements,
                                       tagmeasurements)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.protocol import Protocol, RequestFailure, check_request
from kenning.core.runtime import Runtime
from kenning.dataconverters.modelwrapper_dataconverter import \
    ModelWrapperDataConverter

try:
    from kenning.runtimes.renode import RenodeRuntime
except ImportError:
    RenodeRuntime = type(None)

from kenning.utils.args_manager import serialize_inference
from kenning.utils.class_loader import any_from_json
from kenning.utils.resource_manager import PathOrURI

UNOPTIMIZED_MEASUREMENTS = '__unoptimized__'


class PipelineRunner(object):
    """
    Class responsible for running model optimization pipelines.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataconverter: DataConverter,
        optimizers: List[Optimizer],
        runtime: Runtime,
        protocol: Optional[Protocol] = None,
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        Initializes the PipelineRunner object.

        Parameters
        ----------
        dataset : Dataset
            Dataset object that provides data for inference.
        dataconverter : DataConverter
            DataConverter object that converts data from/to Protocol format.
        optimizers : List[Optimizer]
            List of Optimizer objects that optimize the model.
        runtime : Runtime
            Runtime object that runs the inference.
        protocol : Optional[Protocol]
            Protocol object that provides the communication protocol.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper object that wraps the model.
        """
        self.dataset = dataset
        self.dataconverter = dataconverter
        self.model_wrapper = model_wrapper
        self.optimizers = optimizers
        self.runtime = runtime
        self.protocol = protocol

    @classmethod
    def from_json_cfg(
        cls,
        json_cfg: Dict,
        assert_integrity: bool = True,
        skip_optimizers: bool = False,
        skip_runtime: bool = False,
    ) -> 'PipelineRunner':
        """
        Method that parses a json configuration of an inference pipeline.

        It also checks whether the pipeline is correct in terms of connected
        blocks if `assert_integrity` is set to true.

        Can be used to check whether blocks' parameters and the order of the
        blocks are correct.

        Parameters
        ----------
        json_cfg : Dict
            Configuration of the inference pipeline.
        assert_integrity : bool
            States whether integrity of connected blocks should be checked.
        skip_optimizers : bool
            States whether optimizers should be created.
        skip_runtime : bool
            States whether runtime should be created.

        Returns
        -------
        PipelineRunner :
            PipelineRunner created from provided JSON config.

        Raises
        ------
        AssertionError :
            Raised if required blocks are not provided.
        ValueError :
            Raised if blocks are connected incorrectly.
        jsonschema.exceptions.ValidationError :
            Raised if parameters are incorrect.
        """
        if 'runtime' not in json_cfg:
            skip_runtime = True

        dataset = (
            any_from_json(json_cfg['dataset'])
            if 'dataset' in json_cfg else None
        )

        model_wrapper = (
                any_from_json(json_cfg['model_wrapper'], dataset=dataset)
                if 'model_wrapper' in json_cfg else None
        )

        dataconverter = (
            any_from_json(json_cfg['data_converter'])
            if 'data_converter' in json_cfg else None
        )

        assert model_wrapper or dataconverter, (
            'Provide either dataconverter or model_wrapper.'
        )

        if not dataconverter:
            dataconverter = ModelWrapperDataConverter(model_wrapper)

        optimizers = (
            [
                any_from_json(optimizer_cfg, dataset=dataset)
                for optimizer_cfg in json_cfg.get('optimizers', [])
            ]
            if not skip_optimizers else None
        )

        runtime = (
            any_from_json(json_cfg['runtime'])
            if not skip_runtime else None
        )

        protocol = (
            any_from_json(json_cfg['protocol'])
            if 'protocol' in json_cfg else None
        )

        if assert_integrity:
            cls.assert_io_formats(model_wrapper, optimizers, runtime)

        return cls(
            dataset=dataset,
            dataconverter=dataconverter,
            optimizers=optimizers,
            runtime=runtime,
            protocol=protocol,
            model_wrapper=model_wrapper,
        )

    def add_scenario_configuration_to_measurements(
            self,
            command: List,
            model_path: Optional[PathOrURI]):
        """
        Adds scenario configuration to measurements.

        Parameters
        ----------
        command : List
            Command that was used to run the pipeline.
        model_path : Optional[PathOrURI]
            Path to the compiled model.
        """
        MeasurementsCollector.measurements += {
            'optimizers': [
                dict(zip(
                    ('compiler_framework', 'compiler_version'),
                    optimizer.get_framework_and_version()
                ))
                for optimizer in self.optimizers
            ],
            'command': command,
            'build_cfg': serialize_inference(
                self.dataset,
                self.dataconverter,
                self.optimizers,
                self.protocol,
                self.runtime,
                self.model_wrapper
            ),
        }

        if self.model_wrapper:
            framework, version = (
                    self.model_wrapper.get_framework_and_version()
            )
            MeasurementsCollector.measurements += {
                    'model_framework': framework,
                    'model_version': version
            }

        # TODO: add method for providing metadata to dataset
        if hasattr(self.dataset, 'classnames'):
            MeasurementsCollector.measurements += {
                'class_names': self.dataset.get_class_names()
            }

        if model_path:
            model_path = Path(model_path)
            # If model compressed in ZIP exists use its size
            # It is more accurate for Keras models
            if model_path.with_suffix('.zip').exists():
                model_path = model_path.with_suffix('.zip')

            MeasurementsCollector.measurements += {
                'compiled_model_size': model_path.stat().st_size
            }

    def execute_benchmarks(self, model_path: PathOrURI) -> bool:
        """
        Executes appropriate inference method for benchmarking.

        Parameters
        ----------
        model_path : PathOrURI
            Path to the compiled model.

        Returns
        -------
        bool :
            True if the benchmarks were performed successfully,
            False otherwise.
        """
        if self.runtime:
            if not self.dataset:
                logger.get_logger().error(
                        'The benchmarks cannot be performed without a dataset '
                        'and a runtime or model wrapper'
                        )
                return False
            elif self.protocol:
                if isinstance(self.runtime, RenodeRuntime):
                    return self.runtime.run_client(
                            dataset=self.dataset,
                            modelwrapper=self.model_wrapper,
                            protocol=self.protocol,
                            compiled_model_path=model_path
                    )
                return self._run_client(model_path)
            else:
                return self._run_locally(model_path)
        self.model_wrapper.model_path = model_path
        self.model_wrapper.test_inference()
        return True

    def run(
        self,
        output: Optional[Path] = None,
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        max_target_side_optimizers: int = -1,
        command: List = ['Run in a different environment'],
        run_optimizations: bool = True,
        run_benchmarks: bool = True,
    ) -> int:
        """
        Wrapper function that runs a pipeline using given parameters.

        Parameters
        ----------
        output : Optional[Path]
            Path to the output JSON file with measurements.
        verbosity : Optional[str]
            Verbosity level.
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use
            in the inference (provide a path to save here).
        max_target_side_optimizers : int
            Max number of consecutive target-side optimizers.
        command : Optional[List]
            Command used to run this inference pipeline. It is put in
            the output JSON file.
        run_optimizations : bool
            If False, optimizations will not be executed.
        run_benchmarks : bool
            If False, model will not be tested.

        Returns
        -------
        int :
            The 0 value if the inference was successful, 1 otherwise.

        Raises
        ------
        AssertionError :
            Raised if required blocks are not provided.
        ValueError :
            Raised if blocks are connected incorrectly.
        """
        assert run_optimizations or run_benchmarks, (
            'If both optimizations and benchmarks are skipped, pipeline will '
            'not be executed'
        )
        logger.set_verbosity(verbosity)

        self.assert_io_formats(
            self.model_wrapper,
            self.optimizers,
            self.runtime
        )

        ret = True
        if self.protocol:
            check_request(self.protocol.initialize_client(), 'prepare client')
        model_path = self.handle_optimizations(
            convert_to_onnx,
            run_optimizations
        )
        if output:
            self.add_scenario_configuration_to_measurements(
                    command, model_path
            )
        if run_benchmarks:
            if not output:
                logger.get_logger().warning(
                    'Running benchmarks without defined output -- '
                    'measurements will not be saved'
                )
            ret = self.execute_benchmarks(model_path)

        if output:
            MeasurementsCollector.save_measurements(output)

        if ret:
            return 0
        return 1

    def upload_essentials(self,
                          compiled_model_path: Optional[PathOrURI]) -> bool:
        """
        Wrapper for uploading data to the server.
        Uploads model by default.

        Parameters
        ----------
        compiled_model_path : Optional[PathOrURI]
            Path or URI to the file with a compiled model.

        Returns
        -------
        bool :
            True if succeeded.
        """
        if not compiled_model_path:
            logger.get_logger().warning(
                'No compiled model provided, skipping uploading IO spec'
            )
        else:
            spec_path = self.runtime.get_io_spec_path(compiled_model_path)
            if spec_path.exists():
                self.protocol.upload_io_specification(spec_path)
            else:
                logger.get_logger().info('No Input/Output specification found')
        return self.protocol.upload_model(compiled_model_path)

    def handle_optimizations(
        self,
        convert_to_onnx: Optional[Path] = None,
        run_optimization: bool = True,
        max_target_side_optimizers: int = -1,
    ) -> Optional[Path]:
        """
        Handle model optimization.

        Parameters
        ----------
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use in the
            inference (provide a path to save here).
        run_optimization : bool
            Determines if optimizations should be executed, otherwise last
            compiled model path is returned.
        max_target_side_optimizers : int
            Max number of consecutive target-side optimizers.

        Returns
        -------
        Optional[Path] :
            Path to compiled model.

        Raises
        ------
        RuntimeError :
            When any of the server request fails.
        """
        if not run_optimization:
            if self.optimizers:
                return self.optimizers[-1].compiled_model_path
            elif self.model_wrapper:
                model_path = self.model_wrapper.get_path()
                self.model_wrapper.save_io_specification(model_path)
                return model_path
            else:
                return None

        assert self.model_wrapper, 'Model wrapper is required for optimizations'    # noqa: E501
        model_path = self.model_wrapper.get_path()
        prev_block = self.model_wrapper
        if convert_to_onnx:
            logger.get_logger().warning(
                'Force conversion of the input model to the ONNX format'
            )
            model_path = convert_to_onnx
            prev_block.save_to_onnx(model_path)

        optimizer_idx = 0
        while optimizer_idx < len(self.optimizers):
            next_block = self.optimizers[optimizer_idx]

            model_type = next_block.consult_model_type(
                prev_block,
                force_onnx=(
                    convert_to_onnx is not None and
                    isinstance(prev_block, ModelWrapper)
                )
            )

            if (
                model_type == 'onnx' and
                isinstance(prev_block, ModelWrapper) and
                not convert_to_onnx
            ):
                model_path = Path(tempfile.NamedTemporaryFile().name)
                prev_block.save_to_onnx(model_path)

            prev_block.save_io_specification(model_path)

            if 'target' == next_block.location and self.protocol is not None:
                server_optimizers = []
                while (
                    optimizer_idx < len(self.optimizers) and
                    'target' == self.optimizers[optimizer_idx].location and
                    (
                        len(server_optimizers) < max_target_side_optimizers
                        or max_target_side_optimizers == -1
                    )
                ):
                    server_optimizers.append(self.optimizers[optimizer_idx])
                    optimizer_idx += 1

                optimizers_cfg = {
                    'prev_block': {
                        'model_path': model_path,
                        'model_type': model_type,
                        'io_spec': prev_block.load_io_specification(model_path)
                    },
                    'optimizers': [
                        optimizer.to_json() for optimizer in server_optimizers
                    ]
                }
                optimizers_str = ', '.join(
                    optimizer.__class__.__name__
                    for optimizer in server_optimizers
                )
                logger.get_logger().info(
                    f'Processing blocks: {optimizers_str} on server'
                )

                ret, _ = check_request(
                    self.protocol.upload_optimizers(optimizers_cfg),
                    'upload optimizers config'
                )
                if not ret:
                    raise RuntimeError('Optimizers config upload failed')

                ret, compiled_model = check_request(
                    self.protocol.request_optimization(model_path),
                    'request optimization'
                )
                if not ret or compiled_model is None:
                    raise RuntimeError('Model compilation failed')

                prev_block = server_optimizers[-1]
                with open(prev_block.compiled_model_path, 'wb') as model_f:
                    model_f.write(compiled_model)

            else:
                if 'target' == next_block.location:
                    logger.get_logger().warning(
                        'Ignoring target location parameter for '
                        f'{type(next_block).__name__} as the protocol is not '
                        'provided'
                    )
                logger.get_logger().info(
                    f'Processing block: {type(next_block).__name__} on client'
                )

                next_block.set_input_type(model_type)
                if hasattr(prev_block, 'get_io_specification'):
                    next_block.compile(
                        model_path,
                        prev_block.get_io_specification()
                    )
                else:
                    next_block.compile(model_path)

                prev_block = next_block
                optimizer_idx += 1

            model_path = prev_block.compiled_model_path

        if self.optimizers:
            self.optimizers[-1].save_io_specification(model_path)

        logger.get_logger().info(f'Compiled model path: {model_path}')
        return model_path

    def _run_client(self, compiled_model_path: Optional[Path]) -> bool:
        """
        Main runtime client program.

        The client performance procedure is as follows:

        * connect with the server
        * upload the model
        * send dataset data in a loop to the server:

            * upload input
            * request processing of inputs
            * request predictions for inputs
            * evaluate the response
        * collect performance statistics
        * end connection

        Parameters
        ----------
        compiled_model_path : Optional[Path]
            Path to the model that should be tested.

        Returns
        -------
        bool :
            True if executed successfully.
        """

        if self.protocol is None:
            raise RequestFailure('Protocol is not provided')
        try:
            check_request(
                self.upload_essentials(compiled_model_path),
                'upload essentials',
            )
            measurements = Measurements()
            for X, y in tqdm(self.dataset.iter_test()):
                prepX = tagmeasurements("preprocessing")(self.dataconverter.to_next_block)(X)   # noqa: E501
                check_request(self.protocol.upload_input(prepX), 'send input')
                check_request(
                    self.protocol.request_processing(self.runtime.get_time),
                    'inference',
                )
                _, preds = check_request(
                    self.protocol.download_output(), 'receive output'
                )
                logger.get_logger().debug('Received output')
                posty = tagmeasurements("postprocessing")(self.dataconverter.to_previous_block)(preds)  # noqa: E501
                measurements += self.dataset.evaluate(posty, y)

            measurements += self.protocol.download_statistics()
        except RequestFailure as ex:
            logger.get_logger().fatal(ex)
            return False
        else:
            MeasurementsCollector.measurements += measurements

        self.protocol.disconnect()
        return True

    @systemstatsmeasurements('full_run_statistics')
    def _run_locally(self, compiled_model_path: Path) -> bool:
        """
        Runs inference locally.

        Parameters
        ----------
        compiled_model_path : Path
            Path to the model that should be tested.

        Returns
        -------
        bool :
            True if executed successfully.
        """
        self.model_path = compiled_model_path

        measurements = Measurements()
        try:
            self.runtime.inference_session_start()
            assert self.runtime.prepare_local(), "Cannot prepare local environment"  # noqa: E501
            for X, y in logger.TqdmCallback(
                'runtime', self.dataset.iter_test()
            ):
                prepX = tagmeasurements("preprocessing")(self.dataconverter.to_next_block)(X)   # noqa: E501
                succeed = self.runtime.prepare_input(prepX)
                if not succeed:
                    return False
                self.runtime._run()
                preds = self.runtime.extract_output()
                posty = tagmeasurements("postprocessing")(self.model_wrapper._postprocess_outputs)(preds)   # noqa: E501
                measurements += self.dataset.evaluate(posty, y)
        except KeyboardInterrupt:
            logger.get_logger().info("Stopping benchmark...")
            return False
        finally:
            self.runtime.inference_session_end()
            MeasurementsCollector.measurements += measurements

        return True

    @staticmethod
    def assert_io_formats(
            model_wrapper: Optional[ModelWrapper],
            optimizers: Union[List[Optimizer], Optimizer],
            runtime: Optional[Runtime]):
        """
        Asserts that given blocks can be put together in a pipeline.

        Parameters
        ----------
        model_wrapper : Optional[ModelWrapper]
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

        chain = [model_wrapper] + optimizers + [runtime]
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
                f'No matching formats between two objects: {previous_block} '
                f'and {next_block}\n'
                f'Output block supported formats: {output_formats_str}\n'
                f'Input block supported formats: {input_formats_str}'
            )
