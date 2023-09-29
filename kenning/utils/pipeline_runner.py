# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with pipelines running helper functions.
"""

from pathlib import Path
import tempfile

from typing import Optional, Dict, List, Union

from tqdm import tqdm
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.runtime import Runtime
from kenning.core.protocol import RequestFailure, Protocol, check_request
try:
    from kenning.runtimes.renode import RenodeRuntime
except ImportError:
    RenodeRuntime = None

from kenning.utils.class_loader import any_from_json
from kenning.utils.args_manager import serialize_inference
import kenning.utils.logger as logger
from kenning.core.measurements import (
    Measurements,
    MeasurementsCollector,
    systemstatsmeasurements,
    tagmeasurements
)
from kenning.utils.resource_manager import PathOrURI

UNOPTIMIZED_MEASUREMENTS = '__unoptimized__'


class PipelineRunner(object):
    def __init__(
        self,
        dataset: Dataset,
        model_wrapper: ModelWrapper,
        optimizers: List[Optimizer],
        runtime: Runtime,
        protocol: Optional[Protocol] = None,
    ):
        self.dataset = dataset
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
        ValueError :
            Raised if blocks are connected incorrectly.
        jsonschema.exceptions.ValidationError :
            Raised if parameters are incorrect.
        """
        assert 'model_wrapper' in json_cfg, 'ModelWrapper not provided'
        if 'runtime' not in json_cfg:
            skip_runtime = True

        dataset = (
            any_from_json(json_cfg['dataset'])
            if 'dataset' in json_cfg else None
        )
        model_wrapper = (
            any_from_json(json_cfg['model_wrapper'], dataset=dataset)
        )
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
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime,
            protocol=protocol
        )

    def run(
        self,
        output: Optional[Path] = None,
        verbosity: str = 'INFO',
        convert_to_onnx: Optional[Path] = None,
        command: List = ['Run in a different environment'],
        run_optimizations: bool = True,
        run_benchmarks: bool = True,
        model_name: Optional[str] = None,
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
        command : Optional[List]
            Command used to run this inference pipeline. It is put in
            the output JSON file.
        run_optimizations : bool
            If False, optimizations will not be executed.
        run_benchmarks : bool
            If False, model will not be tested.
        model_name : Optional[str]
            Custom name of the model.

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
            'If both optimizations and benchmarks are skipped, pipeline will '
            'not be executed'
        )
        logger.set_verbosity(verbosity)
        log = logger.get_logger()

        self.assert_io_formats(
            self.model_wrapper,
            self.optimizers,
            self.runtime
        )

        model_framework_tuple = self.model_wrapper.get_framework_and_version()

        if run_benchmarks and not output:
            log.warning(
                'Running benchmarks without defined output -- measurements '
                'will not be saved'
            )

        if output:
            MeasurementsCollector.measurements += {
                'model_framework': model_framework_tuple[0],
                'model_version': model_framework_tuple[1],
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
                    self.model_wrapper,
                    self.optimizers,
                    self.protocol,
                    self.runtime
                ),
            }
            if model_name is not None:
                MeasurementsCollector.measurements += {
                    'model_name': model_name
                }

            # TODO add method for providing metadata to dataset
            if hasattr(self.dataset, 'classnames'):
                MeasurementsCollector.measurements += {
                    'class_names': self.dataset.get_class_names()
                }

        if len(self.optimizers) > 0:
            model_path = self.optimizers[-1].compiled_model_path
        else:
            model_path = self.model_wrapper.get_path()

        ret = True
        if RenodeRuntime is not None and isinstance(self.runtime, RenodeRuntime):  # noqa: E501
            compiled_model_path = self.handle_optimizations(convert_to_onnx)
            self.runtime.run_client(
                dataset=self.dataset,
                modelwrapper=self.model_wrapper,
                protocol=self.protocol,
                compiled_model_path=compiled_model_path
            )
        elif run_benchmarks and self.runtime:
            if not self.dataset:
                log.error(
                    'The benchmarks cannot be performed without a dataset'
                )
                ret = False
            elif self.protocol:
                ret = self._run_client(convert_to_onnx)
            else:
                ret = self._run_locally(convert_to_onnx)
        elif run_benchmarks:
            self.model_wrapper.model_path = model_path
            self.model_wrapper.test_inference()
            ret = True

        if not ret:
            return 1

        if output:
            model_path = Path(model_path)
            # If model compressed in ZIP exists use its size
            # It is more accurate for Keras models
            if model_path.with_suffix(model_path.suffix + '.zip').exists():
                model_path = model_path.with_suffix(model_path.suffix + '.zip')

            MeasurementsCollector.measurements += {
                'compiled_model_size': model_path.stat().st_size
            }

            MeasurementsCollector.save_measurements(output)
        return 0

    def upload_essentials(self, compiled_model_path: PathOrURI) -> bool:
        """
        Wrapper for uploading data to the server.
        Uploads model by default.

        Parameters
        ----------
        compiled_model_path : PathOrURI
            Path or URI to the file with a compiled model.

        Returns
        -------
        bool :
            True if succeeded.
        """
        spec_path = self.runtime.get_io_spec_path(compiled_model_path)
        if spec_path.exists():
            self.protocol.upload_io_specification(spec_path)
        else:
            logger.get_logger().info('No Input/Output specification found')
        return self.protocol.upload_model(compiled_model_path)

    def handle_optimizations(
        self,
        convert_to_onnx: Optional[Path] = None
    ) -> Path:
        """
        Handle model optimization.

        Parameters
        ----------
        dataset : Dataset
            Dataset to be used by optimizers.
        model_wrapper : ModelWrapper
            Model for optimizations.
        optimizers : List[Optimizer]
            List of optimizer for model optimization.
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use in the
            inference (provide a path to save here).

        Returns
        -------
        Path :
            Path to compiled model.

        Raises
        ------
        RuntimeError :
            When any of the server request fails.
        """
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

            logger.get_logger().info(
                f'Processing block: {type(next_block).__name__}'
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

        if not self.optimizers:
            self.model_wrapper.save_io_specification(model_path)

        logger.get_logger().info(f'Compiled model path: {model_path}')
        return model_path

    def _run_client(self, convert_to_onnx: Optional[Path] = None) -> bool:
        """
        Main runtime client program.

        The client performance procedure is as follows:

        * connect with the server
        * run model optimizations
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
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use in the
            inference (provide a path to save here).

        Returns
        -------
        bool :
            True if executed successfully.
        """

        check_request(self.protocol.initialize_client(), 'prepare client')

        compiled_model_path = self.handle_optimizations(convert_to_onnx)

        if self.protocol is None:
            raise RequestFailure('Protocol is not provided')
        try:
            check_request(
                self.upload_essentials(compiled_model_path),
                'upload essentials',
            )
            measurements = Measurements()
            for X, y in tqdm(self.dataset.iter_test()):
                prepX = tagmeasurements("preprocessing")(
                    self.model_wrapper._preprocess_input
                )(
                    X
                )  # noqa: E501
                prepX = self.model_wrapper.convert_input_to_bytes(prepX)
                check_request(self.protocol.upload_input(prepX), 'send input')
                check_request(
                    self.protocol.request_processing(self.runtime.get_time),
                    'inference',
                )
                _, preds = check_request(
                    self.protocol.download_output(), 'receive output'
                )
                logger.get_logger().debug(
                    f'Received output ({len(preds)} bytes)'
                )
                preds = self.model_wrapper.convert_output_from_bytes(preds)
                posty = tagmeasurements("postprocessing")(
                    self.model_wrapper._postprocess_outputs
                )(
                    preds
                )  # noqa: E501
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
    def _run_locally(self, convert_to_onnx: Optional[Path] = None) -> bool:
        """
        Runs inference locally.

        Parameters
        ----------
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use in the
            inference (provide a path to save here).

        Returns
        -------
        bool :
            True if executed successfully.
        """
        self.model_path = self.handle_optimizations(convert_to_onnx)

        measurements = Measurements()
        try:
            self.runtime.inference_session_start()
            self.runtime.prepare_local()
            for X, y in logger.TqdmCallback(
                'runtime', self.dataset.iter_test()
            ):
                prepX = tagmeasurements("preprocessing")(
                    self.model_wrapper._preprocess_input
                )(X)
                prepX = self.model_wrapper.convert_input_to_bytes(prepX)
                succeed = self.runtime.prepare_input(prepX)
                if not succeed:
                    return False
                self.runtime._run()
                preds = self.runtime.extract_output()
                posty = tagmeasurements("postprocessing")(
                    self.model_wrapper._postprocess_outputs
                )(preds)
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
