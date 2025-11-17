# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides runner for optimization flows.
"""

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from kenning.cli.command_template import OPTIMIZE, TEST, CommandTemplate
from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    CompilationError,
    KenningOptimizerError,
    ModelTooLargeError,
    NotSupportedError,
)
from kenning.core.measurements import (
    Measurements,
    MeasurementsCollector,
    tagmeasurements,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import OptimizedModelSizeError, Optimizer
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol, check_request
from kenning.core.report import Report
from kenning.core.runtime import Runtime
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.platforms.local import LocalPlatform
from kenning.runtimes.utils import get_default_runtime
from kenning.utils.class_loader import ConfigKey, objs_from_json
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class PipelineRunner(object):
    """
    Class responsible for running model optimization pipelines.
    """

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        dataconverter: Optional[DataConverter] = None,
        optimizers: List[Optimizer] = [],
        platform: Optional[Platform] = None,
        protocol: Optional[Protocol] = None,
        model_wrapper: Optional[ModelWrapper] = None,
        runtime: Optional[Runtime] = None,
        runtime_builder: Optional[RuntimeBuilder] = None,
        configuration_path: Optional[Path] = None,
        report: Optional[Report] = None,
    ):
        """
        Initializes the PipelineRunner object.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Dataset object that provides data for inference.
        dataconverter : Optional[DataConverter]
            DataConverter object that converts data from/to Protocol format.
        optimizers : List[Optimizer]
            List of Optimizer objects that optimize the model.
        platform : Optional[Platform]
            Platform on which model will be evaluated.
        protocol : Optional[Protocol]
            Protocol object that provides the communication protocol.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper object that wraps the model.
        runtime : Optional[Runtime]
            Runtime object that runs the inference.
        runtime_builder : Optional[RuntimeBuilder]
            RuntimeBuilder object that builds the runtime.
        configuration_path : Optional[Path]
            Path to the file containing configuration.
        report: Optional[Report]
            Report used to gather some information like measurements.

        Raises
        ------
        ValueError
            Raised when invalid values are provided.
        """
        self.dataset = dataset
        self.dataconverter = dataconverter
        self.optimizers = optimizers
        self.platform = platform
        self.protocol = protocol
        self.model_wrapper = model_wrapper
        self.runtime = runtime
        self.runtime_builder = runtime_builder
        self.should_cancel = False
        self.configuration_path = configuration_path

        self.output = None

        self.report = report

        if report is not None:
            self.output = report.measurements[0]

        # resolve defaults
        if (
            self.dataset is None
            and self.model_wrapper is not None
            and self.model_wrapper.default_dataset is not None
        ):
            dataset_cls = self.model_wrapper.default_dataset
            self.dataset = dataset_cls(
                root=Path(f"./build/{type(dataset_cls).__name__}")
            )
            KLogger.info(f"Set dataset to {self.dataset}")

        if self.platform is None:
            self.platform = LocalPlatform()
            KLogger.info("Set platform to Local")

        if self.protocol is None and self.platform.needs_protocol:
            self.protocol = self.platform.get_default_protocol()
            KLogger.info(f"Set protocol to {self.protocol}")

        if self.protocol is not None and self.dataconverter is None:
            io_specification = None

            if self.model_wrapper:
                io_specification = self.model_wrapper.get_io_specification()
            elif self.runtime and hasattr(self.runtime, "model_path"):
                io_specification = self.runtime.get_io_spec_path(
                    self.runtime.model_path
                )

            self.dataconverter = (
                self.protocol.deduce_data_converter_from_io_spec(
                    io_specification
                )
            )

        if not (self.model_wrapper or self.dataconverter):
            raise ValueError("Provide either dataconverter or model_wrapper")

        if self.dataconverter is None:
            self.dataconverter = ModelWrapperDataConverter(self.model_wrapper)

        assert (
            self.model_wrapper or self.dataconverter
        ), "Provide either dataconverter or model_wrapper."

        if self.model_wrapper:
            self.model_wrapper.read_platform(self.platform)

        if self.runtime_builder:
            self.runtime_builder.read_platform(self.platform)

        for optim in optimizers:
            optim.read_platform(self.platform)

    @classmethod
    def from_json_cfg(
        cls,
        json_cfg: Dict,
        assert_integrity: bool = True,
        skip_optimizers: bool = False,
        skip_runtime: bool = False,
        cfg_path: Optional[Path] = None,
        override: Optional[Tuple[argparse.Namespace, List[str]]] = None,
        include_measurements: bool = False,
    ):
        keys = [
            ConfigKey.dataset,
            ConfigKey.platform,
            ConfigKey.protocol,
            ConfigKey.model_wrapper,
            ConfigKey.dataconverter,
            ConfigKey.runtime_builder,
            *([ConfigKey.report] if include_measurements else []),
            *([ConfigKey.runtime] if not skip_runtime else []),
            *([ConfigKey.optimizers] if not skip_optimizers else []),
        ]

        objs = objs_from_json(json_cfg, set(keys), override)

        if assert_integrity:
            cls.assert_io_formats(
                objs.get(ConfigKey.model_wrapper),
                objs[ConfigKey.optimizers],
                objs.get(ConfigKey.runtime),
            )

        return cls.from_objs_dict(objs, configuration_path=cfg_path)

    @classmethod
    def from_objs_dict(cls, objs: Dict[ConfigKey, Any], **kwargs):
        return cls(
            **{key.name: value for key, value in objs.items()}, **kwargs
        )

    def serialize(
        self,
    ) -> Dict:
        """
        Serializes the given objects into a dictionary.

        Returns
        -------
        Dict
            Serialized configuration.
        """
        serialized_dict = {}

        for obj_name in [
            "dataset",
            "dataconverter",
            "optimizers",
            "platform",
            "protocol",
            "model_wrapper",
            "runtime",
            "runtime_builder",
        ]:
            obj = getattr(self, obj_name)
            if obj is None:
                continue

            if isinstance(obj, list):
                serialized_dict[obj_name] = [
                    obj_elem.to_json() for obj_elem in obj
                ]
            else:
                serialized_dict[obj_name] = obj.to_json()

        return serialized_dict

    def add_scenario_configuration_to_measurements(
        self, command: List, model_path: Optional[PathOrURI] = None
    ):
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
            "optimizers": [
                dict(
                    zip(
                        ("compiler_framework", "compiler_version"),
                        optimizer.get_framework_and_version(),
                    )
                )
                for optimizer in self.optimizers
            ],
            "command": command,
            "build_cfg": self.serialize(),
        }

        if self.configuration_path:
            MeasurementsCollector.measurements += {
                "cfg_path": str(self.configuration_path),
            }

        if self.model_wrapper:
            framework, version = self.model_wrapper.get_framework_and_version()
            MeasurementsCollector.measurements += {
                "model_framework": framework,
                "model_version": version,
            }

        # TODO: add method for providing metadata to dataset
        if hasattr(self.dataset, "classnames"):
            MeasurementsCollector.measurements += {
                "class_names": self.dataset.get_class_names()
            }

        model_size = None
        if self.optimizers:
            try:
                model_size = self.optimizers[-1].get_optimized_model_size()
                model_size *= 1024  # Convert to bytes
            except OptimizedModelSizeError as e:
                KLogger.warning(f"Cannot retrieve optimized model size: {e}")
        if not model_size and model_path:
            model_size = Path(model_path).stat().st_size

        if model_size:
            MeasurementsCollector.measurements += {
                "compiled_model_size": model_size
            }

    def run(
        self,
        output: Optional[Path] = None,
        verbosity: str = "INFO",
        convert_to_onnx: Optional[Path] = None,
        max_target_side_optimizers: int = -1,
        command: List = [],
        run_optimizations: bool = True,
        run_benchmarks: bool = True,
    ) -> int:
        """
        Wrapper function that runs a pipeline using given parameters.

        Parameters
        ----------
        output : Optional[Path]
            Path to the output JSON file with measurements.
        verbosity : str
            Verbosity level.
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use
            in the inference (provide a path to save here).
        max_target_side_optimizers : int
            Max number of consecutive target-side optimizers.
        command : List
            Command used to run this inference pipeline. It is put in
            the output JSON file.
        run_optimizations : bool
            If False, optimizations will not be executed.
        run_benchmarks : bool
            If False, model will not be tested.

        Returns
        -------
        int
            The 0 value if the inference was successful, 1 otherwise.

        Raises
        ------
        ValueError
            Raised when invalid values are provided.
        """
        if output is None:
            output = self.output

        if not (run_optimizations or run_benchmarks):
            raise ValueError(
                "If both optimizations and benchmarks are skipped, pipeline "
                "will not be executed"
            )
        KLogger.set_verbosity(verbosity)

        MeasurementsCollector.clear()

        model_framework = self._guess_model_framework(convert_to_onnx)

        CommandTemplate.current_command = OPTIMIZE

        # initialize protocol if needed
        protocol_required_by_optimizers = run_optimizations and any(
            optimizer.location == "target" for optimizer in self.optimizers
        )
        protocol_required = (
            run_benchmarks
            and self.platform is not None
            and self.platform.needs_protocol
        ) or protocol_required_by_optimizers

        if protocol_required and self.protocol is None:
            msg = "Protocol not specified but required"
            KLogger.error(msg)
            raise ValueError(msg)

        if protocol_required_by_optimizers and type(
            self.platform
        ).__name__ in ("BareMetalPlatform", "ZephyrPlatform"):
            msg = (
                f"Invalid configucation, {type(self.platform).__name__} "
                "platform cannot run optimization"
            )
            KLogger.error(msg)
            raise ValueError(msg)

        if protocol_required_by_optimizers:
            check_request(self.protocol.initialize_client(), "prepare client")
            try:
                self.protocol.listen_to_server_logs()
            except NotSupportedError:
                KLogger.warning(
                    "Server logs not available for this protocol:"
                    f" {type(self.protocol)}."
                )

        # handle model optimizations
        model_path = self._handle_optimizations(
            convert_to_onnx, run_optimizations
        )

        # update model io spec after all optimizations (if any)
        if self.optimizers:
            final_io_spec = self.optimizers[-1].load_io_specification(
                model_path
            )
            if final_io_spec is not None:
                self.model_wrapper.io_specification = final_io_spec

        # handle runtime builder
        self._handle_runtime_builder(model_framework, model_path)

        # deduce runtime if not specified
        if (
            run_benchmarks
            and self.runtime is None
            and (
                protocol_required
                or (run_optimizations and len(self.optimizers) > 0)
            )
        ):
            self.runtime = get_default_runtime(model_framework, model_path)
            KLogger.info(f"Deduced runtime {self.runtime}")

        if run_benchmarks:
            CommandTemplate.current_command = TEST

            if self.runtime is None and not protocol_required:
                self.model_wrapper.test_inference()
            else:
                measurements = None
                # handle platform init
                self._handle_platform_init()

                measurements = Measurements()

                try:
                    if (
                        protocol_required
                        and not protocol_required_by_optimizers
                    ):
                        check_request(
                            self.protocol.initialize_client(), "prepare client"
                        )
                        try:
                            self.protocol.listen_to_server_logs()
                        except NotSupportedError:
                            KLogger.warning(
                                "Server logs not available for this protocol:"
                                f" {type(self.protocol)}."
                            )

                    # Handle LLEXT upload
                    self._handle_runtime_upload()

                    self._inference_loop(
                        measurements, model_path, remote=protocol_required
                    )
                except Exception:
                    raise
                finally:
                    if measurements is None:
                        measurements = Measurements()
                    # handle platform deinit
                    self._handle_platform_deinit(measurements)

                MeasurementsCollector.measurements += measurements

        if output:
            self.add_scenario_configuration_to_measurements(
                command, model_path
            )
            MeasurementsCollector.save_measurements(output)

        return 0

    def _handle_runtime_builder(
        self,
        model_framework: Optional[str],
        model_path: Optional[PathOrURI],
    ):
        """
        Handles RuntimeBuilder.

        Parameters
        ----------
        model_framework : Optional[str]
            Framework of the model.
        model_path : Optional[PathOrURI]
            Path to the model.
        """
        if self.runtime_builder is not None:
            if model_framework is not None:
                self.runtime_builder.set_input_framework(model_framework)
            if model_path is not None:
                self.runtime_builder.set_model_path(model_path)
            self.runtime_builder.build()

    def _handle_runtime_upload(self):
        """
        Handles runtime upload i.e. LLEXT.
        """
        llext_path = None
        if (
            self.platform is not None
            and getattr(self.platform, "llext_binary_path", None) is not None
        ):
            # use LLEXT from platform params
            llext_path = self.platform.llext_binary_path
        elif (
            self.runtime_builder is not None and self.runtime_builder.use_llext
        ):
            # use LLEXT built by runtime builder
            llext_path = self.runtime_builder.output_path / "runtime.llext"

        if llext_path is not None and llext_path.exists():
            check_request(
                self.protocol.upload_runtime(llext_path),
                "upload runtime",
            )
        elif llext_path:
            raise FileNotFoundError(
                f"LLEXT binary file does not exist:  {llext_path}"
            )

    def _handle_platform_init(self):
        """
        Handles platform initialization.
        """
        if self.platform is not None:
            self.platform.init()

    def _handle_platform_deinit(self, measurements: Measurements):
        """
        Handles platform deinitialization.

        Parameters
        ----------
        measurements : Measurements
            Measurements to which platform metrics will be added to.
        """
        if self.platform is not None:
            self.platform.deinit(measurements)

    def _handle_optimizations(
        self,
        convert_to_onnx: Optional[Path] = None,
        run_optimization: bool = True,
        run_compatibility_checks: bool = False,
        max_target_side_optimizers: int = -1,
    ) -> Optional[Path]:
        """
        Handles model optimization.

        Parameters
        ----------
        convert_to_onnx : Optional[Path]
            Before compiling the model, convert it to ONNX and use in the
            inference (provide a path to save here).
        run_optimization : bool
            Determines if optimizations should be executed, otherwise last
            compiled model path is returned.
        run_compatibility_checks: bool
            Declares if compatibility checks between optimizers should be run.
            If self.runtime is not defined will deduce runtime
            based on the optimizer.
            Note: will run only on host
        max_target_side_optimizers : int
            Max number of consecutive target-side optimizers.

        Returns
        -------
        Optional[Path]
            Path to compiled model.
            If run_compatibility_checks is set and checks fail,
            will return None.

        Raises
        ------
        ValueError :
            Model wrapper is missing.
        KenningOptimizerError :
            Failed to upload optimizer configuration.
        CompilationError :
            Model Compilation failed.
        """
        if not run_optimization:
            if self.optimizers:
                return self.optimizers[-1].compiled_model_path
            elif self.model_wrapper:
                model_path = self.model_wrapper.get_path()
                self.model_wrapper.save_io_specification(model_path)
                return model_path
            elif self.runtime is not None and hasattr(
                self.runtime, "model_path"
            ):
                return self.runtime.model_path
            else:
                return None

        if self.model_wrapper is None:
            raise ValueError("Model wrapper is required for optimizations")
        model_path = self.model_wrapper.get_path()
        prev_block = self.model_wrapper
        if convert_to_onnx:
            KLogger.warning(
                "Force conversion of the input model to the ONNX format"
            )
            model_path = convert_to_onnx
            prev_block.save_to_onnx(model_path)

        optimizer_idx = 0
        while optimizer_idx < len(self.optimizers):
            next_block = self.optimizers[optimizer_idx]

            model_type = next_block.consult_model_type(
                prev_block,
                force_onnx=(
                    convert_to_onnx is not None
                    and isinstance(prev_block, ModelWrapper)
                ),
            )

            if (
                model_type == "onnx"
                and isinstance(prev_block, ModelWrapper)
                and not convert_to_onnx
            ):
                model_path = Path(tempfile.NamedTemporaryFile().name)
                prev_block.save_to_onnx(model_path)

            prev_block.save_io_specification(model_path)

            if "target" == next_block.location and self.protocol is not None:
                server_optimizers = []
                while (
                    optimizer_idx < len(self.optimizers)
                    and "target" == self.optimizers[optimizer_idx].location
                    and (
                        len(server_optimizers) < max_target_side_optimizers
                        or max_target_side_optimizers == -1
                    )
                ):
                    server_optimizers.append(self.optimizers[optimizer_idx])
                    optimizer_idx += 1

                optimizers_cfg = {
                    "prev_block": {
                        "model_path": model_path,
                        "model_type": model_type,
                        "io_spec": prev_block.load_io_specification(
                            model_path
                        ),
                    },
                    "optimizers": [
                        optimizer.to_json() for optimizer in server_optimizers
                    ],
                }
                optimizers_str = ", ".join(
                    optimizer.__class__.__name__
                    for optimizer in server_optimizers
                )
                KLogger.info(f"Processing blocks: {optimizers_str} on server")

                ret, _ = check_request(
                    self.protocol.upload_optimizers(optimizers_cfg),
                    "upload optimizers config",
                )
                if not ret:
                    raise KenningOptimizerError(
                        "Optimizers config upload failed"
                    )

                ret, compiled_model = check_request(
                    self.protocol.request_optimization(model_path),
                    "request optimization",
                )
                if not ret or compiled_model is None:
                    raise CompilationError("Model compilation failed")

                prev_block = server_optimizers[-1]
                with open(prev_block.compiled_model_path, "wb") as model_f:
                    model_f.write(compiled_model)

            else:
                if "target" == next_block.location:
                    KLogger.warning(
                        "Ignoring target location parameter for "
                        f"{type(next_block).__name__} as the protocol is not "
                        "provided"
                    )
                KLogger.info(
                    f"Processing block: {type(next_block).__name__} on client"
                )

                next_block.set_input_type(model_type)
                next_block.init()
                if run_compatibility_checks:
                    runtime = self.runtime
                    if optimizer_idx == (len(self.optimizers) - 1):
                        runtime = None
                    elif runtime is None:
                        model_framework = self._guess_model_framework(
                            convert_to_onnx
                        )
                        runtime = get_default_runtime(
                            model_framework, model_path
                        )
                    success_check = next_block.run_compatibility_checks(
                        self.platform, runtime, model_path
                    )
                    if not success_check:
                        return None

                if hasattr(prev_block, "get_io_specification"):
                    next_block.compile(
                        model_path, prev_block.get_io_specification()
                    )
                else:
                    next_block.compile(model_path)

                prev_block = next_block
                optimizer_idx += 1

            model_path = prev_block.compiled_model_path

        if self.optimizers:
            self.optimizers[-1].save_io_specification(model_path)

        KLogger.info(f"Compiled model path: {model_path}")
        return model_path

    def _inference_loop(
        self,
        measurements: Measurements,
        model_path: Path,
        remote: bool = False,
    ):
        """
        Executes inference loop.

        Parameters
        ----------
        measurements : Measurements
            Measurements to which metrics will be saved to.
        model_path : Path
            Path to the model used for inference.
        remote : bool
            True if the inference is performed on remote platform.
        """
        KLogger.info("Starting inference loop")

        try:
            if remote:
                self._remote_inference_prepare(model_path)
            else:
                self._local_inference_prepare()

            use_platform_sensor = (
                getattr(self.platform, "sensor", None) is not None
            )

            # prepare iterator for inference
            iterable = (
                range(self.platform.number_of_batches)
                if use_platform_sensor
                else self.dataset.iter_test()
            )

            with LoggerProgressBar() as logger_progress_bar:
                for sample in tqdm(iterable, **logger_progress_bar.kwargs):
                    if self.should_cancel:
                        break

                    if use_platform_sensor:
                        preds = self._sensor_inference_step()
                        y = None
                    else:
                        X, y = sample

                        prepX = tagmeasurements("preprocessing")(
                            self.dataconverter.to_next_block
                        )(X)

                        if remote:
                            preds = self._remote_inference_step(prepX)
                            measurements += self.protocol.download_statistics(
                                final=False
                            )
                        else:
                            preds = self._local_inference_step(prepX)

                    posty = tagmeasurements("postprocessing")(
                        self.dataconverter.to_previous_block
                    )(preds)

                    measurements += self.dataset._evaluate(posty, y)

                    self.platform.inference_step_callback()

        except KeyboardInterrupt:
            KLogger.info("Stopping inference...")
        else:
            if remote:
                measurements += self.protocol.download_statistics(final=True)
        finally:
            if remote:
                self._remote_inference_cleanup()
            else:
                self._local_inference_cleanup()

    def _remote_inference_prepare(self, model_path: Path):
        """
        Uploads model to the remote platform.

        Parameters
        ----------
        model_path : Path
            Path to the model.

        Raises
        ------
        FileNotFoundError
            Raised when IO specification is not found.
        """
        compiled_model_path = None

        if self.runtime is not None:
            spec_path = self.runtime.get_io_spec_path(model_path)
            if not spec_path.exists():
                KLogger.error("No Input/Output specification found")
                raise FileNotFoundError("IO specification not found")
            if (ram_kb := getattr(self.platform, "ram_size_kb", None)) and (
                (model_kb := model_path.stat().st_size // 1024) > ram_kb
            ):
                KLogger.error(
                    f"Model ({model_kb}KB) does not fit "
                    f"into board's RAM ({ram_kb}KB)"
                )
                raise ModelTooLargeError(
                    f"Model too large ({model_kb}KB > {ram_kb}KB)"
                )

            check_request(
                self.protocol.upload_io_specification(spec_path),
                "upload io spec",
            )

            compiled_model_path = (
                self.runtime.preprocess_model_to_upload(model_path)
                if model_path is not None
                else None
            )

        check_request(
            self.protocol.upload_model(compiled_model_path), "upload model"
        )

    def _remote_inference_cleanup(self):
        """
        Cleanups remote inference.
        """
        self.protocol.disconnect()

    def _remote_inference_step(self, X: any) -> any:
        """
        Performs inference step on remote platform.

        Parameters
        ----------
        X : any
            Inference input data.

        Returns
        -------
        any
            Inference output data.
        """
        if self.model_wrapper is not None:
            prepX = self.model_wrapper.convert_input_to_bytes(X)
        else:
            prepX = X
        check_request(self.protocol.upload_input(prepX), "send input")
        check_request(
            self.protocol.request_processing(self.platform.get_time),
            "inference",
        )
        _, preds = check_request(
            self.protocol.download_output(), "receive output"
        )
        KLogger.debug("Received output")
        if self.model_wrapper is not None:
            preds = self.model_wrapper.convert_output_from_bytes(preds)

        return preds

    def _local_inference_prepare(self):
        """
        Starts local inference sessions.
        """
        if self.runtime is not None:
            self.runtime.inference_session_start()
            assert (
                self.runtime.prepare_local()
            ), "Cannot prepare local environment"

    def _local_inference_cleanup(self):
        """
        Stops local inference sessions.
        """
        if self.runtime is not None:
            self.runtime.inference_session_end()

    def _local_inference_step(self, X: any) -> any:
        """
        Performs inference step on local platform.

        Parameters
        ----------
        X : any
            Inference input data.

        Returns
        -------
        any
            Inference output data.
        """
        succeed = self.runtime.load_input(X)
        if not succeed:
            return None
        self.runtime._run()
        preds = self.runtime.extract_output()

        return preds

    def _sensor_inference_step(self) -> any:
        """
        Performs inference step using data read from sensor.

        Returns
        -------
        any
            Inference output data.
        """
        check_request(
            self.protocol.request_processing(self.platform.get_time),
            "inference",
        )
        _, preds = check_request(
            self.protocol.download_output(), "receive output"
        )
        KLogger.debug("Received output")
        if self.model_wrapper is not None:
            preds = self.model_wrapper.convert_output_from_bytes(preds)

        return preds

    @staticmethod
    def assert_io_formats(
        model_wrapper: Optional[ModelWrapper],
        optimizers: List[Optimizer],
        runtime: Optional[Runtime],
    ):
        """
        Asserts that given blocks can be put together in a pipeline.

        Parameters
        ----------
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper of the pipeline.
        optimizers : List[Optimizer]
            Optimizers of the pipeline.
        runtime : Optional[Runtime]
            Runtime of the pipeline.

        Raises
        ------
        ValueError :
            Raised if blocks are incompatible.
        """
        chain = [model_wrapper] + optimizers + [runtime]
        chain = [block for block in chain if block is not None]

        for previous_block, next_block in zip(chain, chain[1:]):
            check_model_type = getattr(next_block, "consult_model_type", None)
            if callable(check_model_type):
                check_model_type(previous_block)
                continue
            elif set(next_block.get_input_formats()) & set(
                previous_block.get_output_formats()
            ):
                continue

            if next_block == runtime:
                KLogger.warning(
                    f"Runtime {next_block} has no matching format with the "
                    f"previous block: {previous_block}\nModel may not run "
                    "correctly"
                )
                continue

            output_formats_str = ", ".join(previous_block.get_output_formats())
            input_formats_str = ", ".join(next_block.get_input_formats())
            raise ValueError(
                f"No matching formats between two objects: {previous_block} "
                f"and {next_block}\n"
                f"Output block supported formats: {output_formats_str}\n"
                f"Input block supported formats: {input_formats_str}"
            )

    def _guess_model_framework(self, convert_to_onnx: bool) -> Optional[str]:
        """
        Retrieves model framework from ModelWrapper and Optimizers.

        Parameters
        ----------
        convert_to_onnx : bool
            Whether model should be converted to ONNX.

        Returns
        -------
        Optional[str]
            Model framework or None if no model specified.
        """

        def first_not_onnx(x):
            for fmt in x.get_output_formats():
                if fmt != "onnx":
                    return fmt

        if self.model_wrapper is None:
            return None

        if len(self.optimizers) == 0:
            if convert_to_onnx:
                return "onnx"

            return first_not_onnx(self.model_wrapper)

        return first_not_onnx(self.optimizers[-1])
