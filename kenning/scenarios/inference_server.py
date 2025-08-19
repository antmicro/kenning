#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs inference server.

It requires implementations of several classes as input:

* Protocol - provides routines for communicating with the client
* Runtime - provides implementation of inference runtime

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.completers import (
    PLATFORMS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
    ClassPathCompleter,
)
from kenning.core.exceptions import ModelNotPreparedError, NotSupportedError
from kenning.core.optimizer import Optimizer
from kenning.core.protocol import Protocol, ServerAction, ServerStatus
from kenning.core.runtime import Runtime
from kenning.platforms.local import LocalPlatform
from kenning.utils.args_manager import (
    report_missing,
    to_namespace_name,
)
from kenning.utils.class_loader import (
    ConfigKey,
    objs_from_argparse,
    objs_from_json,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI

FILE_CONFIG = "Server configuration with JSON/YAML file"
FLAG_CONFIG = "Server configuration with flags"
ARGS_GROUPS = {
    FILE_CONFIG: (
        "Configuration with data defined in JSON/YAML file. This section is "
        f"not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required"
    ),
    FLAG_CONFIG: (
        "Configuration with flags. This section is not compatible with "
        f"'{FILE_CONFIG}'. Arguments with '*' are required."
    ),
}


class InferenceServer(object):
    """
    A class for running the on-device inference server.

    It is used to delegate evaluation of the model to a different
    device if the model is optimized for a different device than
    host.
    """

    def __init__(self, runtime: Runtime, protocol: Protocol):
        self.runtime = runtime
        self.protocol = protocol
        self.close_server_event = Event()
        self.status = ServerStatus(ServerAction.IDLE)

    def close(self):
        """
        Stops the server.
        """
        self.close_server_event.set()

    def run(self):
        """
        Main runtime server program.

        It waits for requests from a single client.

        Based on requests, it loads the model, runs inference and provides
        statistics.
        """

        def client_disconnected_callback():
            self.status = ServerStatus(ServerAction.WAITING_FOR_CLIENT)
            try:
                self.protocol.stop_sending_logs()
            except NotSupportedError:
                KLogger.debug(
                    "Attempted to stop sending logs, but the selected protocol"
                    " does not support sending logs in the first place."
                )

        def client_connected_callback(addr):
            self.status = ServerStatus(ServerAction.CLIENT_CONNECTED)
            try:
                self.protocol.start_sending_logs()
            except NotSupportedError:
                KLogger.warning(
                    "Selected protocol does not support sending logs."
                )

        self.status = ServerStatus(
            ServerAction.WAITING_FOR_CLIENT,
            self.protocol.initialize_server(
                client_connected_callback, client_disconnected_callback
            ),
        )

        if not self.status.success:
            KLogger.error("Server prepare failed")
            return

        KLogger.info("Server started")

        self.protocol.serve(
            self._data_callback,
            self._model_callback,
            self._process_callback,
            self._output_callback,
            self._stats_callback,
            self._io_spec_callback,
            self._optimizers_callback,
            self._unoptimized_model_callback,
            self._optimize_model_callback,
        )
        self.close_server_event.wait()
        try:
            self.protocol.stop_sending_logs()
        except NotSupportedError:
            KLogger.debug(
                "Attempted to stop sending logs, but the selected protocol"
                " does not support sending logs in the first place."
            )
        self.status = ServerStatus(ServerAction.IDLE)
        self.protocol.disconnect()

    def _data_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for uploading an input for inference. You need to
        upload the model before this can be executed.

        Parameters
        ----------
        payload : bytes
            Input data for the model.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.UPLOADING_INPUT)
        self.status.finish_action(self.runtime.load_input_from_bytes(payload))
        return self.status

    def _model_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for preparing a model for inference and starting an
        inference session. You need to upload model iospec, before this can
        be executed.

        Parameters
        ----------
        payload : bytes
            Model data or None, if the model should be loaded from another
            source.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.UPLOADING_MODEL)
        self.runtime.inference_session_start()
        self.status.finish_action(self.runtime.prepare_model(payload))
        return self.status

    def _process_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for processing received input and measuring the
        performance quality. You need to upload input, before this can
        be executed.

        Parameters
        ----------
        payload : bytes
            Not used here.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.PROCESSING_INPUT)
        KLogger.debug("Processing input")
        self.runtime.run()
        KLogger.debug("Input processed")
        self.status.finish_action(True)
        return self.status

    def _output_callback(self) -> Tuple[ServerStatus, Optional[bytes]]:
        """
        Server callback for retrieving model output. You need to call
        '_process_callback' and wait for it to finish, before this can
        be executed.

        Returns
        -------
        Tuple[ServerStatus, Optional[bytes]]
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
            Inference output, or None if output extraction failed.
        """
        self.status.start_action(ServerAction.EXTRACTING_OUTPUT)
        out = None
        try:
            out = self.runtime.upload_output(b"")
        except ModelNotPreparedError:
            self.status.finish_action(False)
        if out:
            self.status.finish_action(True)
        else:
            self.status.finish_action(False)
        return self.status, out

    def _stats_callback(self) -> Tuple[ServerStatus, Optional[bytes]]:
        """
        Server callback for stopping measurements and retrieving stats.

        Returns
        -------
        Tuple[ServerStatus, Optional[bytes]]
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
            Inference statistics, or None in case of failure.
        """
        self.status.start_action(ServerAction.COMPUTING_STATISTICS)
        self.runtime.inference_session_end()
        out = self.runtime.upload_stats(b"")
        self.status.finish_action(True)
        return self.status, out

    def _io_spec_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for preparing model io specification.

        Parameters
        ----------
        payload : bytes
            Input/output specification data or None, if the data
            should be loaded from another source.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.UPLOADING_IOSPEC)
        self.status.finish_action(
            self.runtime.prepare_io_specification(payload)
        )
        return self.status

    def _optimizers_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for loading model optimizers.

        Parameters
        ----------
        payload : bytes
            Serialized JSON with optimizer configuration.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.UPLOADING_OPTIMIZERS)
        from kenning.utils.class_loader import load_class

        json_cfg = yaml.safe_load(payload.decode())

        optimizers_cfg = (
            json_cfg["optimizers"] if "optimizers" in json_cfg else []
        )
        optimizers_cls = [load_class(cfg["type"]) for cfg in optimizers_cfg]
        self.optimizers = [
            cls.from_json(cfg["parameters"], dataset=None)
            for cls, cfg in zip(optimizers_cls, optimizers_cfg)
        ]

        prev_block_cfg = json_cfg["prev_block"]

        class PrevBlockStub(object):
            def __init__(self):
                self.compiled_model_path = Path(prev_block_cfg["model_path"])

            @classmethod
            def get_output_formats(cls):
                return prev_block_cfg["model_type"]

            def save_io_specification(self, model_path: Path):
                with open(Optimizer.get_spec_path(model_path), "w") as spec_f:
                    spec_f.write(json.dumps(prev_block_cfg["io_spec"]))

            def get_io_specification(self) -> Dict[str, Any]:
                return prev_block_cfg["io_spec"]

        self.prev_block = PrevBlockStub()

        optimizers_str = ", ".join(
            optimizer.__class__.__name__ for optimizer in self.optimizers
        )
        KLogger.info(f"Loaded optimizers: {optimizers_str}")

        self.status.finish_action(True)
        return self.status

    def _unoptimized_model_callback(self, payload: bytes) -> ServerStatus:
        """
        Server callback for receiving a model to optimize.

        Parameters
        ----------
        payload : bytes
            Model to optimize.

        Returns
        -------
        ServerStatus
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
        """
        self.status.start_action(ServerAction.UPLOADING_UNOPTIMIZED_MODEL)
        prev_block = self.prev_block
        model_path = prev_block.compiled_model_path
        with open(model_path, "wb") as model_f:
            model_f.write(payload)
        self.status.finish_action(True)
        return self.status

    def _optimize_model_callback(self) -> Tuple[ServerStatus, Optional[bytes]]:
        """
        Server callback for optimizing model. You need to upload optimizers
        and an unoptimized model before this can be executed.

        Returns
        -------
        Tuple[ServerStatus, Optional[bytes]]
            Object containing information about the last task executed by the
            server and whether that task succeeded or not.
            Optimized model, or None if optimization was not successful.
        """
        self.status.start_action(ServerAction.OPTIMIZING_MODEL)
        model_data = None
        try:
            prev_block = self.prev_block
            model_path = prev_block.compiled_model_path
            for optimizer in self.optimizers:
                KLogger.info(f"Processing block: {type(optimizer).__name__}")

                model_type = optimizer.consult_model_type(prev_block)

                prev_block.save_io_specification(model_path)
                optimizer.set_input_type(model_type)
                optimizer.init()
                if hasattr(prev_block, "get_io_specification"):
                    optimizer.compile(
                        model_path, prev_block.get_io_specification()
                    )
                else:
                    optimizer.compile(model_path)

                prev_block = optimizer
                model_path = prev_block.compiled_model_path

            model_path = self.optimizers[-1].compiled_model_path

            with open(model_path, "rb") as model_f:
                model_data = model_f.read()

            self.status.finish_action(True)
        except Exception as e:
            KLogger.error(f"Compilation error: {e}", stack_info=True)
            self.status.finish_action(False)
        return self.status, model_data


class InferenceServerRunner(CommandTemplate):
    """
    Command template for running the inference server.
    """

    parse_all = False
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
            InferenceServerRunner, InferenceServerRunner
        ).configure_parser(parser, command, types, groups)

        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            metavar="CONFIG",
            help="* The path to the input YAML/JSON file with configuration",
            type=ResourceURI,
        ).completer = FilesCompleter(allowednames=("yaml", "yml", "json"))
        groups[FLAG_CONFIG].add_argument(
            "--platform-cls",
            help="Platform-based class that wraps platform being tested",
        ).completer = ClassPathCompleter(PLATFORMS)
        groups[FLAG_CONFIG].add_argument(
            "--protocol-cls",
            help=(
                "* Protocol-based class with the implementation of "
                "communication between inference tester and inference runner"
            ),
        ).completer = ClassPathCompleter(RUNTIME_PROTOCOLS)
        groups[FLAG_CONFIG].add_argument(
            "--runtime-cls",
            help=(
                "* Runtime-based class with the implementation of model "
                "runtime"
            ),
        ).completer = ClassPathCompleter(RUNTIMES)

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        keys = {ConfigKey.platform, ConfigKey.runtime, ConfigKey.protocol}

        if args.json_cfg is not None:
            with open(args.json_cfg, "r") as f:
                json_cfg = yaml.safe_load(f)

            objs = objs_from_json(json_cfg, keys, (args, not_parsed))
        else:

            def required(classes: Dict[ConfigKey, Type]):
                if ConfigKey.runtime not in classes:
                    report_missing([to_namespace_name(ConfigKey.runtime)])

            objs = objs_from_argparse(
                args, not_parsed, keys, required=required
            )

        return InferenceServerRunner._run_server(objs)

    @staticmethod
    def _run_server(objs: Dict[ConfigKey, Any]):
        runtime = objs.get(ConfigKey.runtime)
        protocol = objs.get(ConfigKey.protocol)

        if not protocol:
            platform = objs.get(ConfigKey.platform, LocalPlatform())
            protocol = platform.get_default_protocol()

        formersighandler = signal.getsignal(signal.SIGINT)

        server = InferenceServer(runtime, protocol)

        def sigint_handler(sig, frame):
            server.close()
            KLogger.info(
                "Closing application (press Ctrl-C again for force closing)..."
            )
            signal.signal(signal.SIGINT, formersighandler)

        signal.signal(signal.SIGINT, sigint_handler)

        KLogger.info("Starting server...")
        server.run()


if __name__ == "__main__":
    sys.exit(InferenceServerRunner.scenario_run())
