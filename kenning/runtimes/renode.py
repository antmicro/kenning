# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Renode.
"""

import json
import logging
import queue
import re
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List, Optional

from serial import Serial

import kenning.utils.renode_profiler_parser as profiler_parser
from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements, MeasurementsCollector
from kenning.core.model import ModelWrapper
from kenning.core.protocol import Protocol, RequestFailure, check_request
from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger, LoggerProgressBar, TqdmCallback
from kenning.utils.resource_manager import PathOrURI, ResourceURI

KLogger.add_custom_level(logging.INFO + 1, "RENODE")


class UARTNotFoundInDTSError(Exception):
    """
    Exception raised when UART was not found in devicetree.
    """

    ...


class RenodeRuntimeError(Exception):
    """
    Exception raised when Renode command fails.
    """

    ...


class UARTReader:
    """
    Reads bytes from the provided UART in a separate thread.
    """

    def __init__(self, *args, timeout: float = 0.5, **kwargs):
        self._queue = queue.Queue()

        self._conn = Serial(*args, **kwargs, timeout=timeout)
        self._stop = threading.Event()
        self._thread = None

        self.start()

    def __del__(self):
        self.stop()

    def _create_thread(self, conn: Serial):
        def _reader_thread():
            while not self._stop.is_set():
                content = conn.read()
                content += conn.read_all()
                if content:
                    self._queue.put(content)

        return threading.Thread(target=_reader_thread, daemon=True)

    def read(self, block=False, timeout=None) -> bytes:
        try:
            content = self._queue.get(block=block, timeout=timeout)
            return content
        except queue.Empty:
            return b""

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = self._create_thread(self._conn)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None


class RenodeRuntime(Runtime):
    """
    Runtime subclass that provides and API for testing inference on bare-metal
    runtimes executed on Renode simulated platform.

    This runtime creates Renode platform based on provided resc or Zephyr
    runtime build artifacts. It is required to specify either
    `platform_resc_path` or `zephyr_build_path`. Those parameters are mutually
    exclusive. If the provided resc uses some other files (i.e. csharp sources
    with custom CPU definition) then they should be specified in
    `resc_dependencies` and resc should use them via variables named the same
    as their names lowercased and with dots replaced with underscore
    (i.e. springbokriscv32_cs). The `zephyr_build_path` directory should
    contain devicetree file `<board>_flat.dts` and `<board>.repl`. Those files
    can be generated using `board-repl` cmake target. It also should contain
    runtime binary `zephyr/zephyr.elf`.
    """

    inputtypes = ["iree", "tflite", "tvm"]

    arguments_structure = {
        "runtime_binary_path": {
            "argparse_name": "--runtime-binary-path",
            "description": "Path to bare-metal runtime binary",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "platform_resc_path": {
            "argparse_name": "--platform-resc-path",
            "description": "Path to platform script",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "resc_dependencies": {
            "argparse_name": "--resc-dependencies",
            "description": "Renode script dependencies",
            "type": ResourceURI,
            "is_list": True,
            "default": [],
        },
        "zephyr_build_path": {
            "description": (
                "Path to Zephyr runtime build directory. It should contain "
                "runtime binary and platform repl and dts files."
            ),
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "post_start_commands": {
            "argparse_name": "--post-start-commands",
            "description": (
                "Renode commands executed after starting the machine"
            ),
            "type": str,
            "is_list": True,
            "default": [],
        },
        "runtime_log_uart": {
            "argparse_name": "--runtime-log-uart",
            "description": (
                "Path to UART from which runtime logs should be read. If not "
                "specified, then the logs are read from Renode logs"
            ),
            "type": Path,
            "nullable": True,
        },
        "runtime_log_init_msg": {
            "argparse_name": "--runtime-log-init-msg",
            "description": (
                "Log message produced by runtime after successful "
                "initialization. If specified, Kenning will wait for "
                "such message before starting inference"
            ),
            "type": str,
            "nullable": True,
            "default": None,
        },
        "disable_profiler": {
            "argparse_name": "--disable-profiler",
            "description": "Disables Renode profiler",
            "type": bool,
            "default": False,
        },
        "profiler_dump_path": {
            "argparse_name": "--profiler-dump-path",
            "description": "Path to Renode profiler dump",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "profiler_interval_step": {
            "argparse_name": "--profiler-interval-step",
            "description": "Interval step in ms used to parse profiler data",
            "type": float,
            "default": 10.0,
        },
        "sensor": {
            "argparse_name": "--sensor",
            "description": (
                "Name of the sensor to be used as input. If none then no "
                "sensor is used"
            ),
            "type": str,
            "nullable": True,
            "default": None,
        },
        "batches_count": {
            "argparse_name": "--batches-count",
            "description": "Number of batches to read",
            "type": int,
            "default": 10,
        },
        "llext_binary_path": {
            "argparse_name": "--llext-binary-path",
            "description": "Path to the LLEXT binary",
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
    }

    def __init__(
        self,
        runtime_binary_path: Optional[PathOrURI] = None,
        platform_resc_path: Optional[PathOrURI] = None,
        resc_dependencies: List[ResourceURI] = [],
        zephyr_build_path: Optional[Path] = None,
        post_start_commands: List[str] = [],
        runtime_log_uart: Optional[Path] = None,
        runtime_log_init_msg: Optional[str] = None,
        disable_profiler: bool = False,
        profiler_dump_path: Optional[Path] = None,
        profiler_interval_step: float = 10.0,
        sensor: Optional[str] = None,
        batches_count: int = 100,
        disable_performance_measurements: bool = False,
        llext_binary_path: Optional[PathOrURI] = None,
    ):
        """
        Constructs Renode runtime.

        Parameters
        ----------
        runtime_binary_path : Optional[PathOrURI]
            Path to the runtime binary.
        platform_resc_path : Optional[PathOrURI]
            Path to the Renode script.
        resc_dependencies : List[ResourceURI]
            Renode script dependencies.
        zephyr_build_path : Optional[Path]
            Path to Zephyr runtime build directory. It should contain runtime
            binary and platform repl and dts files.
        post_start_commands : List[str]
            Renode commands executed after starting the machine.
        runtime_log_uart : Optional[Path]
            Path to UART from which runtime logs should be read. If not
            specified, then the logs are read from Renode logs.
        runtime_log_init_msg : Optional[str]
            Log produced by runtime after initialization. If specified, Kenning
            will wait for such message before starting inference.
        disable_profiler : bool
            Disables Renode profiler.
        profiler_dump_path : Optional[Path]
            Path to the Renode profiler dump.
        profiler_interval_step : float
            Interval step in ms used to parse profiler data.
        sensor : Optional[str]
            Name of the sensor to be used as input. If none then no sensor is
            used.
        batches_count : int
            Number of batches to read.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        llext_binary_path : Optional[PathOrURI]
            Path to the LLEXT binary.

        Raises
        ------
        ValueError
            Raised when invalid arguments passed.
        """
        if not ((platform_resc_path is None) ^ (zephyr_build_path is None)):
            raise ValueError(
                "One of platform_resc_path, runtime_build_path should be "
                "specified"
            )
        if platform_resc_path is not None and runtime_binary_path is None:
            raise ValueError("runtime_binary_path should be specified")

        self.runtime_binary_path = runtime_binary_path
        self.platform_resc_path = platform_resc_path
        self.resc_dependencies = resc_dependencies
        self.zephyr_build_path = zephyr_build_path
        self.post_start_commands = post_start_commands
        self.runtime_log_uart = runtime_log_uart
        self.runtime_log_init_msg = runtime_log_init_msg
        self.disable_profiler = disable_profiler
        self.profiler_dump_path = profiler_dump_path
        self.profiler_interval_step = profiler_interval_step
        self.sensor = sensor
        self.batches_count = batches_count
        self.renode_log_file = None
        self.renode_log_file_name = None
        self.renode_log_buffer = ""
        self.uart_log_buffer = ""
        self.uart_log_reader = None
        self.renode_logs = []
        self.llext_binary_path = llext_binary_path
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

        self.machine = None

    def run_client(
        self,
        dataset: Dataset,
        modelwrapper: ModelWrapper,
        dataconverter: DataConverter,
        protocol: Protocol,
        compiled_model_path: PathOrURI,
    ) -> bool:
        self.init_renode()

        protocol.initialize_client()

        # check resc dependencies
        for dependency in self.resc_dependencies:
            assert dependency.is_file(), f"Dependency {dependency} not found"

        try:
            if self.llext_binary_path:
                status = protocol.upload_runtime(self.llext_binary_path)
                self.handle_renode_logs()
                if not status:
                    return False

            spec_path = self.get_io_spec_path(compiled_model_path)
            if spec_path.exists():
                status = protocol.upload_io_specification(spec_path)
                self.handle_renode_logs()
                if not status:
                    return False

            with open(spec_path, "r") as spec_f:
                io_spec = json.load(spec_f)

                self.read_io_specification(io_spec)
                modelwrapper.io_specification = io_spec

            protocol.upload_model(compiled_model_path)
            self.handle_renode_logs()

            measurements = Measurements()

            # get opcode stats before inference
            if not self.disable_performance_measurements:
                pre_opcode_stats = self.get_opcode_stats()

            # prepare iterator for inference
            iterable = (
                range(self.batches_count)
                if self.sensor is not None
                else dataset.iter_test()
            )

            # inference loop
            with LoggerProgressBar() as logger_progress_bar:
                for sample in TqdmCallback(
                    "runtime", iterable, file=logger_progress_bar
                ):
                    try:
                        if self.sensor is None:
                            # provide data to runtime
                            X, _ = sample
                            prepX = dataconverter.to_next_block(X)
                            prepX = modelwrapper.convert_input_to_bytes(prepX)
                            check_request(
                                protocol.upload_input(prepX), "send input"
                            )
                            check_request(
                                protocol.request_processing(self.get_time),
                                "inference",
                            )

                        # get inference output
                        _, preds = check_request(
                            protocol.download_output(), "receive output"
                        )

                        posty = modelwrapper.convert_output_from_bytes(preds)
                        posty = dataconverter.to_previous_block(posty)

                        out_spec = (
                            self.processed_output_spec
                            if self.processed_output_spec
                            else self.output_spec
                        )

                        if self.sensor is not None:
                            measurements += dataset._evaluate(
                                posty, None, out_spec
                            )
                        else:
                            _, y = sample
                            measurements += dataset._evaluate(
                                posty, y, out_spec
                            )

                        self.handle_renode_logs()

                    except KeyboardInterrupt:
                        # break inference loop
                        break

            # get opcode stats after inference
            if not self.disable_performance_measurements:
                post_opcode_stats = self.get_opcode_stats()

                MeasurementsCollector.measurements += {
                    "opcode_counters": self._opcode_stats_diff(
                        pre_opcode_stats, post_opcode_stats
                    )
                }

            MeasurementsCollector.measurements += (
                protocol.download_statistics()
            )

        except RequestFailure as ex:
            KLogger.fatal(ex)
            self.handle_renode_logs()
            return False
        else:
            MeasurementsCollector.measurements += measurements
        finally:
            self.clear_renode()
            self.handle_renode_logs()
        if (
            not self.disable_performance_measurements
            and not self.disable_profiler
        ):
            MeasurementsCollector.measurements += self.get_profiler_stats()

        return True

    def handle_renode_logs(self):
        """
        Captures log from UART and Renode console.
        """
        # capture Renode console logs
        from Antmicro.Renode.Logging import Logger

        Logger.Flush()

        if not self.renode_log_file.closed and self.renode_log_file.readable():
            while True:
                logs = self.renode_log_file.read()
                if not len(logs):
                    break
                self.renode_log_buffer += logs

            *new_logs, self.renode_log_buffer = self.renode_log_buffer.split(
                "\n"
            )
            self.renode_logs.extend(new_logs)
            for new_log in new_logs:
                KLogger.renode(new_log)

        # capture UART logs
        if self.uart_log_reader is not None:
            while True:
                logs = self.uart_log_reader.read()
                if not len(logs):
                    break
                self.uart_log_buffer += logs.decode(errors="ignore")

            *new_logs, self.uart_log_buffer = self.uart_log_buffer.split("\n")
            self.renode_logs.extend(new_logs)
            for new_log in new_logs:
                KLogger.renode(new_log)

    def get_time(self):
        if self.machine is None:
            return 0

        return self.machine.ElapsedVirtualTime.TimeElapsed.TotalSeconds

    def clear_renode(self):
        from pyrenode3.wrappers import Emulation

        self.machine = None
        Emulation().PauseAll()
        if self.uart_log_reader is not None:
            self.uart_log_reader.stop()
        Emulation().clear()

        if (
            self.renode_log_file is not None
            and not self.renode_log_file.closed
        ):
            self.renode_log_file.close()
        if self.log_file_path is not None:
            self.log_file_path.unlink()

    def init_renode(self):
        """
        Initializes Renode process and starts runtime.
        """
        try:
            from pyrenode3.wrappers import Emulation, Monitor  # isort: skip
            from Antmicro.Renode.Logging import Logger
            from Antmicro.Renode.RobotFramework import LogTester
        except ImportError as e:
            msg = (
                "Couldn't initialize pyrenode3. "
                "Ensure that Renode is installed according "
                "to the instructions at https://github.com/antmicro/pyrenode3."
            )
            raise Exception(msg) from e

        if (
            not self.disable_performance_measurements
            and self.profiler_dump_path is None
        ):
            self.profiler_dump_path = Path(
                tempfile.mktemp(prefix="renode_profiler_", suffix=".dump")
            )

        emu = Emulation()
        monitor = Monitor()

        log_tester = LogTester(10)
        Logger.AddBackend(backend=log_tester, name="logTester", overwrite=True)

        self.log_file_path = Path(tempfile.mktemp(prefix="renode_log_"))

        monitor.execute(f"logFile @{self.log_file_path.resolve()}")
        self.renode_log_file = open(self.log_file_path, "r")

        if self.platform_resc_path is not None:
            self._prepare_platform_from_resc()
        else:
            self._prepare_zephyr_platform()

        if (
            not self.disable_performance_measurements
            and not self.disable_profiler
        ):
            self.machine.EnableProfiler(str(self.profiler_dump_path.resolve()))
            KLogger.info(
                f"Profiler dump path: {self.profiler_dump_path.resolve()}"
            )

        if self.runtime_log_uart is not None:
            self.uart_log_reader = UARTReader(
                str(self.runtime_log_uart), 115200
            )

        emu.StartAll()

        for cmd in self.post_start_commands:
            monitor.execute(cmd)

        if self.runtime_log_init_msg is not None:
            KLogger.info("Waiting for runtime init")
            timeout = perf_counter() + 30
            log_idx = 0
            while True:
                self.handle_renode_logs()
                initialized = False
                while log_idx < len(self.renode_logs):
                    if self.runtime_log_init_msg in self.renode_logs[log_idx]:
                        initialized = True
                        break
                    log_idx += 1

                if initialized:
                    break

                if perf_counter() > timeout:
                    KLogger.debug(f"Runtime logs:\n{self.renode_logs}")
                    raise TimeoutError(
                        "Runtime did not initialize in 30 seconds"
                    )

                sleep(0.2)

        Logger.RemoveBackend(log_tester)
        KLogger.info("Renode initialized")

    def get_opcode_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Retrieves opcode counters for all cpus from Renode.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dict where the keys are opcodes and the values are counters.
        """
        KLogger.info("Retrieving opcode counters")

        stats = defaultdict(dict)
        # retrieve opcode counters
        for elem in dir(self.machine.sysbus):
            opcode_counters_getter = getattr(
                getattr(self.machine.sysbus, elem),
                "GetAllOpcodesCounters",
                None,
            )

            if opcode_counters_getter is None:
                continue

            stats_raw = opcode_counters_getter()
            i = 1
            while True:
                try:
                    stats[elem][stats_raw[i, 0]] = int(stats_raw[i, 1])
                except IndexError:
                    break
                i += 1

            if not len(stats):
                KLogger.warning(f"Empty opcode counters for {elem}")

        return stats

    def get_profiler_stats(self) -> Dict[str, List[float]]:
        """
        Parses Renode profiler dump.

        Returns
        -------
        Dict[str, List[float]]
            Stats retrieved from Renode profiler dump.

        Raises
        ------
        FileNotFoundError
            Raised when no Renode profiler dump file was found.
        """
        KLogger.info("Parsing Renode profiler dump")
        if (
            self.profiler_dump_path is None
            or not self.profiler_dump_path.exists()
        ):
            KLogger.error("Missing profiler dump file")
            raise FileNotFoundError

        try:
            timestamps = MeasurementsCollector.measurements.get_values(
                "protocol_inference_step_timestamp"
            )

            durations = MeasurementsCollector.measurements.get_values(
                "protocol_inference_step_timestamp"
            )

            start_timestamp = timestamps[0]
            end_timestamp = timestamps[-1] + durations[-1]
        except KeyError:
            start_timestamp = None
            end_timestamp = None

        parsed_stats = profiler_parser.parse(
            str(self.profiler_dump_path.resolve()).encode(),
            start_timestamp,
            end_timestamp,
            self.profiler_interval_step,
        )

        KLogger.info("Renode profiler dump parsed")

        return parsed_stats

    @staticmethod
    def _opcode_stats_diff(
        opcode_stats_a: Dict[str, Dict[str, int]],
        opcode_stats_b: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, int]]:
        """
        Computes difference of opcode counters. It is assumed that counters
        from second dict are greater.

        Parameters
        ----------
        opcode_stats_a : Dict[str, Dict[str, int]]
            First opcode stats.
        opcode_stats_b : Dict[str, Dict[str, int]]
            Second opcode stats.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Difference between two opcode stats.
        """
        ret = defaultdict(dict)
        for cpu in opcode_stats_b.keys():
            for opcode in opcode_stats_b[cpu].keys():
                ret[cpu][opcode] = opcode_stats_b[cpu][
                    opcode
                ] - opcode_stats_a.get(cpu, {opcode: 0}).get(opcode, 0)

        return ret

    def extract_output(self):
        raise NotImplementedError

    def load_input(self, input_data):
        raise NotImplementedError

    def prepare_model(self, input_data):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def _prepare_platform_from_resc(self):
        """
        Prepares platform based on provided resc.
        """
        from pyrenode3.wrappers import Emulation, Monitor

        emu = Emulation()
        monitor = Monitor()

        monitor.execute(f"$bin=@{self.runtime_binary_path.resolve()}")
        # parse resc dependencies paths
        for dep in self.resc_dependencies:
            dep_name = dep.name.lower().replace(".", "_")
            dep_path = str(dep.resolve())
            monitor.execute(f"${dep_name}=@{dep_path}")

        _, err = monitor.execute_script(str(self.platform_resc_path.resolve()))

        if err:
            raise RenodeRuntimeError("RESC execution error: " + err)

        self.machine = next(iter(emu))

    def _prepare_zephyr_platform(self):
        """
        Prepares platform based on provided Zephyr build artifacts.
        """
        # as it is called only from init_renode, we can assume this import
        # will not raise exception
        from Antmicro.Renode.Exceptions import RecoverableException
        from pyrenode3.wrappers import Emulation

        def find_uart_by_alias(dts: str, alias: str) -> str:
            alias_match = re.findall(
                rf"{alias} = &([a-zA-Z0-9]*);", platform_dts, re.MULTILINE
            )
            if not len(alias_match):
                UARTNotFoundInDTSError(f"{alias} UART not found in devicetree")

            return alias_match[0]

        emu = Emulation()

        # find dts, repl and runtime binary
        try:
            dts_path = next(self.zephyr_build_path.glob("*_flat.dts"))
        except StopIteration:
            KLogger.error(
                f"Devicetree file not found in {self.zephyr_build_path}"
            )
            return False
        try:
            repl_path = next(self.zephyr_build_path.glob("*.repl"))
        except StopIteration:
            KLogger.error(f"repl file not found in {self.zephyr_build_path}")
            return False
        bin_path = self.zephyr_build_path / "zephyr" / "zephyr.elf"
        if not bin_path.exists():
            KLogger.error(
                f"Runtime binary not found in {self.zephyr_build_path}"
            )
            return False

        board = repl_path.stem

        with open(dts_path, "r") as dts_f:
            platform_dts = dts_f.read()

        kcomms_uart = find_uart_by_alias(platform_dts, "kcomms")
        console_uart = find_uart_by_alias(platform_dts, "zephyr,console")

        self.machine = emu.add_mach(board)
        self.machine.load_repl(repl_path.resolve())
        self.machine.load_elf(bin_path.resolve())

        emu.CreateUartPtyTerminal("console_uart_term", "/tmp/uart-log")
        emu.Connector.Connect(
            getattr(self.machine.sysbus, console_uart).internal,
            emu.externals.console_uart_term,
        )

        emu.CreateUartPtyTerminal("kcomms_uart_term", "/tmp/uart")
        emu.Connector.Connect(
            getattr(self.machine.sysbus, kcomms_uart).internal,
            emu.externals.kcomms_uart_term,
        )

        if not self.disable_performance_measurements:
            # enable opcode counters
            for elem in dir(self.machine.sysbus):
                for method_name in (
                    "EnableOpcodesCounting",
                    "EnableRiscvOpcodesCounting",
                    "EnableVectorOpcodesCounting",
                ):
                    try:
                        opcode_counters_enabler = getattr(
                            getattr(self.machine.sysbus, elem),
                            method_name,
                            None,
                        )
                    except TypeError:
                        # property cannot be read
                        continue

                    if opcode_counters_enabler is None:
                        continue

                    try:
                        opcode_counters_enabler()
                        KLogger.info(f"Enabling {method_name} for {elem}")
                    except RecoverableException:
                        KLogger.warning(
                            f"Error enabling {method_name} for {elem}"
                        )
