# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Renode.
"""

import logging
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import tqdm
from serial import Serial

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements, MeasurementsCollector
from kenning.core.model import ModelWrapper
from kenning.core.protocol import Protocol, RequestFailure, check_request
from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger, LoggerProgressBar, TqdmCallback
from kenning.utils.resource_manager import PathOrURI, ResourceURI

KLogger.add_custom_level(logging.INFO + 1, "RENODE")


class RenodeRuntime(Runtime):
    """
    Runtime subclass that provides and API for testing inference on bare-metal
    runtimes executed on Renode simulated platform.
    """

    inputtypes = ["iree"]

    arguments_structure = {
        "runtime_binary_path": {
            "argparse_name": "--runtime-binary-path",
            "description": "Path to bare-metal runtime binary",
            "type": ResourceURI,
        },
        "platform_resc_path": {
            "argparse_name": "--platform-resc-path",
            "description": "Path to platform script",
            "type": ResourceURI,
        },
        "resc_dependencies": {
            "argparse_name": "--resc-dependencies",
            "description": "Renode script dependencies",
            "type": ResourceURI,
            "is_list": True,
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
    }

    def __init__(
        self,
        runtime_binary_path: PathOrURI,
        platform_resc_path: PathOrURI,
        resc_dependencies: List[ResourceURI] = [],
        post_start_commands: List[str] = [],
        runtime_log_uart: Optional[Path] = None,
        runtime_log_init_msg: Optional[str] = None,
        disable_profiler: bool = False,
        profiler_dump_path: Optional[Path] = None,
        profiler_interval_step: float = 10.0,
        sensor: Optional[str] = None,
        batches_count: int = 100,
        disable_performance_measurements: bool = False,
    ):
        """
        Constructs Renode runtime.

        Parameters
        ----------
        runtime_binary_path : PathOrURI
            Path to the runtime binary.
        platform_resc_path : PathOrURI
            Path to the Renode script.
        resc_dependencies : List[ResourceURI]
            Renode script dependencies.
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
        """
        self.runtime_binary_path = runtime_binary_path
        self.platform_resc_path = platform_resc_path
        # check resc dependencies
        for dependency in resc_dependencies:
            assert dependency.is_file(), f"Dependency {dependency} not found"
        self.resc_dependencies = resc_dependencies
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
        self.renode_logs = []
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

        self.machine = None

    def run_client(
        self,
        dataset: Dataset,
        modelwrapper: ModelWrapper,
        protocol: Protocol,
        compiled_model_path: PathOrURI,
    ):
        self.init_renode()

        protocol.initialize_client()

        try:
            spec_path = self.get_io_spec_path(compiled_model_path)
            if spec_path.exists():
                protocol.upload_io_specification(spec_path)
            protocol.upload_model(compiled_model_path)

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
                    if self.sensor is None:
                        # provide data to runtime
                        X, _ = sample
                        prepX = modelwrapper._preprocess_input(X)
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

                    preds = modelwrapper.convert_output_from_bytes(preds)
                    posty = modelwrapper._postprocess_outputs(preds)

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
                        measurements += dataset._evaluate(posty, y, out_spec)

                    self.handle_renode_logs()

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
        if self.runtime_log_uart is not None:
            while True:
                logs = self.runtime_log_uart.read_all()
                if not len(logs):
                    break
                self.uart_log_buffer += logs.decode()

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
        monitor.execute(f"$bin=@{self.runtime_binary_path.resolve()}")

        for dep in self.resc_dependencies:
            dep_name = dep.name.lower().replace(".", "_")
            dep_path = str(dep.resolve())
            monitor.execute(f"${dep_name}=@{dep_path}")

        _, err = monitor.execute_script(str(self.platform_resc_path.resolve()))

        if err:
            raise Exception("RESC execution error: " + err)

        self.machine = next(iter(emu))

        if (
            not self.disable_performance_measurements
            and not self.disable_profiler
        ):
            self.machine.EnableProfiler(str(self.profiler_dump_path.resolve()))
            KLogger.info(
                f"Profiler dump path: {self.profiler_dump_path.resolve()}"
            )

        if self.runtime_log_uart is not None:
            self.runtime_log_uart = Serial(str(self.runtime_log_uart), 115200)

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

    def get_opcode_stats(self) -> Dict[str, int]:
        """
        Retrieves opcode counters from Renode.

        Returns
        -------
        Dict[str, int]
            Dict where the keys are opcodes and the values are counters.
        """
        KLogger.info("Retrieving opcode counters")

        # retrieve opcode counters
        stats_raw = self.machine.sysbus.cpu.GetAllOpcodesCounters()
        stats = {}
        i = 1
        while True:
            try:
                stats[stats_raw[i, 0]] = int(stats_raw[i, 1])
            except IndexError:
                break
            i += 1

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

        parser = _ProfilerDumpParser(
            self.profiler_dump_path,
            self.profiler_interval_step,
            start_timestamp,
            end_timestamp,
        )

        return parser.parse()

    @staticmethod
    def _opcode_stats_diff(
        opcode_stats_a: Dict[str, int], opcode_stats_b: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Computes difference of opcode counters. It is assumed that counters
        from second dict are greater.

        Parameters
        ----------
        opcode_stats_a : Dict[str, int]
            First opcode stats.
        opcode_stats_b : Dict[str, int]
            Seconds opcode stats.

        Returns
        -------
        Dict[str, int]
            Difference between two opcode stats.
        """
        ret = {}
        for opcode in opcode_stats_b.keys():
            ret[opcode] = opcode_stats_b[opcode] - opcode_stats_a.get(
                opcode, 0
            )
        return ret

    def extract_output(self):
        raise NotImplementedError

    def load_input(self, input_data):
        raise NotImplementedError

    def prepare_model(self, input_data):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class _ProfilerDumpParser(object):
    ENTRY_TYPE_INSTRUCTIONS = b"\x00"
    ENTRY_TYPE_MEM0RY = b"\x01"
    ENTRY_TYPE_PERIPHERALS = b"\x02"
    ENTRY_TYPE_EXCEPTIONS = b"\x03"

    ENTRY_HEADER_FORMAT = "<qdc"
    ENTRY_FORMAT_INSTRUCTIONS = "<cQ"
    ENTRY_FORMAT_MEM0RY = "c"
    ENTRY_FORMAT_PERIPHERALS = "<cQ"
    ENTRY_FORMAT_EXCEPTIONS = "Q"

    MEMORY_OPERATION_READ = b"\x02"
    MEMORY_OPERATION_WRITE = b"\x03"

    PERIPHERAL_OPERATION_READ = b"\x00"
    PERIPHERAL_OPERATION_WRITE = b"\x01"

    def __init__(
        self,
        dump_path: Path,
        interval_step: float,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
    ):
        self.dump_path = dump_path
        self.interval_step = interval_step
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

    def parse(self) -> Dict[str, Any]:
        """
        Parses Renode profiler dump.

        Returns
        -------
        Dict[str, Any]
            Dict containing statistics retrieved from the dump file.

        Raises
        ------
        Exception
            Raised when profiler dump could not be parsed
        """
        profiler_timestamps = []
        stats = {
            "executed_instructions": {},
            "memory_accesses": {"read": [], "write": []},
            "peripheral_accesses": {},
            "exceptions": [],
        }

        with (
            LoggerProgressBar() as logger_progress_bar,
            tqdm.tqdm(
                total=self.dump_path.stat().st_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                file=logger_progress_bar,
            ) as progress_bar,
            open(self.dump_path, "rb") as dump_file,
        ):
            # parse header
            cpus, peripherals = self._parse_header(dump_file)

            for cpu in cpus.values():
                stats["executed_instructions"][cpu] = []

            for peripheral in peripherals.keys():
                stats["peripheral_accesses"][peripheral] = {
                    "read": [],
                    "write": [],
                }

            entry = struct.Struct(self.ENTRY_HEADER_FORMAT)
            prev_instr_counter = defaultdict(lambda: 0)

            # parse entries
            entries_counter = 0
            invalid_entries = 0

            while True:
                entry_header = dump_file.read(entry.size)
                if not entry_header:
                    break

                _, virt_time, entry_type = entry.unpack(entry_header)
                virt_time /= 1000

                # ignore entry
                if (
                    self.start_timestamp is not None
                    and self.end_timestamp is not None
                    and not (
                        self.start_timestamp < virt_time
                        and virt_time < self.end_timestamp
                    )
                ):
                    if entry_type == self.ENTRY_TYPE_INSTRUCTIONS:
                        cpu_id, instr_counter = self._read(
                            self.ENTRY_FORMAT_INSTRUCTIONS, dump_file
                        )
                        if cpu_id[0] in cpus:
                            prev_instr_counter[cpus[cpu_id[0]]] = instr_counter
                    elif entry_type == self.ENTRY_TYPE_MEM0RY:
                        self._read(self.ENTRY_FORMAT_MEM0RY, dump_file)
                    elif entry_type == self.ENTRY_TYPE_PERIPHERALS:
                        self._read(self.ENTRY_FORMAT_PERIPHERALS, dump_file)
                    elif entry_type == self.ENTRY_TYPE_EXCEPTIONS:
                        self._read(self.ENTRY_FORMAT_EXCEPTIONS, dump_file)
                    else:
                        raise Exception(
                            "Invalid entry in profiler dump: "
                            f"{entry_type.hex()}"
                        )
                    continue

                # parse entry
                interval_start = virt_time - virt_time % (
                    self.interval_step / 1000
                )
                if (
                    len(profiler_timestamps) == 0
                    or profiler_timestamps[-1] != interval_start
                ):
                    profiler_timestamps.append(interval_start)
                    stats_to_update = [stats]
                    while len(stats_to_update):
                        s = stats_to_update.pop(0)
                        if isinstance(s, list):
                            s.append(0)
                        elif isinstance(s, dict):
                            stats_to_update.extend(s.values())

                if entry_type == self.ENTRY_TYPE_INSTRUCTIONS:
                    # parse executed instruction entry
                    output_list = stats["executed_instructions"]
                    cpu_id, instr_counter = self._read(
                        self.ENTRY_FORMAT_INSTRUCTIONS, dump_file
                    )
                    if cpu_id[0] in cpus:
                        cpu = cpus[cpu_id[0]]
                        output_list = output_list[cpu]

                        output_list[-1] += (
                            instr_counter - prev_instr_counter[cpu]
                        )
                        prev_instr_counter[cpu] = instr_counter
                    else:
                        # invalid cpu id
                        invalid_entries += 1
                        continue

                elif entry_type == self.ENTRY_TYPE_MEM0RY:
                    # parse memory access entry
                    output_list = stats["memory_accesses"]
                    operation = self._read(
                        self.ENTRY_FORMAT_MEM0RY, dump_file
                    )[0]

                    if operation == self.MEMORY_OPERATION_READ:
                        output_list = output_list["read"]
                    elif operation == self.MEMORY_OPERATION_WRITE:
                        output_list = output_list["write"]
                    else:
                        # invalid operation
                        invalid_entries += 1
                        continue

                    output_list[-1] += 1

                elif entry_type == self.ENTRY_TYPE_PERIPHERALS:
                    # parse peripheral access entry
                    output_list = stats["peripheral_accesses"]
                    operation, address = self._read(
                        self.ENTRY_FORMAT_PERIPHERALS, dump_file
                    )

                    peripheral_found = False
                    for peripheral, address_range in peripherals.items():
                        if address_range[0] <= address <= address_range[1]:
                            output_list = output_list[peripheral]
                            peripheral_found = True
                            break

                    if not peripheral_found:
                        # invalid address
                        invalid_entries += 1
                        continue

                    if operation == self.PERIPHERAL_OPERATION_READ:
                        output_list = output_list["read"]
                    elif operation == self.PERIPHERAL_OPERATION_WRITE:
                        output_list = output_list["write"]
                    else:
                        # invalid operation
                        invalid_entries += 1
                        continue

                    output_list[-1] += 1

                elif entry_type == self.ENTRY_TYPE_EXCEPTIONS:
                    # parse exception entry
                    output_list = stats["exceptions"]
                    _ = self._read(self.ENTRY_FORMAT_EXCEPTIONS, dump_file)

                    output_list[-1] += 1

                else:
                    raise Exception(
                        f"Invalid entry in profiler dump: {entry_type.hex()}"
                    )

                entries_counter += 1
                if entries_counter >= 1000:
                    progress_bar.update(dump_file.tell() - progress_bar.n)
                    entries_counter = 0

            progress_bar.update(dump_file.tell() - progress_bar.n)

            if invalid_entries > 0:
                KLogger.warning(
                    f"Found {invalid_entries} invalid entries in profiler "
                    "dump file"
                )

            # multiply counters by 1 sec / interval_step to get counts per sec
            stats_to_update = [stats]
            while len(stats_to_update):
                s = stats_to_update.pop(0)
                if isinstance(s, list):
                    for i in range(len(s)):
                        s[i] *= 1000 / self.interval_step
                elif isinstance(s, dict):
                    stats_to_update.extend(s.values())

        return dict(stats, profiler_timestamps=profiler_timestamps)

    def _parse_header(
        self, file: BinaryIO
    ) -> Tuple[Dict[int, str], Dict[str, Tuple[int, int]]]:
        """
        Parses header of Renode profiler dump.

        Parameters
        ----------
        file : BinaryIO
            File-like object to parse header from.

        Returns
        -------
        Tuple[Dict[int, str], Dict[str, Tuple[int, int]]]
            Tuples of dicts containing cpus and peripherals data.
        """
        cpus = {}
        peripherals = {}
        cpus_count = self._read("i", file)[0]
        for _ in range(cpus_count):
            cpu_id = self._read("i", file)[0]
            cpu_name_len = self._read("i", file)[0]
            cpus[cpu_id] = self._read(f"{cpu_name_len}s", file)[0].decode()

        peripherals_count = self._read("i", file)[0]
        for _ in range(peripherals_count):
            peripheral_name_len = self._read("i", file)[0]
            peripheral_name = self._read(f"{peripheral_name_len}s", file)[
                0
            ].decode()
            peripheral_start_address, peripheral_end_address = self._read(
                "2Q", file
            )
            peripherals[peripheral_name] = (
                peripheral_start_address,
                peripheral_end_address,
            )

        return cpus, peripherals

    @staticmethod
    def _read(format_str: str, file: BinaryIO) -> Tuple[Any, ...]:
        """
        Reads struct of given format from file.

        Parameters
        ----------
        format_str : str
            Format of the struct.
        file : BinaryIO
            File-like object to read struct from.

        Returns
        -------
        Tuple[Any, ...]
            Struct read from file.
        """
        return struct.unpack(
            format_str, file.read(struct.calcsize(format_str))
        )
