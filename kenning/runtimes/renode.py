# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Renode.
"""

import re
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import tqdm
from pyrenode import Pyrenode

from kenning.core.dataset import Dataset
from kenning.core.measurements import (
    Measurements,
    MeasurementsCollector,
)
from kenning.core.model import ModelWrapper
from kenning.core.protocol import Protocol, RequestFailure, check_request
from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI, ResourceURI


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
            "description": "Name of the sensor to be used as input. If none "
            "then no sensor is used",
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
        self.disable_profiler = disable_profiler
        if profiler_dump_path is not None:
            profiler_dump_path = profiler_dump_path.resolve()
        self.profiler_dump_path = profiler_dump_path
        self.profiler_interval_step = profiler_interval_step
        self.sensor = sensor
        self.batches_count = batches_count
        self.renode_handler = None
        self.virtual_time_regex = re.compile(
            r"Elapsed Virtual Time: (\d{2}):(\d{2}):(\d{2}\.\d*)"
        )
        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def run_client(
        self,
        dataset: Dataset,
        modelwrapper: ModelWrapper,
        protocol: Protocol,
        compiled_model_path: PathOrURI,
    ):
        with Pyrenode() as renode_handler:
            self.renode_handler = renode_handler
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
                    for sample in tqdm.tqdm(
                        iterable, file=logger_progress_bar
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

                        if self.sensor is not None:
                            measurements += dataset.evaluate(posty, None)
                        else:
                            _, y = sample
                            measurements += dataset.evaluate(posty, y)

                # get opcode stats after inference
                if not self.disable_performance_measurements:
                    post_opcode_stats = self.get_opcode_stats()

                    MeasurementsCollector.measurements += {
                        "opcode_counters": self._opcode_stats_diff(
                            pre_opcode_stats, post_opcode_stats
                        )
                    }
            except RequestFailure as ex:
                KLogger.fatal(ex)
                return False
            else:
                MeasurementsCollector.measurements += measurements
            finally:
                KLogger.info(self.renode_handler.read_from_renode())
                self.renode_handler = None

        if (
            not self.disable_performance_measurements
            and not self.disable_profiler
        ):
            MeasurementsCollector.measurements += self.get_profiler_stats()

        return True

    def get_time(self):
        if self.renode_handler is None:
            return 0

        elapsed_time_str = self.renode_handler.run_robot_keyword(
            "ExecuteCommand", "machine ElapsedVirtualTime"
        )
        match = self.virtual_time_regex.search(elapsed_time_str)
        if match is None:
            return 0

        groups = match.groups()

        h = int(groups[0])
        m = int(groups[1])
        s = float(groups[2])

        return 3600 * h + 60 * m + s

    def init_renode(self):
        """
        Initializes Renode process and starts runtime.
        """
        if self.renode_handler is None:
            raise ValueError("Renode handler not initialized")

        if (
            not self.disable_performance_measurements
            and self.profiler_dump_path is None
        ):
            self.profiler_dump_path = Path(
                tempfile.mktemp(prefix="renode_profiler_", suffix=".dump")
            )
        self.renode_handler.initialize(read_renode_stdout=True)
        self.renode_handler.run_robot_keyword("CreateLogTester", timeout=5.0)
        self.renode_handler.run_robot_keyword(
            "ExecuteCommand", f"$bin=@{self.runtime_binary_path}"
        )
        for dep in self.resc_dependencies:
            dep_name = dep.name.lower().replace(".", "_")
            dep_path = str(dep.resolve())
            self.renode_handler.run_robot_keyword(
                "ExecuteCommand", f"${dep_name}=@{dep_path}"
            )
        self.renode_handler.run_robot_keyword(
            "ExecuteCommand", f"i @{self.platform_resc_path}"
        )
        self.renode_handler.run_robot_keyword("ExecuteCommand", "start")
        if (
            not self.disable_performance_measurements
            and not self.disable_profiler
        ):
            self.renode_handler.run_robot_keyword(
                "ExecuteCommand",
                f"machine EnableProfiler @{self.profiler_dump_path}",
            )
            KLogger.info(f"Profiler dump path: {self.profiler_dump_path}")
        self.renode_handler.run_robot_keyword(
            "ExecuteCommand", "sysbus.vec_controlblock WriteDoubleWord 0xc 0"
        )
        self.renode_handler.run_robot_keyword(
            "WaitForLogEntry", r".*Runtime started.*", treatAsRegex=True
        )

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
        stats_raw = self.renode_handler.run_robot_keyword(
            "ExecuteCommand", "sysbus.cpu GetAllOpcodesCounters"
        )
        # remove double newlines
        stats_raw = stats_raw.replace("\n\n", "\n")
        lines = stats_raw.split("\n")
        # skip the header
        lines = lines[3:]
        stats = {}
        for line in lines:
            line_split = line.split("|")
            if len(line_split) != 4:
                continue
            _, opcode, counter, _ = line.split("|")
            stats[opcode.strip()] = int(counter.strip())

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

    def extract_output(self) -> List[Any]:
        raise NotImplementedError

    def prepare_input(self, input_data: bytes) -> bool:
        raise NotImplementedError

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
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
