# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for platforms that can be simulated with Renode.
"""

import logging
import tempfile
from abc import ABC
from collections import defaultdict
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List, Optional

import kenning.utils.renode_profiler_parser as profiler_parser
from kenning.core.measurements import Measurements, MeasurementsCollector
from kenning.core.platform import Platform
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI

KLogger.add_custom_level(logging.INFO + 1, "RENODE")


class RenodeSimulationError(Exception):
    """
    Exception raised when Renode command fails.
    """

    ...


class SimulatablePlatform(Platform, ABC):
    """
    Wraps the platform that can be simulated in Renode. This class provides
    methods that are necessary to create Renode simulation.
    """

    arguments_structure = {
        "simulated": {
            "description": (
                "If True, then platform will be simulated in Renode"
            ),
            "type": bool,
            "default": True,
        },
        "runtime_binary_path": {
            "description": "Path to runtime binary",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "platform_resc_path": {
            "description": "Path to platform script",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "resc_dependencies": {
            "description": "Renode script dependencies",
            "type": ResourceURI,
            "is_list": True,
            "nullable": True,
            "default": None,
        },
        "post_start_commands": {
            "description": (
                "Renode commands executed after starting the machine"
            ),
            "type": str,
            "is_list": True,
            "nullable": True,
            "default": None,
        },
        "disable_opcode_counters": {
            "description": "Disables opcode counters",
            "type": bool,
            "default": False,
        },
        "disable_profiler": {
            "description": "Disables Renode profiler",
            "type": bool,
            "default": False,
        },
        "profiler_dump_path": {
            "description": "Path to Renode profiler dump",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "profiler_interval_step": {
            "description": "Interval step in ms used to parse profiler data",
            "type": float,
            "default": 10.0,
        },
        "runtime_init_log_msg": {
            "description": (
                "Log message produced by runtime after successful "
                "initialization. If specified, Kenning will wait for "
                "such message before starting inference"
            ),
            "type": str,
            "nullable": True,
            "default": "Inference server started",
        },
        "runtime_init_timeout": {
            "description": "Timeout in seconds for runtime initialization",
            "type": int,
            "default": 30,
        },
    }

    platform_defaults = dict(
        Platform.platform_defaults,
        runtime_log_init_msg="Inference server started",
        runtime_init_timeout=30,
        resc_dependencies=[],
        post_start_commands=[],
    )

    def __init__(
        self,
        name: Optional[str] = None,
        simulated: bool = True,
        runtime_binary_path: Optional[PathOrURI] = None,
        platform_resc_path: Optional[PathOrURI] = None,
        resc_dependencies: Optional[List[ResourceURI]] = None,
        post_start_commands: Optional[List[str]] = None,
        disable_opcode_counters: bool = False,
        disable_profiler: bool = False,
        profiler_dump_path: Optional[Path] = None,
        profiler_interval_step: float = 10.0,
        runtime_init_log_msg: Optional[str] = None,
        runtime_init_timeout: Optional[int] = None,
    ):
        """
        Constructs simulatable platform.

        Parameters
        ----------
        name : Optional[str]
            Name of the platform.
        simulated : bool
            If True, then platform will be simulated in Renode
        runtime_binary_path : Optional[PathOrURI]
            Path to runtime binary.
        platform_resc_path : Optional[PathOrURI]
            Path to the Renode script.
        resc_dependencies : Optional[List[ResourceURI]]
            Renode script dependencies.
        post_start_commands : Optional[List[str]]
            Renode commands executed after starting the machine.
        disable_opcode_counters : bool
            Disables opcode counters.
        disable_profiler : bool
            Disables Renode profiler.
        profiler_dump_path : Optional[Path]
            Path to the Renode profiler dump.
        profiler_interval_step : float
            Interval step in ms used to parse profiler data.
        runtime_init_log_msg : Optional[str]
            Log message produced by runtime after successful initialization. If
            specified, Kenning will wait for such message before starting
            inference.
        runtime_init_timeout : Optional[int]
            Timeout in seconds for runtime initialization.
        """
        self.simulated = simulated
        self.runtime_binary_path = runtime_binary_path
        self.platform_resc_path = platform_resc_path
        self.resc_dependencies = resc_dependencies
        self.post_start_commands = post_start_commands
        self.disable_profiler = disable_profiler
        self.disable_opcode_counters = disable_opcode_counters
        self.profiler_dump_path = profiler_dump_path
        self.profiler_interval_step = profiler_interval_step
        self.runtime_init_log_msg = runtime_init_log_msg
        self.runtime_init_timeout = runtime_init_timeout

        self.machine = None
        self.opcode_counters = None
        self.renode_log_file = None
        self.renode_log_buffer = ""
        self.renode_logs = []
        self.runtime_log_enabled = False
        self.runtime_log_buffer = ""
        self.runtime_logs = []

        super().__init__(name)

    def init(self):
        super().init()
        if self.simulated:
            self._init_renode()
        else:
            self._init_hardware()

        if self.runtime_log_init_msg is not None:
            self._wait_for_runtime_init()

    def deinit(self, measurements: Measurements):
        if self.simulated:
            self.handle_renode_logs()

            if not self.disable_opcode_counters:
                measurements += {
                    "opcode_counters": self._opcode_stats_diff(
                        self.opcode_counters, self._get_opcode_stats()
                    )
                }

            self._clear_renode()

            if not self.disable_profiler:
                measurements += self._get_profiler_stats()

    def get_time(self):
        if self.simulated:
            if self.machine is None:
                return 0

            return self.machine.ElapsedVirtualTime.TimeElapsed.TotalSeconds

        else:
            return super().get_time()

    def setup_runtime_log_reader(self):
        """
        Setups runtime logs reader.
        """
        ...

    def handle_runtime_logs(self):
        """
        Captures log from runtime.
        """
        ...

    def handle_renode_logs(self):
        """
        Captures log from Renode console.
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
            new_logs = [
                log
                for log in new_logs
                if "Unhandled" not in log and "non existing" not in log
            ]
            self.renode_logs.extend(new_logs)
            for new_log in new_logs:
                KLogger.renode(new_log)

    def inference_step_callback(self):
        if self.simulated:
            self.handle_renode_logs()
        self.handle_runtime_logs()

    def flash_board(self, binary_path: PathOrURI):
        """
        Flashes platform using given binary.

        Parameters
        ----------
        binary_path : PathOrURI
            Path to the binary.
        """
        binary_path = Path(binary_path)

        if self.simulated:
            self.machine.load_elf(str(binary_path.expanduser()))

    def _init_hardware(self):
        ...

    def _init_renode(self):
        """
        Initializes Renode process and starts runtime.
        """
        try:
            from pyrenode3.wrappers import Emulation, Monitor  # isort: skip
        except ImportError as e:
            msg = (
                "Couldn't initialize pyrenode3. "
                "Ensure that Renode is installed according "
                "to the instructions at https://github.com/antmicro/pyrenode3."
            )
            raise Exception(msg) from e

        emu = Emulation()
        monitor = Monitor()

        self.log_file_path = Path(tempfile.mktemp(prefix="renode_log_"))

        monitor.execute(f"logFile @{self.log_file_path.resolve()}")
        self.renode_log_file = open(self.log_file_path, "r")

        self._prepare_renode_platform()

        if not self.disable_profiler:
            self._setup_profiler()

        if not self.disable_opcode_counters:
            self.opcode_counters = self._get_opcode_stats()

        self.setup_runtime_log_reader()

        emu.StartAll()

        for cmd in self.post_start_commands:
            monitor.execute(cmd)

        KLogger.info("Renode initialized")

    def _clear_renode(self):
        """
        Clears Renode simulation.
        """
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

    def _prepare_renode_platform(self):
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
            KLogger.debug(f"Loading RESC dependency {dep_name}={dep_path}")

        _, err = monitor.execute_script(str(self.platform_resc_path.resolve()))

        if err:
            raise RenodeSimulationError("RESC execution error: " + err)

        self.machine = next(iter(emu))

    def _setup_profiler(self):
        """
        Setups Renode profiler.
        """
        if self.profiler_dump_path is None:
            self.profiler_dump_path = Path(
                tempfile.mktemp(prefix="renode_profiler_", suffix=".dump")
            )
        self.profiler_dump_path.parent.mkdir(exist_ok=True)

        self.machine.EnableProfiler(str(self.profiler_dump_path.resolve()))
        KLogger.info(
            f"Profiler dump path: {self.profiler_dump_path.resolve()}"
        )

    def _wait_for_runtime_init(self):
        if self.runtime_log_init_msg is not None and (
            self.runtime_log_enabled or self.simulated
        ):
            KLogger.info("Waiting for runtime init")
            timeout = perf_counter() + self.runtime_init_timeout
            runtime_log_idx = 0
            renode_log_idx = 0
            while True:
                self.handle_renode_logs()
                self.handle_runtime_logs()
                initialized = False
                while runtime_log_idx < len(self.runtime_logs):
                    if (
                        self.runtime_log_init_msg
                        in self.runtime_logs[runtime_log_idx]
                    ):
                        initialized = True
                        break
                    runtime_log_idx += 1

                while renode_log_idx < len(self.renode_logs):
                    if (
                        self.runtime_log_init_msg
                        in self.renode_logs[renode_log_idx]
                    ):
                        initialized = True
                        break
                    renode_log_idx += 1

                if initialized:
                    break

                if perf_counter() > timeout:
                    runtime_logs = "\n".join(self.runtime_logs)
                    renode_logs = "\n".join(self.renode_logs)
                    KLogger.debug(f"Runtime logs:\n{runtime_logs}")
                    KLogger.debug(f"Runtime Renode logs:\n{renode_logs}")
                    raise TimeoutError(
                        "Runtime did not initialize in "
                        f"{self.runtime_init_timeout} seconds"
                    )

                sleep(0.2)

        else:
            # If cannot read logs and await runtime init then just wait for 1
            # second
            sleep(1)

    def _enable_opcode_counters(self):
        # as it is called only from init_renode, we can assume this import
        # will not raise exception
        from Antmicro.Renode.Exceptions import RecoverableException

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
                    KLogger.warning(f"Error enabling {method_name} for {elem}")

    def _get_opcode_stats(self) -> Dict[str, Dict[str, int]]:
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

    def _get_profiler_stats(self) -> Dict[str, List[float]]:
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

        try:
            parsed_stats = profiler_parser.parse(
                str(self.profiler_dump_path.resolve()).encode(),
                start_timestamp,
                end_timestamp,
                self.profiler_interval_step,
            )
        except TypeError:
            KLogger.warning("Could not parse Renode profiler dump")
            parsed_stats = {}
        else:
            KLogger.info("Renode profiler dump parsed")

        if (
            self.profiler_dump_path is not None
            and self.profiler_dump_path.exists()
        ):
            self.profiler_dump_path.unlink()

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
