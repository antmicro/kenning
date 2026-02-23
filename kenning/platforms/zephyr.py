# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for Zephyr platform.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Optional

from kenning.core.exceptions import TargetPlatformCommunicationError
from kenning.core.measurements import Measurements
from kenning.core.optimizer import Optimizer
from kenning.platforms.bare_metal import BareMetalPlatform
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI
from kenning.utils.zpl_suffix import ZplSuffix


def _prepare_traces(
    last_optimizer: Optimizer,
    prepare_input_path: Path,
    prepare_output_path: Path,
    zephyr_build_path: Optional[Path],
    zephyr_base: Optional[Path],
):
    """
    Constructs and executes trace preparation command.

    Parameters
    ----------
    last_optimizer : Optimizer
        The last optimizer from the optimization pipeline.
    prepare_input_path : Path
        Input path in prepare command.
    prepare_output_path : Path
        Output path in prepare command.
    zephyr_build_path : Optional[Path]
        Path to Zephyr build directory.
    zephyr_base : Optional[Path]
    """
    prepare_cmd = [
        "west",
        "zpl-prepare-trace",
    ]

    if zephyr_build_path:
        prepare_cmd.extend(["--build-dir", str(zephyr_build_path)])

    if zephyr_base:
        prepare_cmd.extend(["--zephyr-base", str(zephyr_base)])

    prepare_cmd.extend(last_optimizer.zpl_prepare_cmd_flags())

    prepare_cmd.extend(
        [
            "-o",
            prepare_output_path,
            prepare_input_path,
        ]
    )

    KLogger.info("Traces conversion started.")

    try:
        subprocess.run(prepare_cmd).check_returncode()
    except subprocess.CalledProcessError as e:
        msg = f"West command: '{' '.join(prepare_cmd)}' failed"
        if e.stderr is not None:
            msg += f" with a message:\n\n{e.stderr.decode()}"
        KLogger.error(msg)
    else:
        KLogger.info("Traces conversion completed.")


class ZephyrPlatform(BareMetalPlatform):
    """
    Platform wrapper for Zephyr platform.
    """

    needs_protocol = True

    arguments_structure = {
        "zephyr_build_path": {
            "description": "Path to Zephyr build directory",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "llext_binary_path": {
            "description": "Path to llext binary",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "sensors": {
            "description": "REPL paths to sensors",
            "type": list[str],
            "nullable": True,
            "default": None,
        },
        "sensors_frequency": {
            "description": "Frequency of sensors data",
            "type": float,
            "nullable": True,
            "default": None,
        },
        "enable_zephelin_gdb": {
            "description": "Enable automatic collection of Zephelin traces"
            "using GDB debugger.",
            "type": bool,
            "default": False,
        },
        "enable_zephelin": {
            "description": "Enable automatic collection of Zephelin traces"
            "through the Protocol.",
            "type": bool,
            "default": False,
        },
        "zpl_use_debug_server": {
            "description": "Optionally force debug server to be switched"
            " on or off. When None, debug server is on when simulated=true"
            " and off otherwise",
            "type": bool,
            "default": False,
        },
        "zephyr_base": {
            "description": "Path to Zephyr base directory",
            "type": Path,
            "nullable": True,
            "default": None,
        },
    }

    def __init__(
        self,
        name: Optional[str] = None,
        platforms_definitions: Optional[List[ResourceURI]] = None,
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
        enable_zephelin_gdb: bool = False,
        enable_zephelin: bool = False,
        gdb_port: int = 3333,
        gdb_binary_name: str = "gdb",
        zpl_use_debug_server: bool = False,
        zephyr_base: Optional[Path] = None,
        uart_port: Optional[Path] = None,
        uart_baudrate: int = None,
        uart_log_port: Optional[Path] = None,
        uart_log_baudrate: int = None,
        auto_flash: bool = False,
        openocd_path: Path = "openocd",
        zephyr_build_path: Optional[Path] = None,
        llext_binary_path: Optional[PathOrURI] = None,
        sensor: Optional[str] = None,
        number_of_batches: int = 16,
        sensors: Optional[list[str]] = None,
        sensors_frequency: Optional[float] = None,
    ):
        """
        Constructs Zephyr platform.

        Parameters
        ----------
        name : Optional[str]
            Name of the platform.
        platforms_definitions : Optional[List[ResourceURI]]
            Files with platform definitions
            from the least to the most significant.
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
        enable_zephelin_gdb : bool
            Enable automatic collection of Zephelin traces through GDB.
        enable_zephelin : bool
            Enable automatic collection of Zephelin traces.
        gdb_port : int
            Port number for collecting traces from GDB server.
        gdb_binary_name : str
            Name of system gdb binary.
        zpl_use_debug_server: bool
            Optionally force debug server to be switched
            on or off. When False, debug server is on when simulated=true
            and off otherwise.
        zephyr_base : Optional[Path]
            Path to Zephyr Base directory.
        uart_port : Optional[Path]
            Path to the UART used for communication.
        uart_baudrate : int
            Baudrate of the UART used for communication.
        uart_log_port : Optional[Path]
            Path to the UART used for logging.
        uart_log_baudrate : int
            Baudrate of the UART used for logging.
        auto_flash : bool
            Automatically flashes platform before evaluating model.
            If disabled, the hardware is assumed to be flashed.
        openocd_path : Path
            Path to the OpenOCD.
        zephyr_build_path : Optional[Path]
            Path to Zephyr build directory.
        llext_binary_path : Optional[PathOrURI]
            Path to runtime binary.
        sensor : Optional[str]
            Name of the sensor.
        number_of_batches : int
            Number of batches available.
        sensors : Optional[list[str]]
            List of sensors used for feeding data
        sensors_frequency: Optional[float]
            Sensor data feeding frequency
        """
        self.zephyr_build_path = zephyr_build_path
        self.llext_binary_path = llext_binary_path
        self.enable_zephelin_gdb = enable_zephelin_gdb
        self.enable_zephelin = enable_zephelin
        if enable_zephelin and enable_zephelin_gdb:
            KLogger.error(
                "Parameters `enable_zephelin` and `enable_zephelin_gdb` are"
                " mutually exclusive."
            )
        self.gdb_binary_name = gdb_binary_name

        self.zephyr_base = zephyr_base

        self.no_dbg_server = (
            simulated if not zpl_use_debug_server else not zpl_use_debug_server
        )

        self.sensors = sensors
        self.sensors_frequency = sensors_frequency

        super().__init__(
            name=name,
            platforms_definitions=platforms_definitions,
            simulated=simulated,
            runtime_binary_path=runtime_binary_path,
            platform_resc_path=platform_resc_path,
            resc_dependencies=resc_dependencies,
            post_start_commands=post_start_commands,
            disable_opcode_counters=disable_opcode_counters,
            disable_profiler=disable_profiler,
            profiler_dump_path=profiler_dump_path,
            profiler_interval_step=profiler_interval_step,
            runtime_init_log_msg=runtime_init_log_msg,
            runtime_init_timeout=runtime_init_timeout,
            gdb_port=gdb_port,
            enable_zephelin_gdb=enable_zephelin_gdb,
            uart_port=uart_port,
            uart_baudrate=uart_baudrate,
            uart_log_port=uart_log_port,
            uart_log_baudrate=uart_log_baudrate,
            auto_flash=auto_flash,
            openocd_path=openocd_path,
            sensor=sensor,
            number_of_batches=number_of_batches,
        )

    def post_init(self):
        super().post_init()
        # We remove old trace data, from the previous inference.
        ctf_file_path = Path(
            ZplSuffix.CTF._get_path_with_suffix(Path(self.measurements_path))
        )
        if ctf_file_path.is_file():
            ctf_file_path.unlink()
        tef_file_path = Path(
            ZplSuffix.TRACE_JSON._get_path_with_suffix(
                Path(self.measurements_path)
            )
        )
        if tef_file_path.is_file():
            tef_file_path.unlink()
        if self.protocol is not None and self.enable_zephelin:

            def dump(bytes):
                with open(ctf_file_path, "ab") as ctf_file:
                    ctf_file.write(bytes)

            self.protocol.listen_to_trace_data(dump)

    def deinit(self, measurements: Measurements):
        if self.enable_zephelin_gdb:
            self._deinit_tracing()
        if self.enable_zephelin:
            _prepare_traces(
                self.last_optimizer,
                str(
                    ZplSuffix.CTF._get_path_with_suffix(
                        Path(self.measurements_path)
                    )
                ),
                str(
                    ZplSuffix.TRACE_JSON._get_path_with_suffix(
                        Path(self.measurements_path)
                    )
                ),
                self.zephyr_build_path,
                self.zephyr_base,
            )
        super().deinit(measurements)

    def _deinit_tracing(self):
        if self.zephyr_build_path is None:
            KLogger.warning("No zephyr_build_path specified.")
            KLogger.warning("The trace will not be captured.")
            return None

        if self.measurements_path is None:
            KLogger.warning("No measurements path specified.")
            KLogger.warning("The trace will not be captured.")
            return None

        prepare_input_path = str(
            ZplSuffix.CTF._get_path_with_suffix(Path(self.measurements_path))
        )

        prepare_output_path = str(
            ZplSuffix.TRACE_JSON._get_path_with_suffix(
                Path(self.measurements_path)
            )
        )

        self.cmd = [
            "west",
            "zpl-gdb-capture",
            prepare_input_path,
            *(["--no-debug-server"] if self.no_dbg_server else []),
            f"--gdb={self.gdb_binary_name}",
            f"--gdb-port={self.gdb_port}",
            "--capture-once",
            "--elf-path="
            f"{str(Path(self.zephyr_build_path / 'zephyr' / 'zephyr.elf'))}",
        ]

        KLogger.info("Traces capture started.")

        self.tracing_subprocess = subprocess.Popen(
            self.cmd,
            stderr=subprocess.PIPE,
        )

        _, stderr = self.tracing_subprocess.communicate()

        KLogger.info("Traces capture completed.")

        if self.tracing_subprocess.returncode != 0:
            msg = f"West command: '{' '.join(self.cmd)}' failed"
            if stderr is not None:
                msg += f" with a message:\n\n{stderr.decode()}"
            KLogger.error(msg)
            return None

        _prepare_traces(
            self.last_optimizer,
            prepare_input_path,
            prepare_output_path,
            self.zephyr_build_path,
            self.zephyr_base,
        )

    def _init_hardware(self):
        if (
            self.runtime_binary_path is None
            and self.zephyr_build_path is not None
        ):
            self.runtime_binary_path = Path(
                self.zephyr_build_path / "zephyr" / "zephyr.hex"
            )

        super()._init_hardware()

    def _prepare_renode_platform(self):
        """
        If Zephyr build path is provided, then prepares platform based on that.
        Otherwise fallbacks to initialization with resc file.
        """
        # as it is called only from init_renode, we can assume this import
        # will not raise exception
        from pyrenode3.wrappers import Emulation

        if self.zephyr_build_path is None:
            return super()._prepare_renode_platform()

        def find_uart_by_alias(
            dts: str, alias: str, raise_exception: bool = False
        ) -> str:
            alias_match = re.findall(
                rf"{alias} = &([a-zA-Z0-9]*);", dts, re.MULTILINE
            )
            if not len(alias_match):
                if raise_exception:
                    raise TargetPlatformCommunicationError(
                        f"{alias} UART not found in Zephyr devicetree"
                    )
                return None

            return alias_match[-1]

        emu = Emulation()

        # find dts, repl and runtime binary
        try:
            dts_path = self.zephyr_build_path / "zephyr" / "zephyr.dts"
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

        KLogger.debug(f"REPL path {repl_path.resolve()}")
        KLogger.debug(f"Binary path {bin_path.resolve()}")

        self.machine = emu.add_mach(board)
        self.machine.load_repl(repl_path.resolve())
        self.machine.load_elf(bin_path.resolve())

        if self.uart_port is not None:
            kcomms_uart = find_uart_by_alias(platform_dts, "kcomms", True)
            KLogger.debug(f"Communication UART: {kcomms_uart}")
            emu.CreateUartPtyTerminal(
                "kcomms_uart_term", str(self.uart_port.resolve())
            )
            emu.Connector.Connect(
                getattr(self.machine.sysbus, kcomms_uart).internal,
                emu.externals.kcomms_uart_term,
            )

        if self.uart_log_port is not None:
            console_uart = find_uart_by_alias(platform_dts, "zephyr,console")
            if console_uart != kcomms_uart:
                KLogger.debug(f"Logging UART: {console_uart}")
                emu.CreateUartPtyTerminal(
                    "console_uart_term", str(self.uart_log_port.resolve())
                )
                emu.Connector.Connect(
                    getattr(self.machine.sysbus, console_uart).internal,
                    emu.externals.console_uart_term,
                )
            else:
                self.zephyr_console_enabled = False
        else:
            self.zephyr_console_enabled = False
        if not self.disable_opcode_counters:
            self._enable_opcode_counters()
