# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for Zephyr platform.
"""

import re
from pathlib import Path
from typing import List, Optional

from kenning.platforms.bare_metal import BareMetalPlatform
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class UARTNotFoundInDTSError(Exception):
    """
    Exception raised when UART was not found in devicetree.
    """

    ...


class ZephyrPlatform(BareMetalPlatform):
    """
    Platform wrapper for Zephyr platform.
    """

    needs_protocol = True

    arguments_structure = {
        "zephyr_build_path": {
            "description": "Path to Zephyr build directory",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
        "llext_binary_path": {
            "description": "Path to llext binary",
            "type": ResourceURI,
            "nullable": True,
            "default": None,
        },
    }

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
        uart_port: Optional[Path] = None,
        uart_baudrate: int = None,
        uart_log_port: Optional[Path] = None,
        uart_log_baudrate: int = None,
        openocd_path: Path = "openocd",
        zephyr_build_path: Optional[PathOrURI] = None,
        llext_binary_path: Optional[PathOrURI] = None,
    ):
        """
        Constructs Zephyr platform.

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
        uart_port : Optional[Path]
            Path to the UART used for communication.
        uart_baudrate : int
            Baudrate of the UART used for communication.
        uart_log_port : Optional[Path]
            Path to the UART used for logging.
        uart_log_baudrate : int
            Baudrate of the UART used for logging.
        openocd_path : Path
            Path to the OpenOCD.
        zephyr_build_path : Optional[PathOrURI]
            Path to Zephyr build directory.
        llext_binary_path : Optional[PathOrURI]
            Path to runtime binary.
        """
        self.zephyr_build_path = zephyr_build_path
        self.llext_binary_path = llext_binary_path

        super().__init__(
            name=name,
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
            uart_port=uart_port,
            uart_baudrate=uart_baudrate,
            uart_log_port=uart_log_port,
            uart_log_baudrate=uart_log_baudrate,
            openocd_path=openocd_path,
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
                    raise UARTNotFoundInDTSError(
                        f"{alias} UART not found in devicetree"
                    )
                return None

            return alias_match[-1]

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

        KLogger.debug(f"REPL path {repl_path.resolve()}")
        KLogger.debug(f"Binary path{bin_path.resolve()}")

        self.machine = emu.add_mach(board)
        self.machine.load_repl(repl_path.resolve())
        self.machine.load_elf(bin_path.resolve())

        if self.uart_port is not None:
            kcomms_uart = find_uart_by_alias(platform_dts, "kcomms", True)
            emu.CreateUartPtyTerminal(
                "kcomms_uart_term", str(self.uart_port.resolve())
            )
            emu.Connector.Connect(
                getattr(self.machine.sysbus, kcomms_uart).internal,
                emu.externals.kcomms_uart_term,
            )
            KLogger.debug(f"Communication UART: {kcomms_uart}")

        if self.uart_log_port is not None:
            console_uart = find_uart_by_alias(platform_dts, "zephyr,console")
            emu.CreateUartPtyTerminal(
                "console_uart_term", str(self.uart_log_port.resolve())
            )
            emu.Connector.Connect(
                getattr(self.machine.sysbus, console_uart).internal,
                emu.externals.console_uart_term,
            )
            KLogger.debug(f"Logging UART: {console_uart}")

        if not self.disable_opcode_counters:
            self._enable_opcode_counters()
