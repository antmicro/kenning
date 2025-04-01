# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for bare-metal platform.
"""

import subprocess
import tempfile
from pathlib import Path
from time import sleep
from typing import List, Optional

from kenning.core.measurements import Measurements
from kenning.platforms.simulatable_platform import SimulatablePlatform
from kenning.platforms.utils import UARTReader
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class BareMetalPlatform(SimulatablePlatform):
    """
    Platform wrapper for bare-metal platform.
    """

    needs_protocol = True

    arguments_structure = {
        "uart_port": {
            "description": "Path to the UART used for communication",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "uart_baudrate": {
            "description": "Baudrate of the UART used for communication",
            "type": int,
            "nullable": True,
            "default": None,
        },
        "uart_log_port": {
            "description": "Path to the UART used for logging",
            "type": Path,
            "nullable": True,
            "default": None,
        },
        "uart_log_baudrate": {
            "description": "Baudrate of the UART used for logging",
            "type": int,
            "nullable": True,
            "default": None,
        },
        "auto_flash": {
            "description": "Automatically flashes platform before evaluating model. If disabled, the hardware is assumed to be flashed",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "openocd_path": {
            "description": "Path to the OpenOCD",
            "type": Path,
            "default": "openocd",
        },
    }

    platform_defaults = dict(
        SimulatablePlatform.platform_defaults,
        uart_baudrate=115200,
        uart_log_baudrate=115200,
    )

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
        uart_port: Optional[Path] = None,
        uart_baudrate: int = None,
        uart_log_port: Optional[Path] = None,
        uart_log_baudrate: int = None,
        auto_flash: bool = False,
        openocd_path: Path = "openocd",
    ):
        """
        Constructs bare-metal platform.

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
        """
        self.uart_port = uart_port
        self.uart_baudrate = uart_baudrate
        self.uart_log_port = uart_log_port
        self.uart_log_baudrate = uart_log_baudrate
        self.auto_flash = auto_flash
        self.openocd_path = openocd_path

        # fields obtained from platforms.yml
        self.display_name = None
        self.flash_size_kb = None
        self.ram_size_kb = None
        self.uart_port_wildcard = None
        self.uart_log_port_wildcard = None
        self.compilation_flags = None
        self.openocd_flash_cmd = None

        self.uart_log_reader = None

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
        )

        if self.simulated:
            if self.uart_port is None or self.uart_log_port is None:
                tmp_uart_dir = Path(tempfile.mkdtemp(prefix="renode_uart_"))
                if self.uart_port is None:
                    self.uart_port = tmp_uart_dir / "uart"
                if self.uart_log_port is None:
                    self.uart_log_port = tmp_uart_dir / "uart_log"

        else:
            if self.uart_port is None and self.uart_port_wildcard is not None:
                self.uart_port = self._find_uart_port(self.uart_port_wildcard)

            if (
                self.uart_log_port is None
                and self.uart_log_port_wildcard is not None
            ):
                self.uart_log_port = self._find_uart_port(
                    self.uart_log_port_wildcard
                )

    def deinit(self, measurements: Measurements):
        self.handle_runtime_logs()
        if self.uart_log_reader is not None:
            self.uart_log_reader.stop()
            self.uart_log_reader = None

        super().deinit(measurements)

    def setup_runtime_log_reader(self):
        if self.uart_log_port is not None and self.uart_log_port.exists():
            self.uart_log_reader = UARTReader(
                str(self.uart_log_port.expanduser()),
                self.uart_log_baudrate,
            )

    def handle_runtime_logs(self):
        if self.uart_log_reader is not None:
            while True:
                logs = self.uart_log_reader.read()
                if not len(logs):
                    break
                self.runtime_log_buffer += logs.decode(errors="ignore")

            *new_logs, self.runtime_log_buffer = self.runtime_log_buffer.split(
                "\n"
            )
            self.runtime_logs.extend(new_logs)
            for new_log in new_logs:
                KLogger.debug(f"UART ({self.uart_log_port}): {new_log}")

    def flash_board(self, binary_path: PathOrURI):
        if self.simulated:
            return super().flash_board(binary_path)

        binary_path = Path(binary_path)

        KLogger.info(f"Flashing board {self.name}")
        self._flash_board_openocd(binary_path)

        # wait for platform boot
        sleep(2)

    def get_default_protocol(self):
        from kenning.protocols.uart import UARTProtocol

        return UARTProtocol(
            port=str(self.uart_port.expanduser()),
            baudrate=self.uart_baudrate,
            timeout=30,
        )

    def _init_hardware(self):
        if self.auto_flash:
            if self.openocd_path is None or not hasattr(
                self, "openocd_flash_cmd"
            ):
                raise RuntimeError(
                    "In order to run automatic flash, path to OpenOCD (openocd_flash_cmd) and flash script (openocd_flash_cmd) need to be provided"  # noqa: E501
                )
            if self.runtime_binary_path is not None:
                self.flash_board(self.runtime_binary_path)
            else:
                KLogger.info(
                    "No binary path specified, assuming board is already flashed"  # noqa: E501
                )

        self.setup_runtime_log_reader()

    def _flash_board_openocd(self, binary_path: Path):
        """
        Flashes board using OpenOCD.

        Parameters
        ----------
        binary_path : Path
            Path to the binary.

        Raises
        ------
        RuntimeError
            Raised when OpenOCD fails.
        """
        openocd_cmd = [str(self.openocd_path.expanduser())]

        for cmd in self.openocd_flash_cmd:
            cmd = cmd.replace("{binary_path}", str(binary_path.expanduser()))
            openocd_cmd.append(f"-c {cmd}")

        result = None
        try:
            result = subprocess.run(openocd_cmd, capture_output=True)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            raise RuntimeError("OpenOCD failed") from e
        except Exception as e:
            KLogger.debug(f"Flashing error {e}")
            raise
        finally:
            if result is not None:
                KLogger.debug(f"OpenOCD output:\n{result.stderr.decode()}")

    @staticmethod
    def _find_uart_port(uart_port_wildcard: str) -> Optional[Path]:
        wildcard_as_path = Path(uart_port_wildcard)
        try:
            ret = next(wildcard_as_path.parent.glob(wildcard_as_path.name))
            KLogger.debug(f"Found UART port {ret}")
            return ret
        except StopIteration:
            KLogger.warning(
                "Board UART port could not be found using wildcard "
                f"{uart_port_wildcard}"
            )
            return None
