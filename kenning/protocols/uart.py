# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
UART-based inference communication protocol.
"""

import enum
import json
import selectors
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import serial

from kenning.core.exceptions import NotSupportedError
from kenning.core.measurements import Measurements
from kenning.core.protocol import ServerAction, ServerStatus
from kenning.interfaces.io_interface import IOInterface
from kenning.protocols.bytes_based_protocol import (
    TransmissionFlag,
)
from kenning.protocols.kenning_protocol import (
    KenningProtocol,
)
from kenning.protocols.message import (
    MessageType,
)
from kenning.utils.logger import KLogger

BARE_METAL_IREE_ALLOCATION_STATS_SIZE = 24
RUNTIME_STAT_NAME_MAX_LEN = 32
RUNTIME_ALLOCATION_STATS_KEY = "allocation_stats"

RUNTIME_STAT_FMT = f"{RUNTIME_STAT_NAME_MAX_LEN}sQQ"
STAT_SIZE = struct.calcsize(RUNTIME_STAT_FMT)

# How many seconds to wait for the client to respond to our connection attempt.
CONNECTION_TIMEOUT = 1
# How many times to re-try estabilishing a connection before throwing an error.
CONNECTION_RETRY = 5


class RuntimeStatType(enum.IntEnum):
    """An enum of statistic types returned by the protocol."""

    RUNTIME_STATISTICS_DEFAULT = 0
    RUNTIME_STATISTICS_ALLOCATION = 1
    RUNTIME_STATISTICS_INFERENCE_TIME = 2


def _parse_stats(data: bytes, final: bool = False) -> dict:
    """
    Method used to parse allocation stats sent by runtime.

    Parameters
    ----------
    data : bytes
        Byte array with stats sent by runtime.

    final : bool
            If the inference is finished

    Returns
    -------
    dict
        Parsed stats.

    Raises
    ------
    ValueError
        Raised when passed argument is of invalid size
    """
    if len(data) == BARE_METAL_IREE_ALLOCATION_STATS_SIZE:
        stats = np.frombuffer(data, dtype=np.uint32, count=6)
        stats_json = {
            RUNTIME_ALLOCATION_STATS_KEY: {
                "host_bytes_peak": int(stats[0]),
                "host_bytes_allocated": int(stats[1]),
                "host_bytes_freed": int(stats[2]),
                "device_bytes_peak": int(stats[3]),
                "device_bytes_allocated": int(stats[4]),
                "device_bytes_freed": int(stats[5]),
            }
        }
    elif len(data) % STAT_SIZE == 0:
        stats_json = dict()

        for i in range(0, len(data), STAT_SIZE):
            stat_name, stat_type, stat_value = struct.unpack(
                RUNTIME_STAT_FMT, data[i : i + STAT_SIZE]
            )
            stat_name = stat_name.strip(b"\x00").decode()

            if stat_type == RuntimeStatType.RUNTIME_STATISTICS_DEFAULT:
                if final is not False:
                    continue

                stats_json[stat_name] = stat_value

            elif stat_type == RuntimeStatType.RUNTIME_STATISTICS_ALLOCATION:
                if final is not True:
                    continue

                if RUNTIME_ALLOCATION_STATS_KEY not in stats_json:
                    stats_json[RUNTIME_ALLOCATION_STATS_KEY] = {}
                stats_json[RUNTIME_ALLOCATION_STATS_KEY][
                    stat_name
                ] = stat_value

            elif (
                stat_type == RuntimeStatType.RUNTIME_STATISTICS_INFERENCE_TIME
            ):
                if final is not False:
                    continue

                stats_json[stat_name] = [stat_value / 1e9]

            else:
                raise ValueError(f"Invalid stat type: {stat_type}")
    else:
        raise ValueError(f"Invalid allocations stats size: {len(data)}")

    return stats_json


class UARTProtocol(KenningProtocol):
    """
    An UART-base protocol. It supports only client-side as a server is
    expected to be bare-metal platform.

    It is implemented using pyserial.
    """

    arguments_structure = {
        "port": {
            "description": "The target device name",
            "type": str,
            "required": True,
        },
        "baudrate": {
            "description": "The baud rate",
            "type": int,
            "default": 9600,
        },
        "endianness": {
            "description": "Endianness of the bytes",
            "type": str,
            "default": "little",
        },
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        error_recovery: bool = True,
        timeout: int = -1,
        packet_size: int = 4096,
    ):
        """
        Initializes UARTProtocol.

        Parameters
        ----------
        port : str
            UART port.
        baudrate : int
            UART baudrate.
        error_recovery: bool
            True if checksum verification and error recovery mechanisms are to
            be turned on.
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        packet_size : int
            Size of the packet.
        """
        self.port = port
        self.baudrate = baudrate
        self.collecteddata = bytes()
        self.connection = None
        self.packet_size = packet_size
        super().__init__(timeout, error_recovery)

    def initialize_client(self) -> bool:
        self.connection = serial.Serial(self.port, self.baudrate, timeout=0)
        self.selector.register(
            self.connection,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data,
        )
        if self.connection.is_open:
            self.start()
            _, flags = self.request_blocking(
                MessageType.PING,
                None,
                None,
                [TransmissionFlag.SUCCESS],
                CONNECTION_TIMEOUT,
                CONNECTION_RETRY,
            )
            if flags is None:
                KLogger.error(
                    "Device not responding to connection requests. Device"
                    " reset might be necessary."
                )
                return False
            if TransmissionFlag.FAIL in flags:
                KLogger.error("Device refused connection.")
                return False
            return True
        else:
            return False

    def send_data(self, data: bytes) -> bool:
        if self.connection is None or not self.connection.is_open:
            return False
        return len(data) == self.connection.write(data)

    def receive_data(
        self, connection: serial.Serial, mask: int
    ) -> Optional[bytes]:
        if self.connection is None or not self.connection.is_open:
            return None
        data = self.connection.read(self.packet_size)

        return data

    def disconnect(self):
        if self.connection is not None and self.connection.is_open:
            _, flags = self.request_blocking(
                MessageType.PING,
                None,
                None,
                [TransmissionFlag.FAIL],
                CONNECTION_TIMEOUT,
                CONNECTION_RETRY,
            )
            if flags is None:
                KLogger.error(
                    "Device is not responding to disconnect requests. Device"
                    " reset might be necessary."
                )
            self.stop()
            self.connection.close()

    def initialize_server(self) -> bool:
        raise NotSupportedError(
            "Kenning Server cannot use UART (the protocol is client-side only)"
        )

    def upload_io_specification(self, path: Path) -> bool:
        KLogger.debug("Uploading io specification")
        with open(path, "rb") as io_spec_file:
            io_spec = json.load(io_spec_file)

        data = IOInterface.serialize_io_specification_for_uart(io_spec)

        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_IOSPEC),
            self.request_blocking(MessageType.IO_SPEC, None, data),
        )

    def download_statistics(self, final: bool = False) -> Measurements:
        measurements = Measurements()
        if final is False:
            return measurements

        KLogger.debug("Downloading statistics")

        data, flags = self.request_blocking(MessageType.STATS)
        measurements = Measurements()
        if (
            TransmissionFlag.SUCCESS in flags
            and isinstance(data, bytes)
            and len(data) > 0
        ):
            measurements += _parse_stats(data, final=final)
        return measurements
