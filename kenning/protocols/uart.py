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
from typing import Optional, Tuple

import numpy as np
import serial

from kenning.core.measurements import Measurements
from kenning.interfaces.io_interface import IOInterface
from kenning.protocols.bytes_based_protocol import (
    BytesBasedProtocol,
    Message,
    MessageType,
    ServerStatus,
)
from kenning.utils.logger import KLogger

BARE_METAL_IREE_ALLOCATION_STATS_SIZE = 24
RUNTIME_STAT_NAME_MAX_LEN = 32

RUNTIME_STAT_FMT = f"{RUNTIME_STAT_NAME_MAX_LEN}sQQ"
STAT_SIZE = struct.calcsize(RUNTIME_STAT_FMT)


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
            "allocations": {
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

                if "allocations" not in stats_json:
                    stats_json["allocations"] = {}
                stats_json["allocations"][stat_name] = stat_value

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


class UARTProtocol(BytesBasedProtocol):
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
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: int = -1,
        packet_size: int = 4096,
        endianness: str = "little",
    ):
        """
        Initializes UARTProtocol.

        Parameters
        ----------
        port : str
            UART port.
        baudrate : int
            UART baudrate.
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        packet_size : int
            Size of the packet.
        endianness : str
            Endianness of the communication.
        """
        self.port = port
        self.baudrate = baudrate
        self.collecteddata = bytes()
        self.connection = None
        super().__init__(
            timeout=timeout, packet_size=packet_size, endianness=endianness
        )

    def initialize_client(self) -> bool:
        self.connection = serial.Serial(self.port, self.baudrate, timeout=0)
        self.selector.register(
            self.connection,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data,
        )
        return self.connection.is_open

    def send_data(self, data: bytes) -> bool:
        if self.connection is None or not self.connection.is_open:
            return False
        return len(data) == self.connection.write(data)

    def receive_data(
        self, connection: serial.Serial, mask: int
    ) -> Tuple[ServerStatus, Optional[bytes]]:
        if self.connection is None or not self.connection.is_open:
            return ServerStatus.CLIENT_DISCONNECTED, None

        data = self.connection.read(self.packet_size)

        return ServerStatus.DATA_READY, data

    def disconnect(self):
        if self.connection is not None or self.connection.is_open:
            self.connection.close()

    def upload_io_specification(self, path: Path) -> bool:
        KLogger.debug("Uploading io specification")
        with open(path, "rb") as io_spec_file:
            io_spec = json.load(io_spec_file)

        data = IOInterface.serialize_io_specification_for_uart(io_spec)

        message = Message(MessageType.IO_SPEC, data)

        self.send_message(message)
        return self.receive_confirmation()[0]

    def download_statistics(self, final: bool = False) -> "Measurements":
        KLogger.debug("Downloading statistics")
        self.send_message(Message(MessageType.STATS))
        status, data = self.receive_confirmation()
        measurements = Measurements()
        if status and isinstance(data, bytes) and len(data) > 0:
            measurements += _parse_stats(data, final=final)
        return measurements

    def initialize_server(self) -> bool:
        raise NotImplementedError
