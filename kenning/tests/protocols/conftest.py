# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import random
import socket

import pytest


def random_network_port() -> int:
    """
    Get random free port number within dynamic port range.

    Returns
    -------
    Optional[int]
        Random free port.
    """
    total_tries = 0
    while total_tries < 100:
        port = random.randint(49152, 60999)
        try:
            # check if port is not used
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            total_tries += 1
            continue

    raise RuntimeError("Giving up to find a free socket port.")


@pytest.fixture
def random_byte_data() -> bytes:
    """
    Generates random data in byte format for tests.

    Returns
    -------
    bytes
        Byte array of random data.
    """
    return bytes(random.choices(range(0, 0xFF), k=random.randint(10, 20)))
