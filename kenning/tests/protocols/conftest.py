# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import random
import socket
from typing import Optional

import pytest


def random_network_port() -> Optional[int]:
    """
    Get random free port number within dynamic port range.

    Returns
    -------
    Optional[int] :
        Random free port.
    """
    ports = random.sample(range(49152, 65535), k=5)

    for port in ports:
        try:
            # check if port is not used
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            break
        except OSError:
            continue
    else:
        return None

    return port


@pytest.fixture
def random_byte_data() -> bytes:
    """
    Generates random data in byte format for tests.

    Returns
    -------
    bytes :
        Byte array of random data.
    """
    return bytes(random.choices(range(0, 0xFF), k=random.randint(10, 20)))
