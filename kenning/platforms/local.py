# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for local CPU platform.
"""

from kenning.core.platform import Platform
from kenning.protocols.network import NetworkProtocol


class LocalPlatform(Platform):
    """
    Local platform.
    """

    def get_default_protocol(self):
        return NetworkProtocol(
            host="127.0.0.1",
            port=12345,
            packet_size=32768,
        )
