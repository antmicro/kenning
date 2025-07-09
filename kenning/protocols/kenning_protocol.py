# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Kenning Protocol (a communication protocol for
exchanging inference data between devices).
"""



from abc import ABC

from kenning.protocols.bytes_based_protocol import BytesBasedProtocol
from kenning.protocols.message import (
    Message,
    MessageType,
)

class KenningProtocol(BytesBasedProtocol, ABC):
    """
    Class for managing the flow of Kenning Protocol (unimplemented yet).
    """

    ...
