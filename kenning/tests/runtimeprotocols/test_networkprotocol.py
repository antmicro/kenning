# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol


class TestNetworkProtocol(RuntimeProtocolTests):
    host = ''
    port = 1234
    runtimeprotocolcls = NetworkProtocol

    def initprotocol(self):
        protocol = self.runtimeprotocolcls(self.host, self.port)
        return protocol
