from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol


class TestNetworkProtocol(RuntimeProtocolTests):
    runtimeprotocolcls = NetworkProtocol
