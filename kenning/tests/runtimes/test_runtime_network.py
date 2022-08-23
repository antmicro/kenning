from kenning.compilers.tflite import TFLiteCompiler
from kenning.compilers.tvm import TVMCompiler
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import MessageType
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
import random
import uuid
import pytest


runtimes = [(TFLiteRuntime, TFLiteCompiler), (TVMRuntime, TVMCompiler)]


@pytest.mark.usefixtures('runtimemodel')
@pytest.mark.parametrize('runtimecls,runtimemodel', runtimes,
                         indirect=['runtimemodel'])
@pytest.mark.fast
class TestRuntimeNetwork:
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1235

    def generate_byte_data(self):
        """
        Generates random data in bytes.

        Returns
        ------
        bytes: Generated sequence of bytes
        """
        data = bytes()
        for i in range(random.randint(1, 1000)):
            data += random.randint(0, 9999).to_bytes(4, 'little',
                                                     signed=False)
        return data

    @pytest.fixture
    def runtime(self, runtimecls):
        """
        Initializes runtime object.

        Returns
        -------
        Runtime : Initialized runtime object.
        """
        protocol = self.runtimeprotocolcls(self.host, self.port)
        runtimeobj = runtimecls(protocol, self.runtimemodel)
        yield runtimeobj
        runtimeobj.protocol.disconnect()

    @pytest.fixture
    def server(self):
        """
        Initializes server object.

        Returns
        -------
        RuntimeProtocol : Initialized NetworkProtocol server
        """
        serverobj = self.runtimeprotocolcls(self.host, self.port)
        serverobj.initialize_server()
        yield serverobj
        serverobj.disconnect()

    def test_upload_essentials(self, server, runtime, tmpfolder):
        """
        Tests the `Runtime.upload_essentials()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        runtime : Runtime
            Fixture to get Runtime object
        tmpfolder : Path
            Fixture to get temporary folder for model
        """
        path = tmpfolder / uuid.uuid4().hex
        data = self.generate_byte_data()
        with open(path, 'w') as model:
            print(data, file=model)
        runtime.prepare_client()
        server.accept_client(server.serversocket, None)
        server.send_message(MessageType.OK)
        runtime.upload_essentials(path)

    def test_process_input(self, server, runtime):
        """
        Tests the `Runtime.process_input()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        runtime : Runtime
            Fixture to get Runtime object
        """
        data = self.generate_byte_data()
        runtime.prepare_client()
        assert runtime.prepare_model(None) is True
        assert runtime.prepare_input(data) is True
        runtime.process_input(b'')

    def test_prepare_client(self, server, runtime):
        """
        Tests the `Runtime.prepare_client()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        runtime : Runtime
            Fixture to get Runtime object
        """
        runtime.prepare_client()
        assert runtime.protocol.serversocket is None
        assert runtime.protocol.socket is not None

    def test_prepare_server(self, runtime):
        """
        Tests the `Runtime.prepare_server()` method.

        Parameters
        ----------
        runtime : Runtime
            Fixture to get Runtime object
        """
        runtime.prepare_server()
        assert runtime.protocol.serversocket is not None
        assert runtime.protocol.socket is None
