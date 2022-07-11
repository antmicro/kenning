from kenning.compilers.tflite import TFLiteCompiler
from kenning.compilers.tvm import TVMCompiler
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.runtimes.iree import IREERuntime
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
from runtimetests import RuntimeTests
from tvm import TVMError
import numpy as np
import pytest
import random
import uuid


class TestCoreRuntime(RuntimeTests):
    runtimecls = Runtime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(), *args, **kwargs)
        return runtime

    def test_prepare_input(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_input(b'')

    def test_prepare_model(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model(b'')

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)

    def test_run(self):
        """
        Tests the `Runtime.run()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.run()

    def test_upload_output(self):
        """
        Tests the `Runtime.upload_output()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.upload_output(b'')

    def test_prepare_local(self):
        """
        Tests the `Runtime.prepare_local()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model(b'')

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)


@pytest.mark.usefixtures('runtimemodel')
@pytest.mark.parametrize('runtimemodel', [TFLiteCompiler], indirect=True)
class TestTFLiteRuntime(RuntimeTests):
    runtimecls = TFLiteRuntime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(),
                                  self.runtimemodel,
                                  *args, **kwargs)
        return runtime

    def test_prepare_model(self):
        # Load from file
        runtime = self.initruntime()
        runtime.prepare_model(None)

        # Load from empty byte string
        runtime = self.initruntime()
        assert runtime.prepare_model(b'') is True

        # Doesn't overwrites model file as bytestream is empty
        # Check if written file is empty
        with open(self.runtimemodel, 'rb') as modelfile:
            assert b'' != modelfile.read()

        # Overwrites model file
        # Try to load from incorrect byte stream
        runtime = self.initruntime()
        with pytest.raises(ValueError):
            runtime.prepare_model(b'Kenning')

    def test_prepare_input(self):
        # Correct data, but with wrong shape and datatype
        data = np.arange(100).tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        output = runtime.prepare_input(data)
        assert output is True

        # Correct input shape and datatype
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        assert runtime.prepare_input(data) is True

        # Input has incorrect data type
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.prepare_input(b'')

    def test_run(self):
        # Run without model
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.run()

        # Run without any input
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.run()

        # Run with prepared input
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_input(data)
        runtime.run()

    def test_prepare_local(self):
        runtime = self.initruntime()
        runtime.prepare_local()

        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_local()

    def test_upload_output(self):
        # Test on no model
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.upload_output(b'')

        # Test with model and input
        data = np.zeros(self.inputshapes, dtype=np.float32).tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_input(data)
        runtime.run()
        expected_data = np.zeros(self.outputshapes, dtype=np.float32).tobytes()
        assert runtime.upload_output(b'') == expected_data


@pytest.mark.usefixtures('runtimemodel')
@pytest.mark.parametrize('runtimemodel', [TVMCompiler], indirect=True)
class TestTVMRuntime(RuntimeTests):
    runtimecls = TVMRuntime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(),
                                  self.runtimemodel,
                                  *args, **kwargs)
        return runtime

    def test_prepare_model(self):
        # Load from file
        runtime = self.initruntime()
        assert runtime.prepare_model(None) is True

        # Doesn't overwrites model file because bytestream is empty
        # Load from empty byte stream
        runtime = self.initruntime()
        assert runtime.prepare_model(b'') is True

        # Check if written file is not empty
        with open(self.runtimemodel, 'rb') as modelfile:
            assert b'' != modelfile.read()

    def test_prepare_model_bytes(self):
        # Overwrites model file
        # Try to load from incorrect byte stream
        runtime = self.initruntime()
        with pytest.raises(TVMError):
            runtime.prepare_model(b'Kenning') is False

    def test_prepare_input(self):
        # No model initialized
        data = np.arange(100).tobytes()
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.prepare_input(data)

        # Model is initialized but input is with wrong shape and datatype
        data = np.arange(100).tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        output = runtime.prepare_input(data)
        assert output is False

        # Correct input shape, but wrong datatype
        data = np.arange(25).reshape(self.inputshapes).tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        output = runtime.prepare_input(data)
        assert output is False

        # Correct input shape and datatype
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        output = runtime.prepare_input(data)
        assert output is True

        # Input is empty
        data = b''
        runtime = self.initruntime()
        runtime.prepare_model(None)
        assert runtime.prepare_input(data) is False

        # Incorrect input in byte string
        data = b'Kenning'
        runtime = self.initruntime()
        runtime.prepare_model(None)
        with pytest.raises(ValueError):
            runtime.prepare_input(data)

    def test_run(self):
        # Run without model
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.run()

        # Run without any input
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.run()

        # Run with prepared input
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_input(data)
        runtime.run()

    def test_prepare_local(self):
        runtime = self.initruntime()
        runtime.prepare_local()

        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_local()

    def test_upload_output(self):
        # Test on no model
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.upload_output(b'')

        # Test with model and input
        data = np.zeros((self.inputshapes), dtype=np.float32).tobytes()
        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_input(data)
        runtime.run()
        expected_data = np.zeros(self.outputshapes, dtype=np.float32).tobytes()
        assert runtime.upload_output(b'') == expected_data


class TestTFLiteRuntimeNetwork(TestTFLiteRuntime):
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1234

    def initruntime(self, *args, **kwargs):
        protocol = self.runtimeprotocolcls(self.host, self.port)
        runtime = self.runtimecls(protocol, self.runtimemodel, *args, **kwargs)
        return runtime

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
    def server(self):
        server = self.runtimeprotocolcls(self.host, self.port)
        server.initialize_server()
        yield server
        server.disconnect()

    def test_upload_essentials(self, server, tmpfolder):
        """
        Tests the `Runtime.upload_essentials()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        tmpfolder : Path
            Fixture to get temporary folder for model
        """
        runtime = self.initruntime()
        path = tmpfolder / uuid.uuid4().hex
        data = self.generate_byte_data()
        with open(path, 'w') as model:
            print(data, file=model)
        runtime.prepare_client()
        server.accept_client(server.serversocket, None)
        server.send_message(MessageType.OK)
        runtime.upload_essentials(path)

    def test_process_input(self, server):
        """
        Tests the `Runtime.process_input()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        """
        runtime = self.initruntime()
        data = self.generate_byte_data()
        runtime.prepare_client()
        assert runtime.prepare_model(None) is True
        runtime.process_input(data)

    def test_prepare_client(self, server):
        """
        Tests the `Runtime.prepare_client()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        """
        runtime = self.initruntime()
        runtime.prepare_client()
        assert runtime.protocol.serversocket is None
        assert runtime.protocol.socket is not None

    def test_prepare_server(self):
        """
        Tests the `Runtime.prepare_server()` method.
        """
        runtime = self.initruntime()
        runtime.prepare_server()
        assert runtime.protocol.serversocket is not None
        assert runtime.protocol.socket is None


class TestTVMRuntimeNetwork(TestTVMRuntime):
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1234

    def initruntime(self, *args, **kwargs):
        protocol = self.runtimeprotocolcls(self.host, self.port)
        runtime = self.runtimecls(protocol, self.runtimemodel, *args, **kwargs)
        return runtime

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
    def server(self):
        server = self.runtimeprotocolcls(self.host, self.port)
        server.initialize_server()
        yield server
        server.disconnect()

    def test_upload_essentials(self, server, tmpfolder):
        """
        Tests the `Runtime.upload_essentials()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        tmpfolder : Path
            Fixture to get temporary folder for model
        """
        runtime = self.initruntime()
        path = tmpfolder / uuid.uuid4().hex
        data = self.generate_byte_data()
        with open(path, 'w') as model:
            print(data, file=model)
        runtime.prepare_client()
        server.accept_client(server.serversocket, None)
        server.send_message(MessageType.OK)
        runtime.upload_essentials(path)

    def test_process_input(self, server):
        """
        Tests the `Runtime.process_input()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        """
        runtime = self.initruntime()
        data = self.generate_byte_data()
        runtime.prepare_client()
        assert runtime.prepare_model(None) is True
        runtime.process_input(data)

    def test_prepare_client(self, server):
        """
        Tests the `Runtime.prepare_client()` method.

        Parameters
        ----------
        server : RuntimeProtocol
            Fixture to get NetworkProtocol server
        """
        runtime = self.initruntime()
        runtime.prepare_client()
        assert runtime.protocol.serversocket is None
        assert runtime.protocol.socket is not None

    def test_prepare_server(self):
        """
        Tests the `Runtime.prepare_server()` method.
        """
        runtime = self.initruntime()
        runtime.prepare_server()
        assert runtime.protocol.serversocket is not None
        assert runtime.protocol.socket is None


@pytest.mark.xfail
class TestIREERuntime(RuntimeTests):
    runtimecls = IREERuntime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(),
                                  self.modelpath, *args, **kwargs)
        return runtime
