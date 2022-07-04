from runtimetests import RuntimeTests
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
from kenning.runtimes.iree import IREERuntime
from kenning.compilers.tflite import TFLiteCompiler
from kenning.compilers.tvm import TVMCompiler
from tvm import TVMError
import pytest
import numpy as np


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

    def test_prepare_model_bytes_one(self):
        # Try to load from incorrect byte stream
        runtime = self.initruntime()
        with pytest.raises(ValueError):
            runtime.prepare_model(b'Kenning')

    def test_prepare_model_bytes_two(self):
        # Load from empty byte string
        runtime = self.initruntime()
        assert runtime.prepare_model(b'') is True

        # Check if written file is empty
        with open(self.runtimemodel, 'rb') as modelfile:
            assert b'' != modelfile.read()

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

    def test_prepare_model_bytes_one(self):
        # Load from empty byte stream
        runtime = self.initruntime()
        assert runtime.prepare_model(b'') is True

        # Check if written file is not empty
        with open(self.runtimemodel, 'rb') as modelfile:
            assert b'' != modelfile.read()

    def test_prepare_model_bytes_two(self):
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


@pytest.mark.xfail
class TestIREERuntime(RuntimeTests):
    runtimecls = IREERuntime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(),
                                  self.modelpath, *args, **kwargs)
        return runtime
