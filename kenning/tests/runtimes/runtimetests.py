import pytest
import numpy as np
from tvm import TVMError
from typing import Type
from abc import abstractmethod
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import MeasurementsCollector
from pytest_mock import MockerFixture
from pytest import LogCaptureFixture


@pytest.mark.fast
class RuntimeTests:
    runtimecls: Type[Runtime]
    runtimeprotocolcls: Type[RuntimeProtocol]

    @abstractmethod
    def initruntime(self, *args, **kwargs):
        raise NotImplementedError

    def test_inference_session_start(self):
        """
        Tests the `Runtime.inference_session_start()` method.
        """
        runtime = self.initruntime(collect_performance_data=False)
        runtime.inference_session_start()
        assert runtime.statsmeasurements is None
        MeasurementsCollector.clear()

        runtime = self.initruntime(collect_performance_data=True)
        runtime.inference_session_start()
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None
        MeasurementsCollector.clear()

    def test_inference_session_end(self):
        """
        Tests the `Runtime.inference_session_end()` method.
        """
        runtime = self.initruntime()
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None

    def test_close_server(self):
        """
        Tests the `Runtime.close_server()` method.
        """
        runtime = self.initruntime()
        assert runtime.shouldwork is True

        runtime.close_server()
        assert runtime.shouldwork is False

    def test_prepare_server(self):
        """
        Tests the `Runtime.prepare_server()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_server()

    def test_prepare_client(self):
        """
        Tests the `Runtime.prepare_client()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_client()

    def test_prepare_input(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        raise NotImplementedError

    def test_prepare_model(self):
        """
        Tests the `Runtime.prepare_model()` method.
        """
        raise NotImplementedError

    def test_process_input(self, mocker: MockerFixture,
                           caplog: LogCaptureFixture):
        """
        Tests the `Runtime.process_input()` method.

        Parameters
        ----------
        mocker: MockerFixture
            Fixture to provide changes to source code
        caplog: LogCaptureFixture
            Fixture to read logs
        """
        mocker.patch(
            'kenning.core.runtimeprotocol.RuntimeProtocol.request_success',
            lambda x: None
                     )
        mocker.patch(
            'kenning.core.runtime.Runtime._run',
            lambda x: None
                     )

        import logging
        log_messages = ('Processing input', 'Input processed')
        runtime = self.initruntime()
        caplog.set_level(logging.DEBUG)
        runtime.process_input(b'')
        for i in range(len(log_messages)):
            assert caplog.records[i].msg == log_messages[i]

    def test_run(self):
        """
        Tests the `Runtime.run()` method.
        """
        raise NotImplementedError

    def test_upload_output(self):
        """
        Tests the `Runtime.upload_output()` method.
        """
        raise NotImplementedError

    def test_upload_stats(self):
        """
        Tests the `Runtime.upload_stats()` method.
        """
        runtime = self.initruntime()
        assert b'{}' == runtime.upload_stats(b'')

    def test_upload_essentials(self, mocker: MockerFixture):
        """
        Tests the `Runtime.upload_essentials()` method.

        Parameters
        ----------
        mocker: MockerFixture
            Fixture to provide changes to source code
        """
        from pathlib import Path
        import uuid

        runtime = self.initruntime()
        path = Path(uuid.uuid4().hex)

        def moc_func(self, modelpath: Path):
            assert modelpath == path
            return

        mocker.patch(
            'kenning.core.runtimeprotocol.RuntimeProtocol.upload_model',
            moc_func)
        runtime.upload_essentials(path)

    def test_prepare_local(self):
        """
        Tests the `Runtime.prepare_local()` method.
        """
        raise NotImplementedError


@pytest.mark.usefixtures('runtimemodel')
class RuntimeWithModel(RuntimeTests):
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
        with pytest.raises((ValueError, TVMError)):
            runtime.prepare_model(b'Kenning') is False

    def test_prepare_input(self):
        # No model initialized
        data = np.arange(100).tobytes()
        runtime = self.initruntime()
        with pytest.raises((TypeError, AttributeError)):
            runtime.prepare_input(data)

        # For now we got rid of inpudtype argument from runtimes
        # Model is initialized but input is with wrong shape and datatype
        data = np.arange(99, dtype=np.int8).tobytes()
        runtime = self.initruntime(inputdtype=['float32'])
        runtime.prepare_model(None)
        with pytest.raises(ValueError):
            output = runtime.prepare_input(data)

        # Correct input shape and datatype
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_local()
        output = runtime.prepare_input(data)
        assert output is True

        # Input is empty
        data = b''
        runtime = self.initruntime()
        runtime.prepare_local()
        assert runtime.prepare_input(data) is False

    def test_run(self):
        # Run without model
        runtime = self.initruntime()
        with pytest.raises(AttributeError):
            runtime.run()

        # Run without any input
        runtime = self.initruntime()
        runtime.prepare_local()
        runtime.run()

        # Run with prepared input
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_local()
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
        runtime.prepare_local()
        runtime.prepare_input(data)
        runtime.run()
        expected_data = np.zeros(self.outputshapes, dtype=np.float32).tobytes()
        assert runtime.upload_output(b'') == expected_data
