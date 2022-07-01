import pytest
from typing import Type
from abc import abstractmethod
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import MeasurementsCollector
from pytest_mock import MockerFixture
from pytest import LogCaptureFixture


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
        # TODO: Test with data is being collected
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
        # TODO: Test with data is being collected
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
        runtime.process_input('kenning'.encode())
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
        # TODO: Test on real data with real output
        runtime = self.initruntime()
        assert b'{}' == runtime.upload_stats('kenning'.encode())

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
