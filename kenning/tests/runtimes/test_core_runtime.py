from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from typing import Type
import pytest


@pytest.mark.parametrize("runtimecls", [(Runtime)])
class TestCoreRuntime:

    def test_inference_session_start(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol(), collect_performance_data=False)
        runtime.inference_session_start()
        assert runtime.statsmeasurements is None

        runtime = runtimecls(RuntimeProtocol(), collect_performance_data=True)
        runtime.inference_session_start()
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None
        # TODO: Test with data is being collected

    def test_inference_session_end(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None
        # TODO: Test with data is being collected

    def test_close_server(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        assert runtime.shouldwork is True

        runtime.close_server()
        assert runtime.shouldwork is False

    def test_prepare_server(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.prepare_server()

    def test_prepare_client(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.prepare_client()

    def test_prepare_input(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.prepare_input('kenning'.encode())

    def test_prepare_model(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.prepare_model('kenning'.encode())

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)

    def test_process_input(self, mocker, caplog, runtimecls: Type[Runtime]):
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
        runtime = runtimecls(RuntimeProtocol())
        caplog.set_level(logging.DEBUG)
        runtime.process_input('kenning'.encode())
        for i in range(len(log_messages)):
            assert caplog.records[i].msg == log_messages[i]

    def test_run(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.run()

    def test_upload_output(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.upload_output('kenning'.encode())

    def test_upload_stats(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        assert b'{}' == runtime.upload_stats('kenning'.encode())
        # TODO: Test on real data with real output

    def test_upload_essentials(self, mocker, runtimecls: Type[Runtime]):
        from pathlib import Path
        import uuid

        runtime = runtimecls(RuntimeProtocol())
        path = Path(uuid.uuid4().hex)

        def moc_func(self, modelpath: Path):
            assert modelpath == path
            return

        mocker.patch(
            'kenning.core.runtimeprotocol.RuntimeProtocol.upload_model',
            moc_func)
        runtime.upload_essentials(path)

    def test_prepare_local(self, runtimecls: Type[Runtime]):
        runtime = runtimecls(RuntimeProtocol())
        with pytest.raises(NotImplementedError):
            runtime.prepare_model('kenning'.encode())

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)
