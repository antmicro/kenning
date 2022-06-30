from kenning.core.runtime import Runtime
import pytest


class TestCoreRuntime:
    @pytest.mark.xfail()
    def test_constructor(self):
        assert 0

    @pytest.mark.xfail()
    def test_inference_session_start(self):
        assert 0

    @pytest.mark.xfail()
    def test_inference_session_end(self):
        assert 0

    @pytest.mark.xfail()
    def test_close_server(self):
        assert 0

    def test_prepare_server(self):
        runtime = Runtime(None)
        with pytest.raises(AttributeError):
            runtime.prepare_server()

    @pytest.mark.xfail()
    def test_prepare_client(self):
        assert 0

    @pytest.mark.xfail()
    def test_prepare_input(self):
        assert 0

    @pytest.mark.xfail()
    def test_prepare_model(self):
        assert 0

    @pytest.mark.xfail()
    def test_process_input(self):
        assert 0

    @pytest.mark.xfail()
    def test_run(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_output(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_stats(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_essentials(self):
        assert 0

    @pytest.mark.xfail()
    def test_prepare_local(self):
        assert 0

    @pytest.mark.xfail()
    def test_run_locally(self):
        assert 0

    @pytest.mark.xfail()
    def test_run_client(self):
        assert 0

    @pytest.mark.xfail()
    def test_run_server(self):
        assert 0
