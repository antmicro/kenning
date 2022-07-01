from runtimetests import RuntimeTests
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime
from kenning.runtimes.iree import IREERuntime
from kenning.compilers.tflite import TFLiteCompiler
from kenning.compilers.tvm import TVMCompiler
import pytest


@pytest.mark.fast
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
            runtime.prepare_input('kenning'.encode())

    def test_prepare_model(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model('kenning'.encode())

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
            runtime.upload_output('kenning'.encode())

    def test_prepare_local(self):
        """
        Tests the `Runtime.prepare_local()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model('kenning'.encode())

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
@pytest.mark.xfail
class TestIREERuntime(RuntimeTests):
    runtimecls = IREERuntime
    runtimeprotocolcls = RuntimeProtocol

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimeprotocolcls(),
                                  self.modelpath, *args, **kwargs)
        return runtime
