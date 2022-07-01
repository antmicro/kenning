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
