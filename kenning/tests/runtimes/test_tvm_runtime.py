from runtimetests import RuntimeWithModel
from kenning.runtimes.tvm import TVMRuntime
from kenning.compilers.tvm import TVMCompiler
import pytest


@pytest.mark.parametrize('runtimemodel', [TVMCompiler], indirect=True)
class TestTFLiteRuntime(RuntimeWithModel):
    runtimecls = TVMRuntime
