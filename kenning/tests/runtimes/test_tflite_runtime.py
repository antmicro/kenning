from runtimetests import RuntimeWithModel
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.compilers.tflite import TFLiteCompiler
import pytest


@pytest.mark.parametrize('runtimemodel', [TFLiteCompiler], indirect=True)
class TestTFLiteRuntime(RuntimeWithModel):
    runtimecls = TFLiteRuntime
