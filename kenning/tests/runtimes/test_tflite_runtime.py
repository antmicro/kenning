# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from runtimetests import RuntimeWithModel
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.compilers.tflite import TFLiteCompiler
import pytest


@pytest.mark.xdist_group(name='use_resources')
@pytest.mark.parametrize('runtimemodel', [TFLiteCompiler], indirect=True)
class TestTFLiteRuntime(RuntimeWithModel):
    runtimecls = TFLiteRuntime
