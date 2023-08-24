# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from runtimetests import RuntimeWithModel
from kenning.runtimes.tvm import TVMRuntime
from kenning.optimizers.tvm import TVMCompiler
import pytest


@pytest.mark.parametrize('runtimemodel', [TVMCompiler], indirect=True)
class TestTVMRuntime(RuntimeWithModel):
    runtimecls = TVMRuntime
