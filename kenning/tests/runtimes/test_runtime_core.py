# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.core.runtime import Runtime
from kenning.core.protocol import Protocol
from kenning.runtimes.iree import IREERuntime
from runtimetests import RuntimeTests
import pytest
from unittest.mock import patch


@patch.multiple(Runtime, __abstractmethods__=set())
class TestCoreRuntime(RuntimeTests):
    runtimecls = Runtime

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(*args, **kwargs)
        return runtime

    def test_prepare_input(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_input(b'')

    @pytest.mark.xdist_group(name='use_resources')
    def test_prepare_model(self):
        """
        Tests the `Runtime.prepare_input()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model(b'')

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)

    @pytest.mark.xdist_group(name='use_resources')
    def test_run(self):
        """
        Tests the `Runtime.run()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.run()

    @pytest.mark.xdist_group(name='use_resources')
    def test_upload_output(self):
        """
        Tests the `Runtime.upload_output()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.upload_output(b'')

    @pytest.mark.xdist_group(name='use_resources')
    def test_prepare_local(self):
        """
        Tests the `Runtime.prepare_local()` method.
        """
        runtime = self.initruntime()
        with pytest.raises(NotImplementedError):
            runtime.prepare_model(b'')

        with pytest.raises(NotImplementedError):
            runtime.prepare_model(None)


# FIXME: Implement tests for IREECompiler
@pytest.mark.xfail
class TestIREERuntime(RuntimeTests):
    runtimecls = IREERuntime

    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.model_path, *args, **kwargs)
        return runtime
