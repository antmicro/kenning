# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from tvm import TVMError
from typing import Type
from abc import abstractmethod
from kenning.core.runtime import Runtime
from kenning.core.runtime import ModelNotPreparedError
from kenning.core.runtime import InputNotPreparedError
from kenning.core.protocol import Protocol
from kenning.core.measurements import MeasurementsCollector
from pytest_mock import MockerFixture
from pytest import LogCaptureFixture


@pytest.mark.fast
class RuntimeTests:
    runtimecls: Type[Runtime]
    protocolcls: Type[Protocol]

    @abstractmethod
    def initruntime(self, *args, **kwargs):
        raise NotImplementedError

    def test_inference_session_start(self):
        """
        Tests the `Runtime.inference_session_start()` method.
        """
        runtime = self.initruntime(disable_performance_measurements=True)
        runtime.inference_session_start()
        assert runtime.statsmeasurements is None
        MeasurementsCollector.clear()

        runtime = self.initruntime(disable_performance_measurements=False)
        runtime.inference_session_start()
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None
        MeasurementsCollector.clear()

    def test_inference_session_end(self):
        """
        Tests the `Runtime.inference_session_end()` method.
        """
        runtime = self.initruntime()
        runtime.inference_session_end()
        assert runtime.statsmeasurements is None

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
        runtime = self.initruntime()
        assert b'{}' == runtime.upload_stats(b'')

    def test_prepare_local(self):
        """
        Tests the `Runtime.prepare_local()` method.
        """
        raise NotImplementedError


@pytest.mark.usefixtures('runtimemodel')
class RuntimeWithModel(RuntimeTests):
    def initruntime(self, *args, **kwargs):
        runtime = self.runtimecls(self.runtimemodel, *args, **kwargs)
        return runtime

    def test_prepare_model(self):
        # Load from file
        runtime = self.initruntime()
        assert runtime.prepare_model(None) is True

        # Doesn't overwrites model file because bytestream is empty
        # Load from empty byte stream
        runtime = self.initruntime()
        assert runtime.prepare_model(b'') is True

        # Check if written file is not empty
        with open(self.runtimemodel, 'rb') as modelfile:
            assert b'' != modelfile.read()

    def test_prepare_model_bytes(self):
        # Overwrites model file
        # Try to load from incorrect byte stream
        runtime = self.initruntime()
        with pytest.raises((ValueError, TVMError)):
            runtime.prepare_model(b'Kenning') is False

    def test_prepare_input(self):
        # No model initialized
        data = np.arange(100).tobytes()
        runtime = self.initruntime()
        with pytest.raises(ModelNotPreparedError):
            runtime.prepare_input(data)

        # For now we got rid of inpudtype argument from runtimes
        # Model is initialized but input is with wrong shape and datatype
        # data = np.arange(99, dtype=np.int8).tobytes()
        # runtime = self.initruntime(inputdtype=['float32'])
        # runtime.prepare_model(None)
        # with pytest.raises(ValueError):
        #     output = runtime.prepare_input(data)

        # Correct input shape and datatype
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_local()
        output = runtime.prepare_input(data)
        assert output is True

        # Input is empty
        data = b''
        runtime = self.initruntime()
        runtime.prepare_local()
        assert runtime.prepare_input(data) is False

    def test_run(self):
        # Run without model
        runtime = self.initruntime()
        with pytest.raises(ModelNotPreparedError):
            runtime.run()

        # Run without any input
        runtime = self.initruntime()
        runtime.prepare_local()
        runtime.prepare_model(None)
        with pytest.raises(InputNotPreparedError):
            runtime.run()

        # Run with prepared input
        data = np.arange(25, dtype=np.float32).reshape(self.inputshapes)
        data = data.tobytes()
        runtime = self.initruntime()
        runtime.prepare_local()
        runtime.prepare_input(data)
        runtime.run()

    def test_prepare_local(self):
        runtime = self.initruntime()
        runtime.prepare_local()

        runtime = self.initruntime()
        runtime.prepare_model(None)
        runtime.prepare_local()

    def test_upload_output(self):
        # Test on no model
        runtime = self.initruntime()
        with pytest.raises(ModelNotPreparedError):
            runtime.upload_output(b'')

        # Test with model and input
        data = np.zeros((self.inputshapes), dtype=np.float32).tobytes()
        runtime = self.initruntime()
        runtime.prepare_local()
        runtime.prepare_input(data)
        runtime.run()
        expected_data = np.zeros(self.outputshapes, dtype=np.float32).tobytes()
        assert runtime.upload_output(b'') == expected_data
