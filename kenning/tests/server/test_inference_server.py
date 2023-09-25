# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import threading
from time import sleep
from unittest.mock import patch

import numpy as np
import pytest

from kenning.core.protocol import Message, MessageType, Protocol, ServerStatus
from kenning.core.runtime import Runtime
from kenning.scenarios.inference_server import InferenceServer


@pytest.fixture
def runtime():
    return Runtime()


@pytest.fixture
def protocol():
    return Protocol()


class TestInferenceServerRunner:
    def test_initializer(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server initializer.
        """
        _ = InferenceServer(runtime, protocol)

    def test_close(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server close method.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.should_work = True

        inference_server.close()

        assert not inference_server.should_work

    def test_run(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server close method.
        """
        inference_server = InferenceServer(runtime, protocol)

        with (
            patch.object(
                protocol, 'initialize_server'
            ) as protocol_initialize_server_mock,
            patch.object(
                protocol, 'receive_message'
            ) as protocol_receive_message_mock,
            patch.object(
                protocol, 'disconnect'
            ) as protocol_disconnect_mock
        ):
            protocol_initialize_server_mock.return_value = True
            protocol_receive_message_mock.return_value = (
                ServerStatus.NOTHING, None
            )

            server_thread = threading.Thread(target=inference_server.run)
            server_thread.start()

            protocol_initialize_server_mock.assert_called_once()
            assert inference_server.should_work
            assert server_thread.is_alive()

            inference_server.close()
            server_thread.join()

            assert not inference_server.should_work
            protocol_disconnect_mock.assert_called_once()

    @pytest.mark.parametrize(
        'message_type,callback_name',
        (
            (MessageType.DATA, '_data_callback'),
            (MessageType.MODEL, '_model_callback'),
            (MessageType.PROCESS, '_process_callback'),
            (MessageType.OUTPUT, '_output_callback'),
            (MessageType.STATS, '_stats_callback'),
            (MessageType.IO_SPEC, '_io_spec_callback'),
        )
    )
    def test_callback(
        self,
        runtime: Runtime,
        protocol: Protocol,
        message_type: MessageType,
        callback_name: str
    ):
        """
        Test inference server callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        with (
            patch.object(
                inference_server, callback_name
            ) as server_callback_mock,
            patch.object(
                protocol, 'initialize_server'
            ) as protocol_initialize_server_mock,
            patch.object(
                protocol, 'receive_message'
            ) as protocol_receive_message_mock,
            patch.object(
                protocol, 'disconnect'
            ) as protocol_disconnect_mock
        ):
            inference_server.callbacks[message_type] = server_callback_mock
            protocol_initialize_server_mock.return_value = True
            data = np.random.bytes(128)
            protocol_receive_message_mock.return_value = (
                ServerStatus.DATA_READY, Message(message_type, data)
            )

            server_thread = threading.Thread(target=inference_server.run)
            server_thread.start()

            protocol_initialize_server_mock.assert_called_once()
            assert inference_server.should_work
            assert server_thread.is_alive()

            sleep(.01)

            inference_server.close()
            server_thread.join()

            server_callback_mock.assert_called_with(data)

            assert not inference_server.should_work
            protocol_disconnect_mock.assert_called_once()

    def test_data_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server data callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, 'prepare_input'
            ) as runtime_prepare_input_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            runtime_prepare_input_mock.return_value = True

            inference_server.callbacks[MessageType.DATA](data)

            runtime_prepare_input_mock.assert_called_once_with(data)
            protocol_request_success_mock.assert_called_once()

        with (
            patch.object(
                runtime, 'prepare_input'
            ) as runtime_prepare_input_mock,
            patch.object(
                protocol, 'request_failure'
            ) as protocol_request_failure_mock,
        ):
            runtime_prepare_input_mock.return_value = False

            inference_server.callbacks[MessageType.DATA](data)

            runtime_prepare_input_mock.assert_called_once_with(data)
            protocol_request_failure_mock.assert_called_once()

    def test_model_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server model callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        model = np.random.bytes(128)

        with (
            patch.object(
                runtime, 'prepare_model'
            ) as runtime_prepare_model_mock,
            patch.object(
                runtime, 'inference_session_start'
            ) as runtime_inference_session_start_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            runtime_prepare_model_mock.return_value = True

            inference_server.callbacks[MessageType.MODEL](model)

            runtime_prepare_model_mock.assert_called_once_with(model)
            runtime_inference_session_start_mock.assert_called_once()
            protocol_request_success_mock.assert_called_once()

        with (
            patch.object(
                runtime, 'prepare_model'
            ) as runtime_prepare_model_mock,
            patch.object(
                runtime, 'inference_session_start'
            ) as runtime_inference_session_start_mock,
            patch.object(
                protocol, 'request_failure'
            ) as protocol_request_failure_mock,
        ):
            runtime_prepare_model_mock.return_value = False

            inference_server.callbacks[MessageType.MODEL](model)

            runtime_prepare_model_mock.assert_called_once_with(model)
            runtime_inference_session_start_mock.assert_called_once()
            protocol_request_failure_mock.assert_called_once()

    def test_process_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server process callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        with (
            patch.object(
                runtime, 'run'
            ) as runtime_run_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            inference_server.callbacks[MessageType.PROCESS](b'')

            runtime_run_mock.assert_called_once()
            protocol_request_success_mock.assert_called_once()

    def test_output_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server output callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, 'upload_output'
            ) as runtime_upload_output_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            runtime_upload_output_mock.return_value = data

            inference_server.callbacks[MessageType.OUTPUT](b'')

            runtime_upload_output_mock.assert_called_once()
            protocol_request_success_mock.assert_called_once_with(data)

        with (
            patch.object(
                runtime, 'upload_output'
            ) as runtime_upload_output_mock,
            patch.object(
                protocol, 'request_failure'
            ) as protocol_request_failure_mock,
        ):
            runtime_upload_output_mock.return_value = None

            inference_server.callbacks[MessageType.OUTPUT](b'')

            runtime_upload_output_mock.assert_called_once()
            protocol_request_failure_mock.assert_called_once()

    def test_stats_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server stats callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, 'upload_stats'
            ) as runtime_upload_stats_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            runtime_upload_stats_mock.return_value = data

            inference_server.callbacks[MessageType.STATS](b'')

            runtime_upload_stats_mock.assert_called_once()
            protocol_request_success_mock.assert_called_once_with(data)

    def test_io_spec_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server io_spec callback.
        """
        inference_server = InferenceServer(runtime, protocol)

        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, 'prepare_io_specification'
            ) as runtime_prepare_io_spec_mock,
            patch.object(
                protocol, 'request_success'
            ) as protocol_request_success_mock
        ):
            runtime_prepare_io_spec_mock.return_value = True

            inference_server.callbacks[MessageType.IO_SPEC](data)

            runtime_prepare_io_spec_mock.assert_called_once_with(data)
            protocol_request_success_mock.assert_called_once()

        with (
            patch.object(
                runtime, 'prepare_io_specification'
            ) as runtime_prepare_io_spec_mock,
            patch.object(
                protocol, 'request_failure'
            ) as protocol_request_failure_mock,
        ):
            runtime_prepare_io_spec_mock.return_value = False

            inference_server.callbacks[MessageType.IO_SPEC](data)

            runtime_prepare_io_spec_mock.assert_called_once_with(data)
            protocol_request_failure_mock.assert_called_once()
