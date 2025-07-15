# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

from kenning.core.protocol import (
    Protocol,
    ServerAction,
    ServerDownloadCallback,
    ServerStatus,
    ServerUploadCallback,
)
from kenning.core.runtime import Runtime
from kenning.scenarios.inference_server import InferenceServer


@pytest.fixture
@patch.multiple(Runtime, __abstractmethods__=set())
def runtime():
    return Runtime()


class ProtocolMock(Protocol):
    def initialize_server(
        self, client_connected_callback, client_disconnected_callback
    ):
        return True

    def disconnect(self):
        pass

    def serve(
        self,
        upload_input_callback: Optional[ServerUploadCallback] = None,
        upload_model_callback: Optional[ServerUploadCallback] = None,
        process_input_callback: Optional[ServerUploadCallback] = None,
        download_output_callback: Optional[ServerDownloadCallback] = None,
        download_stats_callback: Optional[ServerDownloadCallback] = None,
        upload_iospec_callback: Optional[ServerUploadCallback] = None,
        upload_optimizers_callback: Optional[ServerUploadCallback] = None,
        upload_unoptimized_model_callback: Optional[
            ServerUploadCallback
        ] = None,
        download_optimized_model_callback: Optional[
            ServerDownloadCallback
        ] = None,
        upload_runtime_callback: Optional[ServerUploadCallback] = None,
    ):
        self.upload_input_callback = upload_input_callback
        self.upload_model_callback = upload_model_callback
        self.process_input_callback = process_input_callback
        self.upload_iospec_callback = upload_iospec_callback
        self.upload_optimizers_callback = upload_optimizers_callback
        self.upload_unoptimized_model_callback = (
            upload_unoptimized_model_callback
        )
        self.download_optimized_model_callback = (
            download_optimized_model_callback
        )
        self.download_stats_callback = download_stats_callback
        self.download_output_callback = download_output_callback


@pytest.fixture
@patch.multiple(ProtocolMock, __abstractmethods__=set())
def protocol():
    return ProtocolMock()


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

        inference_server.close()

        assert inference_server.close_server_event.is_set()
        assert ServerStatus(ServerAction.IDLE) == inference_server.status

    def test_run(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server run method.
        """
        inference_server = InferenceServer(runtime, protocol)

        with (
            patch.object(
                protocol, "initialize_server"
            ) as protocol_initialize_server_mock,
            patch.object(protocol, "serve") as protocol_serve_mock,
            patch.object(protocol, "disconnect") as protocol_disconnect_mock,
        ):
            protocol_initialize_server_mock.return_value = True

            server_thread = threading.Thread(target=inference_server.run)
            server_thread.start()

            assert not inference_server.close_server_event.is_set()
            assert server_thread.is_alive()

            inference_server.close()
            server_thread.join()

            protocol_initialize_server_mock.assert_called_once()
            protocol_serve_mock.assert_called_once()

            assert inference_server.close_server_event.is_set()
            protocol_disconnect_mock.assert_called_once()

    def test_data_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server data callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, "load_input_from_bytes"
            ) as runtime_load_input_mock,
        ):
            runtime_load_input_mock.return_value = True

            assert ServerStatus(
                ServerAction.UPLOADING_INPUT, True
            ) == protocol.upload_input_callback(data)

            runtime_load_input_mock.assert_called_once_with(data)

        with (
            patch.object(
                runtime, "load_input_from_bytes"
            ) as runtime_load_input_mock,
        ):
            runtime_load_input_mock.return_value = False

            assert ServerStatus(
                ServerAction.UPLOADING_INPUT, False
            ) == protocol.upload_input_callback(data)

            runtime_load_input_mock.assert_called_once_with(data)

    def test_model_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server model callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        model = np.random.bytes(128)

        with (
            patch.object(
                runtime, "prepare_model"
            ) as runtime_prepare_model_mock,
            patch.object(
                runtime, "inference_session_start"
            ) as runtime_inference_session_start_mock,
        ):
            runtime_prepare_model_mock.return_value = True

            assert ServerStatus(
                ServerAction.UPLOADING_MODEL, True
            ) == protocol.upload_model_callback(model)

            runtime_prepare_model_mock.assert_called_once_with(model)
            runtime_inference_session_start_mock.assert_called_once()

        with (
            patch.object(
                runtime, "prepare_model"
            ) as runtime_prepare_model_mock,
            patch.object(
                runtime, "inference_session_start"
            ) as runtime_inference_session_start_mock,
        ):
            runtime_prepare_model_mock.return_value = False

            assert ServerStatus(
                ServerAction.UPLOADING_MODEL, False
            ) == protocol.upload_model_callback(model)

            runtime_prepare_model_mock.assert_called_once_with(model)
            runtime_inference_session_start_mock.assert_called_once()

    def test_process_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server process callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        with (
            patch.object(runtime, "run") as runtime_run_mock,
        ):
            assert ServerStatus(
                ServerAction.PROCESSING_INPUT, True
            ) == protocol.process_input_callback(b"")

            runtime_run_mock.assert_called_once()

    def test_output_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server output callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, "upload_output"
            ) as runtime_upload_output_mock,
        ):
            runtime_upload_output_mock.return_value = data

            assert (
                ServerStatus(ServerAction.EXTRACTING_OUTPUT, True),
                data,
            ) == protocol.download_output_callback()

            runtime_upload_output_mock.assert_called_once()

        with (
            patch.object(
                runtime, "upload_output"
            ) as runtime_upload_output_mock,
        ):
            runtime_upload_output_mock.return_value = None

            assert (
                ServerStatus(ServerAction.EXTRACTING_OUTPUT, False),
                None,
            ) == protocol.download_output_callback()

            runtime_upload_output_mock.assert_called_once()

    def test_stats_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server stats callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        data = np.random.bytes(128)

        with (
            patch.object(runtime, "upload_stats") as runtime_upload_stats_mock,
        ):
            runtime_upload_stats_mock.return_value = data

            assert (
                ServerStatus(ServerAction.COMPUTING_STATISTICS, True),
                data,
            ) == protocol.download_stats_callback()

            runtime_upload_stats_mock.assert_called_once()

    def test_io_spec_callback(self, runtime: Runtime, protocol: Protocol):
        """
        Test inference server io_spec callback.
        """
        inference_server = InferenceServer(runtime, protocol)
        inference_server.close()
        inference_server.run()
        data = np.random.bytes(128)

        with (
            patch.object(
                runtime, "prepare_io_specification"
            ) as runtime_prepare_io_spec_mock,
        ):
            runtime_prepare_io_spec_mock.return_value = True

            assert ServerStatus(
                ServerAction.UPLOADING_IOSPEC, True
            ) == protocol.upload_iospec_callback(data)

            runtime_prepare_io_spec_mock.assert_called_once_with(data)

        with (
            patch.object(
                runtime, "prepare_io_specification"
            ) as runtime_prepare_io_spec_mock,
        ):
            runtime_prepare_io_spec_mock.return_value = False

            assert ServerStatus(
                ServerAction.UPLOADING_IOSPEC, False
            ) == protocol.upload_iospec_callback(data)

            runtime_prepare_io_spec_mock.assert_called_once_with(data)
