# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import multiprocessing
import os
import random
import socket
import time

import pytest

from kenning.cli.parser import USED_SUBCOMMANDS
from kenning.core.measurements import Measurements
from kenning.protocols.network import NetworkProtocol
from kenning.scenarios.probe import ProbeServerRunner


def _make_probe_args(port_num, **overrides):
    defaults = {
        "help": False,
        "verbosity": "INFO",
        "protocol_cls": "kenning.protocols.network.NetworkProtocol",
        "pid": os.getpid(),
        "prefix": "session_utilization",
        "step": 0.05,
        "host": "127.0.0.1",
        "port": port_num,
        USED_SUBCOMMANDS: ["probe"],
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestProbeServerRunner:
    def _make_args(self, **overrides):
        defaults = {
            "help": False,
            "verbosity": "INFO",
            "protocol_cls": "kenning.protocols.network.NetworkProtocol",
            "pid": 1,
            "prefix": "session_utilization",
            "step": 0.1,
            "host": "127.0.0.1",
            "port": 12345,
            USED_SUBCOMMANDS: ["probe"],
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_run_with_missing_protocol_cls(self, mocker):
        """
        Test that ProbeServerRunner.run() raises ConfigurationError when
        protocol_cls is not provided.
        """
        args = self._make_args(protocol_cls=None)

        with pytest.raises(Exception):
            ProbeServerRunner.run(args)


def random_network_port():
    ports = random.sample(range(49152, 65535), k=5)
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            break
        except OSError:
            continue
    else:
        return None
    return port


def wait_for_server(host, port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except OSError:
            # The port is already bound (EADDRINUSE), which means the probe
            # server has bound and is listening on it - it is ready. Checking
            # via bind (instead of connecting) avoids consuming the server's
            # single client connection.
            return True
        finally:
            sock.close()
        # Bind succeeded, so the port is still free - the server is not
        # listening yet. Wait and retry.
        time.sleep(0.1)
    return False


class TestProbeClientConnection:
    def test_connect_to_probe_and_download_stats(self):
        """
        Test connecting to probe server as a client and downloading statistics.
        """
        port = random_network_port()
        if port is None:
            pytest.skip("Could not find a free network port")

        host = "127.0.0.1"
        server_error = multiprocessing.Queue()
        close_event = multiprocessing.Event()

        def run_probe_server(
            host,
            port_num,
            error_queue,
            close_event,
        ):
            try:
                args = _make_probe_args(port_num=port_num)
                ProbeServerRunner.run(args)
            except Exception as e:
                error_queue.put(e)
                raise

        process = multiprocessing.Process(
            target=run_probe_server,
            args=(
                host,
                port,
                server_error,
                close_event,
            ),
        )
        process.start()

        try:
            server_ready = wait_for_server(host, port, timeout=10)
            if not server_ready:
                pytest.fail("Probe server did not start within timeout")

            print(f"{host}:{port}")
            client = NetworkProtocol(host, port)
            assert client.initialize_client()

            time.sleep(0.3)

            measurements = client.download_statistics(final=True)
            assert isinstance(measurements, Measurements)

            client.disconnect()
        finally:
            close_event.set()
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

            if not server_error.empty():
                raise server_error.get()

    def test_multiple_clients_connect_and_download_stats(self):
        """
        Test connecting to the probe server as multiple simultaneous clients
        (multi_client mode) and downloading statistics from each of them.
        """
        port = random_network_port()
        if port is None:
            pytest.skip("Could not find a free network port")

        host = "127.0.0.1"
        server_error = multiprocessing.Queue()
        close_event = multiprocessing.Event()

        def run_probe_server(
            host,
            port_num,
            error_queue,
            close_event,
        ):
            try:
                args = _make_probe_args(port_num=port_num, multi_client=True)
                ProbeServerRunner.run(args)
            except Exception as e:
                error_queue.put(e)
                raise

        process = multiprocessing.Process(
            target=run_probe_server,
            args=(
                host,
                port,
                server_error,
                close_event,
            ),
        )
        process.start()

        clients = []
        try:
            server_ready = wait_for_server(host, port, timeout=10)
            if not server_ready:
                pytest.fail("Probe server did not start within timeout")

            # Connect two clients at the same time.
            for _ in range(2):
                client = NetworkProtocol(host, port)
                assert client.initialize_client()
                clients.append(client)

            time.sleep(0.3)

            # Download statistics from each client. This is done sequentially,
            # because in multi_client mode the server broadcasts every
            # response to all connected clients, so concurrent requests would
            # cross each other's responses.
            for client in clients:
                measurements = client.download_statistics(final=True)
                assert isinstance(measurements, Measurements)

            for client in clients:
                client.disconnect()
        finally:
            close_event.set()
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

            if not server_error.empty():
                raise server_error.get()
