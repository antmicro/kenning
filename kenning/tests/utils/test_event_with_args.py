# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from threading import Thread
from time import sleep

from kenning.utils.event_with_args import EventWithArgs


class TestEventWithArgs:
    def test_with_args(self):
        TEST_ARGS = (1, "dfmsdkfk", 92392, [32, 2, 87, -43478])
        event = EventWithArgs()

        def test_thread():
            sleep(1)
            event.set(TEST_ARGS)

        thread = Thread(target=test_thread)
        thread.start()
        received_args = event.wait()
        assert TEST_ARGS == received_args

    def test_without_args(self):
        event = EventWithArgs()

        def test_thread():
            sleep(1)
            event.set()

        thread = Thread(target=test_thread)
        thread.start()
        event.wait()
