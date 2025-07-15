# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Mechanism for sending signals between threads, based on threading.Event,
that allows values to be passed with the event.

Usage:

```
event = EventWithArgs()
```

In the waiting thread:

```
(arg1, arg2) = event.wait()
```

In the notifying thread:

```
event.set((arg1, arg2))
```
"""

from threading import Event
from typing import Tuple


class EventWithArgs(Event):
    """
    A wrapper for threading.Event, allowing to pass a value with the event.
    """

    def __init__(self, default_args: Tuple = (None)):
        """
        Initializes the event.

        Parameters
        ----------
        default_args: Tuple
            Value, that will be return by 'wait', if timeout is reached.
        """
        super().__init__()
        self.args = default_args

    def set(self, args: Tuple = ()):
        """
        Sets the internal flag to 1, notifying all waiting threads (those,
        that called the 'wait' method).

        Parameters
        ----------
        args: Tuple
            Values, that will be passed to the waiting threads.
        """
        self.args = args
        super().set()

    def wait(self, timeout: float = None) -> Tuple:
        """
        Stops the thread until the internal flag is set to 1.

        Parameters
        ----------
        timeout: float
            Waiting timeout in seconds. If set to None, there
            is no timeout.

        Returns
        -------
        Tuple
            Values passed by the notifying thread.
        """
        super().wait(timeout)
        return self.args
