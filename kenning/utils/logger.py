# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing the logging structures.

Module also implements a tqdm loading bar that enables adding callbacks
that are called in specified intervals.

Callbacks are registered and unregistered globally for specific tags and only
tqdm instances of the same tags are going to use those callbacks.
"""

import io
import logging
import urllib.request
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Optional, Type, Union

from tqdm import tqdm


def string_to_verbosity(level: str):
    """
    Maps verbosity string to corresponding logging enum.
    """
    levelconversion = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levelconversion[level]


def set_verbosity(loglevel: str):
    """
    Sets verbosity level.
    """
    logger = logging.getLogger('root')
    logger.setLevel(string_to_verbosity(loglevel))


def get_logger():
    """
    Configures and returns root logger.
    """
    logger = logging.getLogger('root')
    FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] [%(levelname)s] %(message)s'  # noqa: E501
    logging.basicConfig(format=FORMAT)
    return logger

# ----------------
# Tqdm Loading bar


class LoggerProgressBar(io.StringIO):
    """
    Prepares IO stream for TQDM progress bar to run in logging.
    """

    def __init__(self, suppress_new_line=True):
        super().__init__()
        self.logger = get_logger()
        self.buf = ''
        self.prev_terminators = []
        if suppress_new_line:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    self.prev_terminators.append((handler, handler.terminator))
                    handler.terminator = '\r'

    def __enter__(self) -> 'LoggerProgressBar':
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType]) -> bool:
        # restore previous terminator
        for handler, terminator in self.prev_terminators:
            handler.terminator = terminator
        self.logger.log(logging.INFO, '')

        return False

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(logging.INFO, self.buf)


def download_url(url, output_path):
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with (
        LoggerProgressBar() as progress_bar,
        DownloadProgressBar(
            unit='B',
            unit_scale=True,
            miniters=1,
            file=progress_bar,
            desc=url.split('/')[-1],
        ) as t,
    ):
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )

# ----------------
# Tqdm callbacks


class Callback:
    def __init__(self, tag: str, fun: Callable, sec_interval: int, *args: Any):
        """
        Callback that can be registered for a given `tag`. Whenever
        TqdmCallback is used, all registered Callbacks of this tag are
        gathered and invoked every `sec_interval` seconds.

        Parameters
        ----------
        tag : str
            Tag associated with this callback.
        fun : Callable
            Function to be used every `sec_interval` seconds. This function
            takes `format_dict` as the first argument. The rest of the
            arguments are passed using `*args`.
        sec_interval : int
            Specifies the time interval in seconds for the callback function
            to be invoked.
        *args : Any
            Any additional arguments that are passed to the `fun`.
        """
        self.tag = tag
        self.fun = fun
        self.sec_interval = sec_interval
        self.args = args


@dataclass
class CallbackInstance:
    """
    Internal class that specifies a single callback that is used
    in a TqdmCallback instance.

    Attributes
    ----------
    callback : Callback
        Callback that is used.
    last_call_timestamp : int
        Timestamp used to determine when was the callback invoked.
    """

    callback: Callback
    last_call_timestamp: int = 0


class TqdmCallback(tqdm):
    """
    Subclass of tqdm that enables adding callbacks for loading bars.
    """

    callbacks = []

    def __init__(self, tag: str, *args, **kwargs):
        """
        Initializes the class with a tag. All registered callbacks of the same
        tag are used.

        Parameters
        ----------
        tag : str
            Tag of the class.
        """
        super().__init__(*args, **kwargs)
        self.tag = tag
        self.tagged_callbacks = []

        for callback in self.callbacks:
            if callback.tag == self.tag:
                callback.fun(self.format_dict, *callback.args)
                self.tagged_callbacks.append(CallbackInstance(callback))

    @classmethod
    def register_callback(cls, callback: Callback):
        """
        Registers the callback in the static list of callbacks.

        Parameters
        ----------
        callback : Callback
            Callback to be registered.
        """
        cls.callbacks.append(callback)

    @classmethod
    def unregister_callback(cls, callback: Callback):
        """
        Removes the callback from the static list of callbacks.

        Parameters
        ----------
        callback : Callback
            Callback to be unregistered.
        """
        cls.callbacks = [clb for clb in cls.callbacks if clb != callback]

    def update(self, n: Optional[Union[float, int]] = 1) -> bool:
        """
        Updates the displayed progress bar and checks whether any callback
        should be invoked.

        Parameters
        ----------
        n : Optional[Union[int, float]]
            Increment that is added to the internal counter.

        Returns
        -------
        bool :
            True if a `display()` was triggered.
        """
        if not super().update(n):
            return False
        for tagged_callback in self.tagged_callbacks:
            format_dict = self.format_dict
            elapsed = format_dict["elapsed"]
            if (
                elapsed - tagged_callback.last_call_timestamp
                >= tagged_callback.callback.sec_interval
            ):
                tagged_callback.last_call_timestamp = elapsed
                tagged_callback.callback.fun(
                    format_dict, *tagged_callback.callback.args
                )
        return True
