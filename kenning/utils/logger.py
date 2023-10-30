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
from typing import Any, Callable, Dict, List, Optional, Type, Union

import coloredlogs
from tqdm import tqdm

from kenning.utils.singleton import Singleton


class _KLogger(logging.Logger, metaclass=Singleton):
    """
    Kenning Logger class.
    """

    FORMAT = (
        "[%(asctime)-15s {package} %(filename)s:%(lineno)s] [%(levelname)s] "
        "%(message)s"
    )

    def __init__(self):
        """
        Initialize the root logger.
        """
        super().__init__("kenning", "NOTSET")
        coloredlogs.install(
            logger=self,
            level="NOTSET",
            fmt=_KLogger.FORMAT.format(package="kenning"),
        )
        self.configure()

    def configure(self):
        """
        Configure logging formats.
        """
        # set format for existing loggers
        loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        for logger in loggers:
            coloredlogs.install(
                logger=logger,
                level=logger.level,
                fmt=_KLogger.FORMAT.format(package=logger.name),
            )

        # set format for new loggers
        logging_getLogger = logging.getLogger

        def getLogger(name: Optional[str] = None):
            logger = logging_getLogger(name)
            coloredlogs.install(
                logger=logger,
                level=logger.level,
                fmt=_KLogger.FORMAT.format(package=logger.name),
            )
            return logger

        logging.getLogger = getLogger

    def set_verbosity(self, level: str):
        """
        Set verbosity level.

        Parameters
        ----------
        level : str
            The logging level as string.
        """
        self.setLevel(level)
        coloredlogs.adjust_level(self, level)
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            coloredlogs.adjust_level(logger, level)


KLogger = _KLogger()


# ----------------
# Tqdm Loading bar


class LoggerProgressBar(io.StringIO):
    """
    Prepares IO stream for TQDM progress bar to run in logging.
    """

    def __init__(self, suppress_new_line=True):
        super().__init__()
        self.buf = ""
        self.suppress_new_line = suppress_new_line

    def __enter__(self) -> "LoggerProgressBar":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        _KLogger().info(self.buf)

        return False

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        prev_terminators = []
        if self.suppress_new_line:
            for handler in _KLogger().handlers:
                if isinstance(handler, logging.StreamHandler):
                    prev_terminators.append((handler, handler.terminator))
                    handler.terminator = "\r"

        _KLogger().info(self.buf)
        # restore previous terminator
        for handler, terminator in prev_terminators:
            handler.terminator = terminator


def download_url(url: str, output_path: str):
    """
    Downloads the resource and renders the progress bar.

    Parameters
    ----------
    url: str
        URL to file to download
    output_path: str
        Path where to download the file
    """
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with (
        LoggerProgressBar() as logger_progress_bar,
        DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            file=logger_progress_bar,
            desc=url.split("/")[-1],
        ) as t,
    ):
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
        )


# ----------------
# Tqdm callbacks


class Callback:
    """
    A base class for tqdm callbacks.
    """

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

    def __init__(self, tag: str, *args: List, **kwargs: Dict):
        """
        Initializes the class with a tag. All registered callbacks of the same
        tag are used.

        Parameters
        ----------
        tag : str
            Tag of the class.
        *args : List
            Arguments passed to tqdm constructor.
        **kwargs : Dict
            Keyword arguments passed to tqdm constructor.
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

    def update(self, n: Optional[Union[int, float]] = 1) -> bool:
        """
        Updates the displayed progress bar and checks whether any callback
        should be invoked.

        Parameters
        ----------
        n : Optional[Union[int, float]]
            Increment that is added to the internal counter.

        Returns
        -------
        bool
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
