# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing the logging structures.

Module also implements a tqdm loading bar that enables adding callbacks
that are called in specified intervals.

Callbacks are registered and unregistered globally for specific tags and only
tqdm instances of the same tags are going to use those callbacks.
"""

import asyncio
import io
import logging
import os
import sys
import urllib.request
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Type, Union

import coloredlogs
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from kenning.core.exceptions import DownloadError
from kenning.utils.singleton import Singleton

PROGRESS_BAR_STACKLEVEL = 2

CUSTOM_LEVEL_STYLES = {
    "renode": {"color": "blue"},
    "verbose": {"color": "cyan"},
    "device": {"color": "magenta"},
}


class _DuplicateStream(TextIOBase):
    def __init__(self, stream=sys.stderr):
        super().__init__()
        self.stream = stream
        self.client = None
        self.loop = None

    def set_client(self, client, loop):
        self.client = client
        self.loop = loop if loop is not None else asyncio.get_event_loop()

    def write(self, s, /):
        if self.client is not None:
            request = {
                "name": "Kenning Terminal",
                "message": s.replace("\n", "\r\n"),
            }
            # if an error is thrown by the coroutine
            # and not an invalid get(),
            # there will be a warning about an unawaited coroutine
            asyncio.run_coroutine_threadsafe(
                self.client.notify("terminal_write", request),
                self.loop,
            )
        return self.stream.write(s)


DuplicateStream = _DuplicateStream()


class _KLogger(logging.Logger, metaclass=Singleton):
    """
    Kenning Logger class.
    """

    FORMAT = (
        "[%(asctime)-15s {package} %(filename)s:%(lineno)s] [%(levelname)s] "
        "%(message)s"
    )

    custom_log_backends = {}

    def __init__(self):
        """
        Initialize the root logger.
        """
        super().__init__("kenning", "NOTSET")
        # `coloredlogs` treats the given stream that is sys.stdout/sys.stderr
        # differently to avoid corresponding stream duplication. Duplication
        # is desired only for the client
        prev_stderr = sys.stderr
        sys.stderr = DuplicateStream
        coloredlogs.install(
            logger=self,
            level="NOTSET",
            fmt=_KLogger.FORMAT.format(package="kenning"),
            level_styles=dict(
                coloredlogs.DEFAULT_LEVEL_STYLES, **CUSTOM_LEVEL_STYLES
            ),
            stream=sys.stderr,
            isatty=True,
        )
        sys.stderr = prev_stderr
        if os.environ.get("KENNING_ENABLE_ALL_LOGS", False):
            self.configure()

        class CustomBackendDispatcher(logging.Handler):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger

            def emit(self, record):
                for backend in self.logger.custom_log_backends.values():
                    backend(f" [{record.levelname}] {self.format(record)}")

        self.addHandler(CustomBackendDispatcher(self))

    @staticmethod
    def configure():
        """
        Configures logging formats for all loggers.
        """
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
        if os.environ.get("KENNING_ENABLE_ALL_LOGS", False):
            for name in logging.root.manager.loggerDict:
                logger = logging.getLogger(name)
                logger.setLevel(level)
                coloredlogs.adjust_level(logger, level)

    def add_custom_level(self, level_num: int, level_name: str):
        """
        Add custom logging level.

        Parameters
        ----------
        level_num : int
            Integer representing the level.
        level_name : str
            New log level name.

        Raises
        ------
        AttributeError
            Raised if level is already defined.
        """
        if hasattr(logging, level_name):
            raise AttributeError(f"{level_name} already defined")
        if hasattr(logging, level_name.lower()):
            raise AttributeError(f"{level_name.lower()} already defined")
        if hasattr(self, level_name.lower()):
            raise AttributeError(
                f"{level_name.lower()} already defined in KLogger"
            )

        def custom_log(self, message, *args, **kwargs):
            if self.isEnabledFor(level_num):
                self._log(level_num, message, args, **kwargs)

        def custom_log_root(message, *args, **kwargs):
            logging.log(level_num, message, *args, **kwargs)

        logging.addLevelName(level_num, level_name)
        setattr(logging, level_name, level_num)
        setattr(self.__class__, level_name.lower(), custom_log)
        setattr(logging, level_name.lower(), custom_log_root)

    def add_custom_backend(self, name: str, backend: Callable[str, None]):
        """
        Registers a callback function, that will be called each time a log
        is received and will be given the log message as argument.

        Parameters
        ----------
        name : str
            Name under which the callback will be registered.
        backend: Callable[str, None]
            The actual callback function, that will receive log messages in the
            format: [<log level name>] <message>
        """
        self.custom_log_backends[name] = backend

    def remove_custom_backend(self, name: str) -> bool:
        """
        Removes a previously registered custom log backend (callback) based on
        its name.

        Parameters
        ----------
        name : str
            Name of the callback to remove.

        Returns
        -------
        bool
            True if the backend was removed, False if backend does not exist.
        """
        if name in self.custom_log_backends.keys():
            self.custom_log_backends.pop(name)
            return True
        return False

    def error_prepare_exception(
        self, message: str, exception: Exception, *args: Any, **kwargs: Any
    ) -> Exception:
        """
        Log the message with verbosity 'error' and returns an exception
        to be raised.

        This is a wrapper method that should be used whenever an error is
        encountered and the runtime should raise an exception.

        It returns an exception, instead of raising it, so that
        the exception can be tracked back to the original function
        instead of this wrapper.

        Parameters
        ----------
        message : str
            Message to log.
        exception : Exception
            Exception to be prepared to be raised.
        *args : Any
            Arguments to pass to the logging function.
        **kwargs : Any
            Keyword arguments to pass to the logging function.

        Returns
        -------
        Exception
            The exception that was passed as an argument prepared to be raised.
        """
        self.error(message, *args, **kwargs)
        return exception(message)


KLogger = _KLogger()

# ----------------
# Tqdm Loading bar


class LoggerProgressBar(io.StringIO):
    """
    Prepares IO stream for TQDM progress bar to run in logging.
    """

    def __init__(
        self, verbosity: int = logging.INFO, capture_stdout: bool = False
    ):
        super().__init__()
        self.redirect_tqdm = logging_redirect_tqdm(loggers=[KLogger])
        self.redirect_stdout = (
            redirect_stdout(self) if capture_stdout else None
        )
        self.format = ""
        self.verbosity = verbosity
        self.disable = None

    def __enter__(self) -> "LoggerProgressBar":
        self.redirect_tqdm.__enter__()
        fn, lno, func, sinfo = KLogger.findCaller(
            stacklevel=PROGRESS_BAR_STACKLEVEL
        )
        record = KLogger.makeRecord(
            KLogger.name,
            KLogger.level,
            fn=fn,
            lno=lno,
            msg="",
            args=[],
            exc_info=None,
            func=func,
            sinfo=sinfo,
        )
        # Take tqdm-wrapped handler
        self.format = KLogger.handlers[-1].formatter.format(record)
        self.disable = self.verbosity < KLogger.level

        if self.redirect_stdout:
            self.redirect_stdout.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.redirect_tqdm.__exit__(exc_type, exc_value, traceback)
        if self.redirect_stdout:
            self.redirect_stdout.__exit__(exc_type, exc_value, traceback)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {
            "bar_format": self.format + "{l_bar}{bar}{r_bar}",
            "disable": self.disable,
        }


def download_url(url: str, output_path: str):
    """
    Downloads the resource and renders the progress bar.

    Parameters
    ----------
    url: str
        URL to file to download
    output_path: str
        Path where to download the file

    Raises
    ------
    DownloadError
        If resources was not downloaded.
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
            desc=url.split("/")[-1],
            **logger_progress_bar.kwargs,
        ) as t,
    ):
        try:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
        except urllib.request.ContentTooShortError as ex:
            raise DownloadError(f"Error when downloading {url}") from ex
    if not Path(output_path).exists():
        raise DownloadError(f"File from {url} was not downloaded")


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
