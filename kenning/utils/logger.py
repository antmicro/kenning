# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing the logging structures.
"""

import logging
import io
import urllib.request


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


class LoggerProgressBar(io.StringIO):
    """
    Prepares IO stream for TQDM progress bar to run in logging.
    """

    def __init__(self, suppress_new_line=True):
        super().__init__()
        self.logger = get_logger()
        self.buf = ''
        if suppress_new_line:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.terminator = ''

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(logging.INFO, '\r' + self.buf)


def download_url(url, output_path):
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(
            unit='B',
            unit_scale=True,
            miniters=1,
            file=LoggerProgressBar(),
            desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )
