# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides resource manager responsible for downloading and caching resources
"""

import hashlib
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.parse import ParseResult, urlparse

import requests

from kenning.utils.logger import download_url, get_logger
from kenning.utils.singleton import Singleton


class ResourceManager(metaclass=Singleton):
    """
    Download and cache resources used by Kenning.
    """

    CACHE_DIR = Path.home() / '.kenning'

    BASE_URL_SCHEMES = {
        '': None,
        'http': None,
        'https': None,
        'kenning': 'https://dl.antmicro.com/kenning/{path}',
        'file': lambda path: str(Path(path).resolve()),
    }

    def __init__(self):
        """
        Initialize ResourceManager.
        """
        self.url_schemes = ResourceManager.BASE_URL_SCHEMES
        self.log = get_logger()

    def get_resource(
        self, uri: str, output_path: Optional[Path] = None
    ) -> Path:
        """
        Retrieve file and return path to it.

        If the uri points to remote resource, then it is downloaded (if not
        found in cache) and validated.

        Parameters
        ----------
        uri : str
            Resource URI.
        output_path : Optional[Path]
            Path to the output file. If not provided then the path is
            automatically created.

        Returns
        -------
        Path :
            Path to the retrieved resource.

        Raises
        ------
        ChecksumVerifyError :
            Raised when downloaded file has invalid checksum
        """
        # check if file is already cached
        parsed_uri = urlparse(uri)

        if output_path is None:
            parsed_path = parsed_uri.path
            if parsed_path[0] == '/':
                parsed_path = parsed_path[1:]
            output_path = self.CACHE_DIR / parsed_path

        output_path = output_path.resolve()

        resolved_uri = self._resolve_uri(parsed_uri)

        if 'file' == parsed_uri.scheme:
            return Path(resolved_uri)

        if output_path.exists():
            if self._validate_file(resolved_uri, output_path):
                self.log.info(f'Using cached: {output_path}')
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        download_url(resolved_uri, output_path)
        if not self._validate_file(resolved_uri, output_path):
            self.log.error(
                f'Error downloading file {uri} from {resolved_uri}. Invalid '
                'checksum.'
            )
            raise ChecksumVerifyError()

        return output_path

    def add_custom_url_schemes(
        self, custom_url_schemes: Dict[str, Optional[Union[str, Callable]]]
    ):
        """
        Add user defined URL schemes.

        Parameters
        ----------
        custom_url_schemes : Dict[str, Optional[Union[Dict, Callable]]]
            Dictionary with custom url schemes entries. Each entry consists of
            schema and corresponding conversion. Conversion can be None, string
            pattern or callable returning string.
        """
        self.url_schemes = dict(self.url_schemes, **custom_url_schemes)

    def _resolve_uri(self, parsed_uri: ParseResult) -> str:
        """
        Resolve provided URI.

        Parameters
        ----------
        parsed_uri : ParseResult
            Parsed URI.

        Returns
        -------
        str :
            Resolved path to resource.

        Raises
        ------
        ValueError :
            Raised when provided URI has invalid scheme or the defined
            scheme's conversion is invalid.
        """
        try:
            format = self.url_schemes[parsed_uri.scheme]
        except KeyError:
            raise ValueError(
                f'Invalid URI scheme provided: {parsed_uri.scheme}'
            )

        if format is None:
            return parsed_uri.geturl()
        if isinstance(format, str):
            return format.format(path=parsed_uri.path)
        if callable(format):
            return format(parsed_uri.path)

        raise ValueError(f'Invalid conversion for scheme {parsed_uri.scheme}')

    def _validate_file(self, url: str, file_path: Path) -> bool:
        """
        Validate downloaded file.

        Parameters
        ----------
        url : str
            Resource URL.
        file_path : Path
            Path to the local file.

        Returns
        -------
        bool :
            True if file checksum is valid.
        """
        checksum_url = f'{url}.sha256'

        sha = hashlib.sha256()

        response = requests.get(checksum_url, allow_redirects=True)
        if 200 != response.status_code:
            self.log.warning(f'Cannot verify {file_path} checksum, skipping')
            return True

        remote_sha = response.content.decode()

        with open(file_path, 'rb') as file:
            while True:
                data = file.read(sha.block_size)
                if not data:
                    break
                sha.update(data)

        file_sha = sha.hexdigest()

        return file_sha == remote_sha


class ChecksumVerifyError(Exception):
    """
    Exception raised when downloaded file has invalid checksum.
    """

    pass
