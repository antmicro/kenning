# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides resource manager responsible for downloading and caching resources
"""

import hashlib
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List
from urllib.parse import ParseResult, urlparse
from shutil import rmtree

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
            remote_sha_valid = self._validate_file_remote(
                resolved_uri, output_path
            )
            local_sha_valid = self._validate_file_local(output_path)

            if (remote_sha_valid or
                    (remote_sha_valid is None and local_sha_valid)):
                if local_sha_valid is None:
                    self._save_file_checksum(output_path)

                self.log.info(f'Using cached: {output_path}')
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        download_url(resolved_uri, output_path)

        if self._validate_file_remote(resolved_uri, output_path) is False:
            self.log.error(
                f'Error downloading file {uri} from {resolved_uri}. Invalid '
                'checksum.'
            )
            raise ChecksumVerifyError()
        self._save_file_checksum(output_path)

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

    def list_cached_files(self) -> List[Path]:
        """
        Returns list with cached files.

        Returns
        -------
        List[Path] :
            List of cached files.
        """
        result = []

        for cached_file in self.CACHE_DIR.glob('**/*'):
            if cached_file.is_file() and '.sha256' not in cached_file.suffixes:
                result.append(cached_file)

        return result

    def clear_cache(self):
        """
        Removes all cached files
        """
        rmtree(self.CACHE_DIR, ignore_errors=True)
        self.CACHE_DIR.mkdir()

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

    def _validate_file_remote(
            self, url: str, file_path: Path
    ) -> Optional[bool]:
        """
        Validate downloaded file using checksum obtained from remote.

        Parameters
        ----------
        url : str
            Resource URL.
        file_path : Path
            Path to the local file.

        Returns
        -------
        Optional[bool] :
            None if checksum cannot be validate, otherwise True if file
            checksum is valid.
        """
        checksum_url = f'{url}.sha256'

        response = requests.get(checksum_url, allow_redirects=True)
        if 200 != response.status_code:
            self.log.warning(f'Cannot verify {file_path} checksum')
            return None

        remote_sha = response.content.decode().strip()

        file_sha = self._compute_file_checksum(file_path)

        return file_sha == remote_sha

    def _validate_file_local(self, file_path: Path) -> Optional[bool]:
        """
        Validate downloaded file using checksum saved locally after download.

        Parameters
        ----------
        file_path : Path
            Path to the local file.

        Returns
        -------
        Optional[bool] :
            None if checksum cannot be validate, otherwise True if file
            checksum is valid.
        """
        if not file_path.with_suffix('.sha256').exists():
            return None

        with open(file_path.with_suffix('.sha256'), 'r') as checksum_file:
            local_sha = checksum_file.read().strip()

        file_sha = self._compute_file_checksum(file_path)

        return file_sha == local_sha

    def _save_file_checksum(self, file_path: Path):
        """
        Saves file checksum.

        Parameters
        ----------
        file_path : Path
            Path to the local file.
        """

        file_sha = self._compute_file_checksum(file_path)

        with open(file_path.with_suffix('.sha256'), 'w') as checksum_file:
            checksum_file.write(file_sha)

    @staticmethod
    def _compute_file_checksum(file_path: Path) -> str:
        """
        Computes file SHA256 checksum.

        Parameters
        ----------
        file_path : Path
            Path to the local file.

        Returns
        -------
        str :
            Computed checksum as in hex format.
        """
        sha = hashlib.sha256()

        with open(file_path, 'rb') as file:
            while True:
                data = file.read(sha.block_size)
                if not data:
                    break
                sha.update(data)

        return sha.hexdigest()


class Resources(object):
    """
    Dictionary of resources.
    """

    def __init__(self, resources_uri: Dict[str, str]):
        self.resources_uri = resources_uri

    def __getitem__(self, key: str) -> Path:
        if key not in self.resources_uri:
            raise KeyError(f'Invalid resource name: {key}')
        return ResourceManager().get_resource(self.resources_uri[key])

    def __setitem__(self, key: str, value: str):
        if key in self.resources_uri:
            raise KeyError(f'Resource {key} already exists')
        self.resources_uri[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.resources_uri.keys()


class ChecksumVerifyError(Exception):
    """
    Exception raised when downloaded file has invalid checksum.
    """

    pass
