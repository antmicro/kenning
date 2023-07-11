# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides resource manager responsible for downloading and caching resources
"""
import hashlib
from pathlib import Path
from shutil import copy, rmtree
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import ParseResult, urlparse

import requests

from kenning.utils.logger import download_url, get_logger
from kenning.utils.singleton import Singleton


class ResourceManager(metaclass=Singleton):
    """
    Download and cache resources used by Kenning.
    """

    CACHE_DIR = Path.home() / '.kenning'

    MAX_CACHE_SIZE = 50_000_000_000  # 50 GB

    BASE_URL_SCHEMES = {
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

        # no scheme in URI - treat as path string
        if '' == parsed_uri.scheme:
            if output_path is None:
                return Path(uri)
            else:
                copy(uri, output_path)
                return output_path

        resolved_uri = self._resolve_uri(parsed_uri)

        if 'file' == parsed_uri.scheme:
            if output_path is None:
                return Path(resolved_uri)
            else:
                copy(resolved_uri, output_path)
                return output_path

        if output_path is None:
            parsed_path = parsed_uri.path
            if parsed_path[0] == '/':
                parsed_path = parsed_path[1:]
            output_path = self.CACHE_DIR / parsed_path

        output_path = output_path.resolve()

        # file already exists - check if its valid
        if output_path.exists():
            remote_sha_valid = self._validate_file_remote(
                resolved_uri, output_path
            )
            local_sha_valid = self._validate_file_local(output_path)

            if remote_sha_valid or (
                remote_sha_valid is None and local_sha_valid
            ):
                if local_sha_valid is None:
                    self._save_file_checksum(output_path)

                self.log.info(f'Using cached: {output_path}')
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._download_resource(resolved_uri, output_path)

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
        sha_file_path = file_path.with_suffix(file_path.suffix + '.sha256')

        if not sha_file_path.exists():
            return None

        with open(sha_file_path, 'r') as checksum_file:
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

        sha_file_path = file_path.with_suffix(file_path.suffix + '.sha256')

        with open(sha_file_path, 'w') as checksum_file:
            checksum_file.write(file_sha)

    def _download_resource(self, url: str, output_path: Path):
        """
        Downloads resource from given URL.

        Parameters
        ----------
        url : str
            Resource URL.
        output_path : Path
            Path where the resource should be saved.
        """
        if self.CACHE_DIR in output_path.parents:
            response = requests.head(url)
            if response.status_code != 200:
                self.log.warning('Cannot read file size before downloading')

            elif 'Content-Length' in response.headers:
                required_size = int(response.headers['Content-Length'])
                self._free_cache(required_size)

        download_url(url, output_path)

        if self.CACHE_DIR in output_path.parents:
            # free cache in case the file size could not be read before
            # downloading
            self._free_cache(0)

    def _free_cache(self, required_free: int):
        """
        Frees cache space.

        Parameters
        ----------
        required_free : int
            Amount of bytes that need to be available after freeing space.

        Raises
        ------
        ValueError :
            Raised when required free space is bigger that max cache size.
        """
        if required_free > self.MAX_CACHE_SIZE:
            raise ValueError(f'Required free space too big: {required_free} B')
        cached_files = self.list_cached_files()
        cached_files.sort(key=lambda f: f.stat().st_mtime)
        cache_size = sum(f.stat().st_size for f in cached_files)

        while cache_size > self.MAX_CACHE_SIZE - required_free:
            file = cached_files.pop(0)
            cache_size -= file.stat().st_size
            file.unlink()

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
        self._resources_uri = resources_uri

    def __getitem__(self, keys: Union[Tuple[str], str]) -> Path:
        if isinstance(keys, str):
            keys = [keys]

        resources_uri = self._resources_uri
        for key in keys:
            if key not in resources_uri:
                raise KeyError(f'Invalid key: {keys}')
            resources_uri = resources_uri[key]

        return ResourceManager().get_resource(resources_uri)

    def __setitem__(self, key: str, value: str):
        if key in self._resources_uri:
            raise KeyError(f'Resource {key} already exists')
        self._resources_uri[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._resources_uri.keys()

    def keys(self) -> List[Tuple[str, ...]]:
        result = []

        def get_keys(resources_uri: dict, keys: list = []):
            for key, value in resources_uri.items():
                if isinstance(value, str):
                    result.append((*keys, key))
                elif isinstance(value, dict):
                    get_keys(value, keys + [key])

        get_keys(self._resources_uri)

        return result


class ChecksumVerifyError(Exception):
    """
    Exception raised when downloaded file has invalid checksum.
    """

    pass


def get_resource(uri: str, output_path: Path = None) -> Path:
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
    return ResourceManager().get_resource(uri, output_path)
