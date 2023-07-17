# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provide resource manager responsible for downloading and caching resources
"""
import hashlib
import os
from pathlib import Path
from shutil import copy, rmtree
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import ParseResult, urlparse

import requests

from kenning.utils.logger import download_url, get_logger
from kenning.utils.singleton import Singleton


class ResourceManager(metaclass=Singleton):
    """
    Download and cache resources used by Kenning.
    """

    CACHE_DIR = (
        Path(os.environ.get('KENNING_CACHE_DIR', Path.home() / '.kenning'))
        .expanduser()
        .resolve()
    )

    MAX_CACHE_SIZE = 50_000_000_000  # 50 GB

    HASHING_ALGORITHM = 'md5'

    BASE_URL_SCHEMES = {
        'http': None,
        'https': None,
        'kenning': 'https://dl.antmicro.com/kenning/{path}',
        'file': lambda path: Path(path).expanduser().resolve(),
    }

    def __init__(self):
        """
        Initialize ResourceManager.
        """
        self.cache_dir = ResourceManager.CACHE_DIR
        self.url_schemes = ResourceManager.BASE_URL_SCHEMES
        self.max_cache_size = ResourceManager.MAX_CACHE_SIZE
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
                return Path(uri).expanduser().resolve()
            else:
                copy(uri, output_path)
                return output_path

        resolved_uri = self._resolve_uri(parsed_uri)

        if isinstance(resolved_uri, Path):
            if output_path is None:
                return resolved_uri
            else:
                copy(resolved_uri, output_path)
                return output_path

        if output_path is None:
            parsed_path = parsed_uri.path
            if parsed_path[0] == '/':
                parsed_path = parsed_path[1:]
            output_path = self.cache_dir / parsed_path

        output_path = output_path.resolve()

        # file already exists - check if its valid
        if output_path.exists():
            remote_hash_valid = self._validate_file_remote(
                resolved_uri, output_path
            )
            local_hash_valid = self._validate_file_local(output_path)

            if remote_hash_valid or (
                remote_hash_valid is None and local_hash_valid
            ):
                if local_hash_valid is not True:
                    self._save_file_checksum(output_path)

                self.log.info(f'Using cached: {output_path}')
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._download_resource(resolved_uri, output_path)

        if self._validate_file_remote(resolved_uri, output_path) is False:
            raise ChecksumVerifyError(
                f'Error downloading file {uri} from {resolved_uri}. Invalid '
                'checksum.'
            )
        self._save_file_checksum(output_path)

        return output_path

    def set_cache_dir(self, cache_dir_path: Path):
        """
        Set the cache directory path and creates it if not exists.

        Parameters
        ----------
        cache_dir_path : Path
            Path to be set as cache directory.
        """
        self.cache_dir = cache_dir_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        Return list with cached files.

        Returns
        -------
        List[Path] :
            List of cached files.
        """
        result = []

        for cached_file in self.cache_dir.glob('**/*'):
            if (
                cached_file.is_file()
                and f'.{self.HASHING_ALGORITHM}' not in cached_file.suffixes
            ):
                result.append(cached_file.resolve())

        return result

    def clear_cache(self):
        """
        Remove all cached files
        """
        rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir()

    def _resolve_uri(self, parsed_uri: ParseResult) -> Union[str, Path]:
        """
        Resolve provided URI.

        Parameters
        ----------
        parsed_uri : ParseResult
            Parsed URI.

        Union[str, Path]
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
            None if checksum cannot be validated, otherwise True if file
            checksum is valid.
        """
        checksum_url = f'{url}.{self.HASHING_ALGORITHM}'

        response = requests.get(checksum_url, allow_redirects=True)
        if 200 != response.status_code:
            self.log.warning(f'Cannot verify {file_path} checksum')
            return None

        remote_hash = response.content.decode().strip().lower()
        self.log.debug(f'{url} {remote_hash=}')

        file_hash = self._compute_file_checksum(file_path)
        self.log.debug(f'{file_path} {file_hash=}')

        return file_hash == remote_hash

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
            None if checksum cannot be validated, otherwise True if file
            checksum is valid.
        """
        hash_file_path = file_path.with_suffix(
            file_path.suffix + f'.{self.HASHING_ALGORITHM}'
        )

        if not hash_file_path.exists():
            return None

        with open(hash_file_path, 'r') as checksum_file:
            local_hash = checksum_file.read().strip().lower()
        self.log.debug(f'{file_path} {local_hash=}')

        file_hash = self._compute_file_checksum(file_path)
        self.log.debug(f'{file_path} {file_hash=}')

        return file_hash == local_hash

    def _save_file_checksum(self, file_path: Path):
        """
        Save file checksum.

        Parameters
        ----------
        file_path : Path
            Path to the local file.
        """
        file_hash = self._compute_file_checksum(file_path)

        hash_file_path = file_path.with_suffix(
            file_path.suffix + f'.{self.HASHING_ALGORITHM}'
        )

        with open(hash_file_path, 'w') as checksum_file:
            checksum_file.write(file_hash)

    def _download_resource(self, url: str, output_path: Path):
        """
        Download resource from given URL.

        Parameters
        ----------
        url : str
            Resource URL.
        output_path : Path
            Path where the resource should be saved.
        """
        if self.cache_dir in output_path.parents:
            response = requests.head(url)
            if (
                response.status_code == 200
                and 'Content-Length' in response.headers
            ):
                self._free_cache(int(response.headers['Content-Length']))
            else:
                self.log.warning('Cannot read file size before downloading')

        download_url(url, output_path)

        if self.cache_dir in output_path.parents:
            # free cache in case the file size could not be read before
            # downloading
            self._free_cache(0)

    def _free_cache(self, required_free: int):
        """
        Free cache space.

        Parameters
        ----------
        required_free : int
            Amount of bytes that need to be available after freeing space.

        Raises
        ------
        ValueError :
            Raised when required free space is bigger that max cache size.
        """
        if required_free > self.max_cache_size:
            raise ValueError(f'Required free space too big: {required_free} B')
        cached_files = self.list_cached_files()
        cached_files.sort(key=lambda f: f.stat().st_mtime)
        cache_size = sum(f.stat().st_size for f in cached_files)

        while cache_size > self.max_cache_size - required_free:
            file = cached_files.pop(0)
            cache_size -= file.stat().st_size
            file.unlink()

    def _compute_file_checksum(self, file_path: Path) -> str:
        """
        Compute file checksum.

        Parameters
        ----------
        file_path : Path
            Path to the local file.

        Returns
        -------
        str :
            Computed checksum as in hex format.
        """
        hash_algo = getattr(hashlib, self.HASHING_ALGORITHM)()

        with open(file_path, 'rb') as file:
            while True:
                data = file.read(hash_algo.block_size)
                if not data:
                    break
                hash_algo.update(data)

        return hash_algo.hexdigest().lower()


class ResourceURI(Path):
    """
    Handle access to resource used in Kenning.
    """

    _flavour = type(Path())._flavour

    def __new__(cls, uri_or_path: Union[str, Path, 'ResourceURI']):
        if isinstance(uri_or_path, str) and ':/' in uri_or_path:
            uri = urlparse(uri_or_path)
            path = uri.path
        elif isinstance(uri_or_path, cls):
            uri = uri_or_path._uri
            path = Path(uri_or_path)
        else:
            uri = None
            path = Path(uri_or_path).expanduser().resolve()

        instance = super().__new__(cls, path)
        instance._uri = uri

        return instance

    @property
    def uri(self) -> str:
        """
        Get URI of the resource.
        """
        if self._uri is None:
            return None

        return self._uri._replace(path=str(self)).geturl()

    def get_resource(self, output_path: Optional[Path] = None) -> Path:
        """
        Retrieve resource and returns path to it.

        Parameters
        ----------
        output_path : Optional[Path]
            If specified, the resource will be download there.

        Returns
        -------
        Path :
            Path to the downloaded resource.
        """
        if self._uri is None:
            return Path(self)

        return ResourceManager().get_resource(self.uri, output_path)

    def get_path(self) -> Path:
        """
        Return path to the resource.

        Returns
        -------
        Path :
            Path to the resource.
        """
        if self._uri is None:
            return Path(self)

        path = ResourceManager().cache_dir / self.relative_to('/')

        return path

    @property
    def parent(self) -> 'ResourceURI':
        """
        Get parent of the URI.
        """
        ret = super().parent
        ret._uri = self._uri
        return ret

    def with_suffix(self, suffix: str) -> 'ResourceURI':
        """
        Return new URI with changed suffix.

        Parameters
        ----------
        suffix : str
            New suffix to be used.

        Returns
        -------
        ResourceURI :
            URI with changed suffix.
        """
        ret = super().with_suffix(suffix)
        ret._uri = self._uri
        return ret

    def with_name(self, name: str) -> 'ResourceURI':
        """
        Return new URI with changed name.

        Parameters
        ----------
        name : str
            New name to be used.

        Returns
        -------
        ResourceURI :
            URI with changed name.
        """
        ret = super().with_name(name)
        ret._uri = self._uri
        return ret

    def with_stem(self, stem: str) -> 'ResourceURI':
        """
        Return new URI with changed stem.

        Parameters
        ----------
        stem : str
            New stem to be used.

        Returns
        -------
        ResourceURI :
            URI with changed stem.
        """
        ret = super().with_stem(stem)
        ret._uri = self._uri
        return ret


class Resources(object):
    """
    Dictionary of resources.
    """

    def __init__(self, resources_uri: Dict[str, Any]):
        """
        Dictionary of resources. The keys can be strings or tuples of strings.

        Parameters
        ----------
        resources_uri : Dict[str, Any]
            Nested dictionary of resources.
        """
        self._resources_uri = resources_uri

    def __getitem__(self, keys: Union[Tuple[str], str]) -> Path:
        if isinstance(keys, str):
            keys = (keys,)

        resources_uri = self._resources_uri
        for key in keys:
            if not isinstance(resources_uri, dict) or key not in resources_uri:
                raise KeyError(f'Invalid key: {keys}')
            resources_uri = resources_uri[key]

        if isinstance(resources_uri, str):
            return ResourceManager().get_resource(resources_uri)
        if isinstance(resources_uri, ResourceURI):
            return resources_uri.get_resource()

        raise KeyError(f'Invalid key: {keys}')

    def __setitem__(self, keys: Union[Tuple[str], str], value: str):
        if isinstance(keys, str):
            keys = (keys,)
        if keys in self.keys():
            raise KeyError(f'Resource {keys} already exists')
        resources_uri = self._resources_uri
        for key in keys[:-1]:
            resources_uri = resources_uri[key]
        resources_uri[keys[-1]] = value

    def __contains__(self, keys: Union[Tuple[str], str]) -> bool:
        if isinstance(keys, str):
            keys = (keys,)
        return keys in self.keys()

    def __len__(self):
        return len(self.keys())

    def keys(self) -> List[Tuple[str, ...]]:
        """
        Return all resources' keys.

        Returns
        -------
        List[Tuple[str, ...] :
            List of available resources' keys.
        """
        result = []

        def get_keys(resources_uri: dict, keys: list = []):
            for key, value in resources_uri.items():
                if isinstance(value, (str, ResourceURI)):
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