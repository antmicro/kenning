# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provide resource manager responsible for downloading and caching resources.
"""
import hashlib
import os
import re
import tarfile
from importlib.metadata import PackageNotFoundError, version
from inspect import getfullargspec
from pathlib import Path
from shutil import copy, rmtree
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, uses_params
from urllib.request import urlopen
from zipfile import ZipFile

import requests
from tqdm import tqdm

from kenning.utils.logger import KLogger, LoggerProgressBar, download_url
from kenning.utils.singleton import Singleton


def extract_tar(target_dir: Path, src_path: Path):
    """
    Extract the tar file to the provided target directory.

    Parameters
    ----------
    target_dir : Path
        Path to the target directory where extracted files will be saved.
    src_path : Path
        Path to the tar file.
    """
    with tarfile.open(src_path) as tf:
        tf.extractall(target_dir)


def extract_zip(target_dir: Path, src_path: Path):
    """
    Extract the ZIP file to the provided target directory.

    Parameters
    ----------
    target_dir : Path
        Path to the target directory where extracted files will be saved.
    src_path : Path
        Path to the ZIP file.
    """
    with LoggerProgressBar() as logger_progress_bar, ZipFile(
        src_path, "r"
    ) as zip:
        for f in tqdm(
            iterable=zip.namelist(),
            total=len(zip.namelist()),
            file=logger_progress_bar,
        ):
            zip.extract(member=f, path=target_dir)


def _gh_converter(netloc: str, path: str, params_dict: Dict[str, str]) -> str:
    netloc = netloc.split(":")
    return (
        f'https://raw.githubusercontent.com/{netloc[0]}/{netloc[1]}/'
        f'{params_dict["branch"]}{path}'
    )


def _get_cache_dir(env_var: str) -> Path:
    """
    Return cache directory.

    Parameters
    ----------
    env_var : str
        Name of the environment variable with cache dir.

    Returns
    -------
    Path
        Path to the cache.
    """
    cache_dir = os.environ.get(env_var, "")

    if cache_dir:
        cache_dir = Path(cache_dir)
    else:
        cache_dir = Path.home() / ".kenning"

    return cache_dir.expanduser().resolve()


class ResourceManager(metaclass=Singleton):
    """
    Download and cache resources used by Kenning.
    """

    CACHE_DIR_ENV_VAR = "KENNING_CACHE_DIR"
    MAX_CACHE_SIZE_ENV_VAR = "KENNING_MAX_CACHE_SIZE"

    CACHE_DIR = _get_cache_dir(CACHE_DIR_ENV_VAR)

    # 50 GB by default
    MAX_CACHE_SIZE = int(
        os.environ.get(MAX_CACHE_SIZE_ENV_VAR, 50_000_000_000)
    )

    HASHING_ALGORITHM = "md5"

    BASE_URL_SCHEMES = {
        "http": None,
        "https": None,
        "kenning": "https://dl.antmicro.com/kenning/{path}",
        "gh": _gh_converter,
        "demo-scenario": (
            "https://raw.githubusercontent.com/"
            "antmicro/kenning/main/scripts/{path}"
        ),
        "file": lambda uri: Path(uri.path).expanduser().resolve(),
    }

    KENNING_RESOURCES_VERSION_URL = "kenning:///VERSION"

    def __init__(self):
        """
        Initialize ResourceManager.
        """
        self.cache_dir = ResourceManager.CACHE_DIR
        self.url_schemes = ResourceManager.BASE_URL_SCHEMES
        self.max_cache_size = ResourceManager.MAX_CACHE_SIZE
        for schema in self.url_schemes.keys():
            if schema not in uses_params:
                uses_params.append(schema)

        # find parameters indexed by numbers, slices or string
        # (e.g. netloc[0], path[2:], params["branch"])
        self.param_pattern = re.compile(
            r"(\{"  # opening brace
            r"([a-zA-Z][a-zA-Z0-9_]+)"  # match param name
            r"(?:\[((?:"  # opening bracket
            r"(?:[+-]?[0-9]+)?(?::[+-]?[0-9]*){0,2})"  # match number or slice
            r"|(?:\"[a-zA-Z0-9]+\")"  # or match string
            r")\])?"  # closing bracket
            r"\})"  # closing brace
        )
        # find parameters indexed (or not) by anything
        # used only to check if there are some params left after parsing
        self.params_any_index_pattern = re.compile(
            r"(\{([a-zA-Z][a-zA-Z0-9_]+)(?:\[[^\[\]]*\])?\})"
        )

        self.kenning_resources_version_validated = False

    def validate_resources_version(self):
        """
        Retrieve Kenning resources version
        and check if it is compatible with currently used Kenning.
        """
        # Validate version only at the first call
        if self.kenning_resources_version_validated:
            return
        self.kenning_resources_version_validated = True
        current_version = None
        try:
            current_version = version("kenning")
        except PackageNotFoundError:
            try:
                import pkg_resources

                current_version = pkg_resources.get_distribution(
                    "kenning"
                ).version
            except ModuleNotFoundError:
                pass
        uri = self._resolve_uri(
            urlparse(ResourceManager.KENNING_RESOURCES_VERSION_URL)
        )
        try:
            with urlopen(uri) as request:
                resources_version = request.readline().decode("utf-8")
        except HTTPError:
            KLogger.error("Kenning resources version cannot be validated")
            return
        if resources_version.rstrip("\n") != current_version:
            KLogger.error(
                "The newer version of Kenning is available, "
                "some resources may not be compatible with current one"
            )

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
        Path
            Path to the retrieved resource.

        Raises
        ------
        ChecksumVerifyError :
            Raised when downloaded file has invalid checksum
        """
        # check if file is already cached
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme == "kenning":
            self.validate_resources_version()

        # no scheme in URI - treat as path string
        if "" == parsed_uri.scheme:
            if output_path is None:
                return Path(uri).expanduser().resolve()
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                copy(uri, output_path)
                return output_path

        resolved_uri = self._resolve_uri(parsed_uri)

        if isinstance(resolved_uri, Path):
            if output_path is None:
                return resolved_uri
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                copy(resolved_uri, output_path)
                return output_path

        if output_path is None:
            output_path = self.cache_dir / Path(parsed_uri.path).relative_to(
                "/"
            )

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

                KLogger.info(f"Using cached: {output_path}")
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._download_resource(resolved_uri, output_path)

        if self._validate_file_remote(resolved_uri, output_path) is False:
            raise ChecksumVerifyError(
                f"Error downloading file {uri} from {resolved_uri}. Invalid "
                "checksum."
            )
        self._save_file_checksum(output_path)

        return output_path

    def set_max_cache_size(self, max_cache_size: int):
        """
        Set the max cache size.

        Parameters
        ----------
        max_cache_size : int
            Max cache size in bytes.
        """
        self.max_cache_size = max_cache_size

    def set_cache_dir(self, cache_dir_path: Path):
        """
        Set the cache directory path and creates it if not exists.

        Parameters
        ----------
        cache_dir_path : Path
            Path to be set as cache directory.
        """
        self.cache_dir = cache_dir_path.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def add_custom_url_schemes(
        self, custom_url_schemes: Dict[str, Optional[Union[str, Callable]]]
    ):
        """
        Add user defined URL schemes.

        Parameters
        ----------
        custom_url_schemes : Dict[str, Optional[Union[str, Callable]]]
            Dictionary with custom url schemes entries. Each entry consists of
            schema and corresponding conversion. Conversion can be None, string
            pattern or callable returning string.
        """
        self.url_schemes = dict(self.url_schemes, **custom_url_schemes)
        for schema in custom_url_schemes.keys():
            if schema not in uses_params:
                uses_params.append(schema)

    def list_cached_files(self) -> List[Path]:
        """
        Return list with cached files.

        Returns
        -------
        List[Path]
            List of cached files.
        """
        result = []

        for cached_file in self.cache_dir.glob("**/*"):
            if (
                cached_file.is_file()
                and f".{self.HASHING_ALGORITHM}" not in cached_file.suffixes
            ):
                result.append(cached_file.resolve())

        return result

    def clear_cache(self):
        """
        Remove all cached files.
        """
        rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_uri(self, parsed_uri: ParseResult) -> Union[str, Path]:
        """
        Resolve provided URI.

        Parameters
        ----------
        parsed_uri : ParseResult
            Parsed URI.

        Returns
        -------
        Union[str, Path]
            Resolved path to resource.

        Raises
        ------
        ValueError :
            Raised when provided URI has invalid scheme or the defined
            scheme's conversion is invalid.
        """
        try:
            converter = self.url_schemes[parsed_uri.scheme]
        except KeyError:
            raise ValueError(
                f"Invalid URI scheme provided: {parsed_uri.scheme}"
            )

        if converter is None:
            return parsed_uri.geturl()

        if isinstance(converter, str):
            return self._handle_str_converter(parsed_uri, converter)

        if callable(converter):
            return self._handle_callable_converter(parsed_uri, converter)

        raise ValueError(f"Invalid conversion for scheme {parsed_uri.scheme}")

    def _handle_str_converter(
        self, parsed_uri: ParseResult, converter: str
    ) -> str:
        """
        Handle string converter parsing.

        Parameters
        ----------
        parsed_uri : ParseResult
            Parsed URI.
        converter : str
            Converter format.

        Returns
        -------
        str
            Parsed string.

        Raises
        ------
        ValueError
            Raised when conversion of values of arguments fails
        """
        params = set(self.param_pattern.findall(converter))

        for param_str, param, index in params:
            if not hasattr(parsed_uri, param):
                raise ValueError(f"Invalid param name {param}")

            if index == "" and ("[" in param_str or "]" in param_str):
                raise ValueError(f"Invalid index for param: {param_str}")

            param_value = getattr(parsed_uri, param)
            param_as_list = None
            param_as_dict = None
            if "path" == param:
                sep = "/"
                param_as_list = self._param_as_list(param_value, sep)
            elif "netloc" == param:
                sep = "."
                param_as_list = self._param_as_list(param_value, sep)
            elif "params" == param and "=" in param_value:
                sep = ";"
                param_as_dict = self._param_as_dict(param_value, sep)
            elif "query" == param and "=" in param_value:
                sep = "&"
                param_as_dict = self._param_as_dict(param_value, sep)
            elif len(index):
                raise ValueError(
                    f"Invalid conversion for scheme {parsed_uri.scheme}, "
                    f"param {param} does not support indexing and slicing"
                )

            if ":" in index:
                # slice index
                if param_as_list is None:
                    raise ValueError(f"Invalid param: {param}")
                # convert slice str to slice object
                param_slice = slice(
                    *[
                        {True: lambda n: None, False: int}[x == ""](x)
                        for x in (index.split(":") + ["", "", ""])[:3]
                    ]
                )
                value = sep.join(param_as_list[param_slice])
            elif index.isnumeric():
                # numeric index
                if param_as_list is None:
                    raise ValueError(f"Invalid param: {param}")
                value = param_as_list[int(index)]
            elif len(index):
                # string index
                index = index.strip('"')
                if param_as_dict is None:
                    raise ValueError(f"Invalid param: {param}")
                value = param_as_dict[index]
            else:
                value = param_value

            converter = converter.replace(param_str, value)

        # check if there are any params left
        params_check = set(self.params_any_index_pattern.findall(converter))
        if len(params_check):
            raise ValueError(
                "Invalid syntax for params: " f"{[p[1] for p in params_check]}"
            )

        return converter

    def _handle_callable_converter(
        self, parsed_uri: ParseResult, converter: Callable
    ) -> str:
        """
        Handle string converter parsing.

        Parameters
        ----------
        parsed_uri : ParseResult
            Parsed URI.
        converter : Callable
            Converter function.

        Returns
        -------
        str
            Parsed string.

        Raises
        ------
        ValueError
            Raised when invalid parameter name is provided
        """
        callable_arg_spec = getfullargspec(converter)

        args = []

        for arg in callable_arg_spec.args:
            if arg == "uri":
                args.append(parsed_uri)
            elif hasattr(parsed_uri, arg):
                args.append(getattr(parsed_uri, arg))
            elif "_" in arg:
                arg_name, arg_type = arg.rsplit("_", 1)
                if "netloc" == arg_name and "list" == arg_type:
                    args.append(self._param_as_list(parsed_uri.netloc, "."))
                elif "path" == arg_name and "list" == arg_type:
                    args.append(self._param_as_list(parsed_uri.path, "/"))
                elif "params" == arg_name and "dict" == arg_type:
                    args.append(self._param_as_dict(parsed_uri.params, ";"))
                elif "query" == arg_name and "dict" == arg_type:
                    args.append(self._param_as_dict(parsed_uri.query, "&"))
                else:
                    raise ValueError(
                        "Invalid parameter name {arg} in converter"
                    )
            else:
                raise ValueError("Invalid parameter name {arg} in converter")

        return converter(*args)

    @staticmethod
    def _param_as_list(param_value: str, sep: str):
        return param_value.strip(sep).split(sep)

    @staticmethod
    def _param_as_dict(param_value: str, sep: str):
        return {
            name: val
            for name_val in param_value.strip(sep).split(sep)
            for name, val in (name_val.split("="),)
        }

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
        Optional[bool]
            None if checksum cannot be validated, otherwise True if file
            checksum is valid.
        """
        checksum_url = f"{url}.{self.HASHING_ALGORITHM}"

        response = requests.get(checksum_url, allow_redirects=True)
        if 200 != response.status_code:
            KLogger.warning(f"Cannot verify {file_path} checksum")
            return None

        remote_hash = response.content.decode().strip().lower()
        KLogger.debug(f"{url} {remote_hash=}")

        file_hash = self._compute_file_checksum(file_path)
        KLogger.debug(f"{file_path} {file_hash=}")

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
        Optional[bool]
            None if checksum cannot be validated, otherwise True if file
            checksum is valid.
        """
        hash_file_path = file_path.with_suffix(
            file_path.suffix + f".{self.HASHING_ALGORITHM}"
        )

        if not hash_file_path.exists():
            return None

        with open(hash_file_path, "r") as checksum_file:
            local_hash = checksum_file.read().strip().lower()
        KLogger.debug(f"{file_path} {local_hash=}")

        file_hash = self._compute_file_checksum(file_path)
        KLogger.debug(f"{file_path} {file_hash=}")

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
            file_path.suffix + f".{self.HASHING_ALGORITHM}"
        )

        with open(hash_file_path, "w") as checksum_file:
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
                and "Content-Length" in response.headers
            ):
                self._free_cache(int(response.headers["Content-Length"]))
            else:
                KLogger.warning("Cannot read file size before downloading")

        KLogger.debug(f"Downloading {url} to {output_path}")
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
            raise ValueError(f"Required free space too big: {required_free} B")
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
        str
            Computed checksum as in hex format.
        """
        hash_algo = getattr(hashlib, self.HASHING_ALGORITHM)()

        with open(file_path, "rb") as file:
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
    _uri: Optional[ParseResult]

    def __new__(cls, uri_or_path: Union[str, Path, "ResourceURI"]):
        if isinstance(uri_or_path, str) and ":/" in uri_or_path:
            uri = urlparse(uri_or_path)
            path = ResourceManager().cache_dir / Path(uri.path).relative_to(
                "/"
            )
        elif isinstance(uri_or_path, cls):
            uri = uri_or_path._uri
            path = Path(uri_or_path)
        else:
            uri = None
            path = Path(uri_or_path).expanduser().resolve()

        instance = super().__new__(cls, path)
        instance._uri = uri

        if instance.uri is not None:
            try:
                ResourceManager().get_resource(instance.uri, Path(instance))
            except URLError:
                # ignore the exception as the __new__ might be called during
                # URI manipulations (by using with_suffix etc.) and the URI
                # could be invalid as some stage of such operations
                pass

        return instance

    @property
    def uri(self) -> Optional[str]:
        """
        Get URI of the resource.
        """
        if self._uri is None:
            return None

        return self._uri._replace(
            path=f"/{str(Path(self).relative_to(ResourceManager().cache_dir))}"
        ).geturl()

    @property
    def parent(self) -> "ResourceURI":
        """
        Get parent of the URI.
        """
        ret = super().parent
        ret._uri = self._uri
        return ret

    def with_suffix(self, suffix: str) -> "ResourceURI":
        """
        Return new URI with changed suffix.

        Parameters
        ----------
        suffix : str
            New suffix to be used.

        Returns
        -------
        ResourceURI
            URI with changed suffix.
        """
        ret = super().with_suffix(suffix)
        ret._uri = self._uri
        return ResourceURI(ret)

    def with_name(self, name: str) -> "ResourceURI":
        """
        Return new URI with changed name.

        Parameters
        ----------
        name : str
            New name to be used.

        Returns
        -------
        ResourceURI
            URI with changed name.
        """
        ret = super().with_name(name)
        ret._uri = self._uri
        return ResourceURI(ret)

    def with_stem(self, stem: str) -> "ResourceURI":
        """
        Return new URI with changed stem.

        Parameters
        ----------
        stem : str
            New stem to be used.

        Returns
        -------
        ResourceURI
            URI with changed stem.
        """
        ret = super().with_stem(stem)
        ret._uri = self._uri
        return ResourceURI(ret)

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return f"ResourceURI({self.uri})"


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

    def __getitem__(self, keys: Union[Tuple[str, ...], str]) -> Path:
        if isinstance(keys, str):
            keys = (keys,)

        resources_uri = self._resources_uri
        for key in keys:
            if not isinstance(resources_uri, dict) or key not in resources_uri:
                raise KeyError(f"Invalid key: {keys}")
            resources_uri = resources_uri[key]

        if isinstance(resources_uri, str):
            return ResourceManager().get_resource(resources_uri)
        if isinstance(resources_uri, ResourceURI):
            return resources_uri

        raise KeyError(f"Invalid key: {keys}")

    def __setitem__(self, keys: Union[Tuple[str], str], value: str):
        if isinstance(keys, str):
            keys = (keys,)
        if keys in self.keys():
            raise KeyError(f"Resource {keys} already exists")
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
        List[Tuple[str, ...]]
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


PathOrURI = Union[Path, ResourceURI]
