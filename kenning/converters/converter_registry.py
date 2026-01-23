# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides functionality for model converter registration
and alternative conversion path resolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import (
    ConversionError,
    ConverterAlreadyPresent,
    ConverterFormatNotSupported,
)
from kenning.core.model import ResourceURI
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI
from kenning.utils.singleton import Singleton


class ConverterRegistry(metaclass=Singleton):
    """
    Registry for model converters.

    Available conversions between formats
    are stored in graph, which stores
    Nodes with supported formats,
    each node points to other nodes
    describing available posibbilties
    in formats conversions.

    """

    def __init__(self) -> None:
        """
        Initialize the converter registry.
        """
        self._converters: Dict[str, ModelConverter] = {}
        self._graph: Dict[str, List[str]] = {}

    def register(self, converter_cls: ModelConverter) -> None:
        """
        Registers a converter class and extract supported conversions.


        Parameters
        ----------
        converter_cls : ModelConverter
            Converter class defining `source_format` and `to_*` methods.

        Raises
        ------
        ConverterFormatNotSupported
            Raised when trying to use unsupported conversion format.
        ConverterAlreadyPresent
            Raised when trying to register converter which is already
            registered.
        """
        src = converter_cls.source_format

        if src is None:
            raise ConverterFormatNotSupported(
                f"Source format for converter \
                          {converter_cls.__name__} is not defined."
            )

        if src in self._converters.keys():
            raise ConverterAlreadyPresent(
                f"Converter for format {src} is \
                            already present in converter registry."
            )

        self._converters[src] = converter_cls
        self._graph.setdefault(src, [])

        for name in dir(converter_cls):
            if not name.startswith("to_"):
                continue
            # Consider only methods defined in the specific converter class
            if name not in converter_cls.__dict__:
                continue

            dst = name[3:]
            if dst == src:
                continue

            self._graph[src].append(dst)

            self._graph.setdefault(dst, [])

    def _find_all_paths(
        self, src_format: str, dst_format: str
    ) -> List[List[str]]:
        """
        Find all conversion paths between two formats.

        Uses depth-first search on the conversion graph.

        Parameters
        ----------
        src_format : str
            Starting format.
        dst_format : str
            Target format.

        Returns
        -------
        List[List[str]]
            List of all valid conversion paths sorted by length.
        """
        if src_format not in self._graph or dst_format not in self._graph:
            return []

        all_paths = []

        def dfs(node, path):
            if node == dst_format:
                all_paths.append(path)
                return

            for neighbor in self._graph[node]:
                if neighbor not in path:
                    dfs(neighbor, path + [neighbor])

        dfs(src_format, [src_format])

        all_paths.sort(key=len)

        return all_paths

    def convert(
        self,
        model: Union[PathOrURI, Any],
        src_format: str,
        dst_format: str,
        **kwargs: Dict,
    ) -> Any:
        """
        Convert a model between two formats using registered converters.

        Parameters
        ----------
        model : Union[PathOrURI,Any]
            Path or identifier to the source model or model instance.
        src_format : str
            Source format name.
        dst_format : str
            Target format name.
        **kwargs : Dict
            Additional keyword arguments forwarded to each conversion step.

        Returns
        -------
        Any
            Final converted model.

        Raises
        ------
        ConversionError
            If no conversion path exists or all paths fail.
        """
        all_paths = self._find_all_paths(src_format, dst_format)

        if not all_paths:
            raise ConversionError(
                f"No conversion path from {src_format} to {dst_format}"
            )

        last_error = None

        current_model = (
            model if not isinstance(model, (Path, ResourceURI)) else None
        )
        source_path = model if current_model is None else None

        for path in all_paths:
            KLogger.debug(f"Trying conversion path: {' -> '.join(path)}")

            current_format = src_format

            try:
                for next_format in path:
                    converter_cls = self._converters[current_format]
                    converter = converter_cls(source_path)

                    method = getattr(converter, f"to_{next_format}")

                    current_model = method(model=current_model, **kwargs)
                    current_format = next_format

                KLogger.info(
                    f"Successfully converted model at \
                             {model} using {' -> '.join(path)}"
                )

                return current_model

            except Exception as e:
                KLogger.warning(
                    f"Path failed: {' -> '.join(path)}; reason: {e}"
                )
                last_error = e
                continue

        raise ConversionError(
            f"All conversion paths failed from {src_format} to {dst_format}"
        ) from last_error
