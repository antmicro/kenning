# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Darknet model and conversion to other formats.
"""


from typing import Any, TYPE_CHECKING, Dict, Optional, Tuple, Union

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import ConversionError
from kenning.utils.logger import KLogger

if TYPE_CHECKING:
    import tvm


class DarknetConverter(ModelConverter):
    """
    The Darknet model converter.
    """

    source_format = "darknet"

    def to_darknet(
        self,
        libdarknet_path: Optional[str] = None,
        model: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Loads the Darknet model using the libdarknet shared library.

        Parameters
        ----------
        libdarknet_path : Optional[str]
            Path to the darknet shared library (libdarknet.so).
        model : Optional[Any]
            Optional Darknet model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        Any
            A loaded Darknet model instance returned by libdarknet.

        Raises
        ------
        ConversionError
            Raised when libdarknet cannot be loaded.
        """
        from tvm.relay.testing.darknet import __darknetffi__

        if model is not None:
            return model

        if not libdarknet_path:
            KLogger.fatal(
                "The darknet converter requires libdarknet.so library. "
                "Provide the path to it using --libdarknet-path flag"
            )
            raise ConversionError("Provide libdarknet.so library")

        try:
            lib = __darknetffi__.dlopen(str(libdarknet_path))
        except OSError as e:
            raise ConversionError(f"Cannot load libdarknet: {e}")

        cfg_path = str(self.source_model_path.with_suffix(".cfg"))
        weights_path = str(self.source_model_path)

        model = lib.load_network(
            cfg_path.encode("utf-8"),
            weights_path.encode("utf-8"),
            0,
        )

        return model

    def to_tvm(
        self,
        input_shapes: Dict,
        dtypes: Dict,
        libdarknet_path: Optional[str],
        model: Optional[Any] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts darknet file to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        dtypes: Dict
            Mapping from input name to input dtype.
        model : Optional["Any"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        mod: tvm.IRModule
            The relay module.
        params: Union[Dict, str]
            Parameters dictionary to be used by relay module.

        Raises
        ------
        ConversionError
            Raised when libdarknet shared library cannot be loaded.
        IndexError
            Raised when no dtype is provided in the IO specification.
        """
        import tvm.relay as relay

        try:
            dtype = next(iter(dtypes.values()))
        except StopIteration:
            raise IndexError("No dtype in the input specification")

        model = self.to_darknet(model=model, **kwargs)

        return relay.frontend.from_darknet(
            model,
            dtype=dtype,
            shape=input_shapes["input"],
        )
