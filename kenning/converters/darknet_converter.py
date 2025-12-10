# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Darknet model and conversion to other formats.
"""


from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

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

    def to_tvm(
        self,
        input_shapes: Dict,
        dtypes: Dict,
        libdarknet_path: Optional[str],
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts darknet file to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        dtypes: Dict
            Mapping from input name to input dtype.
        libdarknet_path: Optional[str]
            Path to the darknet library.

        Returns
        -------
        mod: tvm.IRModule
            The relay module
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
            dtype = list(dtypes.values())[0]
        except IndexError:
            raise IndexError("No dtype in the input specification")

        from tvm.relay.testing.darknet import __darknetffi__

        if not libdarknet_path:
            KLogger.fatal(
                "The darknet converter requires libdarknet.so library. "
                "Provide the path to it using --libdarknet-path flag"
            )
            raise ConversionError("Provide libdarknet.so library")
        try:
            lib = __darknetffi__.dlopen(str(libdarknet_path))
        except OSError as e:
            raise ConversionError(e)
        net = lib.load_network(
            str(self.source_model_path.with_suffix(".cfg")).encode("utf-8"),
            str(self.source_model_path).encode("utf-8"),
            0,
        )
        return relay.frontend.from_darknet(
            net, dtype=dtype, shape=input_shapes["input"]
        )
