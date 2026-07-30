# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Flax models and conversion to other formats.
"""


from kenning.core.converter import ModelConverter
from kenning.modelwrappers.frameworks.flax import FlaxWrapper


class FlaxConverter(ModelConverter):
    """
    The Flax model converter.
    """

    source_format: str = "flax"

    def to_flax(
        self,
        model: FlaxWrapper,
        **kwargs,
    ) -> FlaxWrapper:
        return model

    def to_mlir(
        self,
        model: FlaxWrapper,
        **kwargs,
    ) -> str:
        """
        Converts Flax model to MLIR.

        Parameters
        ----------
        model : FlaxWrapper
            Model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        str
            Generated MLIR.

        Raises
        ------
        ValueError
            When passed model is not prepared.
        """
        import jax
        import jax.export
        import jax.numpy as jnp

        if model.model is None:
            raise ValueError("Model is not prepared")

        model.model.eval()

        io_spec = model.get_io_specification()
        input_spec = io_spec.get("processed_input", io_spec["input"])

        exported = jax.export.export(jax.jit(lambda x: model.model(x)))(
            jnp.ones(input_spec[0]["shape"])
        )

        return exported.mlir_module()
