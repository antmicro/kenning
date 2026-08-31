# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Custom op handlers for tf2onnx library.

Functions from this module are registered with decorators for internal usage in
the library and are not meant to be used directly.
"""
from typing import Any

import numpy as np
from tf2onnx import utils
from tf2onnx.graph import Graph, Node
from tf2onnx.handler import tf_op, tfl_op


@tfl_op("TFL_GELU", tf_op="Gelu")
class TflDirectOp:
    """
    Class that handles conversion from TFLite to TF ops when the op is
    identical in both representations and the graph doesn't need any
    modifications.
    """

    @classmethod
    def to_tf(cls, ctx: Graph, node: Node, **kwargs: Any):
        """
        Empty method for ops that don't need any graph modifications.

        Parameters
        ----------
        ctx : Graph
            Graph that is being converted.
        node : Node
            Current op node that is being converted.
        **kwargs : Any
            Not used, copied for library compatibility.
        """
        pass


@tf_op(["Gelu"])
class GeluOp:
    """
    Class for converting Gelu Node from generic/TF representation into an ONNX
    compatible one.
    """

    @classmethod
    def version_9(cls, ctx: Graph, node: Node, **kwargs: Any):
        """
        Converts the Gelu Node from generic/TF representation into an ONNX
        compatible one.

        ONNX only supports Gelu directly from opset version 20, which is not
        supported by tf2onnx (or much other tooling), so this function
        decomposes Gelu to the underlying math.

        Parameters
        ----------
        ctx : Graph
            Graph that is being converted.
        node : Node
            Current Gelu op node that is being converted.
        **kwargs : Any
            Not used, copied for library compatibility.
        """
        x = node.input[0]

        # constant names
        prefix = "__kenning__"

        sqrt2 = prefix + "gelu_sqrt2"
        cubic_coef = prefix + "gelu_cubic_coef"
        slope_correction = prefix + "gelu_slope_correction"
        half = prefix + "gelu_half"
        one = prefix + "gelu_one"
        three = prefix + "gelu_three"

        sqrt2_node = ctx.get_node_by_output(sqrt2)

        # Only insert the constants for gelu once
        if sqrt2_node is None:
            input_dtype = utils.map_onnx_to_numpy_type(ctx.get_dtype(x))

            ctx.make_const(sqrt2, np.array([np.sqrt(2)], dtype=input_dtype))
            ctx.make_const(
                cubic_coef,
                np.array([0.044715], dtype=input_dtype),
            )
            ctx.make_const(
                slope_correction,
                np.array([np.sqrt(2 / np.pi)], dtype=input_dtype),
            )
            ctx.make_const(half, np.array([0.5], dtype=input_dtype))
            ctx.make_const(one, np.array([1], dtype=input_dtype))
            ctx.make_const(three, np.array([3], dtype=input_dtype))

        approximate_erf = bool(node.get_attr_value("approximate"))

        if approximate_erf:
            x_pow_three = ctx.make_node("Pow", [x, three])
            pow_corrected = ctx.make_node(
                "Mul", [x_pow_three.output[0], cubic_coef]
            )
            pow_plus_x = ctx.make_node("Add", [pow_corrected.output[0], x])
            mul_scaling_factor = ctx.make_node(
                "Mul", [pow_plus_x.output[0], slope_correction]
            )
            erf = ctx.make_node("Tanh", [mul_scaling_factor.output[0]])
        else:
            x_scaled = ctx.make_node("Div", [x, sqrt2])

            erf = ctx.make_node("Erf", [x_scaled.output[0]])

        erf_plus_one = ctx.make_node("Add", [erf.output[0], one])

        mul_x = ctx.make_node("Mul", [erf_plus_one.output[0], x])

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        old_output = node.output[0]

        ctx.remove_node(node.name)

        ctx.make_node(
            "Mul",
            [mul_x.output[0], half],
            outputs=[old_output],
            shapes=shapes,
            dtypes=dtypes,
        )
