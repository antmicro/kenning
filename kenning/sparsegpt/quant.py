# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Original Quantizer class implementation comes from repository:
https://github.com/AutoGPTQ/AutoGPTQ.

Only nuanced changes were made to the original file.
The original file was licensed under MIT.
"""

import torch
import torch.nn as nn

from kenning.core.exceptions import KenningOptimizerError


class Quantizer(nn.Module):
    """
    Implementation of the GPTQ algorithms. This is based on:
    * GPTQ repository: https://github.com/IST-DASLab/gptq,
    * GPTQ paper: https://arxiv.org/abs/2210.17323.

    Attributes
    ----------
    configured : bool
        Whether the quantizer is configured and ready to be used.
    zero : Optional[torch.Tensor]
        Zero point of the quantizer. Calculated when 'find_params()' is called.
    scale : Optional[torch.Tensor]
        Scale of the quantizer Calculated when 'find_params()' is called.
    """

    def __init__(self):
        """
        Initializes the quantizer.
        """
        super().__init__()

        shape = 1
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.configured = False

        self.zero = None
        self.scale = None

    def configure(
        self,
        bits: int,
        perchannel: bool = False,
        sym: bool = False,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ):
        """
        Configures the quantizer.

        Parameters
        ----------
        bits : int
            Target precision for quantization.
        perchannel : bool
            Whether to use per-channel quantization.
            If False, the same quantization parameters are used
            for all channels.
        sym : bool
            Whether to use symmetric quantization.
        mse : bool
            Whether to use MSE optimization for quantization parameters instead
            of using RTN formulas.
        norm : float
            Norm used for MSE optimization.
        grid : int
            Number of grid points used for MSE optimization.
        maxshrink : float
            Maximum shrinkage of MSE optimization grid.
        """
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.configured = True

    def find_params(self, x: torch.Tensor, weight: bool = False):
        """
        Finds quantization parameters for the input
        tensor using passed configuration.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be quantized.
        weight : bool
            Whether the input tensor is a weight tensor.

        Raises
        ------
        KenningOptimizerError
            If the quantizer is not configured yet.
        """
        if not self.configured:
            raise KenningOptimizerError(
                "Quantizer not configured yet. Call 'configure()' with "
                "appropriate parameters."
            )

        dev = x.device
        shape = x.shape
        self.maxq = self.maxq.to(dev)

        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = (
                    torch.round(-xmin1 / scale1) if not self.sym else self.zero
                )
                q = self._quantize(
                    x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                )
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
        elif len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        elif len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    @staticmethod
    def _quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        maxq: int,
    ) -> torch.Tensor:
        """
        Quantizes the input tensor using passed params.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be quantized.
        scale : torch.Tensor
            Scale of the quantizer.
        zero : torch.Tensor
            Zero point of the quantizer.
        maxq : int
            Maximum value in target precision.

        Returns
        -------
        torch.Tensor
            Quantized tensor.
        """
        if maxq < 0:
            return (x > scale / 2).float() * scale + (
                x < zero / 2
            ).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns quantized tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be quantized.

        Returns
        -------
        torch.Tensor
            Quantized tensor.
        """
        if self.ready():
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def ready(self) -> bool:
        """
        Heuristic to determine whether weights are quantized by checking if
        the scale is not zero.

        Returns
        -------
        bool
            True if the weights are quantized, False otherwise.
        """
        return torch.all(self.scale != 0)
