# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Original SparseGPT class comes from repository:
https://github.com/IST-DASLab/sparsegpt.

Only nuanced changes were made to the original file.
The original file was licensed under Apache-2.0.
"""


import math

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    """
    Implementation of the SparseGPT and GPTQ algorithms.

    This class is used to optimize the weights of a given layer
    in a transformer model. The optimization is done by applying
    the SparseGPT and gptq algorithms to the weights of the layer.

    The layer is passed to the constructor of the class, and then
    the add_batch method is used to add the input data to the
    calibration dataset. After adding all the input data, the
    optimize method is called to optimize the weights of the layer.

    * SparseGPT paper: https://arxiv.org/abs/2301.00774
    * SparseGPT repository: https://github.com/IST-DASLab/sparsegpt.
    * GPTQ paper: https://arxiv.org/abs/2210.17323

    """

    def __init__(self, layer: nn.Module):
        """
        Constructor of the SparseGPT class.

        Parameters
        layer: nn.Module
            The layer to be optimized.

        """
        self.layer = layer
        self.dev = self.layer.weight.device

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.n_samples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Add a batch of input data which is used as a calibration
        data for the optimization process.

        New data can be added multiple times before calling the
        optimize method.

        Parameters
        ----------
        inp : torch.Tensor
            The input data to be added to the calibration dataset.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # Updating Hessian matrix using the new input data
        self.H *= self.n_samples / (self.n_samples + tmp)
        self.n_samples += tmp
        inp = math.sqrt(2 / self.n_samples) * inp.float()
        self.H += inp.matmul(inp.t())

    def optimize(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> tuple[float]:
        """
        Optimize the weights of the layer using the SparseGPT
        and gptq algorithms. The optimization is done in place,
        so the weights of the layer are modified.

        Parameters
        ----------
        sparsity : float
            The desired sparsity level of the optimized weights.
            The value should be between 0 and 1, where 0 means
            no sparsity and 1 means full sparsity.
        prunen : int
            Value used for semi-structured pruning. The parameter
            specifies the number of weights to be pruned in
            each block of weights.
        prunem : int
            Value used for semi-structured pruning. The parameter
            specifies the size of the blocks
        blocksize : int
            Number of weights in each block. The number of columns
            of the Hessian matrix is divided into blocks, and the
            optimization is done separately for each block.
        percdamp : float
            Damping factor used to stabilize the optimization process.

        Returns
        -------
        tuple[float]
            A tuple containing the loss value after the optimization
            and additional information about quantization process.
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        scale = []
        zero = []
        g_idx = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][
                        int(tmp.numel() * sparsity)
                    ]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            if hasattr(self, "quantizer"):
                self.quantizer.find_params(W[:, i1:i2], weight=True)
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                if hasattr(self, "quantizer"):
                    q = self.quantizer.quantize(q.unsqueeze(1)).flatten()

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (
                            torch.diag(Hinv1)[i : (i + prunem)].reshape(
                                (
                                    1,
                                    -1,
                                )
                            )
                        )
                        ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )

                # This has to go after the quantization to make sure that
                # the zeroed weights are not quantized
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if hasattr(self, "quantizer"):
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)

            # act_order is not supported for now, so g_idx
            # is statically created to be compatible with autogptq format
            g_idx = [i // blocksize for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32)

        return torch.sum(Losses).item(), scale, zero, g_idx

    def free(self):
        """
        Free the memory used by the Hessian matrix.
        """
        self.H = None
        torch.cuda.empty_cache()
