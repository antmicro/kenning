# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Ai8x model and conversion to other formats.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from kenning.core.converter import ModelConverter
from kenning.utils.class_loader import append_to_sys_path

if TYPE_CHECKING:
    from kenning.optimizers.ai8x import Ai8xTools

_DEFAULT_DEVICE = "cpu"


class Ai8xConverter(ModelConverter):
    """
    The Ai8x model converter.
    """

    source_format: str = "ai8x"

    def to_ai8x(
        self,
        ai8x_model_path: Path,
        ai8x_tools: "Ai8xTools",
    ) -> None:
        """
        Loads Ai8x compatible PyTorch model.

        Parameters
        ----------
        ai8x_model_path : Path
            Path where ai8x-compatible model will be saved.
        ai8x_tools : Ai8xTools
            Ai8X tools wrapper.
        """
        import torch

        with append_to_sys_path([ai8x_tools.ai8x_training_path]):
            model = torch.load(
                self.source_model_path,
                weights_only=False,
                map_location=torch.device(_DEFAULT_DEVICE),
            )

        if isinstance(model, torch.nn.Module):
            torch.save(
                {
                    "epoch": 0,
                    "state_dict": model.state_dict(),
                },
                ai8x_model_path,
            )
        else:
            torch.save(model, ai8x_model_path)
