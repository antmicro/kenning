# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of Safetensors models and conversion to other formats.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from kenning.core.converter import ModelConverter

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class SafetensorsConverter(ModelConverter):
    """
    The Safetensors model converter.
    """

    source_format: str = "safetensors"

    def to_safetensors(
        self,
        model: Optional["PreTrainedModel"] = None,
        device_map: str = "auto",
        **kwargs,
    ) -> "PreTrainedModel":
        """
        Loads Safetensors model.

        Parameters
        ----------
        model : Optional["PreTrainedModel"]
            Optional model object.
        device_map : str
            Device map for loading the model (e.g. "auto", "cpu", "cuda").
        **kwargs:
            Keyword arguments passed to `AutoModel.from_pretrained`.

        Returns
        -------
        PreTrainedModel
            Loaded Transformers model backed by safetensors.
        """
        if not model:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                str(self.source_model_path),
                use_safetensors=True,
                trust_remote_code=True,
                device_map=device_map,
                **kwargs,
            )

        return model

    def to_tinygrad(
        self,
        model: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Loads Safetensors model into a Tinygrad state dictionary.

        Parameters
        ----------
        model : Optional[Dict]
            Optional model state dictionary.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        Dict
            Loaded Tinygrad state dictionary of tensors.
        """
        if not model:
            from tinygrad.nn.state import safe_load

            model = safe_load(str(self.source_model_path))

        return model

    def to_onnx(
        self,
        io_spec: Dict[str, List[Dict]],
        model: Optional["PreTrainedModel"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":  # noqa: F821
        """
        Converts Safetensors model to ONNX format.

        Parameters
        ----------
        io_spec : Dict[str, List[Dict]]
            Input and output specification.
        model : Optional[PreTrainedModel]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model, a variant of ModelProto.
        """
        import io

        import onnx
        import torch

        if not model:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                str(self.source_model_path), use_safetensors=True
            )
        model.eval()

        input_names = [spec["name"] for spec in io_spec["input"]]
        output_names = [spec["name"] for spec in io_spec["output"]]

        dynamic_axes = {}
        dummy_inputs = []

        for spec in io_spec["input"]:
            shape = tuple(1 if dim == -1 else dim for dim in spec["shape"])
            dtype = getattr(torch, spec["dtype"].replace("float32", "float"))
            dummy_inputs.append(torch.zeros(shape, dtype=dtype))

            dyn_axes = {
                i: "dynamic"
                for i, dim in enumerate(spec["shape"])
                if dim == -1
            }
            if dyn_axes:
                dynamic_axes[spec["name"]] = dyn_axes

        for spec in io_spec["output"]:
            dyn_axes = {
                i: "dynamic"
                for i, dim in enumerate(spec["shape"])
                if dim == -1
            }
            if dyn_axes:
                dynamic_axes[spec["name"]] = dyn_axes

        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            tuple(dummy_inputs),
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            opset_version=kwargs.get("opset_version", 14),
            do_constant_folding=True,
        )

        buffer.seed(0)
        modelproto = onnx.load_model(buffer)

        return modelproto
