# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Enables loading of PyTorch models and conversion to other formats.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from kenning.core.converter import ModelConverter
from kenning.core.exceptions import (
    CompilationError,
    ConversionError,
    ModelNotLoadedError,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI

if TYPE_CHECKING:
    import onnx
    import torch
    import tvm

    from kenning.optimizers.ai8x import Ai8xTools

_DEFAULT_DEVICE = "cpu"


class TorchConverter(ModelConverter):
    """
    The PyTorch model converter.
    """

    source_format: str = "torch"

    def to_torch(
        self,
        model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> "torch.nn.Module":
        """
        Loads PyTorch model.

        Parameters
        ----------
        model : Optional["torch.nn.Module"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        torch.nn.Module
            Torch Model.

        Raises
        ------
        ModelNotLoadedError
            Raised if a full odel cannot be loaded from provided path.
        """
        import torch

        if not model:
            model = torch.load(
                self.source_model_path,
                weights_only=False,
                map_location=torch.device(_DEFAULT_DEVICE),
            )

        if isinstance(model, Dict):
            raise ModelNotLoadedError(
                f"The provided file ({str(self.source_model_path)}) contains "
                "a PyTorch state dictionary with weights of a model. The "
                "architecture of a model is required as well. Save the "
                "full model, both its architecture and weights, and try again."
            )
        model.eval()
        return model

    def to_onnx(
        self,
        input_spec: List[Dict],
        output_names: List,
        model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> "onnx.ModelProto":
        """
        Converts Torch model to ONNX.

        Parameters
        ----------
        input_spec: List[Dict]
            Dictionary representing inputs.
        output_names: List
            Names of outputs to include in the final model.
        model : Optional["torch.nn.Module"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        onnx.ModelProto
            Loaded ONNX model.

        Raises
        ------
        CompilationError
            Raised if the input type of the model is not torch.nn.Module.
        """
        import onnx
        import torch

        if not model:
            model = torch.load(
                str(self.source_model_path),
                map_location=_DEFAULT_DEVICE,
                weights_only=False,
            )

        if not isinstance(model, torch.nn.Module):
            raise CompilationError(
                "ONNX compiler expects the input data of type: "
                f"torch.nn.Module, but got: {type(model).__name__}"
            )

        model.eval()

        sample_input = tuple(
            torch.randn(spec["shape"], device=_DEFAULT_DEVICE)
            for spec in input_spec
        )

        import io

        mem_buffer = io.BytesIO()
        torch.onnx.export(
            model,
            sample_input,
            mem_buffer,
            opset_version=11,
            input_names=[spec["name"] for spec in input_spec],
            output_names=output_names,
        )
        onnx_model = onnx.load_model_from_string(mem_buffer.getvalue())
        return onnx_model

    def to_ai8x(
        self,
        ai8x_model_path: Path,
        ai8x_tools: "Ai8xTools",
        device_id: int,
        model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        """
        Converts torch model into ai8x-compatible model.

        Parameters
        ----------
        ai8x_model_path : Path
            Path where ai8x-compatible model will be saved.
        ai8x_tools : Ai8xTools
            Ai8X tools wrapper.
        device_id : int
            Ai8X device ID.
        model : Optional["torch.nn.Module"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Raises
        ------
        ConversionError
            When model with unsupported layers is passed.
        """
        import json

        import torch

        from kenning.optimizers.ai8x_fuse import fuse_torch_sequential

        model = torch.load(
            self.source_model_path,
            weights_only=False,
            map_location=torch.device(_DEFAULT_DEVICE),
        )

        if not isinstance(model, torch.nn.Sequential):
            raise ConversionError(
                "Only Sequential models are supported,"
                "got {type(model).__name__}"
            )

        ai8x_model = fuse_torch_sequential(
            ai8x_tools.ai8x_training_path, device_id, model
        )

        torch.save(
            {
                "epoch": 0,
                "state_dict": ai8x_model.state_dict(),
            },
            ai8x_model_path,
        )

        ai8x_model_tmp_path = ai8x_model_path.with_name(
            ai8x_model_path.name + "_tmp"
        )
        torch.save(ai8x_model, ai8x_model_tmp_path)

        io_spec = json.loads(
            self.source_model_path.with_suffix(
                self.source_model_path.suffix + ".json"
            ).read_text()
        )

        yaml_cfg_path = ai8x_model_path.resolve().with_suffix(
            ai8x_model_path.suffix + ".yaml"
        )

        ai8x_tools.yamlwriter(
            ai8x_model_tmp_path,
            io_spec.get("processed_input", io_spec["input"])[0]["shape"],
            device_id,
            yaml_cfg_path,
        )

        KLogger.info(f"Model YAML configuration saved in {yaml_cfg_path}")

    def to_tvm(
        self,
        input_shapes: Dict,
        conversion_func: Optional[str],
        model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> Tuple["tvm.IRModule", Union[Dict, str]]:
        """
        Converts Torch file to TVM format.

        Parameters
        ----------
        input_shapes: Dict
            Mapping from input name to input shape.
        conversion_func: Optional[str]
            Model-specific selector of output conversion functions.
        model : Optional["torch.nn.Module"]
            Optional model object.
        **kwargs:
            Keyword arguments passed between conversions.

        Returns
        -------
        mod: tvm.IRModule
            The relay module.
        params: Union[Dict, str]
            Parameters dictionary to be used by relay module.
        """
        import numpy as np
        import torch
        import tvm.relay as relay

        def no_conversion(out_dict):
            """
            Passes model as is to the compiler.
            """
            return out_dict

        # This is a model-specific selector of output conversion functions.
        # It defaults to a no_conversion function that just returns its input
        # It is easily expandable in case it is needed for other models
        if conversion_func == "dict_to_tuple":
            # For PyTorch Mask R-CNN Model
            from kenning.modelwrappers.instance_segmentation.pytorch_coco import (  # noqa: E501
                dict_to_tuple,
            )

            wrapper = dict_to_tuple
        else:  # General case - no conversion is happening
            wrapper = no_conversion

        def mul(x: tuple) -> int:
            """
            Method used to convert shape-representing tuple
            to a 1-dimensional size to allow the model to be inferred with
            an 1-dimensional byte array.

            Parameters
            ----------
            x : tuple
                Tuple describing the regular input shape.

            Returns
            -------
            int
                The size of a 1-dimensional input matching the original shape.
            """
            ret = 1
            for i in list(x):
                ret *= i
            return ret

        class TraceWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inp):
                out = self.model(
                    inp.reshape(input_shapes[list(input_shapes.keys())[0]])
                )
                return wrapper(out[0])

        device = torch.device(
            "cuda" if torch.cuda.is_available() else _DEFAULT_DEVICE
        )

        def model_func(model_path: PathOrURI):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else _DEFAULT_DEVICE
            )
            loaded_model = torch.load(
                str(model_path), map_location=device, weights_only=False
            )
            if not isinstance(loaded_model, torch.nn.Module):
                raise CompilationError(
                    "TVM compiler expects the input data of type: "
                    f"torch.nn.Module, but got: {type(loaded_model).__name__}"
                )
            return loaded_model

        model = model if model else model_func(self.source_model_path)
        wrapped_model = TraceWrapper(model)
        wrapped_model.eval()
        shape = input_shapes[list(input_shapes.keys())[0]]
        sample_input = torch.Tensor(
            np.random.uniform(0.0, 250.0, (mul(shape))),
        )

        sample_input = sample_input.to(device)

        with torch.no_grad():
            wrapped_model(sample_input)
            model_trace = torch.jit.trace(wrapped_model, sample_input)
            model_trace.eval()

        return relay.frontend.from_pytorch(
            model_trace,
            # this is a list of input infos where there is a dict
            # constructed from {input_name: (n-dim tuple-shape)}
            # into {input_name: [product_of_the_dimensions]}
            list(
                {
                    list(input_shapes.keys())[0]: [
                        mul(input_shapes[list(input_shapes.keys())[0]])
                    ]
                }.items()
            ),
        )
