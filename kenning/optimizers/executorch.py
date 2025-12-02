# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implementing the ExecuTorch compiler.
"""

import itertools
import warnings
from math import ceil
from typing import Dict, Generator, List, Literal, Optional, Tuple, TypeVar

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    IOSpecificationMissingEntryError,
    IOSpecificationNotFoundError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.modelwrappers.frameworks.pytorch import Tensor
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI

# PyTorch-specific type hints.
Dim = TypeVar("torch.Dim")
Module = TypeVar("torch.nn.Module")
ExportedProgram = TypeVar("torch.export.ExportedProgram")

warnings.filterwarnings("ignore", message=".*on an already erased node.*")


class ExecuTorchOptimizer(Optimizer):
    """Class implementing ExecuTorch compiler."""

    inputtypes = {"torch": ...}

    outputtypes = ["executorch"]

    arguments_structure = {
        "compiled_model_path": {
            "description": "The path to the compiled model output",
            "type": ResourceURI,
            "required": True,
        },
        "location": {
            "description": "Specifies where optimization should be performed "
            "in client-server scenario",
            "default": "host",
            "enum": Optimizer.locations,
        },
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "torch",
            "enum": list(inputtypes.keys()),
        },
        "quantize": {
            "argparse_name": "--quantize",
            "description": (
                "Whether 8-bit integer quantization should be applied."
            ),
            "type": bool,
            "default": False,
        },
        "dataset_percentage": {
            "argparse_name": "--dataset-percentage",
            "description": (
                "The percentage of the dataset to use for "
                "quantization calibration."
            ),
            "type": float,
            "default": 0.05,
        },
        "backends": {
            "argparse_name": "--backends",
            "description": (
                "What ExecuTorch backends the model will be optimized for."
                " Available: XNNPACK, CoreML"
            ),
            "type": List[str],
            "default": [],
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
        model_framework: Literal["torch"] = "torch",
        quantize: bool = False,
        dataset_percentage: float = 0.05,
        backends: List[str] = [],
    ):
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
            model_wrapper=model_wrapper,
        )
        self.model_framework = model_framework
        self.quantize = quantize
        self.dataset_percentage = dataset_percentage
        self.backends = backends
        self.set_input_type(model_framework)

    def _extract_shapes_from_io_specification(
        self,
        io_specification: Dict[str, List[Dict]],
    ) -> Dict[str, Optional[Dict[int, Dim]]]:
        """
        Extract dynamic shapes from the IO specification.

        Returns a dictionary mapping input names to their dynamic shape
        specifications, suitable for passing directly to torch.export as
        dynamic_shapes.

        Parameters
        ----------
        io_specification : Dict[str, List[Dict]]
            Dictionary with input and output specification.

        Returns
        -------
        Dict[str, Optional[Dict[int, Dim]]]
            Dictionary mapping input names to dynamic shape specifications.
        """
        from torch.export import Dim

        dynamic_shapes = {}
        input_specs = io_specification.get("input", [])
        method_name = io_specification.get("entry_func", "forward")

        # Get the argument names from input_specs.
        arg_names = [
            spec.get("name", f"input_{idx}")
            for idx, spec in enumerate(input_specs)
        ]
        for idx, input_spec in enumerate(input_specs):
            name = arg_names[idx]
            shape = input_spec.get("shape", None)
            if shape is None or not isinstance(shape, (list, tuple)):
                dynamic_shapes[name] = None
                continue

            dims = {}
            for dim_idx, dim_size in enumerate(shape):
                # If dimension is -1, treat as dynamic.
                if isinstance(dim_size, int) and dim_size < 0:
                    dims[dim_idx] = Dim(name=f"{name}_dim{dim_idx}", min=1)
            dynamic_shapes[name] = dims if dims else None

        # Ensure keys match the actual model input argument names.
        # If model has attribute 'forward', get its argument names.
        import inspect

        model_forward = getattr(self, "model", None)
        if model_forward is not None:
            sig = inspect.signature(getattr(model_forward, method_name))
            expected_arg_names = [k for k in sig.parameters if k != "self"]
            # Re-map `dynamic_shapes` keys to match `expected_arg_names`.
            if len(expected_arg_names) == len(dynamic_shapes):
                dynamic_shapes = {
                    expected_arg_names[i]: v
                    for i, v in enumerate(dynamic_shapes.values())
                }
        return dynamic_shapes

    def _generate_sample_inputs(
        self, io_spec: Optional[Dict[str, List[Dict]]]
    ) -> Tensor:
        import torch

        if io_spec is None:
            raise IOSpecificationNotFoundError(
                "Generating sample input requires IO specification to exist."
            )

        input_specs = io_spec.get("processed_input", io_spec.get("input", []))
        for input_spec in input_specs:
            input_shape = input_spec.get("shape", None)
            if input_shape is None or not isinstance(
                input_shape, (tuple, list)
            ):
                raise IOSpecificationMissingEntryError(
                    "`shape` entry is required in the IO specification,"
                    " but it is missing."
                )

            for s in input_shape:
                if type(s) is not int or s <= 0:
                    raise IOSpecificationMissingEntryError(
                        f"All dimensions should be positive integers, got: {s}"
                    )

            shape = tuple(s for s in input_shape)
            return torch.randn(*shape).cpu()

        raise IOSpecificationMissingEntryError(
            "Could not generate sample input from IO specification."
        )

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> None:
        from executorch.exir import to_edge_transform_and_lower
        from torch.export import export

        partitioners = []
        for backend in self.backends:
            if "XNNPACK" == backend:
                import executorch.backends.xnnpack.partition as xnnpart

                partitioners.append(
                    xnnpart.xnnpack_partitioner.XnnpackPartitioner()
                )
            elif "CoreML" == backend:
                from executorch.backends.apple.coreml.partition import (
                    CoreMLPartitioner,
                )

                partitioners.append(CoreMLPartitioner())
            else:
                KLogger.warning(
                    f"Attempted to use unsupported backend: {backend}"
                )

        if not input_model_path.exists():
            resolved_path = input_model_path.resolve()
            raise FileNotFoundError(
                f"There is no model file under {str(resolved_path)}"
            )
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        if io_spec is None:
            raise IOSpecificationNotFoundError(
                "IO specification is required for compilation."
            )
        io_spec["entry_func"] = "forward"
        try:
            from copy import deepcopy

            io_spec_processed = deepcopy(io_spec)

            io_spec_processed["input"] = (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError("No input specification found")

        conversion_kwargs = {
            "io_spec": io_spec_processed,
        }

        input_type = self.get_input_type(input_model_path)
        self.model = converter_registry.convert(
            input_model_path,
            input_type,
            "torch",
            **conversion_kwargs,
        )

        sample_inputs = (self._generate_sample_inputs(io_spec),)
        dynamic_shapes = self._extract_shapes_from_io_specification(
            io_specification=io_spec,
        )

        if not self.quantize:
            exported_program = export(
                self.model, sample_inputs, dynamic_shapes=dynamic_shapes
            )
        else:
            exported_program = self._apply_quantization(io_spec=io_spec)

        executorch_program = to_edge_transform_and_lower(
            exported_program, partitioner=partitioners
        ).to_executorch()

        with open(self.compiled_model_path, "wb") as f:
            f.write(executorch_program.buffer)
            f.flush()

        self.save_io_specification(self.compiled_model_path, io_spec)

    def get_framework_and_version(self) -> Tuple[str, str]:
        from executorch.version import __version__

        return ("executorch", __version__)

    @staticmethod
    def _sample_generator(
        io_spec: Dict,
        dataset: Dataset,
        dataset_percentage: float,
        model_wrapper: Optional[ModelWrapper] = None,
    ) -> Generator["Tensor", None, None]:
        import torch

        if model_wrapper is None:
            raise ValueError("Model wrapper cannot be None.")

        input_shape = tuple(
            io_spec.get("processed_input", io_spec["input"])[0]["shape"]
        )

        for batch in dataset.calibration_dataset_generator(dataset_percentage):
            for sample in batch:
                tensor_sample = torch.as_tensor(sample, dtype=torch.float32)
                yield tensor_sample.reshape(input_shape).cpu()

    def _apply_quantization(
        self,
        io_spec: Dict,
    ) -> ExportedProgram:
        import torch
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        quantization_parameters = get_symmetric_quantization_config(
            is_per_channel=True
        )
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(quantization_parameters)

        if self.dataset is None:
            sample_count = ceil(len(self.dataset) * self.dataset_percentage)
            generate_sample = (
                self._generate_sample_inputs(io_spec)
                for _ in range(sample_count)
            )
        else:
            generate_sample = ExecuTorchOptimizer._sample_generator(
                io_spec=io_spec,
                model_wrapper=self.model_wrapper,
                dataset=self.dataset,
                dataset_percentage=self.dataset_percentage,
            )

        generate_sample, generate_sample_copy = itertools.tee(generate_sample)
        first_sample = next(generate_sample_copy)
        sample_inputs = (first_sample,)
        prepared_model = prepare_pt2e(
            torch.export.export_for_training(
                self.model, sample_inputs
            ).module(),
            quantizer,
        )
        for sample in generate_sample:
            prepared_model(sample)

        quantized_model = convert_pt2e(prepared_model)
        return torch.export.export(quantized_model, sample_inputs)
