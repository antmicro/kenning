# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TVM deep learning compiler.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

import tvm
import tvm.relay as relay

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    CompilationError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    Optimizer,
)
from kenning.core.platform import Platform
from kenning.utils.compiler_flag import merge_compiler_flags
from kenning.utils.logger import KLogger
from kenning.utils.onnx import check_io_spec
from kenning.utils.resource_manager import PathOrURI, ResourceURI
from kenning.utils.zpl_suffix import ZplSuffix

TVM_FUNC_PATTERN = re.compile(
    r"TVM_DLL[\s\n\t]+int32_t[\s\n\t]*(tvmgen_[a-zA-Z0-9_]*)\([^\(\)]*\)"
)


class TVMCompiler(Optimizer):
    """
    The TVM compiler.
    """

    inputtypes = [
        "keras",
        "onnx",
        "darknet",
        "torch",
        "tflite",
        "any",
    ]

    outputtypes = ["tvm"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": inputtypes + ["any"],
        },
        "target": {
            "description": "The kind or tag of the target device",
            "default": None,
        },
        "target_attrs": {
            "description": "The target attributes (like device or arch) - e.g. '-device=arm_cpu -march=armv7e-m'",  # noqa: E501
            "default": "",
        },
        "target_microtvm_board": {
            "description": "The target board",
            "default": None,
            "nullable": True,
        },
        "target_host": {
            "description": "The kind or tag of the host (CPU) target device",
            "type": str,
            "default": None,
            "nullable": True,
        },
        "zephyr_header_template": {
            "argparse_name": "--zephyr-header-template",
            "description": (
                "Path to the template header that will be used to generate "
                "header for TVM ops source for Kenning Zephyr runtime."
            ),
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
        "zephyr_llext_source_template": {
            "description": (
                "Path to the LLEXT source template. If provided model LLEXT "
                "source will be generated."
            ),
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
        "opt_level": {
            "description": "The optimization level of the compilation",
            "default": 3,
            "type": int,
        },
        "libdarknet_path": {
            "argparse_name": "--libdarknet-path",
            "description": "Path to the libdarknet.so library, for darknet models",  # noqa: E501
            "default": "/usr/local/lib/libdarknet.so",
            "type": str,
        },
        "use_tvm_vm": {
            "argparse_name": "--compile-use-vm",
            "description": "At compilation stage use the TVM Relay VirtualMachine",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "conversion_func": {
            "argparse_name": "--output-conversion-function",
            "description": "The type of output conversion function used for PyTorch conversion",  # noqa: E501
            "default": "default",
            "enum": ["default", "dict_to_tuple"],
        },
        "conv2d_data_layout": {
            "description": "Configures the I/O layout for the CONV2D operations",  # noqa: E501
            "type": str,
            "default": "",
        },
        "conv2d_kernel_layout": {
            "description": "Configures the kernel layout for the CONV2D operations",  # noqa: E501
            "type": str,
            "default": "",
        },
        "use_fp16_precision": {
            "argparse_name": "--use-fp16-precision",
            "description": "Applies conversion of FP32 weights to FP16",
            "type": bool,
            "default": False,
        },
        "use_int8_precision": {
            "argparse_name": "--use-int8-precision",
            "description": "Applies conversion of FP32 weights to INT8",
            "type": bool,
            "default": False,
        },
        "use_tensorrt": {
            "argparse_name": "--use-tensorrt",
            "description": "For CUDA targets: delegates supported operations to TensorRT",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "dataset_percentage": {
            "argparse_name": "--dataset-percentage",
            "description": "Tells how much data from the calibration dataset (training or external) will be used for calibration dataset",  # noqa: E501
            "type": float,
            "default": 0.25,
        },
        "module_name": {
            "argparse_name": "--module-name",
            "description": "The name of a module, `tvmgen_{MODULE_NAME}` will be used as prefix for generated functions",  # noqa: E501
            "type": str,
            "nullable": True,
            "default": None,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "any",
        target: Optional[str] = None,
        target_attrs: str = "",
        target_microtvm_board: Optional[str] = None,
        target_host: Optional[str] = None,
        zephyr_header_template: Optional[PathOrURI] = None,
        zephyr_llext_source_template: Optional[PathOrURI] = None,
        opt_level: int = 3,
        libdarknet_path: str = "/usr/local/lib/libdarknet.so",
        use_tvm_vm: bool = False,
        conversion_func: str = "default",
        conv2d_data_layout: str = "",
        conv2d_kernel_layout: str = "",
        use_fp16_precision: bool = False,
        use_int8_precision: bool = False,
        use_tensorrt: bool = False,
        dataset_percentage: float = 0.25,
        module_name: Optional[str] = None,
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        A TVM Compiler wrapper.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        compiled_model_path : PathOrURI
            Path where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        target : Optional[str]
            Target accelerator on which the model will be executed.
        target_attrs : str
            Target attributes.
        target_microtvm_board : Optional[str]
            Target board on which the model will be executed
        target_host : Optional[str]
            CPU architecture of the target (used when target has a host).
        zephyr_header_template : Optional[PathOrURI]
            Path to the template header that will be used to generate header
            for TVM ops source for Kenning Zephyr runtime.
        zephyr_llext_source_template : Optional[PathOrURI]
            Path to the LLEXT source template. If provided model LLEXT source
            will be generated.
        opt_level : int
            Optimization level of compilation.
        libdarknet_path : str
            Path to the libdarknet.so library, used only during conversion
            of darknet model.
        use_tvm_vm : bool
            At compilation stage use the TVM Relay VirtualMachine.
        conversion_func : str
            Output conversion function.
        conv2d_data_layout : str
            Data layout to convert the model to.
            Empty if no conversion is necessary.
            This value must be set if conv2d_kernel_layout is set.
        conv2d_kernel_layout : str
            Kernel layout to convert the model to.
            Empty if no conversion is necessary.
        use_fp16_precision : bool
            Applies conversion of FP32 weights to FP16.
        use_int8_precision : bool
            Applies conversion of FP32 weights to INT8.
        use_tensorrt : bool
            Applies transformations moving supported operations to
            TensorRT kernels.
        dataset_percentage : float
            If use_int8_precision is set, the given percentage of samples
            from the training dataset or external calibration dataset is
            used for calibrating the model.
        module_name : Optional[str]
           The name of a module, `tvmgen_{module_name}` will be used as prefix
           for generated functions and entrypoint function will be names
           as `TVM{ModuleName}SystemLibEntryPoint`.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        assert not (
            use_fp16_precision and use_int8_precision
        ), "Compilation cannot use both FP16 and INT8 conversion"
        assert not (
            use_tensorrt and (use_fp16_precision or use_int8_precision)
        ), "TensorRT usage with FP16 or INT8 passes is not supported"
        if target:
            assert not (
                use_tensorrt and ("cuda" not in target)
            ), "TensorRT is only supported with CUDA target"
        self.model_framework = model_framework

        self.target = target
        self.target_attrs = target_attrs
        self.target_microtvm_board = target_microtvm_board

        self.platform_target = None
        self.platform_target_attrs = []

        self.target_host = target_host
        self.target_host_obj = (
            tvm.target.Target(target_host) if target_host else None
        )
        self.zephyr_header_template = zephyr_header_template
        self.zephyr_llext_source_template = zephyr_llext_source_template

        self.opt_level = opt_level
        self.libdarknet_path = libdarknet_path
        self.use_tvm_vm = use_tvm_vm
        self.conversion_func = conversion_func
        self.set_input_type(model_framework)
        self.conv2d_data_layout = conv2d_data_layout
        self.conv2d_kernel_layout = conv2d_kernel_layout
        self.use_fp16_precision = use_fp16_precision
        self.use_int8_precision = use_int8_precision
        self.use_tensorrt = use_tensorrt
        self.dataset_percentage = dataset_percentage
        self.module_name = module_name
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def init(self):
        import tvm.micro.testing as mtvmt

        target = self._get_target()
        assert target, (
            "Target is not initialized. Ensure it is supplied "
            + "either explicitly or through read_platform()"
        )

        target_attrs = self._get_target_attrs()

        if target in mtvmt.utils.get_supported_platforms():
            if self.target_microtvm_board:
                try:
                    self.target_obj = mtvmt.get_target(
                        target, self.target_microtvm_board
                    )
                    if target_attrs:
                        KLogger.info(
                            "Target chosen from microTVM,"
                            " skipping provided target options"
                        )
                except KeyError:
                    # board not found
                    self.target_obj = tvm.target.Target("c " + target_attrs)
            else:
                self.target_obj = tvm.target.Target("c " + target_attrs)
                self.target_microtvm_board = True
        else:
            self.target_obj = tvm.target.Target(f"{target} {target_attrs}")

        KLogger.debug(f"Using target: {self.target_obj}")

    def compile_model(self, mod, params, outputpath, io_spec):
        # additional regular optimizations applied to models
        transforms = [relay.transform.RemoveUnusedFunctions()]

        target = self._get_target()

        if self.use_int8_precision:

            def generator():
                for sample in self.dataset.calibration_dataset_generator(
                    self.dataset_percentage
                ):
                    # TODO add support for any number of inputs
                    assert len(io_spec["input"]) == 1, (
                        "Currently only single-input models are supported "
                        + "during quantization"
                    )
                    yield {io_spec["input"][0]["name"]: tvm.nd.array(sample)}

            with relay.quantize.qconfig(
                calibrate_mode="kl_divergence", weight_scale="max"
            ):
                mod = relay.quantize.quantize(mod, params, dataset=generator())

        if self.use_fp16_precision:
            transforms.append(relay.transform.ToMixedPrecision())

        if self.conv2d_data_layout != "" or self.conv2d_kernel_layout != "":
            if self.conv2d_kernel_layout == "":
                self.conv2d_kernel_layout = "default"
            KLogger.info(
                "Applying ConvertLayout transform:\n"
                f'DATA LAYOUT   : "{self.conv2d_data_layout}"\n'
                f'KERNEL LAYOUT : "{self.conv2d_kernel_layout}"'
            )

            if self.conv2d_data_layout == "":
                raise CompilationError("conv2d_data_layout cannot be empty")
            transforms.append(
                relay.transform.ConvertLayout(
                    {
                        "nn.conv2d": [
                            self.conv2d_data_layout,
                            self.conv2d_kernel_layout,
                        ],
                        "nn.max_pool2d": [
                            self.conv2d_data_layout,
                            self.conv2d_kernel_layout,
                        ],
                        "qnn.conv2d": [
                            self.conv2d_data_layout,
                            self.conv2d_kernel_layout,
                        ],
                    }
                )
            )

        additional_opts = tvm.transform.Sequential(transforms)

        if self.use_tvm_vm:
            with tvm.transform.PassContext(
                opt_level=self.opt_level, disabled_pass=["FoldScaleAxis"]
            ):
                mod = additional_opts(mod)
                vm_exec = relay.vm.compile(
                    mod, target=self.target_obj, params=params
                )
                bytecode, lib = vm_exec.save()
                with open(str(outputpath) + ".ro", "wb") as file:
                    file.write(bytecode)
                lib.export_library(str(outputpath) + ".so")
        else:
            pass_config = {}
            if target.startswith("zephyr"):
                pass_config["tir.disable_vectorize"] = True

            with tvm.transform.PassContext(
                opt_level=self.opt_level,
                config=pass_config,
                disabled_pass=["AlterOpLayout"],
            ):
                mod = additional_opts(mod)
                if self.use_tensorrt:
                    from tvm.relay.op.contrib.tensorrt import (
                        partition_for_tensorrt,
                    )

                    mod = partition_for_tensorrt(mod, params)
                lib = relay.build(
                    mod,
                    target=self.target_obj,
                    target_host=self.target_host_obj,
                    params=params,
                    mod_name=self.module_name if self.module_name else "",
                )

            if self.target_microtvm_board:
                graph_json = lib.get_graph_json()
                # minify JSON
                graph_json = json.dumps(
                    json.loads(graph_json), separators=(",", ":")
                )
                with open(
                    self.compiled_model_path.with_suffix(".graph.json"), "w"
                ) as graph_f:
                    graph_f.write(graph_json)

                graph_json = graph_json.encode()

                params = tvm.runtime.params.save_param_dict(lib.get_params())

                graph_data = b""

                graph_data += len(graph_json).to_bytes(
                    4, "little", signed=False
                )
                graph_data += len(params).to_bytes(4, "little", signed=False)

                graph_data += graph_json
                graph_data += params

                with open(outputpath, "wb") as graph_f:
                    graph_f.write(graph_data)

                if (
                    self.zephyr_header_template is None
                    and self.zephyr_llext_source_template is None
                ):
                    return None

                # extract TVM functions from source file
                tvm_funcs = list(
                    set(TVM_FUNC_PATTERN.findall(lib.get_lib().get_source()))
                )
                tvm_funcs.sort()
                template_tvmgen_functions = "".join(
                    [f" \\\n    FUNC({func_name})" for func_name in tvm_funcs]
                )
                template_tvmgen_functions_count = f'"\\x{len(tvm_funcs):02x}"'

                if self.zephyr_llext_source_template is None:
                    # write source
                    with open(outputpath.with_suffix(".c"), "w") as ops_f:
                        ops_f.write(lib.get_lib().get_source())

                    # generate header
                    with open(self.zephyr_header_template, "r") as template_f:
                        template = template_f.read()

                    mod_name = (
                        f"_{self.module_name}" if self.module_name else ""
                    )
                    mod_name_macro = mod_name.upper()
                    mod_name_pascal = "".join(
                        [m.title() for m in mod_name.split("_")]
                    )

                    # For compatibility with Kenning Zephyr Runtime we need an
                    # option to preserve default function names, which is done
                    # with an empty string or a None value passes as
                    # 'module_name' parameter.
                    to_replace = [
                        ("{{TVMGEN_FUNCTIONS}}", template_tvmgen_functions),
                        (
                            "{{TVMGEN_FUNCTIONS_COUNT}}",
                            template_tvmgen_functions_count,
                        ),
                        ("{{MOD_NAME}}", mod_name),
                        ("{{MOD_NAME_MACRO}}", mod_name_macro),
                        ("{{MOD_NAME_FUNC}}", mod_name_pascal),
                    ]
                    for pattern, value in to_replace:
                        template = template.replace(pattern, value)

                    with open(outputpath.with_suffix(".h"), "w") as header_f:
                        header_f.write(template)

                else:
                    # write LLEXT source
                    with open(
                        self.zephyr_llext_source_template, "r"
                    ) as template_f:
                        llext_source = template_f.read()

                    llext_source = llext_source.replace(
                        "{{MODEL_SRC}}", lib.get_lib().get_source()
                    )
                    llext_source = llext_source.replace(
                        "{{TVMGEN_FUNCTIONS}}", template_tvmgen_functions
                    )

                    with open(outputpath.with_suffix(".c"), "w") as llext_f:
                        llext_f.write(llext_source)

                    self.llext_model = True

            else:
                lib.export_library(outputpath)
            metadata = lib.function_metadata
            metadata_path = ZplSuffix.TVM_METADATA._get_path_with_suffix(
                Path(outputpath)
            )
            metadata_path.write_text(tvm.ir.save_json(metadata))

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        io_spec_processed = check_io_spec(io_spec)
        input_type = self.get_input_type(input_model_path)

        model_cls = self.get_model_class()

        if model_cls is None:
            KLogger.warning("Cannot get model class from model wrapper.")

        conversion_kwargs = {
            "io_spec": io_spec_processed,
            "model_cls": model_cls,
            "conversion_func": self.conversion_func,
            "libdarknet_path": self.libdarknet_path,
        }

        mod, params = converter_registry.convert(
            input_model_path, input_type, "tvm", **conversion_kwargs, **kwargs
        )

        self.compiled_model_path.parent.mkdir(parents=True, exist_ok=True)
        io_spec["entry_func"] = (
            self.module_name if type(self.module_name) is str else ""
        )
        self.compile_model(mod, params, self.compiled_model_path, io_spec)
        if self.use_fp16_precision or self.use_int8_precision:
            output_dtype = "float16" if self.use_fp16_precision else "int8"
            for id in range(len(io_spec["output"])):
                io_spec["output"][id]["prequantized_dtype"] = io_spec[
                    "output"
                ][id]["dtype"]
                io_spec["output"][id]["dtype"] = output_dtype
        self.save_io_specification(input_model_path, io_spec)

    def read_platform(self, platform: Platform):
        platform_target_attrs = getattr(platform, "compilation_flags", [])

        match type(platform).__name__:
            case "CUDAPlatform":
                self.platform_target = "cuda"
                platform_target_attrs.append(
                    f"-arch={platform.compute_capability}"
                )

            case "ZephyrPlatform":
                self.platform_target = "zephyr"
                self.target_microtvm_board = platform.name
                if self.zephyr_header_template is None:
                    self.zephyr_header_template = ResourceURI(
                        "gh://antmicro:kenning-zephyr-runtime/lib/kenning_inference_lib/runtimes/tvm/generated/model_impl.h.template;branch=main"
                    )

            case _:
                KLogger.warning(
                    f"Unsupported platform: {type(platform).__name__}"
                )
                return None

        self.platform_target_attrs = platform_target_attrs

        KLogger.info(f"Set TVMCompiler target to {self.target}")

    def get_framework_and_version(self):
        return ("tvm", tvm.__version__)

    def zpl_prepare_cmd_flags(self):
        flags = super().zpl_prepare_cmd_flags()
        if getattr(self, "compiled_model_path", None) is not None:
            flags.extend(
                [
                    "--tvm-model-paths",
                    str(
                        ZplSuffix.TVM_GRAPH_JSON._get_path_with_suffix(
                            self.compiled_model_path
                        )
                    ),
                    "--tvm-model-metadata-paths",
                    str(
                        ZplSuffix.TVM_METADATA._get_path_with_suffix(
                            self.compiled_model_path
                        )
                    ),
                ]
            )

        if self.module_name is not None:
            flags.extend(
                [
                    "--tvm-model-op-remove-prefix",
                    f"tvmgen_{self.module_name}_",
                ]
            )

        return flags

    @classmethod
    def get_framework(cls) -> str:
        return "tvm"

    @classmethod
    def get_framework_version(cls) -> str:
        return tvm.__version__

    def _get_target(self):
        return self.target or self.platform_target or "llvm"

    def _get_target_attrs(self, tostring=True):
        return merge_compiler_flags(
            self.target_attrs.split(),
            self.platform_target_attrs,
            tostring=tostring,
        )
