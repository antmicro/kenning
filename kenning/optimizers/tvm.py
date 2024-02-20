# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TVM deep learning compiler.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import onnx
import tvm
import tvm.relay as relay

from kenning.core.dataset import Dataset
from kenning.core.optimizer import (
    CompilationError,
    ConversionError,
    IOSpecificationNotFoundError,
    Optimizer,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


def onnxconversion(
    compiler: "TVMCompiler",
    model_path: PathOrURI,
    input_shapes: Dict,
    dtypes: Dict,
) -> Tuple[tvm.IRModule, Union[Dict, str]]:
    """
    Converts ONNX file to TVM format.

    Parameters
    ----------
    compiler: TVMCompiler
        Compiler used for conversion
    model_path: PathOrURI
        Path to the model to convert
    input_shapes: Dict
        Mapping from input name to input shape
    dtypes: Dict
        Mapping from input name to input dtype

    Returns
    -------
    mod: tvm.IRModule
        The relay module
    params: Union[Dict, str]
        Parameters dictionary to be used by relay module

    Raises
    ------
    IndexError
        Raised when no dtype was provided in the IO specification
    """
    try:
        dtype = list(dtypes.values())[0]
    except IndexError:
        raise IndexError("No dtype in the input specification")

    onnxmodel = onnx.load(model_path)
    return relay.frontend.from_onnx(
        onnxmodel, shape=input_shapes, freeze_params=True, dtype=dtype
    )


def kerasconversion(
    compiler: "TVMCompiler",
    model_path: PathOrURI,
    input_shapes: Dict,
    dtypes: Dict,
) -> Tuple[tvm.IRModule, Union[Dict, str]]:
    """
    Converts Keras file to TVM format.

    Parameters
    ----------
    compiler: TVMCompiler
        Compiler used for conversion
    model_path: PathOrURI
        Path to the model to convert
    input_shapes: Dict
        Mapping from input name to input shape
    dtypes: Dict
        Mapping from input name to input dtype

    Returns
    -------
    mod: tvm.IRModule
        The relay module
    params: Union[Dict, str]
        Parameters dictionary to be used by relay module
    """
    import tensorflow as tf

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(model.summary())
    return relay.frontend.from_keras(model, shape=input_shapes, layout="NHWC")


def no_conversion(out_dict):
    """
    Passes model as is to the compiler.
    """
    return out_dict


def torchconversion(
    compiler: "TVMCompiler",
    model_path: PathOrURI,
    input_shapes: Dict,
    dtypes: Dict,
) -> Tuple[tvm.IRModule, Union[Dict, str]]:
    """
    Converts Torch file to TVM format.

    Parameters
    ----------
    compiler: TVMCompiler
        Compiler used for conversion
    model_path: PathOrURI
        Path to the model to convert
    input_shapes: Dict
        Mapping from input name to input shape
    dtypes: Dict
        Mapping from input name to input dtype

    Returns
    -------
    mod: tvm.IRModule
        The relay module
    params: Union[Dict, str]
        Parameters dictionary to be used by relay module
    """
    import numpy as np
    import torch

    # This is a model-specific selector of output conversion functions.
    # It defaults to a no_conversion function that just returns its input
    # It is easily expandable in case it is needed for other models
    if compiler.conversion_func == "dict_to_tuple":
        # For PyTorch Mask R-CNN Model
        from kenning.modelwrappers.instance_segmentation.pytorch_coco import (
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_func(model_path: PathOrURI):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = torch.load(str(model_path), map_location=device)
        if not isinstance(loaded_model, torch.nn.Module):
            raise CompilationError(
                f"TVM compiler expects the input data of type: torch.nn.Module, but got: {type(loaded_model).__name__}"  # noqa: E501
            )
        return loaded_model

    model = TraceWrapper(model_func(model_path))
    model.eval()
    inp = torch.Tensor(
        np.random.uniform(
            0.0, 250.0, (mul(input_shapes[list(input_shapes.keys())[0]]))
        ),
    )

    inp = inp.to(device)

    with torch.no_grad():
        model(inp)
        model_trace = torch.jit.trace(model, inp)
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


def darknetconversion(
    compiler: "TVMCompiler",
    model_path: PathOrURI,
    input_shapes: Dict,
    dtypes: Dict,
) -> Tuple[tvm.IRModule, Union[Dict, str]]:
    """
    Converts darknet file to TVM format.

    Parameters
    ----------
    compiler: TVMCompiler
        Compiler used for conversion
    model_path: PathOrURI
        Path to the model to convert
    input_shapes: Dict
        Mapping from input name to input shape
    dtypes: Dict
        Mapping from input name to input dtype

    Returns
    -------
    mod: tvm.IRModule
        The relay module
    params: Union[Dict, str]
        Parameters dictionary to be used by relay module

    Raises
    ------
    ConversionError
        Raised when libdarknet shared library cannot be loaded.
    IndexError
        Raised when no dtype is provided in the IO specification
    """
    try:
        dtype = list(dtypes.values())[0]
    except IndexError:
        raise IndexError("No dtype in the input specification")

    from tvm.relay.testing.darknet import __darknetffi__

    if not compiler.libdarknet_path:
        KLogger.fatal(
            "The darknet converter requires libdarknet.so library. Provide "
            "the path to it using --libdarknet-path flag"
        )
        raise ConversionError("Provide libdarknet.so library")
    try:
        lib = __darknetffi__.dlopen(str(compiler.libdarknet_path))
    except OSError as e:
        raise ConversionError(e)
    net = lib.load_network(
        str(model_path.with_suffix(".cfg")).encode("utf-8"),
        str(model_path).encode("utf-8"),
        0,
    )
    return relay.frontend.from_darknet(
        net, dtype=dtype, shape=input_shapes["input"]
    )


def tfliteconversion(
    compiler: "TVMCompiler",
    model_path: PathOrURI,
    input_shapes: Dict,
    dtypes: Dict,
) -> Tuple[tvm.IRModule, Union[Dict, str]]:
    """
    Converts TFLite file to TVM format.

    Parameters
    ----------
    compiler: TVMCompiler
        Compiler used for conversion
    model_path: PathOrURI
        Path to the model to convert
    input_shapes: Dict
        Mapping from input name to input shape
    dtypes: Dict
        Mapping from input name to input dtype

    Returns
    -------
    mod: tvm.IRModule
        The relay module
    params: Union[Dict, str]
        Parameters dictionary to be used by relay module
    """
    with open(model_path, "rb") as f:
        tflite_model_buf = f.read()

    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    return relay.frontend.from_tflite(
        tflite_model, shape_dict=input_shapes, dtype_dict=dtypes
    )


class TVMCompiler(Optimizer):
    """
    The TVM compiler.
    """

    inputtypes = {
        "keras": kerasconversion,
        "onnx": onnxconversion,
        "darknet": darknetconversion,
        "torch": torchconversion,
        "tflite": tfliteconversion,
    }

    outputtypes = ["tvm"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "onnx",
            "enum": list(inputtypes.keys()),
        },
        "target": {
            "description": "The kind or tag of the target device",
            "default": "llvm",
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
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "onnx",
        target: str = "llvm",
        target_microtvm_board: Optional[str] = None,
        target_host: Optional[str] = None,
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
            Framework of the input model, used to select a proper backend.
        target : str
            Target accelerator on which the model will be executed.
        target_microtvm_board : Optional[str]
            Target board on which the model will be executed
        target_host : Optional[str]
            CPU architecture of the target (used when target has a host).
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
        """
        assert not (
            use_fp16_precision and use_int8_precision
        ), "Compilation cannot use both FP16 and INT8 conversion"
        assert not (
            use_tensorrt and (use_fp16_precision or use_int8_precision)
        ), "TensorRT usage with FP16 or INT8 passes is not supported"  # noqa: E501
        assert not (
            use_tensorrt and ("cuda" not in target)
        ), "TensorRT is only supported with CUDA target"
        self.model_framework = model_framework

        self.target = target
        self.target_microtvm_board = target_microtvm_board

        if self.target_microtvm_board:
            import tvm.micro.testing as mtvmt

            self.target_obj = mtvmt.get_target(target, target_microtvm_board)
        else:
            self.target_obj = tvm.target.Target(target)

        self.target_host = target_host
        self.target_host_obj = (
            tvm.target.Target(target_host) if target_host else None
        )

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
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
        )

    def compile_model(self, mod, params, outputpath, io_spec):
        # additional regular optimizations applied to models
        transforms = [relay.transform.RemoveUnusedFunctions()]

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
            if self.target == "zephyr":
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
                )

            if self.target_microtvm_board:
                graph_json = lib.get_graph_json().encode()
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

                with open(outputpath.with_suffix(".c"), "w") as ops_f:
                    ops_f.write(lib.get_lib().get_source())

            else:
                lib.export_library(outputpath)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        try:
            input_spec = (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError("No input specification found")

        inputshapes = {spec["name"]: spec["shape"] for spec in input_spec}
        dtypes = {spec["name"]: spec["dtype"] for spec in input_spec}

        if not inputshapes:
            raise ValueError("No shapes in the input specification")

        mod, params = self.inputtypes[self.inputtype](
            self, input_model_path, inputshapes, dtypes
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

    def get_framework_and_version(self):
        return ("tvm", tvm.__version__)
