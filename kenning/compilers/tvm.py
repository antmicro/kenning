"""
Wrapper for TVM deep learning compiler.
"""

import tvm
import onnx
import tvm.relay as relay
from pathlib import Path
import re

from kenning.core.optimizer import Optimizer, CompilationError
from kenning.core.dataset import Dataset
from kenning.utils.logger import get_logger


def onnxconversion(
        compiler: 'TVMCompiler',
        modelpath: Path,
        input_shapes,
        dtype='float32'):
    onnxmodel = onnx.load(modelpath)
    return relay.frontend.from_onnx(
        onnxmodel,
        shape=input_shapes,
        freeze_params=True,
        dtype=dtype
    )


def kerasconversion(
        compiler: 'TVMCompiler',
        modelpath: Path,
        input_shapes,
        dtype='float32'):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(modelpath))
    print(model.summary())
    return relay.frontend.from_keras(
        model,
        shape=input_shapes,
        layout='NHWC'
    )


def no_conversion(out_dict):
    return out_dict


def torchconversion(
        compiler: 'TVMCompiler',
        modelpath: Path,
        input_shapes,
        dtype='float32'):
    import torch
    import numpy as np

    # This is a model-specific selector of output conversion functions.
    # It defaults to a no_conversion function that just returns its input
    # It is easily expandable in case it is needed for other models
    if compiler.wrapper_function == 'dict_to_tuple':  # For PyTorch Mask R-CNN Model  # noqa: E501
        from kenning.modelwrappers.instance_segmentation.pytorch_coco import dict_to_tuple  # noqa: E501
        wrapper = dict_to_tuple
    else:  # General case- no conversion is happening
        wrapper = no_conversion

    def mul(x: tuple) -> int:
        """
        Method used to convert shape-representing tuple
        to a 1-dimmensional size to allow the model to be inferred with
        an 1-dimmensional byte array

        Parameters
        ----------
        x : tuple
            tuple describing the regular input shape

        Returns
        -------
        int : the size of a 1-dimmensional input matching the original shape
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

    model_func = torch.load
    model = TraceWrapper(model_func(modelpath))
    model.eval()
    inp = torch.Tensor(
        np.random.uniform(
            0.0,
            250.0,
            (mul(input_shapes[list(input_shapes.keys())[0]]))
        )
    )

    with torch.no_grad():
        model(inp)
        model_trace = torch.jit.trace(model, inp)
        model_trace.eval()

    return relay.frontend.from_pytorch(
        model_trace,
        # this is a list of input infos where there is a dict
        # constructed from {input_name: (n-dim tuple-shape)}
        # into {input_name: [product_of_the_dimmensions]}
        list(
            {
                list(input_shapes.keys())[0]:
                [mul(input_shapes[list(input_shapes.keys())[0]])]
            }.items()
        )
    )


def darknetconversion(
        compiler: 'TVMCompiler',
        modelpath: Path,
        input_shapes,
        dtype='float32'):
    from tvm.relay.testing.darknet import __darknetffi__
    if not compiler.libdarknetpath:
        log = get_logger()
        log.fatal(
            'The darknet converter requires libdarknet.so library. ' +
            'Provide the path to it using --libdarknet-path flag')
        raise CompilationError('Provide libdarknet.so library')
    lib = __darknetffi__.dlopen(str(compiler.libdarknetpath))
    net = lib.load_network(
        str(modelpath.with_suffix('.cfg')).encode('utf-8'),
        str(modelpath).encode('utf-8'),
        0
    )
    return relay.frontend.from_darknet(
        net,
        dtype=dtype,
        shape=input_shapes['data']
    )


def tfliteconversion(
        compiler: "TVMCompiler",
        modelpath: Path,
        input_shapes,
        dtype='float32'):

    tflite_model_buf = open(modelpath, "rb").read()

    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    return relay.frontend.from_tflite(
        tflite_model,
        dtype_dict=input_shapes,
        shape_dict={"input": dtype}
    )


class TVMCompiler(Optimizer):
    """
    The TVM compiler.
    """

    outputtypes = []

    inputtypes = {
        'keras': kerasconversion,
        'onnx': onnxconversion,
        'darknet': darknetconversion,
        'torch': torchconversion,
        'tflite': tfliteconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str,
            target: str,
            target_host: str,
            opt_level: int = 2,
            libdarknetpath: str = '/usr/local/lib/libdarknet.so',
            use_tvm_vm: bool = False,
            conversion_func: str = 'default'):
        """
        A TVM Compiler wrapper.

        Parameters
        ----------
        dataset : Dataset
            Dataset object
        compiled_model_path : Path
            Path where compiled model will be saved
        modelframework : str
            Framework of the input model, used to select a proper backend
        target : str
            Target accelerator on which the model will be executed
        target_host : str
            CPU architecture of the target (used when target has a host).
        opt_level : int
            optimization level of compilation
        libdarknetpath : str
            path to the libdarknet.so library, used only during conversion
            of darknet model
        """
        self.set_input_type(modelframework)
        self.target = tvm.target.Target(target)
        self.target_host = (
                tvm.target.Target(target_host) if target_host else None
        )
        self.opt_level = opt_level
        self.libdarknetpath = libdarknetpath
        self.use_tvm_vm = use_tvm_vm

        self.wrapper_function = conversion_func

        super().__init__(dataset, compiled_model_path)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--model-framework',
            help='The input type of the model, framework-wise',
            choices=cls.inputtypes.keys(),
            default='onnx'
        )
        group.add_argument(
            '--target',
            help='The kind or tag of the target device',
            required=True
        )
        group.add_argument(
            '--target-host',
            help='The kind or tag of the host (CPU) target device',
        )
        group.add_argument(
            '--opt-level',
            help='The optimization level of the compilation',
            default=2,
            type=int
        )
        group.add_argument(
            '--libdarknet-path',
            help='Path to the libdarknet.so library, for darknet models',
            type=str
        )
        group.add_argument(
            '--compile-use-vm',
            help='At compilation stage use the TVM Relay VirtualMachine',
            action='store_true'
        )
        group.add_argument(
            '--output-conversion-function',
            help='The type of output conversion function used for PyTorch conversion',  # noqa: E501
            choices=['default', 'dict_to_tuple'],
            default='default'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.model_framework,
            args.target,
            args.target_host,
            args.opt_level,
            args.libdarknet_path,
            args.compile_use_vm,
            args.output_conversion_function
        )

    def set_input_type(self, inputtype: str):
        assert inputtype in self.inputtypes.keys()
        self.inputtype = inputtype

    def compile_model(self, mod, params, outputpath):
        if str(self.target).startswith('cuda'):
            archmatch = re.search(r'-arch=(sm_\d\d)', str(self.target))
            arch = archmatch.group(1) if archmatch else None
            if arch:
                tvm.autotvm.measure.measure_methods.set_cuda_target_arch(arch)
        if self.use_tvm_vm:
            with tvm.transform.PassContext(
                    opt_level=3,
                    disabled_pass=["FoldScaleAxis"]):
                vm_exec = relay.vm.compile(
                    mod,
                    target=self.target,
                    params=params
                )
                bytecode, lib = vm_exec.save()
                with open(str(outputpath)+'.ro', 'wb') as file:
                    file.write(bytecode)
                lib.export_library(str(outputpath)+'.so')
        else:
            with tvm.transform.PassContext(opt_level=self.opt_level):
                lib = relay.build(
                    mod,
                    target=self.target,
                    target_host=self.target_host,
                    params=params
                )
            lib.export_library(outputpath)

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes,
            dtype='float32'):
        mod, params = self.inputtypes[self.inputtype](
            self,
            inputmodelpath,
            inputshapes,
            dtype
        )
        self.compile_model(mod, params, self.compiled_model_path)

    def get_framework_and_version(self):
        return ('tvm', tvm.__version__)

    def get_output_formats(self):
        return self.outputtypes

    def get_input_formats(self):
        return list(self.inputtypes.keys())
