"""
Wrapper for TVM deep learning compiler.
"""

import tvm
import onnx
import tvm.relay as relay
from pathlib import Path
import re

from edge_ai_tester.core.compiler import ModelCompiler, CompilationError
from edge_ai_tester.core.dataset import Dataset
from edge_ai_tester.utils.logger import get_logger


def onnxconversion(compiler: 'TVMCompiler', modelpath: Path, input_shapes, dtype='float32'):
    onnxmodel = onnx.load(modelpath)
    return relay.frontend.from_onnx(
        onnxmodel,
        shape=input_shapes,
        freeze_params=True,
        dtype=dtype)


def kerasconversion(compiler: 'TVMCompiler', modelpath: Path, input_shapes, dtype='float32'):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(modelpath))
    print(model.summary())
    return relay.frontend.from_keras(
        model,
        shape=input_shapes,
        layout='NHWC'
    )


def darknetconversion(compiler: 'TVMCompiler', modelpath: Path, input_shapes, dtype='float32'):
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


class TVMCompiler(ModelCompiler):
    """
    The TVM compiler.
    """

    inputtypes = {
        'onnx': onnxconversion,
        'keras': kerasconversion,
        'darknet': darknetconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str,
            target: str,
            target_host: str,
            opt_level: int = 2,
            libdarknetpath: str = '/usr/local/lib/libdarknet.so'):
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
            choices=(tvm.target.Target.list_kinds() +
                     [key for key, _ in tvm.target.list_tags().items()]),
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
            args.libdarknet_path
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
