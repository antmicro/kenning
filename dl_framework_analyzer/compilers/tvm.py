import tvm
from tvm import te
import tvm.relay as relay
import onnx
from pathlib import Path

from dl_framework_analyzer.core.compiler import ModelCompiler


class TVMCompiler(ModelCompiler):
    """
    The TVM compiler.
    """
    
    inputtypes = {
        'onnx': onnxconversion
    }

    def __init__(
            self,
            modelframework: str,
            target: str,
            target_host: str,
            opt_level=2):
        self.set_input_type(inputtype)
        self.target = tvm.target.Target(target)
        self.target_host = tvm.target.Target(target_host) if target_host else None
        self.opt_level = opt_level

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--model-framework',
            help='The input type of the model, framework-wise',
            choices=cls.inputtypes.keys()
            required=True
        )
        group.add_argument(
            '--target',
            help='The kind or tag of the target device',
            choices=tvm.target.Target.list_kinds() + [key for key, _ in tvm.target.list_tags().items()],
            required=True
        )
        group.add_argument(
            '--target-host',
            help='The kind or tag of the host (CPU) target device',
        )
        group.add_argument(
            '--compiled-model-path',
            help='The path to the compiled model output',
            type=Path,
            required=True
        )
        group.add_argument(
            '--opt-level',
            help='The optimization level of the compilation',
            default=2,
            type=int
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.model_framework,
            args.target,
            args.target_host,
            args.opt_level
        )

    def set_input_type(self, inputtype: str):
        assert inputtype in inputtypes.keys()
        self.inputtype = inputtype

    def onnxconversion(self, inputmodel: Any):
        onnxmodel = onnx.load(inputfile)
        return relay.frontend.from_onnx(onnxmodel)

    def compile_model(self, mod, params, outputpath):
        with tvm.transform.PassContext(opt_level=self.opt_level):
            lib = relay.build(
                mod['main'],
                target=self.target,
                target_host=self.target_host,
                params=params
            )
            lib.export_library(outputpath)

    def compile(self, inputmodel: Any, inputtype: str, outfile: Path):
        assert inputtype in self.inputtypes, 'Not supported model format'
        mod, params = self.inputtypes[inputtype](inputmodel)
        self.compile_model(mod, params, outfile)
