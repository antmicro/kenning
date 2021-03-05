import tvm
from tvm import te
import tvm.relay as relay
import onnx

from dl_framework_analyzer.core.compiler import ModelCompiler


class TVMCompiler(object):
    """
    The TVM compiler.
    """
    
    inputtypes = {
        'onnx': onnxconversion
    }

    def __init__(self, inputtype: str, target: str):
        self.set_input_type(inputtype)
        self.target = target

    def set_input_type(self, inputtype: str):
        assert inputtype in inputtypes.keys()
        self.inputtype = inputtype

    def onnxconversion(self, inputfile):
        onnxmodel = onnx.load(inputfile)
        mod, params = relay.frontend.from_onnx(inputfile)

        with tvm.transform.PassContext(opt_level=1):
            intrp = relay.build_module.create_executor(
                "graph",
                mod,
                tvm.cpu(0),
                target
            )
