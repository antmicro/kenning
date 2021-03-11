"""
The below program tests compiling and running CUDA-based target on Jetson.

It compiles the model in the ONNX format to CUDA-based binary for the Jetson
AGX Xavier.

It requires TVM with CUDA 10.2 support on both compiling computer and Jetson.

It also requires running RPC server on Jetson.

Run:

    python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090

on Jetson AGX Xavier.

"""

import tvm
import tvm.relay as relay
import onnx
from tvm import rpc
from tvm.contrib import graph_runtime as runtime

import numpy as np

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model'
    )
    parser.add_argument(
        'output'
    )
    parser.add_argument(
        'ip_address'
    )
    parser.add_argument(
        'port'
    )
    args = parser.parse_args()
    tvm.autotvm.measure.measure_methods.set_cuda_target_arch('sm_72')
    onnxmodel = onnx.load(args.model)
    mod, params = relay.frontend.from_onnx(
        onnxmodel,
        shape={'input.1': (1, 3, 224, 224)},
        freeze_params=True,
        dtype='float32'
    )
    lib = relay.build(
        mod['main'],
        target=tvm.target.Target('nvidia/jetson-agx-xavier'),
        target_host='llvm -mtriple=aarch64-linux-gnu',
        params=params
    )

    lib.export_library(args.output)

    remote = rpc.connect(args.ip_address, args.port)
    remote.upload(args.output)
    rlib = remote.load_module(args.output)
    ctx = remote.gpu()
    m = runtime.GraphModule(rlib['default'](ctx))
    m.set_input(
        0,
        tvm.nd.array(np.random.randn(1, 3, 224, 224).astype('float32'))
    )
    m.run()
    out = m.get_output(0)
    print(out)
