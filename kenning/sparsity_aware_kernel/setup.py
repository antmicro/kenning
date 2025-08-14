# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from kenning.core.exceptions import KenningOptimizerError

DEBUG_CXX_FLAGS = ["-g"]
OPT_CXX_FLAGS = ["-O2", "-std=c++17"]

DEBUG_NVCC_FLAGS = ["-g", "-G"]
OPT_NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
OPT_CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
OPT_NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


class CUTLASSNotInitialized(KenningOptimizerError):
    """
    Exception raised when 'cutlass_library' cannot be imported.
    """

    pass


try:
    import cutlass_library

    CUTLASS_DIR = Path(cutlass_library.__file__).parent / "source"
except ImportError:
    raise CUTLASSNotInitialized(
        "CUTLASS library was not installed properly. "
        + "Make sure that 'cutlass_library' can be imported in python"
    )

setup(
    name="custom_ext",
    ext_modules=[
        CUDAExtension(
            "custom_ext",
            [
                "custom_ext/gptq/q_compressed_gemm.cu",
                "custom_ext/pybind.cpp",
            ],
            extra_compile_args={"cxx": OPT_CXX_FLAGS, "nvcc": OPT_NVCC_FLAGS},
            include_dirs=[
                f"{CUTLASS_DIR}/include",
                f"{CUTLASS_DIR}/tools/util/include",
                f"{CUTLASS_DIR}/tools/library/include",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
