# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

CORALNPU_TRACE_PARSER_DIR = Path(__file__).parent.resolve()
CORALNPU_TRACE_PROTO_FILE = CORALNPU_TRACE_PARSER_DIR / "coralnpu_trace.proto"


def generate_protobuf(output_dir: Path, output_type: str):
    """
    Generate Protobuf sources for the CoralNPU trace parser.

    Parameters
    ----------
    output_dir : Path
        Directory where generated files should be written.
    output_type : str
        Protobuf output type, e.g. "python" or "cpp".

    Raises
    ------
    RuntimeError
        If the protoc executable is not available.
    """
    protoc = shutil.which("protoc")
    if protoc is None:
        raise RuntimeError(
            "protoc is required to build the CoralNPU trace parser"
        )

    subprocess.check_call(
        [
            protoc,
            f"--proto_path={CORALNPU_TRACE_PARSER_DIR}",
            f"--{output_type}_out={output_dir}",
            CORALNPU_TRACE_PROTO_FILE.name,
        ],
        cwd=CORALNPU_TRACE_PARSER_DIR,
    )


class BuildPy(build_py):
    """
    Build Python package with generated Protobuf bindings.
    """

    def run(self):
        output_file = CORALNPU_TRACE_PARSER_DIR / "coralnpu_trace_pb2.py"

        if (
            not output_file.exists()
            or output_file.stat().st_mtime
            < CORALNPU_TRACE_PROTO_FILE.stat().st_mtime
        ):
            generate_protobuf(CORALNPU_TRACE_PARSER_DIR, "python")

        super().run()


class BuildExt(build_ext):
    """
    Build C++ extension with generated Protobuf bindings.
    """

    def build_extension(self, ext):
        if ext.name == "kenning.utils.coralnpu_trace_parser._parser":
            output_dir = (
                Path(self.build_temp) / "coralnpu_trace_parser"
            ).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            generate_protobuf(output_dir, "cpp")

            ext.sources.append(str(output_dir / "coralnpu_trace.pb.cc"))
            ext.include_dirs.append(str(output_dir))

        super().build_extension(ext)


setup(
    packages=["kenning.utils.coralnpu_trace_parser"],
    package_dir={
        "kenning.utils.coralnpu_trace_parser": ".",
    },
    package_data={
        "kenning.utils.coralnpu_trace_parser": [
            "coralnpu_trace.proto",
        ],
    },
    ext_modules=cythonize(
        [
            Extension(
                "kenning.utils.coralnpu_trace_parser._parser",
                sources=[
                    "caller.pyx",
                    "parser.cpp",
                ],
                include_dirs=[numpy.get_include()],
                language="c++",
                libraries=["protobuf"],
                extra_compile_args=["-O3", "-Os"],
            ),
        ],
    ),
    cmdclass={
        "build_py": BuildPy,
        "build_ext": BuildExt,
    },
)
