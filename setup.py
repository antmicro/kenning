# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy
import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

project_path = (
    Path(__file__).parent / "kenning/sparsity_aware_kernel"
).absolute()

setuptools.setup(
    packages=setuptools.find_packages(),
    long_description=long_description,
    include_package_data=True,
    extras_require={
        "sparsity-aware-kernel": [
            f"kenning-sparsity-aware-kernel @ file://{project_path}"
        ]
    },
    ext_modules=cythonize(
        setuptools.Extension(
            "kenning.modelwrappers.instance_segmentation.cython_nms",
            sources=[
                "kenning/modelwrappers/instance_segmentation/cython_nms.pyx"
            ],
            include_dirs=[numpy.get_include()],
        )
    ),
)
