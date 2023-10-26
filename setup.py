# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import setuptools
import numpy
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    long_description=long_description,
    include_package_data=True,
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
