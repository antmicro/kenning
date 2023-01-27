# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='kenning',
    version='0.0.1',
    packages=setuptools.find_packages(),
    long_description=long_description,
    include_package_data=True,
    description="Kenning - a framework for implementing and testing deployment pipelines for deep learning applications on edge devices",  # noqa: E501
    author='Antmicro Ltd.',
    author_email='contact@antmicro.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'Jinja2>=2.11.2',
        'Pillow>=8.1.0',
        'matplotlib>=3.3.4',
        'numpy~=1.23.5',
        'onnx>=1.7.0',
        'psutil>=5.8.0',
        'scikit_learn>=0.24.1',
        'tqdm>=4.56.2',
        'jsonschema>=4.16.0',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_antmicro_theme @ git+https://github.com/antmicro/sphinx_antmicro_theme.git#egg=sphinx_antmicro_theme',  # noqa: E501
            'sphinxcontrib-napoleon',
            'docutils==0.16',
            'myst-parser'
        ],
        'tensorflow': [
            'onnx_tf~=1.10.0',
            'tensorflow~=2.9.1',
            'tensorflow_addons~=0.17.1',
            'tf2onnx~=1.11.1',
            'tensorflow_probability~=0.17.0',
            'tensorflow_model_optimization~=0.7.3'
        ],
        'torch': [
            'torch~=1.13.0',
            'torchvision~=0.14.0'
        ],
        'mxnet': [
            'gluoncv>=0.10.2',
            'mxnet~=1.9.1'
        ],
        'nvidia_perf': [
            'pynvml>=8.0.4'
        ],
        'object_detection': [
            'boto3>=1.17.5',
            'botocore>=1.20.5',
            'opencv_python>=4.5.2',
            'pandas>=1.2.1',
            'pycocotools'
        ],
        ":python_version<'3.9'": [
            'importlib_resources>=5.1.4'
        ],
        'iree': [
            'iree-compiler>=20220415.108',
            'iree-runtime>=20220415.108',
            'iree-tools-tf>=20220415.108',
            'iree-tools-tflite>=20220415.108'
        ],
        'tvm': [
            'apache-tvm'
        ],
        'onnxruntime': [
            'onnxruntime'
        ],
        'test': [
            'pytest',
            'pytest-mock'
        ]
    },
)
