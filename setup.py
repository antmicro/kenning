import setuptools

with open('README.rst', 'r') as fh:
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
        'numpy>=1.20.0',
        'onnx>=1.7.0',
        'psutil>=5.8.0',
        'scikit_learn>=0.24.1',
        'tqdm>=4.56.2',
        'jsonschema'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_antmicro_theme @ git+https://github.com/antmicro/sphinx_antmicro_theme.git#egg=sphinx_antmicro_theme',  # noqa: E501
            'sphinxcontrib-napoleon',
            'docutils==0.16'
        ],
        'tensorflow': [
            'onnx_tf>=1.7.0',
            'tensorflow>=2.4.1',
            'tensorflow_addons>=0.12.1',
            'tf2onnx>=1.8.3',
            'tensorflow_probability'
        ],
        'torch': [
            'torch>=1.7.1',
            'torchvision>=0.8.2'
        ],
        'mxnet': [
            'gluoncv>=0.10.2',
            'mxnet>=1.8.0'
        ],
        'nvidia_perf': [
            'pynvml>=8.0.4'
        ],
        'object_detection': [
            'boto3>=1.17.5',
            'botocore>=1.20.5',
            'opencv_python>=4.5.2',
            'pandas>=1.2.1'
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
        ]
    },
)
