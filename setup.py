import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='dl_framework_analyzer',
    version='0.0.1',
    packages=setuptools
    description="Deep Learning Frameworks' Analyzer",
    author='Antmicro Ltd.',
    author_email='contact@antmicro.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)


