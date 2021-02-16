import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dl_framework_analyzer',
    version='0.0.1',
    packages=setuptools.find_packages(),
    long_description=long_description,
    include_package_data=True,
    description="Deep Learning Frameworks' Analyzer",
    author='Antmicro Ltd.',
    author_email='contact@antmicro.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.20.0',
        'psutil>=5.8.0',
        'tensorflow>=2.4.1',
        'Jinja2>=2.11.2',
        'matplotlib>=3.3.4',
        'pynvml>=8.0.4',
        'tqdm>=4.56.2'
    ],
)
