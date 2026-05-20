"""
A set of Python bindings for the uv command line tool.
"""


import subprocess
from typing import List

from kenning.core.exceptions import KenningUVError
from kenning.utils.logger import KLogger


def _run_with_error_handling(command: List[str]):
    """
    Runs a command and checks if it succeeded.

    Parameters
    ----------
    command: List[str]
        Command to be ran.

    Raises
    ------
    KenningUVError
        Thrown when the command fails (returns a non-zero code).
    """
    KLogger.debug(f"Running uv command: '{' '.join(command)}'...")
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        KLogger.error(
            f"Command failed: '{' '.join(command)}', error: '{result.stderr}'."
        )
        raise KenningUVError(
            f"Command '{' '.join(command)}' returned a non-zero exit code."
            f" Error output: '{result.stderr}'."
        )


def venv(python: str = None, path: str = None):
    """
    Creates a Python virtual environment using uv.

    Parameters
    ----------
    python: str
        Python version for the environment (eg. '3.11').
    path: str
        Path for the environment. If not specified defaults to '.venv'
    """
    command = ["uv", "venv", "--clear"]
    if path:
        command += [path]
    if python:
        command += ["--python", python]
    _run_with_error_handling(command)


def pip_install(packages: List[str] = None, venv_path: str = None):
    """
    Installs Python packages.

    Parameters
    ----------
    packages: List[str]
        List of packages to install (eg. ['numpy', 'matplotlib']).
    venv_path: str
        Path to the Python binary from the environment the packages will be
        installed to. If not specified, uv defaults will be used.
    """
    command = ["uv", "pip", "install", "--upgrade"]
    if packages:
        command += packages
    if venv_path:
        command += ["--python", venv_path]
    _run_with_error_handling(command)
