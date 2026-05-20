"""
Handling system for virtual environments created with uv.
Environments are handled/identified based on a unique string (name).
"""
import re
import subprocess
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import importlib_resources
import yaml

import kenning.dispatcher.uv_bindings as uv
from kenning.core.exceptions import KenningRequirementsError

# Where the created environments will be stored (relative to home directory).
ENVIRONMENT_FOLDER_NAME = ".kenning-env"

# Length, in bytes, of the checksum used to verify if an environment is
# up-to-date (hash of the Python version and list of dependencies).
CHECKSUM_BYTES = 4

# Default Python version used for all environments, where version is not given
# explicitly.
DEFAULT_PYTHON = "3.12"

# Name of the Kenning project submodule, where special YAML files, describing
# specifications for virtual environments, are to be located.
REQUIREMENTS_MODULE = "kenning.requirements"

# Suffix added to virtual environment name, to get name of the YAML file with
# specifications.
REQUIREMENTS_FILE_SUFFIX = "-requirements.yml"


class VEnvHandler:
    """
    Class serving as a handle for manipulating a virtual environment.
    """

    def __init__(self, name: str):
        if not re.search("^[a-zA-Z0-9_-]+$", name):
            raise KenningRequirementsError(
                f"Virtual environment name '{name}' invalid."
            )
        self.name = name
        # Those variables will be set properly when 'initialize' is called.
        self.python = None
        self.dependencies = None

    def initialize(self):
        """
        Checks status of the virtual environment and builds or rebuilds it if
        necessary.
        """
        environment_specification = self._read_requirements_yaml()
        if not environment_specification:
            raise KenningRequirementsError(
                f"Requirement specification YAML file for the {self.name}"
                f" module is not present in {REQUIREMENTS_MODULE}."
            )
        try:
            self._parse_requirements_yaml(environment_specification)
        except Exception as e:
            raise KenningRequirementsError(
                "Parsing Requirement specification YAML file for the"
                f" {self.name} module failed with: {e}."
            )
        self._setup_venv()

    def execute_module(
        self, module: str, parameters: List[str] = []
    ) -> Tuple[int, str, str]:
        """
        Executes a module from the virtual environment and waits for it to
        finish.

        Parameters
        ----------
        module: str
            Name of the module to run.
        parameters: List[str]
            Arguments for the module.

        Returns
        -------
        Tuple[int, str, str]
            Return code, standard output and standard error output from the
            process being ran.
        """
        res = subprocess.run(
            [self._get_venv_path() / "bin/python", "-m"]
            + [module]
            + parameters,
            capture_output=True,
            text=True,
        )
        return res.returncode, res.stdout, res.stderr

    def start_module(
        self, module: str, parameters: List[str] = []
    ) -> subprocess.Popen:
        """
        Starts a module from the virtual environment.

        Parameters
        ----------
        module: str
            Name of the module to run.
        parameters: List[str]
            Arguments for the module.

        Returns
        -------
        subprocess.Popen
            Process handling object.
        """
        return subprocess.Popen(
            [self._get_venv_path() / "bin/python", "-m"]
            + [module]
            + parameters,
            stdout=subprocess.PIPE,
            text=True,
        )

    def _get_venv_path(self) -> Path:
        """
        Infers the path to the virtual environment based on the name.

        Returns
        -------
        Path
            Path the the virtual environment.
        """
        return Path.home() / ENVIRONMENT_FOLDER_NAME / f".{self.name}"

    def _get_checksum_path(self) -> Path:
        """
        Infers the path to the checksum file of the virtual environment based
        on the name.

        Returns
        -------
        Path
            Path the the checksum file.
        """
        return self._get_venv_path().with_suffix(".checksum")

    def _get_venv_path_str(self) -> str:
        """
        Infers the path to the virtual environment based on name as a string.

        Returns
        -------
        str
            Path the the virtual environment.
        """
        return str(self._get_venv_path())

    def _compute_checksum(self) -> bytes:
        """
        Computes a checksum based on details of the virtual environment.

        Returns
        -------
        bytes
            Bytestream with the checksum.
        """
        return abs(
            zlib.adler32(
                bytes(
                    self.python + " ".join(self.dependencies), encoding="ascii"
                )
            )
        ) % pow(2, CHECKSUM_BYTES * 8)

    def _write_checksum(self):
        """
        Computes a checksum based on details of the virtual environment and
        saves it under the proper path (inferred from the name).
        """
        with open(self._get_checksum_path(), "wb") as file:
            file.write(
                self._compute_checksum().to_bytes(
                    CHECKSUM_BYTES, byteorder="big"
                )
            )

    def _load_checksum(self) -> bytes:
        """
        Loads the virtual environment checksum from the checksum file.

        Returns
        -------
        bytes
            Bytestream with the checksum.
        """
        with open(self._get_checksum_path(), "rb") as file:
            return file.read()

    def _verify_checksum(self) -> bool:
        """
        Checks if the checksum in the file is the same as it should be
        according to the given Python version and dependencies.

        Returns
        -------
        bool
            True if the checksum file exists and contains the correct value,
            False if the checksum file does not exist, or does not contain the
            correct value.
        """
        checksum_file_path = self._get_checksum_path()
        if checksum_file_path.exists():
            return self._compute_checksum() == self._load_checksum()
        return False

    def _create_venv(self):
        """
        Creates a virtual environment and a checksum file for it.
        """
        uv.venv(python=self.python, path=self._get_venv_path_str())
        uv.pip_install(
            packages=self.dependencies, venv_path=self._get_venv_path_str()
        )
        self._write_checksum()

    def _setup_venv(self):
        """
        Checks if the virtual environment already exists, with the correct
        Python version and dependencies, and creates/recreates it, if it
        doesn't.
        """
        if not self._verify_checksum():
            self._create_venv()

    def _parse_requirements_yaml(self, yaml_file: Path):
        """
        Parses a YAML dump, containing information about the venv. Saves the
        relevant virtual environment specification in the object.

        Parameters
        ----------
        yaml_file: Path
            Path to the yaml file.
        """
        with open(yaml_file, "r") as file:
            specs = yaml.safe_load(file.read())
            self.python = DEFAULT_PYTHON
            self.dependencies = []
            if specs:
                if "python" in specs:
                    self.python = specs["python"]
                if "requirements" in specs:
                    self.dependencies = specs["requirements"]

    def _read_requirements_yaml(self) -> Optional[Path]:
        """
        Searches for the requirements YAML file, with name inferred from the
        virtual environment name, in the resource module with those files.

        Returns
        -------
        Optional[Path]
            File path if found. 'None' otherwise.
        """

        def find_and_read_file(
            name: str, traversable: importlib_resources.abc.Traversable
        ) -> Optional[Path]:
            """
            Searches for a file with the given name in the given resource
            module. The search is recursive.

            Parameters
            ----------
            name:str
                Name of the file.
            traversable:importlib_resources.abc.Traversable
                The package resource module to search.

            Returns
            -------
            Optional[Path]
                File resource handle if found. 'None' otherwise.
            """
            # Checking everything in the folder
            contents = traversable.iterdir()
            for entity in contents:
                new_traversable = traversable.joinpath(entity)
                if new_traversable.is_file():
                    # Returning contents of our file, if we found it
                    if entity.name == name:
                        return new_traversable
                elif new_traversable.is_dir():
                    # If we have a subfolder, we call the function recursively,
                    # and return results IF it finds the file
                    candidate = find_and_read_file(name, new_traversable)
                    if candidate:
                        return candidate
            # Return None if we didn't find the file.
            return None

        return find_and_read_file(
            self.name + REQUIREMENTS_FILE_SUFFIX,
            importlib_resources.files(REQUIREMENTS_MODULE),
        )
