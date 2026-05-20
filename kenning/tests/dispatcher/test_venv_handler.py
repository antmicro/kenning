from pathlib import Path

import pytest

import kenning.dispatcher.venv_handler
from kenning.core.exceptions import KenningRequirementsError, KenningUVError
from kenning.dispatcher.venv_handler import (
    ENVIRONMENT_FOLDER_NAME,
    VEnvHandler,
)


class TestVEnvHandler:
    def test_init_wrong_name(self):
        with pytest.raises(KenningRequirementsError):
            VEnvHandler("foo/bar/*/")

    def test_create_venv(self):
        venv = VEnvHandler("test")
        venv.python = "3.12"
        venv.dependencies = ["pyinstrument==5.0.0", "numpy==2.0.0"]

        venv._create_venv()

        with open(
            Path.home() / (ENVIRONMENT_FOLDER_NAME + "/.test.checksum"), "rb"
        ) as file:
            assert b"\xDC\x22\x0B\x33" == file.read()

        code, output, err = venv.execute_module("pyinstrument", ["--version"])
        assert 0 == code
        assert "pyinstrument 5.0.0, on Python 3.12." in output

    def test_setup_venv(self):
        venv = VEnvHandler("test_setup_venv")

        venv.python = "3.12"
        venv.dependencies = ["pyinstrument==5.0.0", "numpy==2.0.0"]

        venv._setup_venv()
        code, output, err = venv.execute_module("pyinstrument", ["--version"])

        assert 0 == code
        assert "pyinstrument 5.0.0, on Python 3.12." in output

        venv.python = "3.12"
        venv.dependencies = ["pyinstrument==5.1.2", "numpy==2.0.0"]

        venv._setup_venv()
        code, output, err = venv.execute_module("pyinstrument", ["--version"])

        assert 0 == code
        assert "pyinstrument 5.1.2, on Python 3.12." in output

        venv.python = "3.10"
        venv.dependencies = ["pyinstrument==5.0.2", "numpy==2.0.0"]

        venv._setup_venv()
        code, output, err = venv.execute_module("pyinstrument", ["--version"])

        assert 0 == code
        assert "pyinstrument 5.0.2, on Python 3.10." in output

    def test_read_yaml(self):
        kenning.dispatcher.venv_handler.REQUIREMENTS_MODULE = (
            "kenning.tests.dispatcher.test-requirement-files"
        )

        venv = VEnvHandler("TestBasic")

        assert (
            "kenning/tests/dispatcher/test-requirement-files/TestBasic-requirements.yml"
            in str(venv._read_requirements_yaml())
        )

    def test_initialize_basic(self):
        kenning.dispatcher.venv_handler.REQUIREMENTS_MODULE = (
            "kenning.tests.dispatcher.test-requirement-files"
        )
        venv = VEnvHandler("TestBasic")

        venv.initialize()
        code, output, err = venv.execute_module("pyinstrument", ["--version"])

        assert 0 == code
        assert "pyinstrument 5.1.2, on Python 3.11." in output

        process = venv.start_module("tqdm", ["--version"])
        process.wait()

        assert "4.67.3" in process.communicate()[0]

    def test_initialize_no_file(self):
        kenning.dispatcher.venv_handler.REQUIREMENTS_MODULE = (
            "kenning.tests.dispatcher.test-requirement-files"
        )
        venv = VEnvHandler("TestThisFileDoesNotExist")

        with pytest.raises(KenningRequirementsError):
            venv.initialize()

    def test_initialize_syntax_error(self):
        kenning.dispatcher.venv_handler.REQUIREMENTS_MODULE = (
            "kenning.tests.dispatcher.test-requirement-files"
        )
        venv = VEnvHandler("TestSyntaxError")

        with pytest.raises(KenningRequirementsError):
            venv.initialize()

    def test_initialize_no_package(self):
        kenning.dispatcher.venv_handler.REQUIREMENTS_MODULE = (
            "kenning.tests.dispatcher.test-requirement-files"
        )
        venv = VEnvHandler("TestNoPackage")

        with pytest.raises(KenningUVError):
            venv.initialize()
