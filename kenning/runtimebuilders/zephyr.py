# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for west.
"""

import logging
import os
import subprocess
import venv
from functools import wraps
from pathlib import Path
from typing import Optional

from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.utils.logger import KLogger


class WestExecutionError(Exception):
    """
    Exception raised when West fails.
    """

    ...


class WestRun:
    """
    Allows for easy execution of West commands.
    """

    def __init__(
        self,
        workspace: "Optional[os.PathLike[str] | str]" = None,
        zephyr_base: "Optional[os.PathLike[str] | str]" = None,
        venv_dir: "Optional[os.PathLike[str] | str]" = None,
    ) -> None:
        """
        Prepares the WestRun object.

        Parameters
        ----------
        workspace : Optional[os.PathLike[str] | str]
            Path to the Zephyr workspace
        zephyr_base : Optional[os.PathLike[str] | str]
            Path to the Zephyr base
        venv_dir : Optional[os.PathLike[str] | str]
            Path to where the venv should be placed
        """
        self._venv_dir = None if venv_dir is None else Path(venv_dir)
        self._zephyr_base = None if zephyr_base is None else Path(zephyr_base)
        self._workspace = Path.cwd() if workspace is None else Path(workspace)

    def ensure_zephyr_base(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._ensure_zephyr_base()
            return func(self, *args, **kwargs)

        return wrapper

    def ensure_venv(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._ensure_venv()
            return func(self, *args, **kwargs)

        return wrapper

    def init(self):
        cmd = [
            self._west_exe,
            "init",
            "-l",
            str(self._workspace),
        ]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            msg = "Zephyr Workspace initialization failed."
            raise WestExecutionError(msg) from e

    @ensure_zephyr_base
    def update(self):
        cmd = [self._west_exe, "update"]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            msg = "Zephyr Workspace update failed."
            raise WestExecutionError(msg) from e

    @ensure_zephyr_base
    @ensure_venv
    def build(
        self,
        board,
        application_dir,
        build_dir=None,
        extra_conf_file=None,
        pristine=True,
    ):
        cmd = [
            self._west_exe,
            "build",
            "--pristine",
            "always" if pristine else "auto",
            "--board",
            board,
        ]

        if build_dir is not None:
            cmd.extend(["--build-dir", str(build_dir)])

        cmd.append(str(application_dir))

        if extra_conf_file is not None:
            cmd.extend(["--", f"-DEXTRA_CONF_FILE={extra_conf_file}"])

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            msg = (
                "Zephyr build failed. Try removing "
                f"'{build_dir}' and try again."
            )
            raise WestExecutionError(msg) from e

    def has_zephyr_base(self) -> bool:
        try:
            self._ensure_zephyr_base()
        except FileNotFoundError:
            return False
        return True

    def _ensure_zephyr_base(self):
        if self._zephyr_base is not None:
            return

        self._zephyr_base = self._search_for_zephyr_base()
        if self._zephyr_base is None:
            msg = "Couldn't find Zephyr base."
            raise FileNotFoundError(msg)

        os.environ["ZEPHYR_BASE"] = str(self._zephyr_base)

    @ensure_zephyr_base
    def _ensure_venv(self):
        if self._venv_dir is None:
            self._venv_dir = self._zephyr_base.parent / ".west-venv"

        if (self._venv_dir / "bin/west").exists():
            return

        self._prepare_venv()

    @ensure_zephyr_base
    def _prepare_venv(self):
        venv.EnvBuilder(clear=True, with_pip=True).create(self._venv_dir)

        cmd = [
            str(self._venv_dir / "bin/pip"),
            "install",
            "-r",
            str(self._zephyr_base / "scripts/requirements.txt"),
        ]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError:
            msg = "Setting up virtual environment for west failed."
            raise RuntimeError(msg)

    def _search_for_zephyr_base(self):
        if (p := os.environ.get("ZEPHYR_BASE", None)) is not None:
            return Path(p)

        for p in [self._workspace, *self._workspace.parents]:
            if (p / ".west").exists():
                return p / "zephyr"

    @property
    def _west_exe(self):
        west_path = self._venv_dir / "bin/west"

        if (west_path).exists():
            return str(west_path)

        return "west"

    @property
    def _subprocess_cfg(self):
        stream = None if KLogger.level <= logging.DEBUG else subprocess.DEVNULL
        return {
            "stdout": stream,
            "stderr": stream,
        }


class ZephyrRuntimeBuilder(RuntimeBuilder):
    """
    RuntimeBuilder for the kenning-zephyr-runtime.
    """

    arguments_structure = {
        "board": {
            "description": "Zephyr name of the target board",
            "type": str,
            "required": True,
        },
        "application_dir": {
            "description": "Workspace relative path to the app directory",
            "type": Path,
            "default": Path("app"),
        },
        "build_dir": {
            "description": "Workspace relative path to the build directory",
            "type": Path,
            "default": Path("build"),
        },
        "venv_dir": {
            "description": "Workspace relative path to west's venv",
            "type": Path,
            "default": Path(".west-venv"),
        },
        "zephyr_base": {
            "description": "Path to the Zephyr base",
            "type": Path,
            "default": None,
            "nullable": True,
        },
    }

    allowed_frameworks = [
        "tflite",
        "tvm",
    ]

    def __init__(
        self,
        workspace: Path,
        board: str,
        runtime_location: Optional[Path] = None,
        model_framework: Optional[str] = None,
        application_dir: Path = Path("app"),
        build_dir: Path = Path("build"),
        venv_dir: Path = Path(".west-venv"),
        zephyr_base: Optional[Path] = None,
    ):
        """
        RuntimeBuilder for the kenning-zephyr-runtime.

        Parameters
        ----------
        workspace: Path
            Location of the project directory.
        board: str
            Name of the target board.
        runtime_location: Optional[Path]
            Destination of the built runtime
        model_framework: Optional[str]
            Selected model framework
        application_dir: Path
            Path to the project's application directory.
            In the kenning-zephyr-runtime it's always './app'.
        build_dir: Path
            Path to the project's build directory.
        venv_dir: Path
            Venv for west
        zephyr_base: Optional[Path]
            Path to the Zephyr base.

        Raises
        ------
        FileNotFoundError
            If the application directory doesn't exist.
        """
        super().__init__(
            workspace=workspace.resolve(),
            runtime_location=runtime_location,
            model_framework=model_framework,
        )

        self.board = board

        self.application_dir = self._fix_relative(application_dir)
        if not self.application_dir.exists():
            msg = (
                "Application directory "
                f"'{self.application_dir}' doesn't exist."
            )
            raise FileNotFoundError(msg)

        self.build_dir = self._fix_relative(build_dir)
        self.build_dir.mkdir(exist_ok=True)

        self.venv_dir = self._fix_relative(venv_dir)

        self._westrun = WestRun(self.workspace, zephyr_base, self.venv_dir)

        if not self._westrun.has_zephyr_base():
            self._westrun.init()

        self._westrun.update()
        KLogger.info(f"Updated Zephyr workspace in '{self.workspace}'")

        self._prepare_modules()
        KLogger.info("Prepared modules")

    def build(self) -> Path:
        self._westrun.build(
            self.board,
            self.application_dir,
            self.build_dir,
            f"{self.model_framework}.conf",
            True,
        )

        runtime_elf = self.build_dir / "zephyr/zephyr.elf"

        if self.runtime_location is not None:
            self.runtime_location.unlink(missing_ok=True)
            self.runtime_location.symlink_to(runtime_elf)
            KLogger.info(
                "Zephyr Runtime was build and "
                f"symlinked to '{self.runtime_location.absolute()}'"
            )
            return self.runtime_location

        KLogger.info(
            "Zephyr Runtime was build and "
            f"is located in '{runtime_elf.absolute()}'"
        )
        return runtime_elf

    def _fix_relative(self, p: Path) -> Path:
        if p.is_relative_to(self.workspace):
            return p
        if not p.is_absolute():
            return self.workspace / p

        msg = f"Invalid path: '{p}'"
        raise ValueError(msg)

    def _prepare_modules(self):
        try:
            subprocess.run(
                ["./scripts/prepare_modules.sh"],
                cwd=self.workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Module preparation failed") from e
