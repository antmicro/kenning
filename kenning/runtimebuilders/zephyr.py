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
from pathlib import Path
from typing import Optional

from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.utils.logger import KLogger


class _WestRun:
    def __init__(
        self, workspace=None, zephyr_base=None, venv_dir=None
    ) -> None:
        self._venv_dir = venv_dir
        self._zephyr_base = zephyr_base
        self._workspace = Path.cwd() if workspace is None else workspace

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
            raise Exception("west init failed.") from e

    def update(self):
        self._ensure_zephyr_base()

        cmd = [self._west_exe, "update"]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            raise Exception("west update failed.") from e

    def build(
        self,
        board,
        application_dir,
        build_dir=None,
        extra_conf_file=None,
        pristine=True,
    ):
        self._ensure_zephyr_base()
        self._ensure_venv()

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
                "Zephyr build failed. Try removing"
                f"'{build_dir}' and try again."
            )
            raise Exception(msg) from e

    def has_zephyr_base(self) -> bool:
        try:
            self._ensure_zephyr_base()
        except Exception:
            return False
        return True

    def _ensure_zephyr_base(self):
        if self._zephyr_base is not None:
            return

        self._zephyr_base = self._search_for_zephyr_base()
        if self._zephyr_base is None:
            raise Exception("zephyr base not found")

        os.environ["ZEPHYR_BASE"] = str(self._zephyr_base)

    def _ensure_venv(self):
        self._ensure_zephyr_base()

        if self._venv_dir is None:
            self._venv_dir = self._zephyr_base.parent / ".west-venv"

        if (self._venv_dir / "bin/west").exists():
            return

        self._prepare_venv()

    def _prepare_venv(self):
        self._ensure_zephyr_base()

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
            raise Exception("venv setup failed")

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
            "description": "Zephyr board",
            "type": str,
            "required": True,
        },
        "application_dir": {
            "description": "Application source directory",
            "type": Path,
            "default": Path("./app"),
        },
        "build_dir": {
            "description": "Build directory",
            "type": Path,
            "default": Path("./build"),
        },
        "venv_dir": {
            "description": "venv directory for west",
            "type": Path,
            "default": Path("./venv"),
        },
        "zephyr_base": {
            "description": "The path to the Zephyr base",
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
                "Application source directory "
                f"'{self.application_dir}' doesn't exist."
            )
            raise FileNotFoundError(msg)

        self.build_dir = self._fix_relative(build_dir)
        self.build_dir.mkdir(exist_ok=True)

        self.venv_dir = self._fix_relative(venv_dir)

        self._westrun = _WestRun(workspace, zephyr_base, self.venv_dir)

        if not self._westrun.has_zephyr_base():
            self._westrun.init()

        self._westrun.update()
        KLogger.info("Updated Zephyr")

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

        KLogger.info("Built runtime")

        runtime_elf = self.build_dir / "zephyr/zephyr.elf"

        if self.runtime_location is not None:
            self.runtime_location.unlink(missing_ok=True)
            self.runtime_location.symlink_to(runtime_elf)
            return self.runtime_location

        return runtime_elf

    def _fix_relative(self, p: Path) -> Path:
        if p.is_relative_to(self.workspace):
            p = p.relative_to(self.workspace)
        else:
            p = self.workspace / p

        return p

    def _prepare_modules(self):
        try:
            subprocess.run(
                ["./scripts/prepare_modules.sh"],
                cwd=self.workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise Exception("failed to prepare modules") from e
