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
        application_dir: Path = Path("./app"),
        build_dir: Path = Path("./build"),
        venv_dir: Path = Path("./venv"),
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
        self.board = board
        self.workspace = workspace.resolve()

        self.application_dir = self._fix_relative(workspace, application_dir)
        if not self.application_dir.exists():
            msg = (
                "Application source directory "
                f'"{self.application_dir}" doesn\'t exist.'
            )
            raise FileNotFoundError(msg)

        self.build_dir = self._fix_relative(workspace, build_dir)
        self.build_dir.mkdir(exist_ok=True)

        self.venv_dir = self._fix_relative(workspace, venv_dir)

        self.model_framework = model_framework

        self.zephyr_base = None

        if zephyr_base is not None:
            self.zephyr_base = Path(zephyr_base)

        if self.zephyr_base is None:
            self.zephyr_base = self._search_for_zephyr_base()
            KLogger.info(f"Found Zephyr at {self.zephyr_base}")

        if self.zephyr_base is None:
            self.zephyr_base = self._init_zephyr()
            KLogger.info(f"Initialized Zephyr at {self.zephyr_base}")

        os.environ["ZEPHYR_BASE"] = str(self.zephyr_base)

        self._update_zephyr()
        KLogger.info("Updated Zephyr")

        self._prepare_venv()
        KLogger.info("Prepared venv")

        self._prepare_modules()
        KLogger.info("Prepared modules")

        super().__init__(
            workspace=workspace,
            runtime_location=runtime_location,
            model_framework=model_framework,
        )

    def build(self) -> Path:
        self._build_project(f"{self.model_framework}.conf")
        KLogger.info("Built runtime")

        runtime_elf = self.build_dir / "zephyr/zephyr.elf"

        if self.runtime_location is not None:
            self.runtime_location.unlink(missing_ok=True)
            self.runtime_location.symlink_to(runtime_elf)
            return self.runtime_location

        return runtime_elf

    def _fix_relative(self, base: Path, p: Path) -> Path:
        if p.is_relative_to(base):
            p = p.relative_to(base)
        else:
            p = base / p

        return p

    def _search_for_zephyr_base(self):
        if (p := os.environ.get("ZEPHYR_BASE", None)) is not None:
            return Path(p)

        for p in self.workspace.parents:
            if (p / ".west").exists():
                return p / "zephyr"

    def _init_zephyr(self):
        try:
            subprocess.run(
                [self._west_path, "init", "-l", str(self.workspace)],
                **self._subprocess_cfg,
            ).check_returncode()
        except subprocess.CalledProcessError as e:
            raise Exception("west init failed.") from e

        return self.workspace.parent / "zephyr"

    def _update_zephyr(self):
        try:
            subprocess.run(
                [self._west_path, "update"], **self._subprocess_cfg
            ).check_returncode()
        except subprocess.CalledProcessError as e:
            raise Exception("west update failed.") from e

    def _prepare_modules(self):
        try:
            subprocess.run(
                ["./scripts/prepare_modules.sh"],
                cwd=self.workspace,
                **self._subprocess_cfg,
            )
        except subprocess.CalledProcessError as e:
            raise Exception("failed to prepare modules") from e

    def _build_project(self, extra_conf_file, pristine=True):
        cmd = [
            self._west_path,
            "build",
            "--pristine",
            "always" if pristine else "auto",
            "--board",
            self.board,
            "--build-dir",
            self.build_dir,
            self.application_dir,
            "--",
            f"-DEXTRA_CONF_FILE={extra_conf_file}",
        ]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            msg = (
                "Zephyr build failed. Try removing"
                f'"{self.build_dir}" and try again.'
            )
            raise Exception(msg) from e

    def _prepare_venv(self):
        venv.EnvBuilder(clear=True, with_pip=True).create(self.venv_dir)

        cmd = [
            str(self.venv_dir / "bin/pip"),
            "install",
            "-r",
            str(self.zephyr_base / "scripts/requirements.txt"),
        ]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError:
            raise Exception("venv setup failed")

    @property
    def _west_path(self):
        if (self.venv_dir / "bin/west").exists():
            return str(self.venv_dir / "bin/west")

        return "west"

    @property
    def _subprocess_cfg(self):
        stream = None if KLogger.level <= logging.DEBUG else subprocess.DEVNULL
        return {
            "stdout": stream,
            "stderr": stream,
        }
