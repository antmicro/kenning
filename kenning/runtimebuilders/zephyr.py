# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a runtime builder capable of compiling the Kenning Zephyr runtime.
"""

import logging
import os
import shutil
import subprocess
import venv
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union

from kenning.core.platform import Platform
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
        workspace: "Optional[Union[os.PathLike[str], str]]" = None,
        zephyr_base: "Optional[Union[os.PathLike[str], str]]" = None,
        venv_dir: "Optional[Union[os.PathLike[str], str]]" = None,
    ) -> None:
        """
        Prepares the WestRun object.

        Parameters
        ----------
        workspace : Optional[Union[os.PathLike[str], str]]
            Path to the Zephyr workspace
        zephyr_base : Optional[Union[os.PathLike[str], str]]
            Path to the Zephyr base
        venv_dir : Optional[Union[os.PathLike[str], str]]
            Path to where the venv should be placed
        """
        self._venv_dir = None if venv_dir is None else Path(venv_dir)
        self._zephyr_base = None if zephyr_base is None else Path(zephyr_base)
        self._workspace = Path.cwd() if workspace is None else Path(workspace)

    def _ensure_zephyr_base(func):
        """
        Ensures that Zephyr base is found before the decorated
        function is called.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._ensure_zephyr_base_helper()
            return func(self, *args, **kwargs)

        return wrapper

    def _ensure_venv(func):
        """
        Ensures that virtual environment for West is created before
        the decorated function is called.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._ensure_venv_helper()
            return func(self, *args, **kwargs)

        return wrapper

    def init(self):
        """
        Wrapper for 'west init'.
        """
        self._subprocess_west_run(["init", "-l", str(self._workspace)])

    @_ensure_zephyr_base
    def update(self):
        """
        Wrapper for 'west update'.
        """
        self._subprocess_west_run(["update"])

    @_ensure_zephyr_base
    @_ensure_venv
    def build(
        self,
        board: str,
        application_dir: "Union[os.PathLike[str], Path]",
        build_dir: "Optional[Union[os.PathLike[str], Path]]" = None,
        extra_conf_file: "Optional[str]" = None,
        extra_build_args: List[str] = [],
        pristine: str = "auto",
    ):
        """
        Wrapper for 'west build'.

        Parameters
        ----------
        board : str
            Name of the board passed to '--board'.
        application_dir : Union[os.PathLike[str], Path]
            Path to the application dir (usually './app').
        build_dir : Optional[Union[os.PathLike[str], Path]]
            Path to where the build directory should be located.
        extra_conf_file : Optional[str]
            Name of the additional .conf file.
        extra_build_args : List[str]
            Extra arguments passed to west.
        pristine : str
            Whether build should be pristine. One of "always", "never", "auto".
        """
        params = [
            "build",
            str(application_dir),
            "--pristine",
            pristine,
            "--board",
            board,
        ]

        if build_dir is not None:
            params.extend(["--build-dir", str(build_dir)])

        params.append("--")
        if extra_conf_file is not None:
            params.append(f"-DEXTRA_CONF_FILE={extra_conf_file}")

        params.extend(extra_build_args)

        self._subprocess_west_run(params)

    def build_target(
        self,
        target: str,
        build_dir: "Optional[Union[os.PathLike[str], Path]]" = None,
    ):
        params = [
            "build",
            "-t",
            target,
        ]

        if build_dir is not None:
            params.extend(["--build-dir", str(build_dir)])

        self._subprocess_west_run(params)

    def has_zephyr_base(self) -> bool:
        """
        Checks if Zephyr base exists.
        """
        try:
            self._ensure_zephyr_base_helper()
        except FileNotFoundError:
            return False
        return True

    def _ensure_zephyr_base_helper(self):
        if self._zephyr_base is not None:
            return

        self._zephyr_base = self._search_for_zephyr_base()
        if self._zephyr_base is None:
            msg = "Couldn't find Zephyr base."
            raise FileNotFoundError(msg)

        os.environ["ZEPHYR_BASE"] = str(self._zephyr_base)

    @_ensure_zephyr_base
    def _ensure_venv_helper(self):
        if self._venv_dir is None:
            self._venv_dir = self._zephyr_base.parent / ".west-venv"

        if (self._venv_dir / "bin/west").exists():
            return

        self._prepare_venv()

    @_ensure_zephyr_base
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

    def _get_west_executable(self):
        west_path = self._venv_dir / "bin/west"

        if (west_path).exists():
            return str(west_path)

        return "west"

    def _subprocess_west_run(self, params):
        cmd = [self._get_west_executable(), *params]

        try:
            subprocess.run(cmd, **self._subprocess_cfg).check_returncode()
        except subprocess.CalledProcessError as e:
            msg = f"Command: '{' '.join(cmd)}' failed"
            if e.stderr is not None:
                msg += f" with a message:\n\n{e.stderr.decode()}"
            raise WestExecutionError(msg) from e

    @property
    def _subprocess_cfg(self):
        cfg = {}
        if KLogger.level > logging.DEBUG:
            cfg["capture_output"] = True

        return cfg


class ZephyrRuntimeBuilder(RuntimeBuilder):
    """
    RuntimeBuilder for the kenning-zephyr-runtime.
    """

    arguments_structure = {
        "board": {
            "description": "Zephyr name of the target board",
            "type": str,
            "default": None,
            "nullable": True,
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
            "description": "Workspace relative path to West's venv",
            "type": Path,
            "default": Path(".west-venv"),
        },
        "zephyr_base": {
            "description": "Path to the Zephyr base",
            "type": Path,
            "default": None,
            "nullable": True,
        },
        "extra_targets": {
            "description": "Extra targets to be built",
            "type": list[str],
            "default": [],
        },
        "extra_build_args": {
            "description": "Extra build arguments",
            "type": list[str],
            "default": [],
        },
        "use_llext": {
            "description": "Whether LLEXT should be used",
            "type": bool,
            "default": False,
        },
        "run_west_update": {
            "description": "Whether to update the project before build",
            "type": bool,
        },
    }

    allowed_frameworks = [
        "tflite",
        "tvm",
    ]

    def __init__(
        self,
        workspace: Path,
        board: Optional[str] = None,
        output_path: Optional[Path] = None,
        model_framework: Optional[str] = None,
        application_dir: Path = Path("app"),
        build_dir: Path = Path("build"),
        venv_dir: Path = Path(".west-venv"),
        zephyr_base: Optional[Path] = None,
        extra_targets: List[str] = [],
        extra_build_args: List[str] = [],
        use_llext: bool = False,
        run_west_update: bool = False,
    ):
        """
        RuntimeBuilder for the kenning-zephyr-runtime.

        Parameters
        ----------
        workspace : Path
            Location of the project directory.
        board : Optional[str]
            Name of the target board.
        output_path: Optional[Path]
            Destination of the built binaries.
        model_framework : Optional[str]
            Selected model framework
        application_dir : Path
            Path to the project's application directory.
            In the kenning-zephyr-runtime it's always './app'.
        build_dir : Path
            Path to the project's build directory.
        venv_dir : Path
            Virtual environment for West
        zephyr_base : Optional[Path]
            Path to the Zephyr base.
        extra_targets : List[str]
            Extra targets to be built.
        extra_build_args : List[str]
            Extra build arguments.
        use_llext : bool
            Whether LLEXT should be used
        run_west_update : bool
            Whether to update the project before build.

        Raises
        ------
        FileNotFoundError
            If the application directory doesn't exist.
        """
        super().__init__(
            workspace=workspace.resolve(),
            output_path=output_path,
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

        self.extra_targets = extra_targets
        self.extra_build_args = extra_build_args
        self.use_llext = use_llext

        self._westrun = WestRun(self.workspace, zephyr_base, self.venv_dir)

        if not self._westrun.has_zephyr_base():
            self._westrun.init()

        if run_west_update:
            self._westrun.update()
            KLogger.info(f"Updated Zephyr workspace in '{self.workspace}'")

            self._prepare_modules()
            KLogger.info("Prepared modules")

    def build(self) -> Optional[Path]:
        output_files = {}

        if self.board is None:
            KLogger.error("Board must be specified before build")
            return None

        if self.use_llext:
            extra_conf_file = f"llext_{self.model_framework}.conf;llext.conf"

            llext_path = (
                self.build_dir / "llext" / f"{self.model_framework}.llext"
            )

            output_files[str(llext_path)] = "runtime.llext"
        else:
            extra_conf_file = f"{self.model_framework}.conf"

        # add board-specific config if it exists
        board_specific_conf = (
            self.workspace
            / "app"
            / "boards"
            / f"{self.board.replace('/', '_')}.conf"
        )
        if board_specific_conf.exists():
            KLogger.info(f"Adding board-specific config {board_specific_conf}")
            extra_conf_file = (
                f"{extra_conf_file};{board_specific_conf.resolve()}"
            )

        extra_build_args = [
            f"-DMODULE_EXT_ROOT={self.workspace}",
        ]

        if self.model_path is not None:
            extra_build_args.append(
                f'-DCONFIG_KENNING_MODEL_PATH="{self.model_path.resolve()}"'
            )

        extra_build_args.extend(self.extra_build_args)

        if "board-repl" in self.extra_targets:
            board = self.board.split("/")[0]
            output_files[f"{board}.repl"] = f"{board}.repl"
            output_files[f"{board}_flat.dts"] = f"{board}_flat.dts"

        self._westrun.build(
            self.board,
            self.application_dir,
            build_dir=self.build_dir,
            extra_conf_file=extra_conf_file,
            extra_build_args=extra_build_args,
            pristine="auto",
        )

        for target in self.extra_targets:
            self._westrun.build_target(target, self.build_dir)

        runtime_elf = "zephyr/zephyr.elf"
        output_files[runtime_elf] = runtime_elf

        runtime_hex = "zephyr/zephyr.hex"
        output_files[runtime_hex] = runtime_hex

        if self.output_path is not None:
            for src_file, dest_file in output_files.items():
                src_path = self.build_dir / src_file
                dest_path = self.output_path / dest_file

                dest_path.parent.mkdir(exist_ok=True, parents=True)
                dest_path.unlink(missing_ok=True)

                shutil.copy(src_path, dest_path)

                KLogger.info(f"Output file {src_file} copied to {dest_path}.")

            return self.output_path / runtime_elf

        KLogger.info(
            "Zephyr Runtime was build and "
            f"is located in '{Path(runtime_elf).absolute()}'"
        )
        return Path(runtime_elf)

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

    def read_platform(self, platform: Platform):
        if (type(self).__name__ == "ZephyrRuntimeBuilder") and type(
            platform
        ).__name__ in (
            "BareMetalPlatform",
            "ZephyrPlatform",
        ):
            platform.zephyr_build_path = self.output_path
            KLogger.info(
                "Set platform Zephyr build path to "
                f"{platform.zephyr_build_path}"
            )
            self.board = platform.name
            KLogger.info(f"Set runtime builder board to {self.board}")
