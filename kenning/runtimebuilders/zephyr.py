import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class ZephyrRuntimeBuilder(RuntimeBuilder):
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
        runtime_location: PathOrURI,
        board: str,
        model_framework: Optional[str] = None,
        application_dir: Path = Path("./app"),
        build_dir: Path = Path("./build"),
        zephyr_base: Optional[PathOrURI] = None,
    ):
        self.board = board
        self.workspace = workspace.resolve()

        self.application_dir = self._fix_relative(workspace, application_dir)
        if not self.application_dir.exists():
            raise FileNotFoundError(
                f'Application source directory "{self.application_dir}" doesn\'t exist.'
            )

        self.build_dir = self._fix_relative(workspace, build_dir)
        self.build_dir.mkdir(exist_ok=True)

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

        self._prepare_modules()
        KLogger.info("Prepared modules")

        super().__init__(
            workspace=workspace,
            runtime_location=runtime_location,
            model_framework=model_framework,
        )

    def build(self):
        self._build_project(f"{self.model_framework}.conf")
        self.runtime_location.unlink(missing_ok=True)
        self.runtime_location.symlink_to(self.build_dir / "zephyr/zephyr.elf")
        KLogger.info("Built runtime")

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
                ["west", "init", "-l", str(self.workspace)],
                **self._subprocess_cfg,
            ).check_returncode()
            # subprocess.run(["west", "zephyr-export"]).check_returncode()
        except subprocess.CalledProcessError as e:
            raise Exception("west init failed.") from e

        return self.workspace.parent / "zephyr"

    def _update_zephyr(self):
        try:
            subprocess.run(
                ["west", "update"], **self._subprocess_cfg
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
            "west",
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
            raise Exception(
                f'Zephyr build failed. Try removing "{self.build_dir}" and try again.'
            ) from e

    @property
    def _subprocess_cfg(self):
        stream = None if KLogger.level <= logging.DEBUG else subprocess.DEVNULL
        return {
            "stdout": stream,
            "stderr": stream,
        }
