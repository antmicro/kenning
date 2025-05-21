# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with utilities for platforms definition generation,
based on the Zephyr DTS.
"""

import os
import re
import subprocess
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path, PosixPath
from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict, Generator, List, Optional

from dts2repl.dtlib import DTError, Node
from dts2repl.dts2repl import (
    get_dt,
    get_node_prop,
    get_reg,
    parse_phandles_and_nums,
)

from kenning import resources
from kenning.utils.logger import KLogger

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

ZEPHYR_BASE = os.environ.get("ZEPHYR_BASE", None)
if ZEPHYR_BASE is None:
    raise Exception("`ZEPHYR_BASE` env variable is not defined")
ZEPHYR_BASE = Path(ZEPHYR_BASE)
if not ZEPHYR_BASE.exists():
    raise Exception(
        f"`ZEPHYR_BASE` ({ZEPHYR_BASE}) points to non-existing directory"
    )

sys.path.insert(0, str((ZEPHYR_BASE / "scripts").absolute()))
sys.path.insert(0, str((ZEPHYR_BASE / "doc" / "_scripts").absolute()))

# Python modules from Zephyr, require additional PYTHONPATH
import list_boards  # noqa: E402
import zephyr_module  # noqa: E402

ZEPHYR_URL = "https://raw.githubusercontent.com/zephyrproject-rtos/zephyr/refs/heads/main/"

# Standard directories for Zephyr SDK
POSSIBLE_ZEPHYR_SDK_LOCATIONS = [
    "~",
    "~/.local",
    "~/.local/opt",
    "~/bin",
    "/opt",
    "/usr/local",
]

# Overrides for ARM cpu
ARM_CPU_OVERRIDES = {
    "cortex-m4f": "cortex-m4",
    "cortex-m0+": "cortex-m0plus",
    "cortex-m33f": "cortex-m33",
    "cortex-r8f": "cortex-r8",
    "cortex-r5f": "cortex-r5",
}

# Zephyr memory compatible entries
SRAM_COMP = ["mmio-sram"]
DTCM_COMP = ["arm,dtcm", "nxp,imx-dtcm"]
CCM_COMP = ["nordic,nrf-ccm", "st,stm32-ccm", "arc,dccm"]
OCM_COMP = ["xlnx,zynq-ocm"]


def get_image_path(board_dir: Path, board_name: str) -> Optional[str]:
    """
    Returns a path to the image.

    Parameters
    ----------
    board_dir : Path
        Directory with the board (in the Zephyr repo).
    board_name : str
        Name of the board.

    Returns
    -------
    Optional[str]
        URL address to the image.
    """
    for root, dirs, files in os.walk(board_dir, topdown=False):
        for name in files:
            if re.match(rf".*?({board_name})?.*?\.(jpg|jpeg|webp|png)", name):
                image_path = Path(root) / name
                return ZEPHYR_URL + str(
                    PosixPath(image_path.relative_to(ZEPHYR_BASE))
                )
    return None


def get_zephyr_sdk() -> Optional[Path]:
    """
    Finds the location of Zephyr SDK.

    Returns
    -------
    Optional[Path]
        Location of found Zephyr SDK or None.
    """
    full_sdk_path = os.environ.get("ZEPHYR_SDK_INSTALL_DIR", None)
    if full_sdk_path is not None:
        return Path(full_sdk_path)
    zephyr_sdk = "zephyr-sdk*"
    for possible_location in POSSIBLE_ZEPHYR_SDK_LOCATIONS:
        path = next(
            Path(possible_location).expanduser().glob(zephyr_sdk), None
        )
        if path is not None:
            full_sdk_path = path
            break
    return full_sdk_path


def get_arm_arch(
    arm_gcc: Path, aarch64_gcc: Path, cpu_name: str
) -> Optional[str]:
    """
    Gets architecture based on the cpu name.

    Parameters
    ----------
    arm_gcc : Path
        Path to the arm-zephyr-eabi-gcc.
    aarch64_gcc : Path
        Path to the aarch64-zephyr-elf-gcc.
    cpu_name : str
        Name of the cpu.

    Returns
    -------
    Optional[str]
        Found architecture or None.
    """
    gcc_args = [
        f"-mcpu={ARM_CPU_OVERRIDES.get(cpu_name, cpu_name)}",
        "-Q",
        "--help=target",
    ]
    try:
        output = subprocess.run(
            [str(arm_gcc), *gcc_args],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        try:
            output = subprocess.run(
                [str(aarch64_gcc), *gcc_args],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            KLogger.warn(f"Cannot find arch for cpu {cpu_name}")
            return None
    match = re.search(rb"-march=\s*(\S+)", output.stdout)
    if match is None:
        return None
    return match.group(1).decode()


def can_be_ram(node: Node) -> bool:
    """
    Detect whether DTS node can be a RAM.

    Parameters
    ----------
    node : Node
        Node of DTS.

    Return
    ------
    bool
        Whether node can be RAM.
    """
    if not node.name.startswith("memory@"):
        return False

    compatible = list(get_node_prop(node, "compatible", default=[]))
    if "zephyr,memory-region" not in compatible or not any(
        comp in compatible
        for comp in SRAM_COMP + DTCM_COMP + CCM_COMP + OCM_COMP
    ):
        return False
    return True


def run_one_cmake(board_target: str, source_dir: Path, out_dir: Path):
    """
    Runs CMake script generating flat DTS for the board.

    Parameters
    ----------
    board_target : str
        Zephyr target specifying board, its version and revision.
    source_dir : Path
        Directory with defined CMakeList.
    out_dir : Path
        Directory for output files.
    """
    try:
        subprocess.run(
            [
                "cmake",
                f"-DBOARD={board_target}",
                f"-DBOARD_ROOT={str(ZEPHYR_BASE)}",
                "-B",
                str(
                    out_dir
                    / board_target.replace("/", "_")
                    .replace("@", "_")
                    .replace(".", "_")
                ),
                str(source_dir),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as er:
        KLogger.warn(f"Error building flat DTS: {er.stderr}")


def get_all_variants(
    var: Optional[list_boards.Variant]
) -> Generator[str, None, None]:
    """
    Recursively generates all variants of a board.

    Parameters
    ----------
    var : Optional[list_boards.Variant]
        Variant of a board, can contain another variants.

    Yields
    ------
    str
        Combined variant of a board.
    """
    if var is None:
        yield ""
        return
    for v in var.variants + [None]:
        if v:
            for subvar in get_all_variants(v):
                yield var.name + (f"/{subvar}" if subvar else "")
        else:
            yield var.name + ""


def generate_flat_dts(
    boards: List[list_boards.Board], tmpdir: Path
) -> Dict[str, List[str]]:
    """
    Generates flat DTS for Zephyr boards.

    Parameters
    ----------
    boards : List[list_boards.Board]
        List containing all Zephyr boards.
    tmpdir : Path
        Directory where generated files will be stored.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping board names to the available board targets.
    """
    board_devicetrees = {}
    with path(resources, "zephyr_dts") as p:
        zephyr_dts = Path(p)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    done_boards = set()
    for board in boards.values():
        for revision, soc in product(
            board.revisions if board.revisions else [None],
            board.socs,
        ):
            for cpu_cluster, variant in product(
                soc.cpuclusters if soc.cpuclusters else [None],
                soc.variants + [None],
            ):
                for cluster_var in (
                    cpu_cluster.variants if cpu_cluster else []
                ) + [None]:
                    if variant is not None and cluster_var is not None:
                        KLogger.error(
                            "Two variants from CPU cluster and variants "
                            f"received for {board.name}"
                        )
                    var = variant or cluster_var

                    for combined_var in get_all_variants(var):
                        rev_str = ""
                        if revision and revision.name:
                            rev_str = f"@{revision.name}"
                        parts = [
                            f"{board.name}{rev_str}",
                            soc.name,
                        ]
                        if cpu_cluster:
                            parts.append(cpu_cluster.name)
                        if combined_var:
                            parts.append(combined_var)

                        board_target = "/".join(parts)
                        if board_target in done_boards:
                            continue
                        done_boards.add(board_target)
                        if board.name not in board_devicetrees:
                            board_devicetrees[board.name] = []
                        board_devicetrees[board.name].append(board_target)
                        executor.submit(
                            run_one_cmake, board_target, zephyr_dts, tmpdir
                        )
    executor.shutdown()
    return board_devicetrees


def get_platforms_definitions() -> Dict:
    """
    Generates platform definitions based on Zephyr DTS.

    Returns
    -------
    Dict
        Platform definitions.
    """
    module_settings = {
        "arch_root": [ZEPHYR_BASE],
        "board_root": [ZEPHYR_BASE],
        "soc_root": [ZEPHYR_BASE],
    }

    for module in zephyr_module.parse_modules(ZEPHYR_BASE):
        for key in module_settings:
            root = module.meta.get("build", {}).get("settings", {}).get(key)
            if root is not None:
                module_settings[key].append(Path(module.project) / root)

    Args = namedtuple(
        "args",
        ["arch_roots", "board_roots", "soc_roots", "board_dir", "board"],
    )
    args_find_boards = Args(
        arch_roots=module_settings["arch_root"],
        board_roots=module_settings["board_root"],
        soc_roots=module_settings["soc_root"],
        board_dir=[],
        board=None,
    )

    boards = list_boards.find_v2_boards(args_find_boards)
    board_catalog = {}

    zephyr_sdk = get_zephyr_sdk()
    arm_gcc = None
    if zephyr_sdk:
        arm_gcc = (
            zephyr_sdk / "arm-zephyr-eabi" / "bin" / "arm-zephyr-eabi-gcc"
        )
        aarch64_gcc = (
            zephyr_sdk
            / "aarch64-zephyr-elf"
            / "bin"
            / "aarch64-zephyr-elf-gcc"
        )
    if not zephyr_sdk or not arm_gcc.exists() or not aarch64_gcc.exists():
        KLogger.warn(
            "Cannot find path to ARM gcc, "
            "please make sure the Zephyr SDK is installed correctly. "
            "Compilation flags will not be generated"
        )

    KLogger.info("Running script creating flat DTS for all Zephyr boards")
    tmpdir = Path(mkdtemp())
    board_devicetrees = generate_flat_dts(boards, tmpdir)

    for board in boards.values():
        if board.name not in board_devicetrees:
            KLogger.error(f"Cannot find DTS for {board.name}")
            continue

        full_name = board.full_name or board.name

        # Use pre-gathered build info and DTS files
        for board_target in board_devicetrees[board.name]:
            dts_path = next(
                (
                    tmpdir
                    / board_target.replace("/", "_")
                    .replace("@", "_")
                    .replace(".", "_")
                ).glob("**/zephyr.dts.pre"),
                None,
            )
            if dts_path is None:
                KLogger.error(f"DTS file for {board_target} not found")
                continue
            dt = get_dt(dts_path)
            if board_target in board_catalog:
                KLogger.error(
                    f"{board_target} already exist " "in the catalog, skipping"
                )
                continue

            # Try to find baudrate
            baudrate = None
            chosen = dt.get_node("/chosen")
            try:
                zephyr_console = get_node_prop(chosen, "zephyr,console")
                baudrate = parse_phandles_and_nums(
                    dt, zephyr_console, "current-speed"
                )[0]
            except AttributeError:
                KLogger.warn(
                    f"target {board_target} does not define zephyr console"
                )
            except KeyError:
                KLogger.warn(
                    f"target {board_target} does not contain baudrate"
                )

            # Try to find compilation flags for ARM cpus
            compilation_flags = None
            if arm_gcc:
                compatibles = []
                try:
                    cpus = dt.get_node("/cpus")
                    for cpu in cpus.node_iter():
                        if (
                            get_node_prop(  # is cpu
                                cpu, "device_type", [None]
                            )[0]
                            == "cpu"
                            and get_node_prop(  # is not disabled
                                cpu, "status", None
                            )
                            != "disabled"
                        ):
                            compatibles.extend(
                                get_node_prop(cpu, "compatible")
                            )
                except (AttributeError, KeyError, DTError):
                    KLogger.warn(f"Cannot find CPU for {board_target}")
                # Filter out non ARM cpus
                compatibles = [
                    c[4:] for c in set(compatibles) if c.startswith("arm,")
                ]
                # Get architectures
                archs = [
                    (a, c)
                    for a, c in [
                        (get_arm_arch(arm_gcc, aarch64_gcc, c), c)
                        for c in compatibles
                    ]
                    if a is not None
                ]
                # Prepare compilation flags
                if archs:
                    compilation_flags = [
                        f"-march={archs[0][0]}",
                        f"-mcpu={archs[0][1]}",
                        "-keys=arm_cpu,cpu",
                        "-device=arm_cpu",
                        f"-model={board.socs[0].name}",
                    ]
                elif compatibles:
                    KLogger.warn(
                        f"Cannot find arch for {board_target} "
                        f"cpus ({compatibles})"
                    )

            # Get size of the memory, based on sram and memory-regions
            ram_nodes = [n for n in dt.node_iter() if can_be_ram(n)]
            ram = 0
            try:
                sram = get_node_prop(chosen, "zephyr,sram")
                if sram not in ram_nodes:
                    ram += next(get_reg(sram))[1] // 1024
            except (AttributeError, KeyError, TypeError):
                pass
            for node in ram_nodes:
                try:
                    ram += next(get_reg(node))[1] // 1024
                except (AttributeError, KeyError, TypeError):
                    pass

            # Get size of the flash
            flash = None
            try:
                flash_reg = get_node_prop(chosen, "zephyr,flash")
                flash = next(get_reg(flash_reg))[1] // 1024
            except (AttributeError, KeyError, TypeError):
                pass
            if ram is None and flash is None:
                KLogger.warning(
                    f"Cannot retrieve flash ({flash}) "
                    f"and ram ({ram}) for {board_target}"
                )

            # Prepare board suffix based on revision and version
            suffix = []
            if board.revisions:
                for r in board.revisions:
                    if f"@{r.name}" in board_target:
                        suffix.append(f"rev. {r.name}")
            if len(board_devicetrees[board.name]) > 1:
                if "/" in board_target:
                    suffix.append(board_target.split("/", 1)[1])
                else:
                    suffix.append(board_target)

            image = get_image_path(board.dir, board.name)

            board_catalog[board_target] = {
                k: v
                for k, v in (
                    (
                        "display_name",
                        full_name
                        + (f" ({', '.join(suffix)})" if suffix else ""),
                    ),
                    ("uart_log_baudrate", baudrate),
                    ("ram_size_kb", ram),
                    ("flash_size_kb", flash),
                    ("compilation_flags", compilation_flags),
                    ("image", image),
                    ("default_platform", "ZephyrPlatform"),
                    ("default_optimizer", ["TFLiteCompiler", "TVMCompiler"]),
                )
                if v is not None
            }

    # Cleanup the temporary directory
    rmtree(tmpdir)

    return board_catalog
