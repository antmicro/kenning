# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from kenning.utils.logger import KLogger

GROUND_TRUTH_PATH = Path(__file__).with_name("platforms_ground_truth.yml")


def run_sys(cmd: List[str]):
    KLogger.info(f"\n=== RUN: {cmd}")
    subprocess.run(cmd, check=True)


@pytest.fixture(scope="session")
def ground_truth_platforms() -> Dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        pytest.xfail(f"Missing ground truth file: {GROUND_TRUTH_PATH}")
    with GROUND_TRUTH_PATH.open("r") as fd:
        return yaml.safe_load(fd) or {}


@pytest.fixture(scope="session")
def generated_platforms(request) -> Dict[str, Any]:
    val = request.config.getoption("--generated-platforms-path", skip=True)
    path = Path(val)
    if not path.exists():
        pytest.fail(f"--platforms-path path does not exist: {path}")
    with path.open("r") as fd:
        return yaml.safe_load(fd) or {}


@pytest.fixture(scope="session")
def zephyr_workspace():
    workspace = Path("zephyr-workspace")
    workspace.mkdir(exist_ok=True)
    return workspace


@pytest.fixture(scope="session")
def kenning_zephyr_runtime_repo(zephyr_workspace: Path):
    repo_dir = "kenning-zephyr-runtime"
    if not (zephyr_workspace / repo_dir).exists():
        run_sys(
            [
                "git",
                "clone",
                "https://github.com/antmicro/kenning-zephyr-runtime.git",
                str(zephyr_workspace / repo_dir),
            ]
        )
    return zephyr_workspace / repo_dir


@pytest.fixture()
def zephyr_simple_scenarios(
    request, kenning_zephyr_runtime_repo, ground_truth_platforms, tmp_path
):
    val = request.config.getoption("--generated-platforms-path", skip=True)
    scenario_schema = yaml.safe_load(
        f"""
platform:
  type: ZephyrPlatform
  parameters:
    name: FIELD_TO_BE_SET
    simulated: true
    platforms_definitions: [{Path(val)}]
    zephyr_build_path: ./build
dataset:
  type: MagicWandDataset
  parameters:
    dataset_root: ./output/MagicWandDataset
optimizers:
- type: TFLiteCompiler
  parameters:
    compiled_model_path: ./output/magic-wand.tflite
    inference_input_type: float32
    inference_output_type: float32
"""
    )

    fetch_model_commands = [
        f"cd {kenning_zephyr_runtime_repo}"
        "wget 'https://dl.antmicro.com/kenning/models/classification/magic_wand.h5.json'",
        "wget 'https://dl.antmicro.com/kenning/models/classification/magic_wand.tflite'",
        "mkdir -p output",
        "ls",
        "mv magic_wand.h5.json output/magic_wand.tflite.json",
        "mv magic_wand.tflite output/magic_wand.tflite",
    ]

    run_sys(["bash", "-lc", ";".join(fetch_model_commands)])

    def make_scenario(name):
        scenario = deepcopy(scenario_schema)
        scenario["platform"]["parameters"]["name"] = name
        return scenario

    SCENARIOS = []
    for i, name in enumerate(ground_truth_platforms):
        scenerio_path = tmp_path / str(i)
        with open(scenerio_path, "w") as fd:
            yaml.safe_dump(make_scenario(name), fd)
        SCENARIOS.append((name, scenerio_path))

    return SCENARIOS


@pytest.fixture(scope="session")
def zephyr_env(request, kenning_zephyr_runtime_repo: Path) -> Dict[str, Any]:
    """
    Prepared environment for kenning zephyr runtime scenarios.
    Fetches repository, runs scripts, sets python env.

    Returns Dict with set environment.
    """
    request.config.getoption("--generated-platforms-path", skip=True)

    snapshot_name = ".env_snapshot"
    snapshot_file = kenning_zephyr_runtime_repo / snapshot_name
    get_env_commends = [
        "set -euo pipefail",
        f"cd {kenning_zephyr_runtime_repo}",
        "./scripts/prepare_zephyr_env.sh",
        "source .venv/bin/activate",
        "./scripts/prepare_modules.sh",
        "source ./scripts/prepare_renode.sh",
        f"env > {snapshot_name}",
        f"echo '=== ENV SNAPSHOT wrote to {snapshot_name}'",
    ]

    run_sys(["bash", "-lc", ";".join(get_env_commends)])

    if not snapshot_file.exists():
        raise RuntimeError(f"Missing env snapshot file: {snapshot_file}")

    env_snapshot = {}
    for entry in snapshot_file.read_text().split("\n"):
        if not entry:
            continue
        key, _, val = entry.partition("=")
        if not key or key == "_":
            continue
        env_snapshot[key] = val

    return env_snapshot


class TestGeneratedPlatformsSpecs:
    def includes(tested_dict: Dict, testing_dict: Dict) -> Tuple[bool, List]:
        """
        Determines if tested Dict constrains testing Dict.

        The inclusion is in the sense that for  "x" to contain "y"
        then "x" must have all the paths from the root to the leafs
        that are in "y".

        Function returns at the first met mismatch.

        Parameters
        ----------
        tested_dict : Dict[str, Any]
            Dictionary which is tested to contain "testing_dict"
        testing_dict : Dict[str, Any]
            Dictionary with which "tested_dict" is tested

        Returns
        -------
        Tuple[bool, List]
            Pair of a success status and a trace
            to missmaching fild in "tested_dir"
        """

        def _includes(
            tested_val, testing_val, _trace_acc=[]
        ) -> Tuple[bool, List]:
            if not isinstance(tested_val, type(testing_val)):
                return False, _trace_acc
            elif isinstance(tested_val, List):
                for val in testing_val:
                    try:
                        idx = testing_val.index(val)
                        testing_val.pop(idx)
                    except ValueError:
                        return False, _trace_acc
                return True, []
            elif isinstance(tested_val, Dict):
                if not set(testing_val.keys()).issubset(
                    set(tested_val.keys())
                ):
                    return False, _trace_acc
                for key in testing_val:
                    success, trace = _includes(
                        tested_val[key], testing_val[key], _trace_acc + [key]
                    )
                    if not success:
                        return False, trace
                return True, []
            else:
                if tested_val == testing_val:
                    return True, []
                return False, _trace_acc

        return _includes(tested_dict, testing_dict)

    def test_generated_platforms_values(
        self, generated_platforms, ground_truth_platforms
    ):
        res, trace = TestGeneratedPlatformsSpecs.includes(
            generated_platforms, ground_truth_platforms
        )
        assert res, f"platforms spec mismatch at: {trace}"


class TestGeneratedPlatformsScenarios:
    def test_inference_starting(
        self, zephyr_simple_scenarios, zephyr_env, tmpfolder
    ):
        """
        Tests whether inference starts for generated platform specification.
        """
        for scenario, name in zephyr_simple_scenarios:
            # niesprawdzone
            res = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "west",
                    "build",
                    "--board",
                    name,
                    "app",
                    "--",
                    "-DEXTRA_CONF_FILE=tflite.conf",
                ],
                env=zephyr_env,
            )
            print(res)
            print()
            print(res.args)
            print(res.returncode)
            print()
            res = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "kenning",
                    "test",
                    "--cfg",
                    scenario,
                    "--measurements",
                    tmpfolder / f"measurements_{name}",
                ],
                env=zephyr_env,
            )

            print(res)
            print()
            print(res.args)
            print(res.returncode)
            print()
