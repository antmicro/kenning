import os
from glob import glob
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.9", "3.10"]
PYTEST_CPU_ONLY = os.environ.get("NOX_PYTEST_CPU_ONLY", "n") != "n"


def prepare_pyrenode(session: nox.Session):
    renode_dir = session.create_tmp()
    with session.chdir(renode_dir):
        session.run_install(
            "wget",
            "https://builds.renode.io/renode-latest.linux-portable-dotnet.tar.gz",
            "-O",
            "renode-latest.linux-portable-dotnet.tar.gz",
            external=True,
        )
        session.run_install(
            "tar",
            "-xf",
            "renode-latest.linux-portable-dotnet.tar.gz",
            external=True,
        )

        renode_bin = Path(glob("renode_*-dotnet_portable/renode")[0]).resolve()

        session.env["PYRENODE_RUNTIME"] = "coreclr"
        session.env["PYRENODE_BIN"] = renode_bin

        session.log(f"Using Renode from: '{renode_bin}'.")


def prepare_kenning(session: nox.Session, device: str):
    optional_dependencies = [
        "docs",
        "tensorflow",
        "torch",
        "mxnet",
        "object_detection",
        "speech_to_text",
        "tflite",
        "tvm",
        "iree",
        "onnxruntime",
        "test",
        "real_time_visualization",
        "pipeline_manager",
        "reports",
        "uart",
        "renode",
        "zephyr",
        "nni",
        "ros2",
        "albumentations",
        "llm",
    ]

    extra_indices = []

    if device == "any":
        optional_dependencies.append("nvidia_perf")

    if device == "cpu":
        extra_indices.append("https://download.pytorch.org/whl/cpu")

    deps_str = ",".join(optional_dependencies)
    indices_strs = [f"--extra-index-url={url}" for url in extra_indices]
    session.install(*indices_strs, f".[{deps_str}]")


def fix_pyximport(session: nox.Session):
    session.run(
        "python",
        "-c",
        """
import numpy as np
import pyximport;
pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True
);
from kenning.modelwrappers.instance_segmentation.cython_nms import (
    nms,
);
""",
    )


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("device", ["cpu", "any"])
def run_pytest(session: nox.Session, device):
    prepare_kenning(session, device)
    prepare_pyrenode(session)

    if PYTEST_CPU_ONLY and device != "cpu":
        session.log("Skipping pytest")
        return

    fix_pyximport(session)

    report_path = Path("pytest-reports") / f"{session.name}.json"
    session.run(
        "pytest",
        "kenning",
        "--ignore=kenning/tests/core/test_model.py",
        "--ignore=kenning/tests/utils/test_class_loader.py",
        "-n=auto",
        "-m",
        "(not docs_gallery) and (not docs)",
        f"--report-log={report_path}",
    )


@nox.session(python=PYTHON_VERSIONS)
def run_gallery_tests(session: nox.Session):
    session.install(".[test,pipeline_manager]")

    report_path = Path("pytest-reports") / f"{session.name}.json"
    session.run(
        "pytest",
        "kenning/tests/docs/test_snippets.py",
        "--capture=fd",
        "-n=4",
        "-m",
        "docs_gallery",
        f"--report-log={report_path}",
    )
