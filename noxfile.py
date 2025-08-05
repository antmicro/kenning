"""
Test Kenning on multiple Python versions.
"""
import os
from glob import glob
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
PYTEST_CPU_ONLY = os.environ.get("NOX_PYTEST_CPU_ONLY", "n") != "n"
PYTEST_EXPLICIT_DOWNLOAD = (
    os.environ.get("NOX_PYTEST_EXPLICIT_DOWNLOAD", "n") != "n"
)

KENNING_DEPS_DIR = Path("kenning-deps").resolve()

nox.options.sessions = ["run_pytest", "run_gallery_tests"]


def _prepare_pyrenode(session: nox.Session):
    """
    Installs Renode for pyrenode3.
    """
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


def _prepare_pip_params(session: nox.Session, device: str):
    """
    Installs Kenning with all dependencies.
    """
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
        "anomaly_detection",
        "auto_pytorch",
    ]

    extra_indices = []

    if device == "any":
        optional_dependencies.append("nvidia_perf")

    if device == "cpu":
        extra_indices.append("https://download.pytorch.org/whl/cpu")

    deps_str = ",".join(optional_dependencies)
    indices_strs = [f"--extra-index-url={url}" for url in extra_indices]
    return [*indices_strs, f".[{deps_str}]"]


def _fix_pyximport(session: nox.Session):
    """
    Fixes pyximport related crashes by initializing `$HOME/.pyxbld`.
    """
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


def _fix_name(name):
    """
    Converts concrete session name into a suitable filename. For example,
    `run_pytest-3.10(device='cpu')` is converted into `run_pytest-3.10-cpu`.
    """
    namever, _, args = name.partition("(")
    name, _, ver = namever.partition("-")
    args = args.rstrip(")")

    params = []
    params.append(name)
    if ver:
        params.append(ver)

    if args:
        for arg in args.split(","):
            arg = arg.strip()
            _, v = arg.split("=")
            v = v.strip("'\"")
            params.append(v)

    return "-".join(params)


def _prepare_kenning(session: nox.Session, device):
    if not PYTEST_EXPLICIT_DOWNLOAD:
        pip_params = _prepare_pip_params(session, device)
        session.install(*pip_params)
        return

    for path in KENNING_DEPS_DIR.glob("*"):
        path = Path(path)
        name, ver, *params = path.name.split("-")

        if session.python == ver and device in params:
            wheels = path.glob("*")
            deps = []
            for dep in wheels:
                # TODO pip >= 24 does not allow ".tip" suffix present in
                # sphinx_immaterial
                if (
                    "sphinx_immaterial-0.0.post1.tip-py3-none-any.whl"
                    == dep.name
                ):
                    newpath = (
                        dep.parent
                        / "sphinx_immaterial-0.0.post1-py3-none-any.whl"
                    )
                    dep.rename(newpath)
                    dep = newpath
                deps.append(dep)
            session.install("--no-deps", ".", *deps)
            return


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("device", ["cpu", "any"])
def get_deps(session: nox.Session, device):
    """
    Downloads Kenning dependencies.
    """
    pip_params = _prepare_pip_params(session, device)
    name = _fix_name(session.name)
    deps_path = KENNING_DEPS_DIR / name
    session.run("pip", "download", f"--dest={deps_path}", *pip_params)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("device", ["cpu", "any"])
def run_pytest(session: nox.Session, device):
    """
    Install Kenning with all dependencies and run pytest.
    """
    _prepare_kenning(session, device)
    _prepare_pyrenode(session)

    name = _fix_name(session.name)

    requirements_path = Path("requirements") / f"{name}.txt"
    requirements_path.parent.mkdir(exist_ok=True)
    requirements_path.write_text(session.run("pip", "freeze", silent=True))

    if PYTEST_CPU_ONLY and device != "cpu":
        session.log("Skipping pytest")
        return

    _fix_pyximport(session)

    report_path = Path("pytest-reports") / f"{name}.json"

    session.run(
        "pytest",
        "kenning",
        "--ignore=kenning/tests/utils/test_class_loader.py",
        "-n=auto",
        "--cov=kenning",
        "--cov-report=html",
        "--timeout=720",
        "-m",
        "(not snippets) and (not gpu) and (not automl) and (not compat_matrix)",  # noqa: E501
        f"--report-log={report_path}",
    )


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("specification", ["cpu", "gpu", "all"])
def run_gallery_tests(session: nox.Session, specification):
    """
    Install Kenning with minimal dependencies and run gallery tests.
    """
    session.install(".[test,pipeline_manager]")

    name = _fix_name(session.name)

    pattern_md = (
        "docs/source/gallery/*.md"
        if not session.posargs
        else session.posargs[0]
    )

    marks = "snippets"
    if specification == "cpu":
        marks = "(snippets) and (not gpu)"
    elif specification == "gpu":
        marks = "(snippets) and (gpu)"
    elif specification == "all":
        marks = "snippets"

    report_path = Path("pytest-reports") / f"{name}.json"
    test_docs_log_dir = Path("log_docs") / f"{name}"
    test_docs_log_dir.mkdir(parents=True)
    session.run(
        "pytest",
        "kenning/tests/docs/test_snippets.py",
        "--input-file-pattern",
        pattern_md,
        "-m",
        marks,
        "--save-tmp-pattern",
        "_autoPyTorch_tmp",
        "--capture=fd",
        "-n=4",
        f"--report-log={report_path}",
        f"--test-docs-log-dir={test_docs_log_dir}",
    )
