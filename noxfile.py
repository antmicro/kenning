"""
Test Kenning on multiple Python versions.
"""
import os
from glob import glob
from pathlib import Path

import nox
from nox_uv import session

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
PYTEST_CPU_ONLY = os.environ.get("NOX_PYTEST_CPU_ONLY", "n") != "n"
PYTEST_EXPLICIT_DOWNLOAD = (
    os.environ.get("NOX_PYTEST_EXPLICIT_DOWNLOAD", "n") != "n"
)

KENNING_DEPS_DIR = Path("kenning-deps").resolve()

nox.options.sessions = ["run_pytest", "run_gallery_tests"]
nox.options.default_venv_backend = "uv"


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

        renode_bin = Path(glob("renode_*-portable/renode")[0]).resolve()

        session.env["PYRENODE_RUNTIME"] = "coreclr"
        session.env["PYRENODE_BIN"] = renode_bin

        session.log(f"Using Renode from: '{renode_bin}'.")


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


@session(python=PYTHON_VERSIONS, uv_sync_locked=False)
@nox.parametrize("device", ["cpu", "any"])
def get_deps(session: nox.Session, device):
    """
    Downloads Kenning dependencies.
    """
    name = _fix_name(session.name)
    deps_path = KENNING_DEPS_DIR / name
    session.run(
        "uv", "tool", "install", ".[all]", env={"UV_TOOL_DIR": deps_path}
    )


@session(python=PYTHON_VERSIONS, uv_all_extras=True, uv_sync_locked=False)
@nox.parametrize("device", ["cpu", "any"])
def run_pytest(session: nox.Session, device):
    """
    Install Kenning with all dependencies and run pytest.
    """
    _prepare_pyrenode(session)

    # Build cython extensions in-place
    session.run("python", "setup.py", "build_ext", "--inplace")

    name = _fix_name(session.name)

    requirements_path = Path("requirements") / f"{name}.txt"
    requirements_path.parent.mkdir(exist_ok=True)
    requirements_path.write_text(
        session.run("uv", "pip", "freeze", silent=True)
    )

    if PYTEST_CPU_ONLY and device != "cpu":
        session.log("Skipping pytest")
        return

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


@session(
    python=PYTHON_VERSIONS,
    uv_extras=["test", "pipeline_manager"],
    uv_sync_locked=False,
)
@nox.parametrize("specification", ["cpu", "gpu", "ros", "all"])
def run_gallery_tests(session: nox.Session, specification):
    """
    Install Kenning with minimal dependencies and run gallery tests.
    """
    name = _fix_name(session.name)

    # unset UV_PYTHON and VIRTUAL_ENV to
    # allow usage of session venv
    envs = {"UV_PYTHON": None, "VIRTUAL_ENV": None}

    pattern_md = (
        "docs/source/gallery/*.md"
        if not session.posargs
        else session.posargs[0]
    )

    marks = "snippets"
    if specification == "cpu":
        marks = "(snippets) and (not gpu)"
    elif specification == "gpu":
        marks = "(snippets) and (gpu) and (not ros)"
    elif specification == "ros":
        marks = "(snippets) and (gpu) and (ros)"
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
        env=envs,
    )


@session(
    python=PYTHON_VERSIONS,
    uv_extras=[
        "test",
        "pipeline_manager",
        "tvm",
        "tensorflow",
        "reports",
        "renode",
    ],
    uv_sync_locked=False,
)
def test_generate_platforms(session: nox.Session):
    """
    Install Kenning with all dependencies and run pytest.
    """
    name = _fix_name(session.name)

    report_path = Path("platform-tests-reports") / f"{name}.json"

    # get generated platforms file path
    platforms = os.getenv("PLATFORMS_PATH")

    session.run(
        "pytest",
        "kenning/tests/scenarios/test_generate_platforms.py",
        "--generated-platforms-path",
        platforms,
        "--timeout=720",
        f"--report-log={report_path}",
    )
