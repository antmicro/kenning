# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
import venv
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import pexpect
import pytest
from filelock import FileLock
from tuttest import Snippet, get_snippets

# Regexes for changing Kenning installation to local version
KENNING_LINK_RE = (
    # Optional group with Kenning module name, finished with @
    r"("
    r"kenning(\[[^\]]*\])?"
    r"\s*@\s+"
    r")?"
    # Git and Http prefix
    r"git\+https"
    # Kenning included in the repository URL
    r".*kenning.*\.git"
)
PIP_INSTALL_RE = r"pip3? (?:.* )?install (?:.*?)?"
# Regex for detecting Kenning installation from Git
PIP_INSTALL_KENNING_RE = PIP_INSTALL_RE + KENNING_LINK_RE
# Regex for detecting Kenning installation from current folder
PIP_INSTALL_KENNING_LOCAL_RE = (
    # The main pip installation with flags
    rf"({PIP_INSTALL_RE})"
    # The current dir (represented by the dot,
    # cannot be followed by subdirectories), optionally prepended with quote
    r"(?:['\"]*\.(?='|\"|\[|$))"
    # Kenning optional dependencies, all finished with optional quote
    r"(\[?[^\]]*\])?['\"]?"
)

# List of snippet executables types
EXECUTABLE_TYPES = ("bash",)
# Kenning repo root path
KENNING_ROOT_PATH = "./kenning"
# Pip install lockfile
PIP_LOCK_FILE = Path(os.getcwd()).resolve() / ".PIP_LOCK"
# Mapping of markdown files to separate working directories
WORKING_DIRS: Dict[str, Path] = {}
# Mapping of markdown files to subshells
SHELLS: Dict[str, Dict[int, pexpect.spawn]] = defaultdict(dict)
# Template for command checking if previous process was successful
COMMAND_CHECK = "if [[ $? == 0 ]] ; then echo '{}'; else echo '{}'; fi"
# Template of regex detecting status of previous command
EXPECT_RE = "(?<!'){}(?!')"
# Regex for splitting multiline scripts
NEW_LINE_RE = re.compile("(?<!\\\\)\n", flags=re.MULTILINE)
# Default timeout of command execution
DEFAULT_TIMEOUT = 60 * 120  # 120 min
# Directory with datasets (relative to Kenning)
DATASET_DIR = "build"
# Possible arguments for snippet
SNIPPET_ARGUMENTS = ("test-skip", "timeout", "name", "terminal", "save-as")
# Key of the snippet's positional arguments
SNIPPET_POSITIONAL_ARG = "__arg"
# Key of the snippet's metadata containing markdown directory
SNIPPET_MARKDOWN_PARENT = "__markdown_dir"


def extract_snippet_args(snippet: Snippet):
    """
    Extracts all arguments specified after the language definition
    and save them to metadata.

    If language is wrapped in curly brackets, first argument will be stored
    in separate argument `SNIPPET_POSITIONAL_ARG`.

    Parameters
    ----------
    snippet : Snippet
        Snippet object
    """
    args = snippet.lang.split(" ", 1)
    snippet.lang = args[0]
    args = shlex.split(args[1]) if len(args) == 2 else []
    for i, arg in enumerate(args):
        if "=" in arg:
            arg = arg.split("=", 1)
        else:
            arg = (arg, True)
        if arg[0] in SNIPPET_ARGUMENTS:
            snippet.meta[arg[0]] = arg[1]
        elif (
            i == 0
            and snippet.lang.startswith("{")
            and snippet.lang.endswith("}")
        ):
            snippet.meta[SNIPPET_POSITIONAL_ARG] = arg[0]
        else:
            raise KeyError(f"Snippet cannot have {arg[0]} argument")


def find_string_in_file(filename: Path, text: str):
    with open(filename) as f:
        if text in f.read():
            return True
    return False


def get_all_snippets(
    markdown_pattern: str,
) -> Generator[Tuple[str, str, Snippet], None, None]:
    """
    Finds all executable snippets from gallery of examples
    and dumps named JSON snippets to files.

    Parameters
    ----------
    markdown_pattern : str
        Pattern with markdowns, has to be supported by `glob`.

    Yields
    ------
    str :
        Name of the markdown file with snippets.
    str :
        Name of the found snippet.
    Snippet :
        Found snippet.
    """
    for markdown in glob(markdown_pattern):
        markdown = Path(markdown)

        python_snippet = None
        last_snippet_name = None
        for name, snippet in get_snippets(str(markdown)).items():
            snippet.meta["depends"] = []
            snippet.meta["terminal"] = int(snippet.meta.get("terminal", 0))

            # Parse args from language
            extract_snippet_args(snippet)
            if "save-as" in snippet.meta:
                if snippet.lang == "{literalinclude}":
                    # Take path to the markdown directory
                    # starting with docs/source/...
                    from_docs = []
                    for p in markdown.resolve().parents:
                        from_docs.append(p.name)
                        if p.name == "docs":
                            break
                    snippet.meta[SNIPPET_MARKDOWN_PARENT] = Path(
                        *from_docs[::-1]
                    )
                if last_snippet_name:
                    snippet.meta["depends"].append(last_snippet_name)
                last_snippet_name = name
                yield markdown.with_suffix("").name, name, snippet
            if snippet.lang not in EXECUTABLE_TYPES + ("python",):
                continue

            # Snippet should not be executed
            if snippet.meta.get("test-skip", False):
                continue

            # Append values to snippet's content
            if "append_before" in snippet.meta:
                snippet.text = (
                    f"{snippet.meta['append_before']} {snippet.text}"
                )
            if "append_after" in snippet.meta:
                snippet.text = f"{snippet.text} {snippet.meta['append_after']}"

            # Snippet is executable -- yield it
            if snippet.lang in EXECUTABLE_TYPES:
                # Split multiline snippet
                for id, line in enumerate(NEW_LINE_RE.split(snippet.text)):
                    # Skip empty and commented lines
                    if line.strip() == "" or line.lstrip().startswith("#"):
                        continue
                    line_snippet = copy.deepcopy(snippet)
                    line_snippet.text = line
                    # Set previous snippet as dependency
                    if last_snippet_name:
                        line_snippet.meta["depends"].append(last_snippet_name)
                    last_snippet_name = f"{name}_{id}"
                    yield markdown.stem, last_snippet_name, line_snippet
            # Python snippet -- combine and yield at the end of function
            elif snippet.lang == "python":
                if python_snippet:
                    python_snippet.text += snippet.text + "\n"
                else:
                    python_snippet = snippet
                    python_snippet.text += "\n"

        # Yield combined python snippets
        if python_snippet:
            yield markdown.stem, name, python_snippet


def execute_script_and_wait(
    shell: pexpect.spawn,
    script: str,
    timeout: Optional[float] = None,
) -> bool:
    """
    Wrapper that sends commands to subshell and waits for them to end.

    It also make sure that process is killed when timeout happens.

    Parameters
    ----------
    shell : pexpect.spawn
        Interface for communication with subshell.
    script : str
        Script that should be executed.
    timeout : Optional[float]
        How long process can run.

    Returns
    -------
    bool
        True if the script ended successfully
    """
    success, failure = uuid.uuid4(), uuid.uuid4()
    check_cmd = COMMAND_CHECK.format(success, failure)
    expect_list = [
        re.compile(EXPECT_RE.format(success)),
        re.compile(EXPECT_RE.format(failure)),
    ]
    try:
        script = script.split("\n", 1)
        content = None
        if not script[0].endswith("\\") and len(script) > 1:
            content = script[1]
            script = script[0]
        else:
            script = "\n".join(script)

        if not script.rstrip().endswith(" &"):
            # Use check command twice to make sure it used
            shell.sendline(f"{script} && \\\n {check_cmd}")
            if content:
                shell.sendline(content)
            shell.sendline(check_cmd)
        else:
            # Running command in background
            shell.sendline(f"{script} {check_cmd}")
        # Wait for end of script
        index = shell.expect_list(
            expect_list, timeout=timeout if timeout else -1
        )
        return index == 0
    except pexpect.TIMEOUT:
        # Send SIGTERM
        shell.sendcontrol("\\")
        # Wait for program's end
        shell.sendline(check_cmd)
        shell.expect_list(expect_list)
        raise


def get_working_directory(markdown: str, tmpfolder: Path) -> Path:
    """
    Returns path to the virtual environment.

    Prepares working directory and returns path to it.

    Parameters
    ----------
    markdown : str
        Name of the markdown file.
    tmpfolder : Path
        Path to the temporary folder.

    Returns
    -------
    Path
        Path to the working directory.
    """
    lock_path = tmpfolder / f"{markdown}_wd.lock"

    with FileLock(lock_path):
        if markdown in WORKING_DIRS:
            return WORKING_DIRS[markdown]

        working_dir_path = tmpfolder / f"{markdown}_wd"

        venv_path = working_dir_path / "venv"
        if not venv_path.exists():
            venv.create(venv_path, with_pip=True, upgrade_deps=True)
        kenning_path = working_dir_path / "kenning"
        if not kenning_path.exists():
            patch_file = "temp.patch"
            subprocess.check_output(["git", "clone", "/ci", str(kenning_path)])
            with open(str(kenning_path / patch_file), "wb") as file:
                file.write(subprocess.check_output(["git", "diff"]))
            subprocess.check_output(
                ["git", "apply", patch_file, "--allow-empty"],
                cwd=str(kenning_path),
            )
            os.remove(str(kenning_path / patch_file))
        WORKING_DIRS[markdown] = working_dir_path
        return working_dir_path


def get_subshell(
    markdown: str,
    _id: int,
    working_dir: Path,
    log_dir: Optional[Path] = None,
) -> pexpect.spawn:
    """
    Returns existing subshell associated with markdown file and ID
    or create new one.

    Parameters
    ----------
    markdown : str
        Name of the markdown file.
    id : int
        ID of the subshell.
    working_dir : Path
        Path to the working directory.
    log_dir : Optional[Path]
        Path to folder where subshell logs will be saved.

    Returns
    -------
    pexpect.spawn
        Interface for communication with subshell.
    """
    if _id in SHELLS[markdown]:
        return SHELLS[markdown][_id]

    SHELLS[markdown][_id] = pexpect.spawn(
        shutil.which("bash"),
        cwd=str(working_dir),
        timeout=DEFAULT_TIMEOUT,
        encoding="utf-8",
        echo=False,
        use_poll=True,
    )
    if log_dir:
        tmp_file = open(f"{log_dir}/{markdown}_{_id}.log", "w")
        SHELLS[markdown][_id].logfile_read = tmp_file

    # Activate virtual environment
    venv_path = working_dir / "venv"
    if not execute_script_and_wait(
        SHELLS[markdown][_id], f'source {venv_path / "bin" / "activate"}'
    ):
        raise Exception("Virtual environment cannot be activated")

    # Link directories with datasets
    os.makedirs(working_dir / DATASET_DIR, exist_ok=True)
    for checksum_path in glob(f"{DATASET_DIR}/**/DATASET_CHECKSUM"):
        checksum_path = Path(checksum_path)
        target_path = working_dir / DATASET_DIR / checksum_path.parent.name
        if target_path.exists():
            continue
        os.symlink(checksum_path.parent.absolute(), target_path)

    return SHELLS[markdown][_id]


def create_script(snippet: Snippet, working_dir: Path) -> str:
    """
    Extract script from snippet and prepare it to be executed.

    It also detect `pip install kenning` and change it to use local version.

    Parameters
    ----------
    snippet : Snippet
        Snippet with script information.
    gallery_snippet : bool
        Whether snippet is from gallery.
    working_dir : Path
        The path to the temporary directory with Kenning clone.

    Returns
    -------
    str
        Prepared script
    """
    kenning_path = working_dir / "kenning"
    script = None
    if snippet.lang == "bash":
        # If `pip install` change it to install local Kenning version
        pip_install = re.match(PIP_INSTALL_KENNING_RE, snippet.text)
        if pip_install:
            snippet.text = re.sub(
                KENNING_LINK_RE, rf"{kenning_path}\2", snippet.text
            )
        else:
            snippet.text = re.sub(
                PIP_INSTALL_KENNING_LOCAL_RE,
                rf"\1'{kenning_path}\2'",
                snippet.text,
            )
        script = snippet.text
    elif snippet.lang == "python":
        # Dump python script to file and change script to run it
        _, tmpfile = tempfile.mkstemp()
        with open(tmpfile, "w") as fd:
            fd.write(snippet.text)
        script = f"python {tmpfile}"
    elif "save-as" in snippet.meta:
        save_as = Path(snippet.meta["save-as"])
        script = f"mkdir -p {save_as.parent}"
        if SNIPPET_MARKDOWN_PARENT in snippet.meta and isinstance(
            snippet.meta[SNIPPET_MARKDOWN_PARENT], Path
        ):
            full_path = (
                kenning_path
                / snippet.meta[SNIPPET_MARKDOWN_PARENT]
                / snippet.meta[SNIPPET_POSITIONAL_ARG]
            )
            script += f" && cp {full_path} {save_as}"
        else:
            text = snippet.text.split("\n")
            # Remove directive's options
            if text[0].startswith("---"):
                for i, line in enumerate(text[1:]):
                    if not line.startswith("---"):
                        continue
                    text = text[i + 2 :]
                    break
            text = "\n".join(text)

            script += f' && cat <<EOF > "{save_as}"\n'
            script += text
            script += "\nEOF\n"

    return script


class TestDocsSnippets:
    @pytest.mark.parametrize(
        "markdown",
        [
            pytest.param(
                markdown,
                marks=[
                    pytest.mark.xdist_group(f"TestDocsGallery_{markdown}"),
                    pytest.mark.order(-1),
                    pytest.mark.snippets,
                ],
            )
            for markdown in map(
                lambda p: Path(p).with_suffix("").name,
                glob(str(pytest.input_file_pattern)),
            )
        ],
    )
    def test_cleanup(self, markdown: str):
        """
        Checks if subshell is alive after the tests and cleanup used resources.

        Parameters
        ----------
        self : TestDocsSnippets
            Instance of TestDocsSnippets class
        markdown : str
            Name of markdown file
        """
        shells = SHELLS[markdown]
        for shell in shells.values():
            if not shell.isalive():
                pytest.fail(reason="Shell does not work after the tests")
            if shell.logfile_read:
                shell.logfile_read.close()
            shell.terminate(force=True)

    @pytest.mark.parametrize(
        "script,snippet,markdown",
        [
            pytest.param(
                create_script(
                    snippet,
                    get_working_directory(
                        markdown, pytest.test_directory / "tmp"
                    ),
                ),
                snippet,
                markdown,
                id=f"{markdown}_{snippet_name}",
                marks=[
                    pytest.mark.xdist_group(f"TestDocsGallery_{markdown}"),
                    pytest.mark.dependency(
                        name=f"_{markdown}_{snippet_name}",
                        depends=[
                            f"_{markdown}_{dep}"
                            for dep in snippet.meta["depends"]
                        ],
                    ),
                    pytest.mark.snippets,
                ],
            )
            for markdown, snippet_name, snippet in get_all_snippets(
                str(pytest.input_file_pattern)
            )
        ],
    )
    def test_snippet(
        self,
        script: str,
        snippet: Snippet,
        markdown: str,
        docs_log_dir: Optional[Path],
    ):
        """
        Test for snippet from documentation.

        It is run in separate environment and uses pseudo-tty.

        Parameters
        ----------
        self : TestDocsSnippets
            Instance of TestDocsSnippets class
        script : str
            Script that should be tested.
        snippet : Snippet
            Snippet from which script was extracted.
        markdown : str
            Name of the markdown file containing snippet.

        Fixtures
        --------
        docs_log_dir : Optional[Path]
            Path to folder where subshell logs will be saved.
        """
        working_dir = get_working_directory(
            markdown, pytest.test_directory / "tmp"
        )
        subshell = get_subshell(
            markdown, snippet.meta["terminal"], working_dir, docs_log_dir
        )

        timeout = (
            float(snippet.meta["timeout"])
            if "timeout" in snippet.meta
            else None
        )
        try:
            if subshell.logfile_read:
                subshell.logfile_read.write(
                    f'\n\n{"-" * 32}\n\n{script}\n\n{"-" * 32}\n\n'
                )
            success = execute_script_and_wait(subshell, script, timeout)
            if not success:
                pytest.fail(reason=f"'{script}' returned non-zero code")
        except pexpect.TIMEOUT:
            # Fail if timeout was not expected
            if not timeout:
                pytest.fail(reason=f"Unexpected timeout during: '{snippet}'")
        except pexpect.EOF:
            pytest.fail(
                reason=f"'{script}' finished without printing status message"
            )
