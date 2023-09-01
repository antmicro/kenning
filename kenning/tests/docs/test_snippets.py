# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import pexpect
import tempfile
import venv
import os
import re
import uuid
import copy
import shutil
from glob import glob
from pathlib import Path
from tuttest import get_snippets, Snippet
from typing import Generator, Tuple, Dict, Optional
from collections import defaultdict

# Regex for extracting arguments from snippets
ARGS_RE = re.compile(r'([^ =]+)(=([^ ]+))?')
# Regex for changing Kenning installtion to local version
KENNING_LINK_RE = r'(kenning(\[[^\]]+\]) @ )?git\+https.*\.git'
# Regex for detecting Kenning installation
PIP_INSTALL_KENNING_RE = r'pip (.* )?install .*' + KENNING_LINK_RE
# Patterns of markdown files
DOCS_DIR = Path(__file__).parent / "source"
DOCS_MARKDOWNS = str(DOCS_DIR / "*.md")
GALLERY_DIR = DOCS_DIR / "gallery"
GALLERY_MARKDOWNS = str(GALLERY_DIR / "*.md")
# List of snippet executables types
EXECUTABLE_TYPES = ('bash',)
# Mapping of markdown files to separate virtual environment
VENVS: Dict[str, Path] = {}
# Mapping of markdown files to subshells
SHELLS: Dict[str, Dict[int, pexpect.spawn]] = defaultdict(dict)
# Template for command checking if previous process was successful
COMMAND_CHECK = "if [[ $? == 0 ]] ; then echo '{}'; else echo '{}'; fi"
# Template of regex detecting status of previous command
EXPECT_RE = "(?<!echo '){}"
# Regex for splitting multiline scripts
NEW_LINE_RE = re.compile('(?<!\\\\)\n', flags=re.MULTILINE)
# Default timeout of command execution
DEFAULT_TIMEOUT = 60 * 15  # 15 min
# Path to the environment for testing snippets from documentation
DOCS_VENV = os.environ.get("KENNING_DOCS_VENV")
# Possible arguments for snippet
SNIPPET_ARGUMENTS = ('skip', 'timeout', 'name', 'terminal')


def get_all_snippets(
    markdown_pattern: str,
) -> Generator[Tuple[Path, str, Snippet, Path], None, None]:
    """
    Finds all executable snippets from gallery of examples
    and dumps named JSON snippets to files.

    Parameters
    ----------
    markdown_pattern : str
        Pattern with markdowns, has to be supported by `glob`

    Yields
    ------
    Path :
        Name of the markdown file with snippets
    str :
        Name of the found snippet
    Snippet :
        Found snippet
    """
    for markdown in glob(markdown_pattern):
        markdown = Path(markdown)

        python_snippet = None
        last_snippet_name = None
        for name, snippet in get_snippets(str(markdown)).items():
            # Parse args from language
            args = snippet.lang.split(' ', 1)
            snippet.lang = args[0]
            if snippet.lang not in EXECUTABLE_TYPES + ('python',):
                continue
            args = ARGS_RE.findall(args[1]) if len(args) == 2 else []
            for arg in args:
                if arg[0] in SNIPPET_ARGUMENTS:
                    snippet.meta[arg[0]] = arg[2] if arg[2] else True
                else:
                    raise KeyError(f"Snippet cannot have {arg[0]} argument")

            snippet.meta['depends'] = []
            snippet.meta['terminal'] = int(snippet.meta.get('terminal', 0))

            # Snippet should not be executed
            if snippet.meta.get('skip', False):
                continue

            # Append values to snippet's conntent
            if 'append_before' in snippet.meta:
                snippet.text = \
                    f"{snippet.meta['append_before']} {snippet.text}"
            if 'append_after' in snippet.meta:
                snippet.text = \
                    f"{snippet.text} {snippet.meta['append_after']}"

            # Snippet is executable -- yield it
            if snippet.lang in EXECUTABLE_TYPES:
                # Split multiline snippet
                for id, line in enumerate(NEW_LINE_RE.split(snippet.text)):
                    line_snippet = copy.deepcopy(snippet)
                    line_snippet.text = line
                    # Set previous snippet as dependency
                    if last_snippet_name:
                        line_snippet.meta['depends'].append(last_snippet_name)
                    last_snippet_name = f"{name}_{id}"
                    yield (markdown.with_suffix('').name,
                           last_snippet_name, line_snippet)
            # Python snippet -- combain and yield at the end of function
            elif snippet.lang == "python":
                if python_snippet:
                    python_snippet.text += snippet.text + '\n'
                else:
                    python_snippet = snippet
                    python_snippet.text += '\n'

        # Yield combined python snippets
        if python_snippet:
            yield markdown.with_suffix('').name, name, python_snippet


def execute_script_and_wait(
    shell: pexpect.spawn, script: str, timeout: Optional[float] = None,
) -> bool:
    """
    Wrapper that sends commands to subshell and waits for them to end.

    It also make sure that process is killed when timeout happens.

    Parameters
    ----------
    shell : pexpect.spawn
        Interface for communication with subshell
    script : str
        Script that should be executed
    timeout : Optional[float]
        How long process can run

    Returns
    -------
    bool :
        Succesfulness of the script
    """
    success, failure = uuid.uuid4(), uuid.uuid4()
    check_cmd = COMMAND_CHECK.format(success, failure)
    expect_list = [
        re.compile(EXPECT_RE.format(success)),
        re.compile(EXPECT_RE.format(failure)),
    ]
    try:
        # Use check command twice to make sure it used
        shell.sendline(f"{script} && {check_cmd}")
        shell.sendline(check_cmd)
        # Wait for end of script
        index = shell.expect_list(
            expect_list, timeout=timeout if timeout else -1
        )
        return index == 0
    except pexpect.TIMEOUT:
        # Send SIGTERM
        shell.sendcontrol('\\')
        # Wait for program's end
        shell.sendline(check_cmd)
        shell.expect_list(expect_list)
        raise


def get_venv(markdown: str, tmpfolder: Path) -> Path:
    """
    Returns path to the virtual environment.

    Returns path for existing virtual environment assosiated with markdown
    file or create a new one.

    Parameters
    ----------
    markdown : str
        Name of the markdown file
    tmpfolder : Path
        Path to the temporary folder

    Returns
    -------
    Path :
        Path to the virtual environment
    """
    if markdown not in VENVS:
        VENVS[markdown] = tmpfolder / f"venv_{markdown}"
        venv.create(VENVS[markdown], with_pip=True, upgrade_deps=True)
    return VENVS[markdown]


def get_subshell(
    markdown: str,
    id: int,
    separate_venv: bool,
    tmpfolder: Path,
    log_dir: Optional[str] = None,
) -> pexpect.spawn:
    """
    Returns existing subshell assosiated with markdown file and ID
    or create new one.

    Parameters
    ----------
    markdown : str
        Name of the markdown file
    id : int
        ID of the subshell
    separate_venv : bool
        Should subshell use separate virtual environment
    tmpfolder : Path
        Path to the temporary folder
    log_dir : Optional[Path]
        Path to folder where subshell logs will be saved

    Returns
    -------
    pexpect.spawn :
        Interface for communication with subshell
    """
    if id in SHELLS[markdown]:
        return SHELLS[markdown][id]

    SHELLS[markdown][id] = pexpect.spawn(
        shutil.which('bash'),
        timeout=DEFAULT_TIMEOUT,
        encoding='utf-8',
        echo=False,
        use_poll=True,
    )
    if log_dir:
        tmp_file = open(f'{log_dir}/{markdown}_{id}.log', 'w')
        SHELLS[markdown][id].logfile_read = tmp_file

    # Activate virtual environment
    if separate_venv:
        _venv = get_venv(markdown, tmpfolder)
    elif DOCS_VENV:
        _venv = Path(DOCS_VENV)
    if _venv and not execute_script_and_wait(
            SHELLS[markdown][id], f"source {_venv / 'bin' / 'activate'}"):
        raise Exception('Virtual environment cannot be activated')

    # Create copy of the repo and change working directory
    repo_copy = tmpfolder / f"{markdown}_kenning"
    if not (
        repo_copy.exists() or execute_script_and_wait(
            SHELLS[markdown][id], f"git clone . {repo_copy}")
    ) or not execute_script_and_wait(SHELLS[markdown][id], f"cd {repo_copy}"):
        raise Exception('Copy of the repository cannot be created')

    return SHELLS[markdown][id]


def create_script(snippet: Snippet) -> str:
    """
    Extract script from snippet and prepare it to be executed.

    It also detect `pip install kenning` and change it to use local version.

    Parameters
    ----------
    snippet : Snippet
        Snippet with script information
    """
    script = None
    if snippet.lang == "bash":
        # If `pip install` change it to intall local Kenning version
        pip_install = re.match(PIP_INSTALL_KENNING_RE, snippet.text)
        if pip_install:
            snippet.text = re.sub(KENNING_LINK_RE, r'.\2', snippet.text)
        script = snippet.text
    elif snippet.lang == "python":
        # Dump python script to file and change script to run it
        _, tmpfile = tempfile.mkstemp()
        with open(tmpfile, 'w') as fd:
            fd.write(snippet.text)
        script = f'python {tmpfile}'

    return script


def factory_test_snippet(markdown_pattern: str, docs_gallery: bool):
    """
    Factory creating tests for snippets from documentation.

    Parameters
    ----------
    markdown_pattern : str
        Defines from which files snippets should be extracted.
    docs_gallery : bool
        Defines if markdowns are part of gallery, marks tests as `docs_gallery`
        and run then in separate environment

    Returns
    -------
    Callable :
        Parametrized test
    """
    tmpfolder = pytest.test_directory / 'tmp'

    @pytest.mark.parametrize('script,snippet,markdown,separate_venv', [
        pytest.param(
            create_script(snippet),
            snippet, markdown, docs_gallery,
            id=f"{markdown}_{snippet_name}",
            marks=[
                pytest.mark.xdist_group(f"TestDocsGallery_{markdown}"),
                pytest.mark.dependency(
                    name=f"_{markdown}_{snippet_name}",
                    depends=[
                        f"_{markdown}_{dep}"
                        for dep in snippet.meta["depends"]
                    ]
                ),
            ] + (
                [pytest.mark.docs_gallery] if docs_gallery
                else [pytest.mark.docs]
            )
        ) for markdown, snippet_name, snippet
        in get_all_snippets(markdown_pattern)
    ])
    def _test_snippet(
        self,
        script: str,
        snippet: Snippet,
        markdown: str,
        separate_venv: bool,
        docs_log_dir: Optional[Path],
    ):
        """
        Test for snippet from documentation.

        It is run in separate environment and uses pseudo-tty.

        Parameters
        ----------
        script : str
            Script that should be tested
        snippet : Snippet
            Snippet from which script was extracted
        markdown : str
            Name of the markdown file containing snippet
        separate_venv : bool
            Should test use separate environment

        Fixtures
        --------
        docs_log_dir : Optional[Path]
            Path to folder where subshell logs will be saved
        """
        subshell = get_subshell(
            markdown, snippet.meta['terminal'],
            separate_venv, tmpfolder, docs_log_dir
        )

        timeout = float(snippet.meta['timeout']) \
            if 'timeout' in snippet.meta else None
        try:
            success = execute_script_and_wait(subshell, script, timeout)
            if not success:
                pytest.fail(reason=f"'{script}' returned non-zero code")
        except pexpect.TIMEOUT:
            # Fail if timeout was not expected
            if not timeout:
                pytest.fail(reason=f"Unexpected timeout during: '{snippet}'")
        except pexpect.EOF:
            pytest.fail(
                reason=f"'{script}' finished without printing status message")

    return _test_snippet


def factory_cleanup(markdown_pattern: str, docs_gallery: bool):
    """
    Factory creating tests for releasing resources.

    Parameters
    ----------
    markdown_pattern : str
        Defines for which files cleanup should be created.
    docs_gallery : bool
        Defines if markdowns are part of gallery
        and marks tests as `docs_gallery`

    Returns
    -------
    Callable :
        Parametrized test
    """

    @pytest.mark.parametrize('markdown', [
        pytest.param(
            markdown,
            marks=[
                pytest.mark.xdist_group(f"TestDocsGallery_{markdown}"),
                pytest.mark.order(-1),
            ] + (
                [pytest.mark.docs_gallery] if docs_gallery
                else [pytest.mark.docs]
            )
        ) for markdown in map(
            lambda p: Path(p).with_suffix('').name, glob(markdown_pattern))
    ])
    def _cleanup(self, markdown: str):
        """
        Checks if subshell is alive after the tests and cleanup used resources.

        Parameters
        ----------
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

    return _cleanup


class TestDocsSnippets:

    test_snippet = factory_test_snippet(DOCS_MARKDOWNS, False)
    test_gallery_snippet = factory_test_snippet(GALLERY_MARKDOWNS, True)
    test_cleanup = factory_cleanup(DOCS_MARKDOWNS, False)
    test_cleanup_gallery = factory_cleanup(GALLERY_MARKDOWNS, True)
