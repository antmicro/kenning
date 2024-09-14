# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def empty_file_path() -> Generator[Path, None, None]:
    """
    Fixture that returns path to a new temporary file that is closed
    automatically after using the fixture.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
