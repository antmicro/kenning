# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.fixture
def docs_log_dir(request: pytest.FixtureRequest) -> Path:
    log_dir = Path(request.config.getoption("--test-docs-log-dir"))
    log_dir.mkdir(exist_ok=True)
    return log_dir
