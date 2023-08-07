# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path


@pytest.fixture
def docs_log_dir(request) -> Path:
    log_dir = Path(request.config.getoption('--test-docs-log-dir'))
    log_dir.mkdir(exist_ok=True)
    return log_dir
