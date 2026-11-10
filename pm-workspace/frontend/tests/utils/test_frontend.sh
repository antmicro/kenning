#!/bin/bash

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# A script used for testing pipeline manager frontend.


#!/bin/bash

set -ex

npx playwright install --with-deps
npx playwright test --list
TOTAL_NUM_TESTS=$(python3 tests/utils/read_total_tests.py test-results.json)
echo "Found $TOTAL_NUM_TESTS tests"
npm run test
EXECUTED_TESTS=$(python3 tests/utils/read_total_tests.py test-results.json)
echo "Executed $EXECUTED_TESTS tests"
[[ $TOTAL_NUM_TESTS -eq $EXECUTED_TESTS ]] && exit 0 || exit 1