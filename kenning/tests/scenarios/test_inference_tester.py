# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from pytest_mock import MockerFixture

from kenning.cli.command_template import OPTIMIZE, TEST
from kenning.cli.config import USED_SUBCOMMANDS
from kenning.core.measurements import Measurements
from kenning.scenarios.inference_tester import InferenceTester

MeasurementsType = Tuple[Dict[str, Any], Dict[str, Any]]


@pytest.fixture
def measurements() -> MeasurementsType:
    """Creates sample measurements."""
    return (
        {"compiled_model_path": 32414736},
        {"compiled_model_path": 4615960},
    )


@pytest.fixture
def measurements_expected(
    mocker: MockerFixture,
    measurements: MeasurementsType,
) -> MeasurementsType:
    """
    Mocks the entire `PipelineRunner` and substitutes relevant methods and
    attributes.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest mocker object.
    measurements : MeasurementsType
        Measurements to mock.

    Returns
    -------
    MeasurementsType
        Expected measurements.
    """
    module = "kenning.scenarios.inference_tester"
    name = f"{module}.PipelineRunner"
    mock = mocker.Mock()

    def run(output, *args, **kwargs):
        Path(output).write_text(
            json.dumps(
                measurements[0]
                if kwargs.get("run_optimizations") is False
                else measurements[1]
            )
        )
        return 0

    mocker.patch(name, return_value=mock)
    mocker.patch(f"{name}.from_json_cfg", return_value=mock)
    mock.run = run
    mock.optimizers = [None]
    mock.dataset = mocker.Mock()
    mock.output = None

    name = f"{module}.get_command"
    mocker.patch(name, return_value="")

    return measurements


def test_measurements_evaluate_unoptimized(
    measurements_expected: MeasurementsType,
    tmpfolder: Path,
):
    """
    Tests the `InferenceTester` when "--evaluate-unoptimized" argument is
    passed.

    Expects `UNOPTIMIZED` attribute to carry corresponding measurements from
    unoptimized model.

    Parameters
    ----------
    measurements_expected : MeasurementsType
        A pair of mocked 'PipelineRunner' and expected measurements.
    tmpfolder : Path
        Path to the temporary folder.
    """
    unoptimized_expected, optimized_expected = measurements_expected
    output = tmpfolder / "output.json"
    json_cfg = tmpfolder / "cfg.json"
    json_cfg.write_text("{}")

    mock_args = Namespace(
        **{USED_SUBCOMMANDS: [OPTIMIZE, TEST]},
        measurements=[output],
        verbosity="",
        help=False,
        json_cfg=json_cfg,
        evaluate_unoptimized=True,
        compiler_cls=None,
    )

    assert (
        InferenceTester.run(mock_args) == 0
    ), "InferenceTester.run did not finish successfully"

    with output.open() as outputfile:
        outputdata = json.load(outputfile)
    unoptimized_outputdata = outputdata.pop(Measurements.UNOPTIMIZED)

    assert outputdata == optimized_expected
    assert unoptimized_outputdata == unoptimized_expected
