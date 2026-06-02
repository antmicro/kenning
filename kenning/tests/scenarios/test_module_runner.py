# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from unittest.mock import MagicMock, patch

import pytest

from kenning.core.exceptions import KenningError
from kenning.scenarios.module_runner import (
    ModuleRunner,
)


class DummyModule:
    def __init__(self, init_var):
        self.init_var = init_var

    def some_method(self):
        # These are not initialized in __init__
        self.uninit_var_1 = "test"
        self.uninit_var_2 = 123


class TestModuleRunner:
    def test_configure_parser(self):
        """
        Ensures that the argument parser is correctly configured with the
        expected arguments.
        """
        parser, groups = ModuleRunner.configure_parser()

        # Extract argument destinations
        args_dests = [action.dest for action in parser._actions]
        assert "verbosity" in args_dests
        assert "module" in args_dests
        assert "cfg" in args_dests

        module_action = next(a for a in parser._actions if a.dest == "module")
        assert module_action.required is True

    @patch("kenning.scenarios.module_runner.load_class")
    def test_run_missing_config(self, mock_load_class):
        """
        KenningError is raised when --cfg argument is missing.
        """
        args = argparse.Namespace(module="test.module", cfg=None, no_wait=True)

        with pytest.raises(KenningError) as exc_info:
            ModuleRunner.run(args)

        assert "Missing --cfg argument" in str(exc_info.value)

    @patch("kenning.scenarios.module_runner.load_class")
    @patch("kenning.scenarios.module_runner.inject_uninitialized_attrs")
    @patch("kenning.scenarios.module_runner.threading.Event.wait")
    def test_run_keyboard_interrupt(
        self, mock_wait, mock_inject, mock_load_class
    ):
        """
        Verifies that the run method safely catches a KeyboardInterrupt.
        """
        mock_module_cls = MagicMock()
        mock_load_class.return_value = mock_module_cls
        mock_module_cls.build_from_config.return_value = MagicMock()

        args = argparse.Namespace(
            module="test.module", cfg='{"key": "value"}', no_wait=False
        )

        # Simulate a user pressing Ctrl+C
        mock_wait.side_effect = KeyboardInterrupt()

        # Execution should pass without throwing KeyboardInterrupt
        ModuleRunner.run(args)

        mock_wait.assert_called_once()
        mock_module_cls.build_from_config.assert_called_once_with(
            {"key": "value"}
        )
