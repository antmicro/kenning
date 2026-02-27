# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implements loggers for autopytorch progress tracking.
"""

import time
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

import torch
from autoPyTorch.utils.progress_tracker import (
    EpochTracker,
    TrainingProgressTracker,
)
from rich.table import Table
from smac.utils.constants import MAXINT

from kenning.utils.logger import RichStatus

import pandas as pd

class AutoMLRichStatus(RichStatus):
    """
    Subclass for RichStatus specific for AutoML search.
    """

    def __init__(
        self, enable_live: bool = True, keep_history: bool = True
    ) -> None:
        """
        Initializes an instance of the class.

        Parameters
        ----------
        enable_live: bool
            If False, live rendering is disabled (headless mode).
        keep_history: bool
            If True, will keep track of all the table updates over
            the duration of the search.
        """
        self.current_values = {}
        self.best_values = {}
        self.keep_history = keep_history
        self.history: list[dict] = []
        super().__init__(enable_live=enable_live)

    def _log_current_values(self) -> None:
        """
        Store a snapshot of `current_values` into history.
        """
        if not self.keep_history:
            return

        with self.state_lock:
            snapshot = dict(self.current_values)
        snapshot["timestamp"] = datetime.utcnow()
        self.history.append(snapshot)

    def get_history_df(self) -> pd.DataFrame:
        """
        Return the logged history as a pandas DataFrame.
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def save_history_csv(self, path: Path) -> None:
        """
        Save logged history to CSV file

        Parameters
        ----------
        path: str
            Path to save the CSV file.
        """
        df = self.get_history_df()
        if df.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True)

    def _make_table(self) -> Table:
        """
        Internal function for creating the `rich` table.
        """
        table = Table(expand=True)
        table.add_column("Metric")
        table.add_column("Value (Best so far)")
        table.add_column("Value (current)")

        def add_row(key):
            table.add_row(
                key,
                self._fmt_table_entry(self.best_values[key]),
                self._fmt_table_entry(self.current_values[key]),
            )

        if "Model" in self.best_values and "Model" in self.current_values:
            add_row("Model")
            add_row("Iterations")

            with self.state_lock:
                for key in self.current_values.keys():
                    if key not in ["Model", "Iterations"]:
                        add_row(key)
        return table

    def update_table(self, new_table: Optional[dict] = None) -> None:
        if new_table:
            with self.state_lock:
                self.current_values = new_table
        self._log_current_values()
        super().update_table(new_table)


class RichTrainingProgressTracker(TrainingProgressTracker):
    """
    Progress Tracker containing the RichLogger.
    """

    def __init__(
        self,
        richlogger: AutoMLRichStatus,
        total_time_expected_seconds: float,
        task,
    ):
        self.richlogger = richlogger

        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}

        self.iters = 0

        self.task = task
        super().__init__(total_time_expected_seconds)

    def __enter__(self):
        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}

        new_table = {}
        new_table["Iterations"] = 0
        new_table["Model"] = "None"

        self.best_values = new_table
        self.current_values = new_table

        self.richlogger.update_table()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def report_time(self, time_passed: Optional[float] = None):
        cur_time = time.time()
        if not time_passed:
            time_passed = cur_time - self.last_time_updated
        self.last_time_updated = cur_time
        self.richlogger.advance(self.task, advance=time_passed)

    def report_progress(
        self, time_passed: float, metrics: Dict, model: str, cost
    ) -> None:
        self.iters += 1

        metrics["Model"] = model
        metrics["Iterations"] = self.iters

        if self.lowest_cost >= cost or self.richlogger.best_values == {}:
            self.best_metrics = metrics
            self.lowest_cost = cost

            self.richlogger.best_values = metrics

        self.richlogger.current_values = metrics
        self.richlogger.update_table()

        self.total_time_passed += time_passed


class RichEpochTrainingStepLogger(EpochTracker):
    """
    Progress Tracker for each training step.
    """

    def __init__(
        self,
        richlogger: AutoMLRichStatus,
        task,
        progress_tracker=None,
    ):
        self.richlogger = richlogger
        self.task = task
        self.progress_tracker = progress_tracker
        super().__init__()

    def __enter__(self):
        self.richlogger.reset_task(self.task, "Training", self.total_steps)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def report_step_progress(
        self,
        loss: float,
        batch_size: int,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        additional_info: Optional[Dict] = None,
    ) -> None:
        if self.progress_tracker:
            self.progress_tracker.report_time()
        self.richlogger.advance(self.task)


class RichEpochEvaluationStepLogger(EpochTracker):
    """
    Tracks and logs progress of one epoch evaluation in autoPyTorch.
    """

    def __init__(
        self,
        richlogger: AutoMLRichStatus,
        task,
        progress_tracker=None,
    ):
        self.richlogger = richlogger
        self.task = task
        self.progress_tracker = progress_tracker
        super().__init__()

    def __enter__(self):
        self.richlogger.reset_task(self.task, "Evaluation", self.total_steps)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def report_step_progress(
        self,
        loss: float,
        batch_size: int,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        additional_info: Optional[Dict] = None,
    ) -> None:
        if self.progress_tracker:
            self.progress_tracker.report_time()
        self.richlogger.advance(self.task)
