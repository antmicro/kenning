# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implements loggers for autopytorch progress tracking.
"""

import time
from typing import Any, Dict, List, Optional

import torch
from autoPyTorch.utils.progress_tracker import (
    EpochTracker,
    TrainingProgressTracker,
)
from rich.progress import Progress
from rich.table import Table
from smac.utils.constants import MAXINT

from kenning.utils.logger import KLogger, RichStatus


class AutoMLRichStatus(RichStatus):
    """
    Subclass for RichStatus specific for AutoML search.
    """

    def __init__(
        self,
        enable_live: bool = True,
        exclude_metrics: List[str] = [],
    ) -> None:
        """
        Initializes an instance of the class.

        Parameters
        ----------
        enable_live: bool
            If False, live rendering is disabled (headless mode).
        exclude_metrics: List[str]
            The metrics that will be excluded in the table and final
            CSV report.
        """
        self.current_values = {}
        self.best_values = {}
        self.exclude_metrics = exclude_metrics
        super().__init__(enable_live=enable_live)

    def _make_table(self) -> Table:
        """
        Internal function for creating the `rich` table.
        """
        table = Table(expand=True)
        table.add_column("Metric")
        table.add_column("Value (Best so far)")
        table.add_column("Value (current)")

        def add_row(key):
            if key in self.exclude_metrics:
                return

            table.add_row(
                key,
                self._fmt_table_entry(self.best_values[key]),
                self._fmt_table_entry(self.current_values[key]),
            )

        if "Model" in self.best_values and "Model" in self.current_values:
            # Ensure this always goes on top.
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
        super().update_table(new_table)


class RichTrainingProgressTracker(TrainingProgressTracker):
    """
    Progress Tracker containing the RichLogger.
    """

    def __init__(
        self,
        richlogger: AutoMLRichStatus,
        total_time_expected_seconds: float,
        progress_bar: Progress,
    ):
        """
        Initialize a new `RichTrainingProgressTracker` instance.

        Parameters
        ----------
        richlogger: AutoMLRichStatus
            Instance of `AutoMLRichStatus` to use.

        total_time_expected_seconds: float
            Total time that search is going to take place.

        progress_bar: Progress
            The progress bar to use for the training progress tracker.
        """
        self.richlogger = richlogger

        self.lowest_cost = MAXINT
        self.best_metrics = {}

        self.iters = 0

        self.progress_bar = progress_bar
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
        # TODO: Currently broken. Resets the progress bar
        #       to 0 when used from within `RichEpochStepLogger`.
        cur_time = time.time()

        if not time_passed:
            time_passed = cur_time - self.last_time_updated
        self.last_time_updated = cur_time

    def report_progress(
        self, time_passed: float, metrics: dict, model: str, cost: Any
    ) -> None:
        """
        Function that AutoPyTorch calls to report progress to kenning.

        Parameters
        ----------
        time_passed: float
            How much time has passed since last call.

        metrics: dict
            Metrics obtained for the training step.

        model: str
            Name of the model used.

        cost: Any
            Value of the metric used to determine the performance
            of the model.
        """
        self.iters += 1

        metrics["Model"] = model
        metrics["Iterations"] = self.iters

        if self.lowest_cost >= cost or self.richlogger.best_values == {}:
            self.best_metrics = metrics
            self.lowest_cost = cost

            self.richlogger.best_values = metrics

        self.richlogger.current_values = metrics

        KLogger.info(
            (
                f"AutoML Iteration {self.iters} on {model} "
                f"(time passed: {time_passed:.3f}s, "
                f"cost: {cost:.3f})"
            )
        )

        if not self.richlogger.enable_live:
            for k, v in metrics.items():
                vfmt = self.richlogger._fmt_table_entry(v)
                KLogger.info(f"- {k} = {vfmt}")

        self.richlogger.update_table()

        self.progress_bar.advance(amount=time_passed)

        self.total_time_passed += time_passed


class RichEpochStepLogger(EpochTracker):
    """
    Progress Tracker for each Epoch.
    """

    def __init__(
        self,
        richlogger: AutoMLRichStatus,
        progress_bar: Progress,
    ):
        self.richlogger = richlogger
        self.progress_bar = progress_bar
        super().__init__()

    def __enter__(self):
        self.progress_bar.start_task(self.total_steps)
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
        self.progress_bar.advance(amount=1)
