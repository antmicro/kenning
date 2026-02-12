# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implements loggers for autopytorch progress tracking.
"""

import time
from logging import Logger
from typing import Dict, Optional

import torch
from autoPyTorch.utils.progress_tracker import (
    EpochTracker,
    TrainingProgressTracker,
)
from smac.utils.constants import MAXINT
from tqdm import tqdm

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from kenning.utils.logger import KLogger, LoggerProgressBar, RichLogger


class TrainingProgressLogger(TrainingProgressTracker):
    """
    Progress tracker for the whole autoPyTorch
    training and evaluation process.
    """

    def __init__(
        self,
        total_time_expected_seconds: float,
        logger: Logger = KLogger,
    ):
        self.logger = logger

        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        self.pbar = None
        TrainingProgressTracker.__init__(self, total_time_expected_seconds)

    def __enter__(self):
        with LoggerProgressBar() as logger_progress_bar:
            self.pbar = tqdm(
                total=self.total_time,
                desc="Time limit",
                position=0,
                **logger_progress_bar.kwargs,
            )
        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        cur_perc = self.pbar.n
        total_perc = self.total_time
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()
        super().__exit__(exc_type, exc_val, exc_tb)

    def report_time(self, time_passed: Optional[float] = None):
        cur_time = time.time()
        if not time_passed:
            time_passed = cur_time - self.last_time_updated
        self.last_time_updated = cur_time
        self.pbar.update(time_passed)

    def report_progress(
        self, time_passed: float, metrics: Dict, model: str, cost
    ) -> None:
        self.total_time_passed = self.total_time_passed + time_passed
        self.report_time(time_passed)
        if cost < self.lowest_cost:
            self.lowest_cost = cost
            self.best_metrics = metrics

        self.logger.info(f"Model backbone: {model}")
        if self.best_metrics != {}:
            self.logger.info("so-far best configuration metrics:")
            for name, metric in self.best_metrics.items():
                self.logger.info(f"- {name}: {metric}")


class EpochTrainingStepLogger(EpochTracker):
    """
    Tracks and logs progress of one epoch training in autoPyTorch.
    """

    def __init__(
        self,
        progress_tracker: Optional[TrainingProgressLogger] = None,
        logger: Logger = KLogger,
    ):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.pbar = None

        EpochTracker.__init__(self)

    def __enter__(self):
        with LoggerProgressBar() as logger_progress_bar:
            self.pbar = tqdm(
                desc="Epoch training",
                total=self.total_steps,
                position=1,
                **logger_progress_bar.kwargs,
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.pbar:
            return
        cur_perc = self.pbar.n
        total_perc = int(self.total_steps)
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()
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
        self.pbar.update(1)


class EpochEvaluationStepLogger(EpochTracker):
    """
    Tracks and logs progress of one epoch evaluation in autoPyTorch.
    """

    def __init__(
        self,
        progress_tracker: Optional[TrainingProgressLogger] = None,
        logger: Logger = KLogger,
    ):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.pbar = None

        EpochTracker.__init__(self)

    def __enter__(self):
        with LoggerProgressBar() as logger_progress_bar:
            self.pbar = tqdm(
                desc="Epoch evaluation",
                total=self.total_steps,
                position=1,
                **logger_progress_bar.kwargs,
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.pbar:
            return
        cur_perc = self.pbar.n
        total_perc = int(self.total_steps)
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()
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
        self.pbar.update(1)

class RichLoggerTrainingProgressTracker(TrainingProgressTracker):
    def __init__(
        self,
        richlogger: RichLogger,
        total_time_expected_seconds: float,
    ):
        self.richlogger = richlogger

        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        
        super().__init__(total_time_expected_seconds)

    def __enter__(self):
        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def report_progress(
        self, time_passed: float, metrics: Dict, model: str, cost
    ) -> None:
        model_str = f'Model backbone: {model}'
        metrics_str = 'No metrics yet'
        if self.best_metrics != {}:
            best_metrics = 'so-far best configuration metrics:'
            accuracy_str = f'- accuracy: {metrics["accuracy"]}'
            f1_str = f'- f1: {metrics["f1"]}'
            f1_macro = f'- f1_macro: {metrics["f1_macro"]}'
            f1_micro = f'- f1_micro: {metrics["f1_micro"]}'
            log_loss = f'- log loss: {metrics["log_loss"]}'
            metrics_str = f'{best_metrics}\n{accuracy_str}\n{f1_str}\n{f1_macro}\n{f1_micro}\n{log_loss}'
        msg_str = f'{model_str}\n{metrics_str}'
        self.richlogger.enqueue_bar(msg_str)
