import kenning.utils.logger
from autoPyTorch.utils.progress_tracker import (
    TrainingProgressTracker, EpochTracker
)
from smac.runhistory.runhistory import RunInfo, RunValue, RunHistory
from smac.utils.constants import MAXINT
from logging import Logger
from kenning.utils.logger import KLogger, LoggerProgressBar
from tqdm import tqdm
from typing import Optional, Dict
import time
import torch

class TrainingProgressLogger(TrainingProgressTracker):
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
                **logger_progress_bar.kwargs
            )
        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        return super().__enter__()


    def __exit__(self, exc_type, exc_val, exc_tb ):
        cur_perc = self.pbar.n
        total_perc = self.total_time
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()
        super().__exit__(exc_type, exc_val, exc_tb )


    def report_time( self, time_passed: Optional[float] = None):
        cur_time = time.time()
        if not time_passed:
            time_passed = cur_time - self.last_time_updated
        self.last_time_updated = cur_time
        self.pbar.update(time_passed)


    def report_progress(
        self,
        time_passed: float,
        metrics: Dict,
        model: str,
        cost
    ) -> None:
        self.total_time_passed = self.total_time_passed + time_passed
        self.report_time(time_passed)
        if cost < self.lowest_cost:
            self.lowest_cost = cost
            self.best_metrics = metrics

        self.logger.info(f"Model backbone: {model}")
        if self.best_metrics != {}:
            self.logger.info(f"so-far best configuration metrics:")
            for name, metric in self.best_metrics.items():
                self.logger.info(f"- {name}: {metric}")


class EpochTrainingStepLogger(EpochTracker):
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
                **logger_progress_bar.kwargs
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
        additional_info: Optional[Dict] = None
    ) -> None:
        if self.progress_tracker:
            self.progress_tracker.report_time()
        self.pbar.update(1)

class EpochEvaluationStepLogger(EpochTracker):
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
                **logger_progress_bar.kwargs
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
        additional_info: Optional[Dict] = None
    ) -> None:
        if self.progress_tracker:
            self.progress_tracker.report_time()
        self.pbar.update(1)
