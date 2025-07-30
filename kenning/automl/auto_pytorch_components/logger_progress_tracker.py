import kenning.utils.logger
from autoPyTorch.utils.progress_tracker import (
    TrainingProgressTracker, TrainingEpochTracker
)
from smac.runhistory.runhistory import RunInfo, RunValue, RunHistory
from smac.utils.constants import MAXINT
from logging import Logger
from kenning.utils.logger import KLogger, LoggerProgressBar
from tqdm import tqdm
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
        self.total_time=int(total_time_expected_seconds),
        self.pbar = None

        TrainingProgressTracker.__init__(self, total_time_expected_seconds)


    def __enter__(self):
        with LoggerProgressBar() as logger_progress_bar:
            self.pbar = tqdm(
                total=self.total_time,
                position=0,
                **logger_progress_bar.kwargs
            )
        self.total_time_passed = 0
        self.lowest_cost = MAXINT
        self.best_metrics = {}
        return super().__enter__()


    def __exit__(self, exc_type, exc_val, exc_tb ):
        cur_perc = self.pbar.n
        total_perc = int(self.total_time)
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()
        super().__exit__(exc_type, exc_val, exc_tb )



    def report_progress(
        self,
        time_passed: float,
        metrics: dict,
        model: str,
        cost
    ) -> None:
        self.total_time_passed = self.total_time_passed + time_passed
        self.pbar.update(int(time_passed))
        if cost < self.lowest_cost:
            self.lowest_cost = cost
            self.best_metrics = metrics

        self.logger.info(f"Model backbone: {model}")
        if self.best_metrics != {}:
            self.logger.info(f"so-far best configuration metrics:")
            for name, metric in self.best_metrics.items():
                self.logger.info(f"- {name}: {metric}")


