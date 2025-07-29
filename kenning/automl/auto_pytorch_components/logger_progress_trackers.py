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


    def on_start(self):
        with LoggerProgressBar() as logger_progress_bar:
            self.pbar = tqdm(
                total=self.total_time,
                position=0,
                **logger_progress_bar.kwargs
            )

        self.time_passed = 0

    def __del__(self):
        cur_perc = self.pbar.n
        total_perc = int(self.total_time)
        self.pbar.update(total_perc - cur_perc)
        self.pbar.refresh()
        self.pbar.close()

    def report_progress(
        self,
        total_time_passed: float,
        total_time_left: float
    ) -> None:
        update_time = total_time_passed - self.time_passed
        self.time_passed = total_time_passed
        self.pbar.update(int(update_time))
        self.logger.info("time passed: %f, time left: %f", total_time_passed, total_time_left)

