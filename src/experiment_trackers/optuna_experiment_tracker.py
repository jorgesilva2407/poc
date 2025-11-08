"""Optuna Experiment Tracker Module."""

import os
import json

from src.experiment_trackers.experiment_tracker import (
    ExperimentTracker,
    ExperimentTrackerBuilder,
)


class OptunaExperimentTracker(ExperimentTracker):
    """
    Logs parameters and metrics to Optuna.
    Writes HPO metrics to a specified file.
    """

    def __init__(self, metric_file: str):
        self.metric_file = metric_file

    def log_params(self, params: dict):
        pass

    def log_metrics(self, metrics: dict):
        pass

    def report_hpo_metric(self, metric_name: str, metric_value: float, step: int):
        """
        Writes to a temporary file that will be read in the optuna optimization job.
        """
        os.makedirs(os.path.dirname(self.metric_file), exist_ok=True)
        metric_record = {
            "metric": metric_name,
            "value": metric_value,
        }
        with open(self.metric_file, "w", encoding="utf-8") as f:
            json.dump(metric_record, f)


class OptunaExperimentTrackerBuilder(ExperimentTrackerBuilder):
    """
    Builds the OptunaExperimentTracker.
    """

    @property
    def argparser(self):
        parser = super().argparser
        parser.add_argument(
            "--optuna-metric-file",
            type=str,
            required=True,
            help="File path to write the Optuna HPO metric.",
        )
        return parser

    def _build(self) -> ExperimentTracker:
        metric_file = self._cli_args["optuna_metric_file"]
        return OptunaExperimentTracker(metric_file=metric_file)
