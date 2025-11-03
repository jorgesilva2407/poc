"""A default, do-nothing experiment tracker."""

import sys

from src.experiment_trackers.experiment_tracker import (
    ExperimentTracker,
    ExperimentTrackerBuilder,
)


class NoOpExperimentTracker(ExperimentTracker):
    """
    A tracker that performs no operations.
    """

    def log_params(self, params: dict):
        print(f"Local-only (params): {params}", file=sys.stderr)

    def log_metrics(self, metrics: dict):
        print(f"Local-only (metrics): {metrics}", file=sys.stderr)

    def report_hpo_metric(self, metric_name: str, metric_value: float, step: int):
        print(
            f"Local-only (HPO Metric): {metric_name} = {metric_value:.4f} at step {step}",
            file=sys.stderr,
        )


class NoOpExperimentTrackerBuilder(ExperimentTrackerBuilder):
    """
    Builds the NoOpExperimentTracker.
    """

    @property
    def argparser(self):
        return super().argparser

    def _build(self) -> ExperimentTracker:
        return NoOpExperimentTracker()
