"""Abstract interface for an experiment tracker."""

from abc import ABC, abstractmethod
from src.configurable_builder import ConfigurableBuilder


class ExperimentTracker(ABC):
    """
    Interface for logging parameters and metrics to an experiment tracking platform.
    """

    @abstractmethod
    def log_params(self, params: dict):
        """Logs a dictionary of hyperparameters."""

    @abstractmethod
    def log_metrics(self, metrics: dict):
        """Logs a dictionary of final (summary) metrics."""

    @abstractmethod
    def report_hpo_metric(self, metric_name: str, metric_value: float, step: int):
        """Reports the primary metric for a hyperparameter tuning trial."""


class ExperimentTrackerBuilder(ConfigurableBuilder[ExperimentTracker], ABC):
    """
    Abstract builder for creating an ExperimentTracker instance.
    """

    @property
    def argparser(self):
        """
        Returns the ArgumentParser instance for CLI argument definition.

        Returns:
            ArgumentParser: The argument parser for the builder.
        """
        return super().argparser

    def build(self) -> ExperimentTracker:
        """
        Builds and returns the ExperimentTracker using the configured state and runtime parameters.

        Args:
            run_name (str): An identifier for the experiment run.

        Returns:
            ExperimentTracker: An instance of the built ExperimentTracker.
        """
        if self._cli_args is None:
            raise ValueError(
                "Builder is not configured. Call with_configuration() first."
            )
        return self._build()

    @abstractmethod
    def _build(self) -> ExperimentTracker:
        """Internal method to build the ExperimentTracker instance."""
