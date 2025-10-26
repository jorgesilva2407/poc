"""
Abstract base classes for logging training progress, losses, and metrics.
"""

from abc import ABC, abstractmethod
from enum import Enum


class DatasetType(Enum):
    """
    Enum for different dataset types.
    """

    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"


class Logger(ABC):
    """
    Abstract base class for logging training progress, losses, and metrics.
    """

    @abstractmethod
    def loss(self, loss_name: str, value: float, epoch: int, dataset: DatasetType):
        """Log the loss value for a given epoch."""

    @abstractmethod
    def metrics(self, metrics: dict, epoch: int, dataset: DatasetType):
        """Log various metrics for a given epoch."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LoggerBuilder(ABC):
    """
    Abstract base class for building Logger instances.
    """

    @abstractmethod
    def build(self, model_id: str) -> Logger:
        """Build and return a Logger instance."""
