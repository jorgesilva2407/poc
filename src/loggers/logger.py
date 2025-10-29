"""
Abstract base classes for logging training progress, losses, and metrics.
"""

from enum import Enum
from abc import ABC, abstractmethod

from src.configurable_builder import ConfigurableBuilder


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

    def close(self):
        """Close any resources held by the logger, if necessary."""


class LoggerBuilder(ConfigurableBuilder[Logger], ABC):
    """
    Abstract base class for building Logger instances.
    """
