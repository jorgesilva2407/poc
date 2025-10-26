from abc import ABC, abstractmethod
from enum import Enum


class DatasetType(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"


class Logger(ABC):
    @abstractmethod
    def loss(self, loss_name: str, value: float, epoch: int, dataset: DatasetType):
        """Log the loss value for a given epoch."""
        pass

    @abstractmethod
    def metrics(self, metrics: dict, epoch: int, dataset: DatasetType):
        """Log various metrics for a given epoch."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LoggerBuilder(ABC):
    @abstractmethod
    def build(self, model_id: str) -> Logger:
        """Build and return a Logger instance."""
        pass
