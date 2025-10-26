"""
TensorBoard logger implementation.
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.loggers.logger import Logger, LoggerBuilder, DatasetType


class TensorBoardLogger(Logger):
    """
    TensorBoard logger implementation.
    """

    def __init__(self, writer):
        self.writer = writer

    def loss(self, loss_name: str, value: float, epoch: int, dataset: DatasetType):
        tag = f"Loss/{loss_name}/{dataset.value}"
        self.writer.add_scalar(tag, value, epoch)

    def metrics(self, metrics: dict, epoch: int, dataset: DatasetType):
        for metric_name, metric_value in metrics.items():
            tag = f"Metrics/{metric_name}/{dataset.value}"
            self.writer.add_scalar(tag, metric_value, epoch)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()


class TensorBoardLoggerBuilder(LoggerBuilder):
    """
    Builder for TensorBoardLogger instances.
    """

    def __init__(self, tensorboard_log_dir: Path):
        self.tensorboard_log_dir = tensorboard_log_dir

    def build(self, model_id: str) -> Logger:
        writer = SummaryWriter(log_dir=self.tensorboard_log_dir / model_id)
        return TensorBoardLogger(writer)
