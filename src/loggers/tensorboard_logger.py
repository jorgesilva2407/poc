"""
TensorBoard logger implementation.
"""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.environment import ENVIRONMENT
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

    def close(self):
        self.writer.close()


class TensorBoardLoggerBuilder(LoggerBuilder):
    """
    Builder for TensorBoardLogger instances.
    """

    @property
    def argparser(self):
        parser = super().argparser
        parser.add_argument(
            "--tensorboard-log-dir",
            type=str,
            default=ENVIRONMENT.get(ENVIRONMENT.VARIABLES.TENSORBOARD_LOG_DIR),
            help="Directory to store TensorBoard logs.",
        )
        return parser

    def _build(self, model_id: str) -> Logger:
        log_dir = self._cli_args["tensorboard_log_dir"]
        writer = SummaryWriter(log_dir=Path(log_dir) / model_id)
        return TensorBoardLogger(writer)
