"""
Trainer module for training, validating, and testing recommender models.
"""

import os
from typing import TypedDict

import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.recommender import Recommender
from src.losses.pairwise_losses import PairwiseLoss
from src.metrics.listwise_metrics import ListwiseMetric
from src.loggers.logger import LoggerBuilder, DatasetType
from src.datasets.recommendation_dataset import TripletSample
from src.artifacts_savers.artifacts_saver import (
    ArtifactsSaver,
    ArtifactsSaverBuilder,
)


class RunResults(TypedDict):
    """Results of a training run."""

    best_metric_name: str
    best_val_metric: float
    best_epoch: int
    test_metrics: dict[str, float]


class Trainer:
    """
    Trainer for recommender models.
    Args:
        model (Recommender): The recommender model to be trained.
        train_loader (DataLoader[TripletSample]): DataLoader for training data.
        val_loader (DataLoader[TripletSample]): DataLoader for validation data.
        test_loader (DataLoader[TripletSample]): DataLoader for test data.
        validation_subset_ratio (float): Ratio of the validation set to use for subset validation.
        optimizer (Optimizer): Optimizer for training the model.
        epochs (int): Number of training epochs.
        loss (PairwiseLoss): Pairwise loss function.
        metrics (list[ListwiseMetric]): List of listwise metrics to evaluate.
        early_stopping_metric (str): Metric name to monitor for early stopping.
        early_stopping_delta (float): Minimum change in the monitored metric to qualify as an
            improvement.
        early_stopping_patience (int): Number of epochs with no improvement after which training
            will be stopped.
        maximize_metric (bool): Whether to maximize or minimize the early stopping metric.
        logger_builder (LoggerBuilder): Builder for creating a logger instance.
        artifacts_saver_builder (ArtifactsSaverBuilder): Builder for creating an artifacts saver
            instance.
        device (torch.device): Device to run the training on.
    """

    WARMUP_EPOCHS = 10

    _model: Recommender
    _train_loader: DataLoader[TripletSample]
    _val_loader: DataLoader[TripletSample]
    _test_loader: DataLoader[TripletSample]
    _optimizer: Optimizer
    _epochs: int
    _loss: PairwiseLoss
    _metrics: list[ListwiseMetric]
    _early_stopping_metric: str
    _early_stopping_delta: float
    _early_stopping_patience: int
    _maximize_metric: bool
    _logger: LoggerBuilder
    _artifacts_saver: ArtifactsSaver
    _device: torch.device

    _epochs_without_improvement: int = 0
    _best_val_metric: float | None = None
    _best_epoch: int = 0
    _checkpoint_path: str = "/tmp/best_model_checkpoint.pt"

    def __init__(
        self,
        model: Recommender,
        train_loader: DataLoader[TripletSample],
        val_loader: DataLoader[TripletSample],
        test_loader: DataLoader[TripletSample],
        optimizer: Optimizer,
        epochs: int,
        loss: PairwiseLoss,
        metrics: list[ListwiseMetric],
        early_stopping_metric: str,
        early_stopping_delta: float,
        early_stopping_patience: int,
        maximize_metric: bool,
        logger_builder: LoggerBuilder,
        artifacts_saver_builder: ArtifactsSaverBuilder,
        device: torch.device,
    ):
        self._model = model.to(device)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._optimizer = optimizer
        self._epochs = epochs
        self._loss = loss
        self._metrics = metrics
        self._device = device
        self._early_stopping_metric = early_stopping_metric
        self._early_stopping_delta = early_stopping_delta
        self._early_stopping_patience = early_stopping_patience
        self._maximize_metric = maximize_metric

        model_id = self._model_hparams_str()
        self._logger = logger_builder.build(model_id)
        self._artifacts_saver = artifacts_saver_builder.build(model_id)

    def run(self) -> RunResults:
        """
        Run the training, validation, and testing process.
        """
        for epoch in range(1, self._epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics, should_early_stop = self._validate_epoch(epoch)

            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                + f"Val Loss = {val_loss:.4f}, "
                + ", ".join(
                    [f"Val {name} = {value:.4f}" for name, value in val_metrics.items()]
                ),
            )
            self._logger.loss(self._loss.name, train_loss, epoch, DatasetType.TRAIN)
            self._logger.loss(self._loss.name, val_loss, epoch, DatasetType.VALIDATION)
            self._logger.metrics(val_metrics, epoch, DatasetType.VALIDATION)

            if should_early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    + f"No improvement in {self._early_stopping_patience} consecutive "
                    + "full validations.",
                )
                break

        # Load the best model state before testing
        self._load_checkpoint(self._checkpoint_path)
        print(f"Loaded best model from epoch {self._best_epoch}")

        test_loss, test_metrics, test_user_metrics = self._test()

        self._logger.loss(self._loss.name, test_loss, 0, DatasetType.TEST)
        self._logger.metrics(test_metrics, 0, DatasetType.TEST)

        self._artifacts_saver.save_artifacts(
            self._get_hparams(),
            self._model,
            test_loss,
            test_metrics,
            test_user_metrics,
        )

        return {
            "best_metric_name": self._early_stopping_metric,
            "best_val_metric": self._best_val_metric,
            "best_epoch": self._best_epoch,
            "test_metrics": test_metrics,
        }

    def _train_epoch(self, epoch: int) -> float:
        self._model.train()

        total_loss = 0.0
        total_samples = 0

        for user, pos_item, neg_item in tqdm(
            self._train_loader, desc=f"Epoch {epoch} - Training"
        ):
            user = user.to(self._device)  # Dim (n,)
            pos_item = pos_item.to(self._device)  # Dim (n,)
            neg_item = neg_item.squeeze(-1).to(self._device)  # Dim (n,)

            pos_scores = self._model(user, pos_item)  # Dim (n,)
            neg_scores = self._model(user, neg_item)  # Dim (n,)

            loss_value = self._loss(pos_scores, neg_scores)  # Dim (1,)
            batch_size = user.size(0)

            self._optimizer.zero_grad()
            loss_value.backward()
            self._optimizer.step()

            total_loss += loss_value.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def _validate_epoch(self, epoch: int) -> tuple[float, dict[str, float], bool]:
        val_loss, val_metrics, _ = self._evaluate(
            self._val_loader, description=f"Epoch {epoch} - Validation"
        )
        should_early_stop = self._should_early_stop(
            val_metrics[self._early_stopping_metric],
            epoch,
        )
        return val_loss, val_metrics, should_early_stop

    def _test(self) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
        return self._evaluate(self._test_loader, description="Testing")

    @torch.no_grad()
    def _evaluate(
        self, data_loader: DataLoader[TripletSample], description: str
    ) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
        self._model.eval()

        total_loss = 0.0
        user_metrics: dict[str, list[torch.Tensor]] = {
            metric.name: [] for metric in self._metrics
        }
        total_samples = 0

        for user, pos_item, neg_items in tqdm(data_loader, desc=description):
            user = user.to(self._device)  # Dim (n,)
            pos_item = pos_item.to(self._device)  # Dim (n,)
            neg_items = neg_items.to(self._device)  # Dim (n, m)

            pos_scores = self._model(user, pos_item)  # Dim (n,)

            batch_size, num_negatives = neg_items.shape
            user_expanded = (
                user.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
            )  # Dim (n*m,)
            neg_items_flat = neg_items.reshape(-1)  # Dim (n*m,)
            neg_scores_flat = self._model(user_expanded, neg_items_flat)  # Dim (n*m,)
            neg_scores = neg_scores_flat.view(batch_size, num_negatives)  # Dim (n, m)

            # Only use the first negative sample for loss computation
            loss_value = self._loss(pos_scores, neg_scores[:, 0])  # Dim (1,)

            for metric in self._metrics:
                metric_value: torch.Tensor = metric(pos_scores, neg_scores)  # Dim (n,)
                user_metrics[metric.name].append(metric_value)

            total_loss += loss_value.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        user_metrics = {
            name: torch.cat(values, dim=0) for name, values in user_metrics.items()
        }
        avg_metrics = {
            name: values.mean().item() for name, values in user_metrics.items()
        }
        return avg_loss, avg_metrics, user_metrics

    def _should_early_stop(self, current_metric: float, epoch: int) -> bool:
        if self._best_val_metric is None:
            self._best_val_metric = current_metric
            self._best_epoch = epoch
            self._save_checkpoint(self._checkpoint_path)
            return False

        improved = (
            current_metric > self._best_val_metric + self._early_stopping_delta
            if self._maximize_metric
            else current_metric < self._best_val_metric - self._early_stopping_delta
        )

        if improved:
            self._best_val_metric = current_metric
            self._epochs_without_improvement = 0
            self._best_epoch = epoch
            self._save_checkpoint(self._checkpoint_path)
        else:
            if epoch <= self.WARMUP_EPOCHS:
                return False
            self._epochs_without_improvement += 1

        return self._epochs_without_improvement >= self._early_stopping_patience

    def _get_hparams(self) -> dict[str, int | float | str]:
        hparams = {}
        hparams.update(self._model.hparams)
        opt_params = self._optimizer.param_groups[0]
        hparams["lr"] = opt_params["lr"]
        hparams["weight_decay"] = opt_params["weight_decay"]
        hparams["batch_size"] = self._train_loader.batch_size
        return hparams

    def _model_hparams_str(
        self,
    ) -> str:
        hparams = self._get_hparams()
        hparams_str = "_".join([f"{key}-{value}" for key, value in hparams.items()])
        return f"{self._model.name}-{hparams_str}"

    def _save_checkpoint(self, path: str) -> None:
        """Save current model weights to a temporary file (minimal version)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._model.state_dict(), path)

    def _load_checkpoint(self, path: str) -> None:
        """Load model weights from a temporary file if it exists."""
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self._device)
            self._model.load_state_dict(state_dict)
