"""
Trainer module for training, validating, and testing recommender models.
"""

import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.recommender import Recommender
from src.losses.pairwise_losses import PairwiseLoss
from src.metrics.listwise_metrics import ListwiseMetric
from src.loggers.logger import LoggerBuilder, DatasetType
from datasets.recommendation_dataset import TripletSample
from artifacts_saver.artifacts_saver import ArtifactsSaverBuilder, ArtifactsSaver


class TrainerBuilder:
    """
    Builder for creating a Trainer instance.
    """

    def __init__(self):
        self.required_keys = {
            "model",
            "train_loader",
            "val_loader",
            "test_loader",
            "optimizer",
            "epochs",
            "loss",
            "metrics",
            "early_stopping_metric",
            "early_stopping_delta",
            "early_stopping_patience",
            "maximize_metric",
            "logger_builder",
            "artifacts_saver_builder",
            "device",
        }
        self._config = {}

    def with_model(self, model: Recommender) -> "TrainerBuilder":
        """
        Set the recommender model for the trainer.
        """
        self._config["model"] = model
        return self

    def with_data_loaders(
        self,
        train_loader: DataLoader[TripletSample],
        val_loader: DataLoader[TripletSample],
        test_loader: DataLoader[TripletSample],
    ) -> "TrainerBuilder":
        """
        Set the data loaders for the trainer.
        """
        self._config["train_loader"] = train_loader
        self._config["val_loader"] = val_loader
        self._config["test_loader"] = test_loader
        return self

    def with_optimizer(
        self, optimizer: Optimizer, epochs: int = 50
    ) -> "TrainerBuilder":
        """
        Set the optimizer and number of epochs for the trainer.
        """
        self._config["optimizer"] = optimizer
        self._config["epochs"] = epochs
        return self

    def with_loss(self, loss: PairwiseLoss) -> "TrainerBuilder":
        """
        Set the loss function for the trainer.
        """
        self._config["loss"] = loss
        return self

    def with_metrics(self, metrics: list[ListwiseMetric]) -> "TrainerBuilder":
        """
        Set the evaluation metrics for the trainer.
        """
        self._config["metrics"] = metrics
        return self

    def with_early_stopping(
        self, metric: str, delta: float, maximize: bool, patience: int = 3
    ) -> "TrainerBuilder":
        """
        Set the early stopping criteria for the trainer.
        """
        self._config["early_stopping_metric"] = metric
        self._config["early_stopping_delta"] = delta
        self._config["maximize_metric"] = maximize
        self._config["early_stopping_patience"] = patience
        return self

    def with_logger_builder(self, logger_builder: LoggerBuilder) -> "TrainerBuilder":
        """
        Set the logger builder for the trainer.
        """
        self._config["logger_builder"] = logger_builder
        return self

    def with_artifacts_saver_builder(
        self, artifacts_saver_builder: ArtifactsSaverBuilder
    ) -> "TrainerBuilder":
        """
        Set the artifacts saver builder for the trainer.
        """
        self._config["artifacts_saver_builder"] = artifacts_saver_builder
        return self

    def with_device(self, device: torch.device) -> "TrainerBuilder":
        """
        Set the device for the trainer.
        """
        self._config["device"] = device
        return self

    def build(self) -> "Trainer":
        """
        Build the trainer with the configured parameters.
        """
        missing_keys = self.required_keys - self._config.keys()
        if missing_keys:
            raise ValueError(f"Missing configuration for keys: {missing_keys}")
        return Trainer(**self._config)


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

    model: Recommender
    train_loader: DataLoader[TripletSample]
    val_loader: DataLoader[TripletSample]
    test_loader: DataLoader[TripletSample]
    optimizer: Optimizer
    epochs: int
    loss: PairwiseLoss
    metrics: list[ListwiseMetric]
    early_stopping_metric: str
    early_stopping_delta: float
    early_stopping_patience: int
    maximize_metric: bool
    logger_builder: LoggerBuilder
    artifacts_saver: ArtifactsSaver
    device: torch.device

    epochs_without_improvement: int = 0
    best_val_metric: float | None = None

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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.maximize_metric = maximize_metric

        model_id = self._model_hparams_str()
        self.logger_builder = logger_builder
        self.artifacts_saver = artifacts_saver_builder.build(model_id)

    def run(self):
        """
        Run the training, validation, and testing process.
        """
        with self.logger_builder.build(self._model_hparams_str()) as logger:
            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(epoch)
                val_loss, val_metrics, should_early_stop = self._validate_epoch(epoch)

                logger.loss(self.loss.name, train_loss, epoch, DatasetType.TRAIN)
                logger.loss(self.loss.name, val_loss, epoch, DatasetType.VALIDATION)
                logger.metrics(val_metrics, epoch, DatasetType.VALIDATION)

                if should_early_stop:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        + f"No improvement in {self.early_stopping_patience} consecutive "
                        + "full validations."
                    )
                    break

            test_loss, test_metrics, test_user_metrics = self._test()

            logger.loss(self.loss.name, test_loss, 0, DatasetType.TEST)
            logger.metrics(test_metrics, 0, DatasetType.TEST)

            self.artifacts_saver.save_artifacts(
                self._get_hparams(),
                self.model,
                test_loss,
                test_metrics,
                test_user_metrics,
            )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for user, pos_item, neg_item in tqdm(
            self.train_loader, desc=f"Epoch {epoch} - Training"
        ):
            user = user.to(self.device)  # Dim (n,)
            pos_item = pos_item.to(self.device)  # Dim (n,)
            neg_item = neg_item.to(self.device)  # Dim (n,)

            pos_scores = self.model(user, pos_item)  # Dim (n,)
            neg_scores = self.model(user, neg_item)  # Dim (n,)

            loss_value = self.loss(pos_scores, neg_scores)  # Dim (1,)
            batch_size = user.size(0)

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def _validate_epoch(self, epoch: int) -> tuple[float, dict[str, float], bool]:
        val_loss, val_metrics, _ = self._evaluate(
            self.val_loader, description=f"Epoch {epoch} - Validation"
        )
        should_early_stop = self._should_early_stop(
            val_metrics[self.early_stopping_metric]
        )
        return val_loss, val_metrics, should_early_stop

    def _test(self) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
        return self._evaluate(self.test_loader, description="Testing")

    def _evaluate(
        self, data_loader: DataLoader[TripletSample], description: str
    ) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
        self.model.eval()

        total_loss = 0.0
        user_metrics: dict[str, list[torch.Tensor]] = {
            metric.name: [] for metric in self.metrics
        }
        total_samples = 0

        for user, pos_item, neg_items in tqdm(data_loader, desc=description):
            user = user.to(self.device)  # Dim (n,)
            pos_item = pos_item.to(self.device)  # Dim (n,)
            neg_items = neg_items.to(self.device)  # Dim (n, m)

            pos_scores = self.model(user, pos_item)  # Dim (n,)

            batch_size, num_negatives = neg_items.shape
            user_expanded = (
                user.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
            )  # Dim (n*m,)
            neg_items_flat = neg_items.reshape(-1)  # Dim (n*m,)
            neg_scores_flat = self.model(user_expanded, neg_items_flat)  # Dim (n*m,)
            neg_scores = neg_scores_flat.view(batch_size, num_negatives)  # Dim (n, m)

            # Only use the first negative sample for loss computation
            loss_value = self.loss(pos_scores, neg_scores[:, 0])  # Dim (1,)

            for metric in self.metrics:
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

    def _should_early_stop(self, current_metric: float) -> bool:
        if self.best_val_metric is None:
            self.best_val_metric = current_metric
            return False

        improved = (
            current_metric > self.best_val_metric + self.early_stopping_delta
            if self.maximize_metric
            else current_metric < self.best_val_metric - self.early_stopping_delta
        )

        if improved:
            self.best_val_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.early_stopping_patience

    def _get_hparams(self) -> dict[str, int | float | str]:
        hparams = {}
        hparams.update(self.model.hparams)
        opt_params = self.optimizer.param_groups[0]
        hparams["lr"] = opt_params["lr"]
        hparams["weight_decay"] = opt_params["weight_decay"]
        hparams["batch_size"] = self.train_loader.batch_size
        return hparams

    def _model_hparams_str(
        self,
    ) -> str:
        hparams = self._get_hparams()
        hparams_str = "_".join([f"{key}-{value}" for key, value in hparams.items()])
        return f"{self.model.name}-{hparams_str}"
