import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from google.cloud import storage
from torch.optim import Optimizer
from src.models.recommender import Recommender
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from src.losses.pairwise_losses import PairwiseLoss
from src.metrics.listwise_metrics import ListwiseMetrics
from src.datasets.negative_sampling_dataset import TripletSample


class Trainer:
    TENSORBOARD_LOG_DIR = os.getenv("TENSORBOARD_LOG_DIR", "runs")
    LOCAL_ARTIFACT_SAVE_DIR = os.getenv("LOCAL_ARTIFACT_SAVE_DIR", "artifacts")
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "my-bucket")
    GCP_BLOB_BASE_PATH = os.getenv("GCP_BLOB_BASE_PATH", "artifacts")
    IS_RUNNING_ON_CLOUD = os.getenv("IS_RUNNING_ON_CLOUD", "false").lower() == "true"

    model: Recommender
    train_loader: DataLoader[TripletSample]
    val_loader: DataLoader[TripletSample]
    test_loader: DataLoader[TripletSample]
    optimizer: Optimizer
    loss: PairwiseLoss
    metrics: list[ListwiseMetrics]
    device: torch.device

    epochs: int = 50
    validation_subset_ratio: float = 0.1
    early_stopping_patience: int = 3
    epochs_without_improvement: int = 0
    best_val_metric: float = None

    def __init__(
        self,
        model: Recommender,
        train_loader: DataLoader[TripletSample],
        val_loader: DataLoader[TripletSample],
        test_loader: DataLoader[TripletSample],
        optimizer: Optimizer,
        loss: PairwiseLoss,
        metrics: list[ListwiseMetrics],
        early_stopping_metric: str,
        early_stopping_delta: float,
        maximize_metric: bool,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.subset_val_loader = self._create_subset_val_loader(
            self.validation_subset_ratio
        )
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_delta = early_stopping_delta
        self.maximize_metric = maximize_metric

        model_dir = self._build_model_dir()
        self.writer = SummaryWriter(Path(self._get_tb_log_dir()) / model_dir)
        self.local_artifacts_path = Path(self.LOCAL_ARTIFACT_SAVE_DIR) / model_dir
        self.bucket = storage.Client().bucket(self.GCP_BUCKET_NAME)
        self.gcloud_artifacts_path = Path(self.GCP_BLOB_BASE_PATH) / model_dir

    def _get_tb_log_dir(self) -> Path:
        if self.IS_RUNNING_ON_CLOUD:
            return Path(f"gs://{self.GCP_BUCKET_NAME}") / self.TENSORBOARD_LOG_DIR
        else:
            return Path(self.TENSORBOARD_LOG_DIR)

    def run(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics, _, should_early_stop = self._validate_epoch(epoch)

            self.writer.add_scalar(f"Loss/{self.loss.name}/Train", train_loss, epoch)
            self.writer.add_scalar(f"Loss/{self.loss.name}/Val", val_loss, epoch)

            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f"Metric/{metric_name}/Val", metric_value, epoch)

            if should_early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch}. No improvement in {self.early_stopping_patience} consecutive full validations."
                )
                break

        test_loss, test_metrics, test_user_metrics = self._test()

        self.writer.add_scalar(f"Loss/{self.loss.name}/Test", test_loss, 0)
        for metric_name, metric_value in test_metrics.items():
            self.writer.add_scalar(f"Metric/{metric_name}/Test", metric_value, 0)

        self._save_artifacts(test_loss, test_metrics, test_user_metrics)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for user, pos_item, neg_item in tqdm(
            self.train_loader, desc=f"Epoch {epoch} - Training"
        ):
            user = user.to(self.device)
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
        if epoch % 5 == 0:
            val_loss, val_metrics, _ = self._evaluate(
                self.val_loader,
                description=f"Epoch {epoch} - Validation (Full)",
            )
            should_early_stop = self._should_early_stop(
                val_metrics[self.early_stopping_metric]
            )
            return val_loss, val_metrics, should_early_stop
        else:
            val_loss, val_metrics, _ = self._evaluate(
                self.subset_val_loader,
                description=f"Epoch {epoch} - Validation (Subset {self.validation_subset_ratio * 100}%)",
            )
            return val_loss, val_metrics, False

    def _test(self) -> tuple[float, dict[str, float]]:
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

    def _build_model_dir(
        self,
    ) -> str:
        hparams = self._get_hparams()
        hparams_str = "_".join([f"{key}-{value}" for key, value in hparams.items()])
        return f"{self.model.name}-{hparams_str}"

    def _create_subset_val_loader(
        self, subset_ratio: float
    ) -> DataLoader[TripletSample]:
        dataset = self.val_loader.dataset
        subset_size = int(subset_ratio * len(self.val_loader.dataset))
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        subset = Subset(dataset, subset_indices)
        return DataLoader(subset, batch_size=self.val_loader.batch_size, shuffle=False)

    def _save_artifacts(
        self,
        avg_loss: float,
        avg_metrics: dict[str, float],
        user_metrics: dict[str, torch.Tensor],
    ) -> None:
        print("Saving artifacts to cloud storage...")
        self._save_model()
        self._save_metrics(avg_loss, avg_metrics)
        self._save_user_metrics(user_metrics)
        print("Artifacts saved successfully.")

    def _save_model(self) -> None:
        model_save_path = self.local_artifacts_path / "model.pth"
        self.local_artifacts_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        if self.IS_RUNNING_ON_CLOUD:
            self._save_on_cloud(
                model_save_path, self.gcloud_artifacts_path / "model.pth"
            )

    def _save_metrics(self, avg_loss: float, avg_metrics: dict[str, float]) -> None:
        result = {}
        result["hparams"] = self._get_hparams()
        result["loss"] = avg_loss
        result["metrics"] = avg_metrics
        metrics_save_path = self.local_artifacts_path / "metrics.json"
        self.local_artifacts_path.mkdir(parents=True, exist_ok=True)
        with open(metrics_save_path, "w") as f:
            json.dump(result, f, indent=2)
        if self.IS_RUNNING_ON_CLOUD:
            self._save_on_cloud(
                metrics_save_path, self.gcloud_artifacts_path / "metrics.json"
            )

    def _save_user_metrics(self, user_metrics: dict[str, torch.Tensor]) -> None:
        metrics_artifact_path = self.local_artifacts_path / "user_metrics"
        metrics_artifact_path.mkdir(parents=True, exist_ok=True)
        for metric_name, values in user_metrics.items():
            metric_save_path = metrics_artifact_path / f"{metric_name}.pt"
            torch.save(values.cpu(), metric_save_path)
            if self.IS_RUNNING_ON_CLOUD:
                self._save_on_cloud(
                    metric_save_path,
                    self.gcloud_artifacts_path / "user_metrics" / f"{metric_name}.pt",
                )

    def _save_on_cloud(self, local_path: Path, gcloud_path: Path) -> None:
        blob = self.bucket.blob(str(gcloud_path))
        blob.upload_from_filename(str(local_path))
