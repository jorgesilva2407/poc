import torch
from tqdm import tqdm
from pathlib import Path
from torch.optim import Optimizer
from typing import List, Tuple, Dict
from src.models.recommender import Recommender
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from src.losses.pairwise_losses import PairwiseLoss
from src.metrics.listwise_metrics import ListwiseMetrics
from src.datasets.negative_sampling_dataset import TripletSample


class Trainer:
    def __init__(
        self,
        model: Recommender,
        train_loader: DataLoader[TripletSample],
        val_loader: DataLoader[TripletSample],
        test_loader: DataLoader[TripletSample],
        optimizer: Optimizer,
        loss: PairwiseLoss,
        metrics: List[ListwiseMetrics],
        device: torch.device,
        epochs: int = 50,
        base_log_dir: str = "runs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.epochs = epochs
        log_dir = self._build_log_dir(
            base_log_dir,
            model,
            optimizer,
            train_loader.batch_size,
        )
        self.writer = SummaryWriter(log_dir)

    def run(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)

            self.writer.add_scalar(f"Loss/{self.loss.name}/Train", train_loss, epoch)
            self.writer.add_scalar(f"Loss/{self.loss.name}/Val", val_loss, epoch)

            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f"Metric/{metric_name}/Val", metric_value, epoch)

        test_loss, test_metrics = self._test()
        self.writer.add_scalar(f"Loss/{self.loss.name}/Test", test_loss, 0)
        for metric_name, metric_value in test_metrics.items():
            self.writer.add_scalar(f"Metric/{metric_name}/Test", metric_value, 0)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for user, pos_item, neg_item in tqdm(
            self.train_loader, desc=f"Epoch {epoch} - Training"
        ):
            user = user.to(self.device)
            pos_item = pos_item.to(self.device)  # Dim (n,)
            neg_item = neg_item.to(self.device)  # Dim (n,)

            pos_scores = self.model(user, pos_item)  # Dim (n,)
            neg_scores = self.model(user, neg_item)  # Dim (n,)

            loss_value = self.loss(pos_scores, neg_scores)  # Dim (1,)

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        if epoch % 5 == 0:
            return self._evaluate(
                self.val_loader,
                description=f"Epoch {epoch} - Validation (Full)",
            )
        else:
            subset_ratio = 0.1
            subset_loader = self._create_subset_val_loader(subset_ratio)
            return self._evaluate(
                subset_loader,
                description=f"Epoch {epoch} - Validation (Subset {subset_ratio * 100}%)",
            )

    def _test(self) -> Tuple[float, Dict[str, float]]:
        return self._evaluate(self.test_loader, description="Testing")

    def _evaluate(
        self, data_loader: DataLoader[TripletSample], description: str
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()

        total_loss = 0.0
        total_metrics = {metric.name: 0.0 for metric in self.metrics}
        num_batches = 0

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
                metric_value = metric(pos_scores, neg_scores)  # Dim (n,)
                total_metrics[metric.name] += metric_value.mean().item()

            total_loss += loss_value.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_metrics = {
            name: value / num_batches for name, value in total_metrics.items()
        }
        return avg_loss, avg_metrics

    def _build_log_dir(
        self,
        base_dir: str,
        model: Recommender,
        optimizer: Optimizer,
        batch_size: int,
    ) -> str:
        opt_params = optimizer.param_groups[0]
        opt_str = f"lr={opt_params['lr']}-wd={opt_params['weight_decay']}"
        model_hparams = "-".join([f"{k}={v}" for k, v in model.hparams().items()])
        return (
            Path(base_dir) / f"{model.name}-{opt_str}-{model_hparams}-bs={batch_size}"
        )

    def _create_subset_val_loader(
        self, subset_ratio: float
    ) -> DataLoader[TripletSample]:
        dataset = self.val_loader.dataset
        subset_size = int(subset_ratio * len(self.val_loader.dataset))
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        subset = Subset(dataset, subset_indices)
        return DataLoader(subset, batch_size=self.val_loader.batch_size, shuffle=False)
