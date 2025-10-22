from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from src.metrics.metrics import Metric


class RecommenderTrainer:
    def __init__(
        self,
        model: Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        optimizer: Optimizer,
        loss_function: Module,
        val_metrics: list[Metric],
        device: torch.device,
        num_epochs: int,
        writer: SummaryWriter,
    ):
        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.val_metrics = val_metrics
        self.device = device
        self.num_epochs = num_epochs
        self.writer = writer

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            total_loss = 0
            for batch in tqdm(
                self.train_data_loader, desc=f"Training Epoch {epoch}/{self.num_epochs}"
            ):
                loss = self._train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(self.train_data_loader)
            print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Training Loss", avg_loss, epoch)

            self._validate(epoch)

    def _train_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self, epoch):
        self.model.eval()
        total_metrics = {metric.get_name(): 0.0 for metric in self.val_metrics}
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_data_loader, desc="Validating", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                val_loss += loss.item()
                for metric in self.val_metrics:
                    total_metrics[metric.get_name()] += metric.compute(outputs, targets)

        avg_val_loss = val_loss / len(self.val_data_loader)
        self.writer.add_scalar("Validation/Loss", avg_val_loss, epoch)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        for name, total_value in total_metrics.items():
            avg_value = total_value / len(self.val_data_loader)
            print(f"Validation {name}: {avg_value:.4f}")
            self.writer.add_scalar(f"Validation/{name}", avg_value, epoch)
