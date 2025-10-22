import torch
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes the metric value for a batch.
        """
        pass

    def get_name(self) -> str:
        return self.name
