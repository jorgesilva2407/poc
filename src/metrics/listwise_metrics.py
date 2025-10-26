"""
Abstract base class for listwise metric functions.
"""

from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class ListwiseMetric(ABC, Module):
    """
    Abstract base class for listwise metric functions.
    """

    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """
        Compute the metric.

        Args:
            pos_scores (torch.Tensor): Tensor of shape (n,)
                Predicted scores for positive items.
            neg_scores (torch.Tensor): Tensor of shape (n, m)
                Predicted scores for negative items per positive sample.

        Returns:
            torch.Tensor: Tensor of shape (n,)
                Computed metric.
        """
