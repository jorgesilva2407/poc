"""
Abstract base class for pairwise loss functions.
"""

from abc import ABC, abstractmethod
from torch.nn import Module


class PairwiseLoss(ABC, Module):
    """
    Abstract base class for pairwise loss functions.
    """

    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @property
    def name(self) -> str:
        """
        Get the name of the pairwise loss function.

        Returns:
            str: Name of the pairwise loss function.
        """
        return self.name

    @abstractmethod
    def forward(self, pos_scores, neg_scores):
        """
        Compute the pairwise loss.

        Args:
            pos_scores (torch.Tensor): Tensor of shape (n,)
                Predicted scores for positive items.
            neg_scores (torch.Tensor): Tensor of shape (n,)
                Predicted scores for negative items.

        Returns:
            torch.Tensor: Scalar tensor (0-dimensional)
                Computed pairwise loss.
        """
