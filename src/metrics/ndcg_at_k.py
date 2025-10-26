"""
Normalized Discounted Cumulative Gain at rank k (NDCG@k) metric implementation.
"""

import torch

from src.metrics.listwise_metrics import ListwiseMetric


class NDCGAtK(ListwiseMetric):
    """
    Normalized Discounted Cumulative Gain at rank k (NDCG@k) metric for one positive item.

    Args:
        k (int): The rank at which to compute the NDCG.
    """

    def __init__(self, k: int):
        super().__init__(f"NDCG@{k}")
        self.k = k

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        Compute the NDCG@k metric.

        Args:
            pos_scores (torch.Tensor): Tensor of shape (n,)
                Predicted scores for positive items.
            neg_scores (torch.Tensor): Tensor of shape (n, m)
                Predicted scores for negative items per positive sample.

        Returns:
            torch.Tensor: Tensor of shape (n,)
                Computed NDCG@k metric.
        """
        higher_counts = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
        ranks = higher_counts + 1  # Rank is number of higher scores + 1
        ndcgs = torch.where(
            ranks <= self.k,
            1.0 / torch.log2(ranks.float() + 1),
            torch.zeros_like(ranks, dtype=torch.float),
        )
        return ndcgs
