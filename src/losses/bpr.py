"""
Bayesian Personalized Ranking (BPR) loss implementation.
"""

import torch
import torch.nn.functional as F

from src.losses.pairwise_losses import PairwiseLoss


class BPRLoss(PairwiseLoss):
    """
    Bayesian Personalized Ranking (BPR) loss for pairwise ranking.
    """

    def __init__(self):
        super().__init__("BPR")

    def forward(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        # pylint: disable=not-callable
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        # pylint: enable=not-callable
