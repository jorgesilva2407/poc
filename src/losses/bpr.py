import torch
import torch.nn.functional as F
from src.losses.pairwise_losses import PairwiseLoss


class BPRLoss(PairwiseLoss):
    def __init__(self):
        super().__init__("BPR")

    def forward(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
