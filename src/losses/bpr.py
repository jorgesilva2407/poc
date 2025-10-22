import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        """
        Compute the Bayesian Personalized Ranking (BPR) loss.

        Args:
            pos_scores (torch.Tensor): Predicted scores for positive items.
            neg_scores (torch.Tensor): Predicted scores for negative items.

        Returns:
            torch.Tensor: Computed BPR loss.
        """
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
