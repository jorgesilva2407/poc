"""Neural OR logical module implementation."""

import torch
import torch.nn as nn


class NeuralOR(nn.Module):
    """Neural OR logical module."""

    def __init__(self, event_embedding_dim: int, hidden_dim: int):
        super(NeuralOR, self).__init__()
        self.or_network = nn.Sequential(
            nn.Linear(2 * event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_embedding_dim),
        )

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Neural OR module."""
        combined_input = torch.cat((inputs1, inputs2), dim=-1)
        output = self.or_network(combined_input)
        return output
