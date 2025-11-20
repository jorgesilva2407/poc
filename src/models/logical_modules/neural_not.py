"""Neural NOT logical module implementation."""

import torch
import torch.nn as nn


class NeuralNOT(nn.Module):
    """Neural NOT logical module."""

    def __init__(self, event_embedding_dim: int, hidden_dim: int):
        super(NeuralNOT, self).__init__()
        self.not_network = nn.Sequential(
            nn.Linear(event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Neural NOT module."""
        output = self.not_network(inputs)
        return output
