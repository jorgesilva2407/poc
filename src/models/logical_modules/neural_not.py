"""Neural NOT logical module implementation."""

import torch
import torch.nn as nn

from src.models.logical_modules.interfaces import LogicalNOT, NeuralLogicalModule


class NeuralNOT(LogicalNOT, NeuralLogicalModule):
    """Neural NOT logical module."""

    def __init__(self, event_embedding_dim: int, hidden_dim: int):
        super(NeuralNOT, self).__init__()
        self.not_network = nn.Sequential(
            nn.Linear(event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_embedding_dim),
        )
        self._init_linear_weights()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Neural NOT module."""
        output = self.not_network(input_tensor)
        return output
