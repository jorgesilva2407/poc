"""Neural OR logical module implementation."""

import torch
import torch.nn as nn

from src.models.logical_modules.interfaces import LogicalOR, NeuralLogicalModule


class NeuralOR(LogicalOR, NeuralLogicalModule):
    """Neural OR logical module."""

    def __init__(self, event_embedding_dim: int, hidden_dim: int, dropout_rate: float):
        super(NeuralOR, self).__init__()
        self.or_network = nn.Sequential(
            nn.Linear(2 * event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, event_embedding_dim),
        )
        self._init_linear_weights()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Neural OR module."""
        combined_input = torch.cat((input1, input2), dim=-1)
        output = self.or_network(combined_input)
        return output
