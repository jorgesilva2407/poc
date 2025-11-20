"""Simple lookup encoder for users and items."""

import torch
import torch.nn as nn

from src.models.encoders.interface import Encoder


class LookupEncoder(Encoder):
    """Simple lookup encoder for users and items."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to get embeddings for given indices."""
        return self.embedding(ids)
