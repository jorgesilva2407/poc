"""Interfaces for Graph Collaborative Reasoning (GCR) encoders."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    """
    Interface for entity encoders (User/Item).
    Must implement a forward pass mapping IDs to embeddings and expose embedding_dim.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the size of the output embedding vector."""

    @abstractmethod
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids (torch.Tensor): Tensor of entity IDs.
        Returns:
            torch.Tensor: Tensor of embeddings.
        """
