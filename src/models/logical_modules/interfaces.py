"""Interfaces for Graph Collaborative Reasoning (GCR) logical modules."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LogicalOR(nn.Module, ABC):
    """
    Interface for Logical OR modules.
    """

    @abstractmethod
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input1 (torch.Tensor): First logical term.
            input2 (torch.Tensor): Second logical term.
        Returns:
            torch.Tensor: Result of OR operation.
        """


class LogicalAND(nn.Module, ABC):
    """
    Interface for Logical AND modules.
    """

    @abstractmethod
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input1 (torch.Tensor): First logical term.
            input2 (torch.Tensor): Second logical term.
        Returns:
            torch.Tensor: Result of AND operation.
        """


class LogicalNOT(nn.Module, ABC):
    """
    Interface for Logical NOT modules.
    """

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Logical term to negate.
        Returns:
            torch.Tensor: Result of NOT operation.
        """
