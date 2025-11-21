"""Fuzzy Logic modules for Graph Collaborative Reasoning."""

import torch

from src.models.logical_modules.interfaces import LogicalOR, LogicalNOT


class FuzzyNOT(LogicalNOT):
    """
    Standard Fuzzy Negation: N(x) = 1 - x
    Used by Product, Gödel, and Łukasiewicz systems.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Ensure constraints (numerical stability)
        return 1.0 - input_tensor


class FuzzyProductOR(LogicalOR):
    """
    Product Logic Disjunction (OR).
    Formula: x + y - x * y.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return input1 + input2 - (input1 * input2)


class FuzzyGodelOR(LogicalOR):
    """
    Gödel Logic Disjunction (OR).
    Formula: max(x, y).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return torch.max(input1, input2)


class FuzzyLukasiewiczOR(LogicalOR):
    """
    Łukasiewicz Logic Disjunction (OR).
    Formula: min(x + y, 1).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return torch.min(input1 + input2, torch.tensor(1.0, device=input1.device))
