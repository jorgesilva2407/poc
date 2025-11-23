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


class FuzzySmoothOR(LogicalOR):
    """
    Smooth Logic Disjunction (OR) using LogSumExp.
    Formula: (1/k) * ln(exp(k*x) + exp(k*y))

    Attributes:
        k (nn.Parameter): Stiffness parameter.
                          High k (>20) behaves like Max (prevents saturation).
                          Low k (<5) behaves like Sum (improves gradients).
    """

    def __init__(self, stiffness: float):
        super().__init__()
        self.k = stiffness

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # torch.logaddexp is numerically stable (avoids overflow for high k)
        # Formula: (1/k) * logaddexp(k*a, k*b)
        return (1.0 / self.k) * torch.logaddexp(self.k * input1, self.k * input2)
