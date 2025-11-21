"""Initialization strategies for different network components."""

import torch.nn as nn


def initialize_kaiming_relu(m):
    """
    Kaiming Normal initialization for ReLU networks.
    Maintains variance for 'relu' non-linearity.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_xavier_logic(m):
    """
    Xavier Uniform initialization.
    Ideal for Logical Modules to prevent early saturation of gates,
    keeping gradients flowing during the initial phases.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
