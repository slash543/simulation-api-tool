"""
PyTorch MLP model for predicting contact pressure from insertion speed.

Architecture
------------
A simple fully-connected network with configurable hidden layers, ReLU
activations, and optional Batch Normalisation.

Input:  1 feature  — insertion speed (mm/s)
Output: 1 value    — predicted peak contact pressure (kPa)

Usage::

    from digital_twin_ui.ml.model import PressureMLP

    model = PressureMLP(hidden_dims=[64, 128, 64])
    y = model(torch.tensor([[5.0]]))   # shape (1, 1)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PressureMLP(nn.Module):
    """
    Multi-layer perceptron for contact pressure prediction.

    Args:
        input_dim: Number of input features (default 1 — speed only).
        hidden_dims: Sequence of hidden-layer widths.
        output_dim: Number of output values (default 1 — max pressure).
        dropout: Dropout probability applied after each hidden layer (0 = off).
        batch_norm: Whether to insert BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (64, 128, 256),
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        in_features = input_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden

        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
