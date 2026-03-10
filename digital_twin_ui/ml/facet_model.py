"""
Multi-output MLP for per-facet pressure and area prediction.

Input:  [facet_id_normalized, speed_mm_s]  (2 features)
Output: [contact_pressure, facet_area]     (2 targets)

Usage::

    from digital_twin_ui.ml.facet_model import FacetPressureMLP

    model = FacetPressureMLP(hidden_dims=[64, 128, 64])
    x = torch.tensor([[0.5, 5.0]])   # normalized facet_id=0.5, speed=5.0
    y = model(x)                     # shape (1, 2): [pressure, area]
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FacetPressureMLP(nn.Module):
    """
    Multi-layer perceptron for per-facet (contact_pressure, facet_area) prediction.

    Args:
        input_dim: Number of input features (default 2: facet_id_norm + speed).
        hidden_dims: Sequence of hidden-layer widths.
        output_dim: Number of output values (default 2: pressure + area).
        dropout: Dropout probability (0 = disabled).
        batch_norm: Whether to add BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 128, 256, 128, 64),
        output_dim: int = 2,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        in_feat = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_feat, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_feat = h
        layers.append(nn.Linear(in_feat, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
