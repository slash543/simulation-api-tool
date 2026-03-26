"""MLP surrogate model (PyTorch). Architecture is fully config-driven."""
import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

_ACTIVATIONS: dict[str, type] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Fully-connected network with configurable depth, width, and activation."""

    def __init__(self, input_dim: int, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        layer_sizes: list[int] = mcfg["layers"]
        act_cls = _ACTIVATIONS.get(mcfg.get("activation", "relu"), nn.ReLU)
        dropout_p: float = float(mcfg.get("dropout", 0.0))

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        log.info(
            "MLP built — input_dim=%d  layers=%s  activation=%s  dropout=%.2f",
            input_dim, layer_sizes, mcfg.get("activation"), dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
