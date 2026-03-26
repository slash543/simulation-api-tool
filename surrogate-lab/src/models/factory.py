"""
Model factory — instantiate any registered model by config 'type' string.

To add a new model:
    1. Implement it as an nn.Module in src/models/
    2. Call register_model("MyModel", MyModel) before training
"""
import torch.nn as nn

from src.models.mlp import MLP

_REGISTRY: dict[str, type] = {
    "MLP": MLP,
    # Future hooks:
    # "GNN": GNN,
    # "FNO": FNO,
}


def build_model(input_dim: int, cfg: dict) -> nn.Module:
    model_type: str = cfg["model"]["type"]
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Registered types: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_type](input_dim, cfg)


def register_model(name: str, cls: type) -> None:
    """Register a custom model class so it can be used via config."""
    _REGISTRY[name] = cls
