"""Tests for src/models/mlp.py and src/models/factory.py"""
import pytest
import torch
import torch.nn as nn

from src.models.factory import _REGISTRY, build_model, register_model
from src.models.mlp import MLP


class TestMLP:
    def test_forward_output_shape(self, minimal_cfg):
        model = MLP(input_dim=5, cfg=minimal_cfg)
        x = torch.randn(16, 5)
        out = model(x)
        assert out.shape == (16,)

    def test_single_sample(self, minimal_cfg):
        model = MLP(input_dim=5, cfg=minimal_cfg)
        x = torch.randn(1, 5)
        out = model(x)
        assert out.shape == (1,)

    def test_custom_layers(self, minimal_cfg):
        minimal_cfg["model"]["layers"] = [64, 32, 16]
        model = MLP(input_dim=5, cfg=minimal_cfg)
        x = torch.randn(8, 5)
        assert model(x).shape == (8,)

    def test_single_layer(self, minimal_cfg):
        minimal_cfg["model"]["layers"] = [32]
        model = MLP(input_dim=5, cfg=minimal_cfg)
        assert model(torch.randn(4, 5)).shape == (4,)

    def test_dropout_included(self, minimal_cfg):
        minimal_cfg["model"]["dropout"] = 0.5
        model = MLP(input_dim=5, cfg=minimal_cfg)
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.net.modules())
        assert has_dropout

    def test_no_dropout_when_zero(self, minimal_cfg):
        minimal_cfg["model"]["dropout"] = 0.0
        model = MLP(input_dim=5, cfg=minimal_cfg)
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.net.modules())
        assert not has_dropout

    @pytest.mark.parametrize("activation", ["relu", "tanh", "elu", "gelu"])
    def test_supported_activations(self, activation, minimal_cfg):
        minimal_cfg["model"]["activation"] = activation
        model = MLP(input_dim=5, cfg=minimal_cfg)
        out = model(torch.randn(4, 5))
        assert out.shape == (4,)

    def test_output_is_float(self, minimal_cfg):
        model = MLP(input_dim=5, cfg=minimal_cfg)
        out = model(torch.randn(8, 5))
        assert out.dtype == torch.float32

    def test_gradients_flow(self, minimal_cfg):
        model = MLP(input_dim=5, cfg=minimal_cfg)
        x = torch.randn(8, 5)
        y = torch.randn(8)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_different_input_dims(self, minimal_cfg):
        for dim in [1, 3, 10, 50]:
            model = MLP(input_dim=dim, cfg=minimal_cfg)
            assert model(torch.randn(4, dim)).shape == (4,)


class TestFactory:
    def test_build_mlp(self, minimal_cfg):
        model = build_model(input_dim=5, cfg=minimal_cfg)
        assert isinstance(model, MLP)

    def test_unknown_type_raises(self, minimal_cfg):
        minimal_cfg["model"]["type"] = "UnknownModel"
        with pytest.raises(ValueError, match="UnknownModel"):
            build_model(input_dim=5, cfg=minimal_cfg)

    def test_register_and_use_custom_model(self, minimal_cfg):
        class DummyModel(nn.Module):
            def __init__(self, input_dim, cfg):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return self.linear(x).squeeze(-1)

        register_model("Dummy", DummyModel)
        minimal_cfg["model"]["type"] = "Dummy"
        model = build_model(input_dim=5, cfg=minimal_cfg)
        assert isinstance(model, DummyModel)

        # Cleanup
        del _REGISTRY["Dummy"]

    def test_registered_model_is_callable(self, minimal_cfg):
        model = build_model(input_dim=5, cfg=minimal_cfg)
        out = model(torch.randn(4, 5))
        assert out.shape == (4,)
