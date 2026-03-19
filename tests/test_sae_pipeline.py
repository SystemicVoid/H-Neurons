"""Regression tests for the SAE extraction and intervention helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_activations import select_token_activations
from intervene_sae import SAEFeatureScaler
from spike_sae_feasibility import inspect_sae


class TestSelectTokenActivations:
    def test_answer_tokens_returns_answer_slice(self):
        activations = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        regions = {
            "input": (0, 2),
            "output": (2, 6),
            "answer_tokens": (3, 5),
        }

        selected = select_token_activations(activations, "answer_tokens", regions)

        assert selected is not None
        assert selected.squeeze(-1).tolist() == [[3.0, 4.0]]

    def test_all_except_answer_tokens_returns_complement(self):
        activations = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        regions = {
            "input": (0, 2),
            "output": (2, 6),
            "answer_tokens": (2, 4),
        }

        selected = select_token_activations(
            activations, "all_except_answer_tokens", regions
        )

        assert selected is not None
        assert selected.squeeze(-1).tolist() == [[0.0, 1.0, 4.0, 5.0]]

    def test_all_except_answer_tokens_returns_none_without_answer_span(self):
        activations = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        regions = {
            "input": (0, 1),
            "output": (1, 4),
            "answer_tokens": None,
        }

        selected = select_token_activations(
            activations, "all_except_answer_tokens", regions
        )

        assert selected is None


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_feedforward_layernorm = torch.nn.Identity()

    def forward(self, x):
        return self.post_feedforward_layernorm(x)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])

    def forward(self, x):
        return self.layers[0](x)


class FakeSAE:
    def __init__(self):
        self.device = "cpu"
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, h):
        self.encode_calls += 1
        return h.clone()

    def decode(self, features):
        self.decode_calls += 1
        return features.clone()


class TestSAEFeatureScaler:
    def test_alpha_one_is_exact_noop(self):
        model = FakeModel()
        sae = FakeSAE()
        scaler = SAEFeatureScaler(
            model=model,
            saes={0: sae},
            target_features={0: [1]},
            device="cpu",
        )
        scaler.alpha = 1.0

        x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        y = model(x)

        assert torch.equal(y, x)
        assert sae.encode_calls == 0
        assert sae.decode_calls == 0

    def test_non_control_alpha_scales_only_target_feature(self):
        model = FakeModel()
        sae = FakeSAE()
        scaler = SAEFeatureScaler(
            model=model,
            saes={0: sae},
            target_features={0: [1]},
            device="cpu",
        )
        scaler.alpha = 2.0

        x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        y = model(x)

        assert torch.equal(y, torch.tensor([[[1.0, 4.0, 3.0]]], dtype=torch.float32))
        assert sae.encode_calls == 1
        assert sae.decode_calls == 1


class FakeCfg:
    def to_dict(self):
        return {"d_in": 4, "d_sae": 6, "hook_name": "post_feedforward_layernorm"}


class FakePretrainedSAE:
    def __init__(self):
        self.cfg = FakeCfg()
        self.W_enc = torch.zeros(4, 6)
        self.W_dec = torch.zeros(6, 4)
        self.b_enc = torch.zeros(6)
        self.b_dec = torch.zeros(4)

    def encode(self, x):
        return torch.zeros(x.shape[0], x.shape[1], 6, device=x.device)

    def decode(self, features):
        return torch.zeros(
            features.shape[0], features.shape[1], 4, device=features.device
        )


class FakeSAEClass:
    @staticmethod
    def from_pretrained(**kwargs):
        return FakePretrainedSAE()


class TestSpikeSaeFeasibility:
    def test_inspect_sae_supports_single_object_api(self, monkeypatch):
        fake_module = types.ModuleType("sae_lens")
        fake_module.SAE = FakeSAEClass
        monkeypatch.setitem(sys.modules, "sae_lens", fake_module)

        sae, cfg = inspect_sae("release", "layer_0_width_16k_l0_small")

        assert isinstance(sae, FakePretrainedSAE)
        assert cfg["d_in"] == 4
        assert cfg["d_sae"] == 6
