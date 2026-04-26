#!/usr/bin/env python
"""Tests for scope.nn_pytorch (PyTorch port of the SCoPe DNN).

Unit tests (no GPU needed) cover layer arithmetic, forward shapes,
training-loop integration, and layout options.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scope.nn_pytorch import (
    ConvBlock,
    DenseBlock,
    ScopeNet,
    SeparableConv2d,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class TestDenseBlock:
    def test_shape_single(self):
        m = DenseBlock(10, 20, repetitions=1)
        x = torch.randn(3, 10)
        assert m(x).shape == (3, 20)

    def test_shape_repeated(self):
        m = DenseBlock(10, 20, repetitions=3)
        x = torch.randn(3, 10)
        assert m(x).shape == (3, 20)

    def test_relu_nonneg(self):
        m = DenseBlock(10, 20, repetitions=1, activation="relu")
        x = torch.randn(5, 10)
        assert (m(x) >= 0).all()

    def test_sigmoid_range(self):
        m = DenseBlock(10, 20, repetitions=1, activation="sigmoid")
        x = torch.randn(5, 10)
        y = m(x)
        assert ((y >= 0) & (y <= 1)).all()


class TestSeparableConv2d:
    def test_shape_preserves_batch(self):
        m = SeparableConv2d(4, 8, kernel_size=3)
        x = torch.randn(2, 4, 10, 10)
        # 10 - (3-1) = 8 after valid conv
        assert m(x).shape == (2, 8, 8, 8)

    def test_channels_out(self):
        m = SeparableConv2d(4, 32, kernel_size=3)
        x = torch.randn(2, 4, 26, 26)
        assert m(x).shape == (2, 32, 24, 24)


class TestConvBlock:
    def test_maxpool_halves(self):
        m = ConvBlock(in_channels=1, filters=8, kernel_size=3, pool_size=2)
        x = torch.randn(2, 1, 26, 26)
        y = m(x)
        # After conv: 24×24, after pool 2: 12×12
        assert y.shape == (2, 8, 12, 12)

    def test_repetitions_stack(self):
        m = ConvBlock(in_channels=1, filters=8, kernel_size=3, repetitions=2)
        x = torch.randn(2, 1, 30, 30)
        # two 3×3 conv shave 4 each dim, pool halves → (30-4)/2 = 13
        assert m(x).shape == (2, 8, 13, 13)


# ---------------------------------------------------------------------------
# ScopeNet
# ---------------------------------------------------------------------------
class TestScopeNet:
    def test_forward_shape(self):
        m = ScopeNet(n_features=40)
        feats = torch.randn(4, 40)
        dmdt = torch.randn(4, 1, 26, 26)
        y = m({"features": feats, "dmdt": dmdt})
        assert y.shape == (4, 1)
        assert ((y >= 0) & (y <= 1)).all()

    def test_custom_feature_size(self):
        m = ScopeNet(n_features=47)
        feats = torch.randn(3, 47)
        dmdt = torch.randn(3, 1, 26, 26)
        y = m({"features": feats, "dmdt": dmdt})
        assert y.shape == (3, 1)

    def test_channels_last_accepted(self):
        m = ScopeNet(n_features=40, channels_last=True)
        feats = torch.randn(2, 40)
        dmdt_chlast = torch.randn(2, 26, 26, 1)
        y = m({"features": feats, "dmdt": dmdt_chlast})
        assert y.shape == (2, 1)

    def test_dense_only(self):
        m = ScopeNet(n_features=40, dense_branch=True, conv_branch=False)
        feats = torch.randn(4, 40)
        y = m({"features": feats})
        assert y.shape == (4, 1)

    def test_conv_only(self):
        m = ScopeNet(dense_branch=False, conv_branch=True)
        dmdt = torch.randn(4, 1, 26, 26)
        y = m({"dmdt": dmdt})
        assert y.shape == (4, 1)

    def test_both_branches_off_raises(self):
        with pytest.raises(ValueError):
            ScopeNet(dense_branch=False, conv_branch=False)

    def test_param_count_stable(self):
        """Freeze the parameter count so accidental architecture changes are noticed."""
        m = ScopeNet(n_features=40)
        n = sum(p.numel() for p in m.parameters())
        assert n == 20_506  # snapshot: PyTorch port with SeparableConv2d

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        m1 = ScopeNet(n_features=40).eval()  # no dropout randomness
        torch.manual_seed(0)
        m2 = ScopeNet(n_features=40).eval()
        feats = torch.randn(2, 40)
        dmdt = torch.randn(2, 1, 26, 26)
        y1 = m1({"features": feats, "dmdt": dmdt})
        y2 = m2({"features": feats, "dmdt": dmdt})
        assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Training-loop integration (tiny end-to-end)
# ---------------------------------------------------------------------------
class TestTrainingStep:
    def test_loss_decreases_on_trivial_problem(self):
        """Train for a few steps on a linearly-separable problem; loss must drop."""
        torch.manual_seed(42)
        m = ScopeNet(n_features=8, dmdt_shape=(26, 26))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        # Deterministic signal: feature[0] > 0  -> label=1
        N = 128
        feats = torch.randn(N, 8)
        labels = (feats[:, 0] > 0).float().view(-1, 1)
        dmdt = torch.randn(N, 1, 26, 26)

        initial = None
        for step in range(20):
            opt.zero_grad()
            pred = m({"features": feats, "dmdt": dmdt})
            loss = torch.nn.functional.binary_cross_entropy(pred, labels)
            if initial is None:
                initial = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < initial  # non-trivial learning happened


# ---------------------------------------------------------------------------
# Optional: CUDA forward (only if a GPU is free)
# ---------------------------------------------------------------------------
def _cuda_is_free() -> bool:
    if not torch.cuda.is_available():
        return False
    # Probe a tiny allocation; if it fails, some other process is monopolising the GPU
    try:
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_is_free(), reason="CUDA not available or busy")
def test_cuda_forward():
    m = ScopeNet(n_features=40).cuda()
    feats = torch.randn(2, 40, device="cuda")
    dmdt = torch.randn(2, 1, 26, 26, device="cuda")
    y = m({"features": feats, "dmdt": dmdt})
    assert y.device.type == "cuda"
    assert y.shape == (2, 1)
