"""PyTorch port of the SCoPe feature+dmdt DNN.

Layer-for-layer translation of :class:`scope.nn.ScopeNet` (TF/Keras) into
PyTorch, with the same interface:

  model({"features": (B, F), "dmdt": (B, 26, 26, 1)}) -> (B, 1) sigmoid score

Primary use case: retrain the same architecture on the new WISE NEP feature
set without pulling TensorFlow as a dependency, and as a baseline for the
AppleCider-style transformer branch that will follow.

Notes on the faithful port
--------------------------
* Keras ``SeparableConv2D`` = depthwise + pointwise conv with shared bias.
  PyTorch has no built-in SeparableConv2D so we compose nn.Conv2d
  (depthwise, groups=in_channels) → nn.Conv2d (1×1 pointwise).
* ``tf.keras.layers.GlobalAveragePooling2D`` → ``F.adaptive_avg_pool2d(, 1)``.
* Keras default weight init is Glorot/Xavier; PyTorch default is Kaiming.
  We apply Xavier init explicitly for the dense layers so weight-magnitude
  statistics match the TF model on the first forward pass.
* Channels-last (TF) vs channels-first (PyTorch): we expect callers to pass
  dmdt as (B, 1, 26, 26). A ``channels_last=True`` kwarg on the model will
  accept (B, 26, 26, 1) and transpose internally.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Keras-style repeated Dense layer block."""

    def __init__(self, in_features: int, units: int,
                 repetitions: int = 1, activation: str = "relu"):
        super().__init__()
        self.activation_name = activation
        layers = []
        for i in range(repetitions):
            layers.append(nn.Linear(in_features if i == 0 else units, units))
        self.layers = nn.ModuleList(layers)
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(x)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "tanh":
            return torch.tanh(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self._act(layer(x))
        return x


class SeparableConv2d(nn.Module):
    """Keras-style SeparableConv2D = depthwise + pointwise conv."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=0,
                                    groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """Keras-style repeated SeparableConv block + MaxPool."""

    def __init__(self, in_channels: int, filters: int, kernel_size: int,
                 pool_size: int = 2, repetitions: int = 1,
                 activation: str = "relu"):
        super().__init__()
        self.activation_name = activation
        self.convs = nn.ModuleList()
        for i in range(repetitions):
            c_in = in_channels if i == 0 else filters
            self.convs.append(SeparableConv2d(c_in, filters, kernel_size))
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x)) if self.activation_name == "relu" else conv(x)
        return self.pool(x)


class ScopeNet(nn.Module):
    """PyTorch port of :class:`scope.nn.ScopeNet`.

    Parameters
    ----------
    n_features : int, default 40
        Dimension of the scalar feature vector (dense branch input).
    dmdt_shape : tuple, default (26, 26)
        Spatial shape of the dm-dt histogram. 1 channel is assumed.
    dense_branch : bool
    conv_branch : bool
    dropout_rate : float
    channels_last : bool
        If True, accept dmdt as (B, H, W, 1) (TF layout); else (B, 1, H, W).
    """

    def __init__(
        self,
        n_features: int = 40,
        dmdt_shape=(26, 26),
        dense_branch: bool = True,
        conv_branch: bool = True,
        dropout_rate: float = 0.25,
        channels_last: bool = False,
    ):
        super().__init__()
        if not (dense_branch or conv_branch):
            raise ValueError("Model must have at least one branch")
        self.dense_branch = dense_branch
        self.conv_branch = conv_branch
        self.channels_last = channels_last

        # Dense branch
        if dense_branch:
            self.dense_0 = DenseBlock(n_features, 256, repetitions=1)
            self.dense_1 = DenseBlock(256, 32, repetitions=1)
            self.dropout_0 = nn.Dropout(dropout_rate)

        # Conv branch
        if conv_branch:
            self.conv_0 = ConvBlock(in_channels=1, filters=16, kernel_size=3)
            self.conv_1 = ConvBlock(in_channels=16, filters=32, kernel_size=3)
            self.dropout_1 = nn.Dropout(dropout_rate)
            self.dropout_2 = nn.Dropout(dropout_rate)

        # Fuse
        fuse_dim = 0
        if dense_branch:
            fuse_dim += 32
        if conv_branch:
            fuse_dim += 32
        self.dense_2 = DenseBlock(fuse_dim, 16, repetitions=1)
        self.head = nn.Linear(16, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_dense = x_conv = None
        if self.dense_branch:
            feats = inputs["features"]
            x_dense = self.dense_0(feats)
            x_dense = self.dropout_0(x_dense)
            x_dense = self.dense_1(x_dense)
        if self.conv_branch:
            dmdt = inputs["dmdt"]
            if self.channels_last:
                # (B, H, W, 1) -> (B, 1, H, W)
                dmdt = dmdt.permute(0, 3, 1, 2).contiguous()
            x_conv = self.conv_0(dmdt)
            x_conv = self.dropout_1(x_conv)
            x_conv = self.conv_1(x_conv)
            x_conv = self.dropout_2(x_conv)
            x_conv = F.adaptive_avg_pool2d(x_conv, 1).flatten(1)
        if self.dense_branch and self.conv_branch:
            x = torch.cat([x_dense, x_conv], dim=-1)
        elif self.dense_branch:
            x = x_dense
        else:
            x = x_conv
        x = self.dense_2(x)
        return torch.sigmoid(self.head(x))


def load_keras_weights(model: ScopeNet, keras_h5_path: str) -> None:
    """Load weights from a Keras .h5 checkpoint into the PyTorch model.

    Only implemented for the default architecture. Falls back to a clear
    error message if the layer shapes don't match — the user should retrain
    or inspect mismatches.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required to load Keras weights") from e

    raise NotImplementedError(
        "Weight translation from Keras checkpoints is not yet implemented. "
        "For now, retrain from scratch in PyTorch using the same training "
        "data split. (If needed, use `tf2onnx` to export and `onnx2torch` "
        "to import.)"
    )


if __name__ == "__main__":
    # Quick smoke-test with random inputs
    model = ScopeNet(n_features=47, dmdt_shape=(26, 26))
    B = 4
    feats = torch.randn(B, 47)
    dmdt = torch.randn(B, 1, 26, 26)
    out = model({"features": feats, "dmdt": dmdt})
    print("ScopeNet PyTorch forward:", out.shape, out.squeeze().tolist())
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams:,}")
