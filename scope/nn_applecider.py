"""AppleCider-style transformer encoder for irregular-cadence light curves.

Ported from Felipe Fontinele Nunes' AppleCider (SN classification) to the
WISE NEP variable-star classification setting. Three-branch model:

    1. Event transformer    : (B, L, C) event tensor (variable-length,
                              padded), Time2Vec on dt, Transformer-with-CLS
    2. Folded event branch  : same encoder but on a phase-sorted view of
                              the LC (gated by a period-quality scalar so
                              low-confidence periods don't drive the CLS)
    3. Scalar feature MLP   : hand-crafted features (periods, stats, etc.)

Fused to a multi-head sigmoid output — one binary head per SCoPe DNN node.

Event-tensor channels (canonical order)
---------------------------------------
    0: dt          — time since first detection (days)
    1: dt_prev     — gap from previous detection (days)
    2: log_flux    — log of relative flux
    3: log_flux_err— log of the fractional flux error
    4..: band one-hot (2 channels by default for W1/W2;
        configurable via ``n_bands``)

Design notes
------------
* Time2Vec is applied to the ``dt`` channel and *added* to the linear
  projection of the raw event tensor — consistent with AppleCider.
* We use ``batch_first=True`` Transformer layers so sequences have shape
  ``(B, L, d_model)`` end-to-end.
* The CLS token is prepended, and the ``src_key_padding_mask`` is extended
  with a ``False`` at the CLS position so the transformer can attend to it
  from every LC token.
* The folded branch is optional (``use_folded=False`` by default). When
  enabled it is weighted by a sigmoid of ``period_quality`` so aperiodic
  sources effectively bypass it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time2Vec (scalar -> d_model)
# ---------------------------------------------------------------------------
class Time2Vec(nn.Module):
    """Map scalar time t -> d_model-dim feature vector.

    v0 = w0 * t + b0         (linear)
    vi = sin(wi * t + bi)    for i = 1 .. d_model-1  (periodic)
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < 2:
            raise ValueError("Time2Vec requires d_model >= 2")
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (..., L) -> (..., L, d_model)
        v0 = (self.w0 * t + self.b0).unsqueeze(-1)
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([v0, vp], dim=-1)


# ---------------------------------------------------------------------------
# Event-tensor transformer branch
# ---------------------------------------------------------------------------
class EventTransformer(nn.Module):
    """Transformer encoder + CLS token over a padded event tensor."""

    def __init__(
        self,
        n_event_channels: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(n_event_channels, d_model)
        self.time2vec = Time2Vec(d_model)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, events: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Return (B, d_model) CLS vector.

        Parameters
        ----------
        events : (B, L, C)
        pad_mask : (B, L) bool, True at padded positions
        """
        B, _, _ = events.shape
        h = self.in_proj(events) + self.time2vec(events[..., 0])
        cls = self.cls_tok.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        # Extend pad mask by one token for CLS (never padded)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=full_mask)
        return self.norm(h[:, 0])


# ---------------------------------------------------------------------------
# Scalar-feature MLP branch
# ---------------------------------------------------------------------------
class ScalarMLP(nn.Module):
    """Two-layer MLP over hand-crafted features (periods, stats, etc.)."""

    def __init__(self, n_scalar: int, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_scalar, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
        )

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        return self.net(scalars)


# ---------------------------------------------------------------------------
# Full classifier
# ---------------------------------------------------------------------------
class AppleCiderClassifier(nn.Module):
    """AppleCider-style multi-label variable-star classifier.

    Parameters
    ----------
    n_classes : int
        Number of binary outputs (one sigmoid head per SCoPe taxonomy node).
    n_scalar : int
        Dimension of the scalar feature vector.
    n_event_channels : int
        Channels of the event tensor (default 6 for WISE: dt, dt_prev, logf,
        logfe, onehot_W1, onehot_W2).
    d_model, n_heads, n_layers : Transformer hyperparameters.
    dropout : Dropout rate for encoder + MLP.
    use_folded : If True, add a second identical transformer branch for the
        folded event tensor, gated by a scalar ``period_quality`` input.
    """

    def __init__(
        self,
        n_classes: int,
        n_scalar: int,
        n_event_channels: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_folded: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_folded = use_folded

        self.event_enc = EventTransformer(
            n_event_channels, d_model, n_heads, n_layers, dropout,
        )
        self.scalar_mlp = ScalarMLP(n_scalar, d_model, dropout)
        branches = 2  # event + scalar
        if use_folded:
            self.folded_enc = EventTransformer(
                n_event_channels, d_model, n_heads, n_layers, dropout,
            )
            branches = 3

        self.fuse = nn.Sequential(
            nn.Linear(d_model * branches, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute raw logits (B, n_classes).

        Required input keys
        -------------------
        events        : (B, L, C)
        event_mask    : (B, L) bool (True = padded)
        scalars       : (B, F)

        Optional (required if use_folded=True)
        --------------------------------------
        folded_events  : (B, L', C)
        folded_mask    : (B, L') bool
        period_quality : (B,) float in [0, 1]
        """
        event_cls = self.event_enc(inputs["events"], inputs["event_mask"])
        scalar_vec = self.scalar_mlp(inputs["scalars"])

        parts = [event_cls, scalar_vec]
        if self.use_folded:
            folded_cls = self.folded_enc(
                inputs["folded_events"], inputs["folded_mask"],
            )
            # Gate by period quality so aperiodic sources silence this branch
            gate = inputs["period_quality"].view(-1, 1)
            parts.append(folded_cls * gate)

        fused = self.fuse(torch.cat(parts, dim=-1))
        return self.head(fused)  # raw logits — caller applies BCEWithLogits / sigmoid


# ---------------------------------------------------------------------------
# Event-tensor builder (helper)
# ---------------------------------------------------------------------------
def build_event_tensor(
    hjd: "torch.Tensor | np.ndarray",
    mag: "torch.Tensor | np.ndarray",
    magerr: "torch.Tensor | np.ndarray",
    band: "torch.Tensor | np.ndarray",
    n_bands: int = 2,
    time_scale: float = 100.0,
) -> torch.Tensor:
    """Convert a single-source light curve into an (L, 4 + n_bands) event tensor.

    Band IDs are mapped to ``range(n_bands)`` positional one-hot; pass the
    integer ID matching the ordering used in your caller (e.g. 0=W1, 1=W2).

    Returns a float32 tensor.
    """
    hjd = torch.as_tensor(hjd, dtype=torch.float32)
    mag = torch.as_tensor(mag, dtype=torch.float32)
    magerr = torch.as_tensor(magerr, dtype=torch.float32)
    band = torch.as_tensor(band, dtype=torch.long)

    order = torch.argsort(hjd)
    hjd, mag, magerr, band = hjd[order], mag[order], magerr[order], band[order]

    dt = (hjd - hjd[0]) / time_scale
    dt_prev = torch.cat([torch.zeros(1), torch.diff(hjd)]) / time_scale

    # Flux relative to median; propagate magerr -> log flux error
    med = torch.median(mag)
    f = torch.pow(10.0, -0.4 * (mag - med))
    df = 0.4 * float(torch.log(torch.tensor(10.0))) * f * magerr
    log_f = torch.log(f.clamp_min(1e-8))
    log_fe = torch.log((df / f.clamp_min(1e-8)).clamp_min(1e-8))

    onehot = F.one_hot(band.clamp(0, n_bands - 1), num_classes=n_bands).float()

    scalar_channels = torch.stack([dt, dt_prev, log_f, log_fe], dim=-1)
    return torch.cat([scalar_channels, onehot], dim=-1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L, n_scalar = 4, 500, 47
    n_classes = 45
    model = AppleCiderClassifier(
        n_classes=n_classes, n_scalar=n_scalar, d_model=64,
        n_heads=4, n_layers=3, use_folded=False,
    )
    inp = {
        "events": torch.randn(B, L, 6),
        "event_mask": torch.zeros(B, L, dtype=torch.bool),
        "scalars": torch.randn(B, n_scalar),
    }
    out = model(inp)
    print(f"logits: {out.shape}, params: {sum(p.numel() for p in model.parameters()):,}")
