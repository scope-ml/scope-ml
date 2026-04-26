"""Physical period-range priors for SCoPe variable-star classes.

Each class has a canonical orbital/pulsation period range (in days). During
training we add a penalty term that pushes the predicted probability toward
zero when the source's period falls outside this range. This prevents the
scalar-feature shortcut where a classifier fires "RR Lyrae" on a source with
a 200-day period just because other features looked RR-Lyr-ish.

Ranges are conservative — set slightly wider than strict literature bounds so
edge cases (Blazhko RR Lyr, metal-poor Cepheids, etc.) aren't suppressed.
A range of ``None`` means "no period prior applies" (AGN, flares, YSO, etc.).

Usage
-----
>>> from scope.class_priors import CLASS_PERIOD_RANGES, period_prior_penalty
>>> lo, hi = CLASS_PERIOD_RANGES["rrlyr"]    # (0.15, 1.1)
>>> pen = period_prior_penalty(pred, period, (lo, hi))
>>> loss = bce_loss + 0.5 * pen
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

# (P_min_days, P_max_days) — None means no period prior applies
CLASS_PERIOD_RANGES: Dict[str, Optional[Tuple[float, float]]] = {
    # Eclipsing / rotational binaries
    "wuma":  (0.15, 1.5),    # W UMa contact binaries
    "ew":    (0.15, 1.5),    # Alias of above (contact-binary "EW")
    "ea":    (0.4,  60.0),   # Algol types (EA) can extend to tens of days
    "eb":    (0.3,  60.0),   # β Lyr types
    "e":     (0.15, 60.0),   # Generic eclipsing parent
    "bis":   (0.1,  200.0),  # Generic binary — loose
    "emsms": (0.2,  10.0),   # Detached MS-MS eclipsing
    "el":    (0.1,  30.0),   # Ellipsoidal (tidally distorted rotators)
    # RR Lyrae
    "rrlyr": (0.15, 1.1),    # Parent (RRab + RRc + RRd)
    "rrab":  (0.35, 1.1),    # Fundamental-mode: sawtooth
    "rrc":   (0.15, 0.5),    # First-overtone: sinusoidal
    "rrd":   (0.3,  0.5),    # Double-mode
    "rrblz": (0.35, 1.1),    # Blazhko modulation
    # Pulsators — short period
    "dscu":  (0.02, 0.3),    # δ Scuti: 30 min – 7 hr
    # Pulsators — long period
    "ceph":  (1.0,  100.0),  # Classical Cepheid
    "ceph2": (1.0,  100.0),  # Type II / Anomalous
    "blher": (1.0,  10.0),   # BL Her
    "wvir":  (10.0, 150.0),  # W Virginis
    "lpv":   (50.0, 2000.0), # Generic LPV
    "mir":   (80.0, 2000.0), # Mira
    "srv":   (30.0, 2000.0), # Semi-regular
    "osarg": (10.0, 200.0),  # OGLE Small Amplitude Red Giants
    "saw":   (0.3,  1000.0), # Sawtooth morphology parent
    "sin":   (0.05, 1000.0), # Sinusoidal morphology parent
    "puls":  (0.02, 2000.0), # Generic pulsator — loose (spans δ Scu to Miras)
    # Rotational
    "rscvn": (1.0,  100.0),  # Active binary chromospheric variability
    # Compact / accreting — non-orbital variability common but some periodic
    "cv":    (0.03, 1.5),    # CV orbital (WZ Sge 1.3 h, dwarf novae days)
    "agn":   None,           # Aperiodic — no period prior
    "yso":   None,           # Flaring, irregular, or rotation; many regimes
    "fla":   None,           # Flare star — rotation could be periodic but loose
    "longt": (100.0, 5000.0),# Long-timescale variability
    # Structural / umbrella
    "pnp":   None,           # "Periodic/non-periodic" umbrella — no prior
    "wisepnp": None,         # WISE-observable periodicity (coh_best / coh_agree)
    "vnv":   None,           # "Variable/non-variable" umbrella
    "bogus": None,
    "blend": None,
    "ext":   None,
    "wp":    None,
    "dp":    None,
    "dip":   None,
    "mp":    None,
    "bright": None,
    "mapr": None,
    "i":     None,
    "hp":    None,
    "blyr":  (0.3,  60.0),   # Beta Lyrae subtype of eclipsing
    "wp_dnn": None,
}


def period_prior_penalty(
    pred: torch.Tensor,
    period_days: torch.Tensor,
    period_range: Optional[Tuple[float, float]],
    transition_sharpness: float = 5.0,
) -> torch.Tensor:
    """Soft out-of-range penalty: push ``pred`` toward 0 outside the canonical
    period range.

    Parameters
    ----------
    pred : (B,) or (B, 1)
        Sigmoid probabilities.
    period_days : (B,)
        Source periods in days (positive; replace 0 / NaN with a small
        positive value beforehand).
    period_range : (P_min, P_max) or None
        If ``None``, return a zero-penalty tensor.
    transition_sharpness : float
        Width of the soft-mask transition in log-period units. Default 5 →
        ~20% of a decade.

    Returns
    -------
    scalar loss term ≥ 0
    """
    if period_range is None:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    lo, hi = period_range
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    log_p = torch.log(period_days.clamp_min(1e-6))
    # soft rectangular window [lo, hi] using two logistic gates
    in_range = (
        torch.sigmoid((log_p - log_lo) * transition_sharpness)
        * torch.sigmoid((log_hi - log_p) * transition_sharpness)
    )
    out_of_range = 1.0 - in_range
    # Penalty = mean predicted probability × out-of-range mass
    return (pred.view(-1) * out_of_range.view(-1)).mean()
