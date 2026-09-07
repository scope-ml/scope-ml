#!/usr/bin/env python3
"""Fourier re-ranking of trial periods for RR Lyrae and Cepheids.

The eclipsing binary model in eclipsing.py is the wrong shape for a pulsator:
RR Lyrae and Cepheids are sawtooth, not two dips in a flat line.  Gaia handles
them in a separate pipeline (SOS Cep&RRL) with a truncated Fourier series, and
the same split applies here.

The measured case for doing this: the correct period is in our stored candidate
list for 98.8 per cent of RR Lyrae and 92.3 per cent of Cepheids, while the
pipeline currently reports it for 38.9 per cent and 0 per cent.  As with the
eclipsing binaries, the selector is the bottleneck rather than the search.

Model at trial period P, per band b, for harmonic order K:

    m_b(phi) = a_b0 + sum_{k=1..K} [ a_bk cos(2 pi k phi) + b_bk sin(2 pi k phi) ]

This is entirely linear, so unlike the eclipse fit there is no optimiser at all
and no starting values to get wrong.  Each band gets its own coefficients since
pulsation amplitude is strongly chromatic, while the period is shared.

The order K is chosen by BIC alongside the period, which is what distinguishes
a sawtooth (needing several harmonics) from a sinusoid (needing one).  Fits are
unweighted, for the same reason as the eclipse model: the Rubin uncertainties
are unreliable.
"""
import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco if not args or not callable(args[0]) else args[0]


MIN_PER_BAND = 25
# Gaia SOS Cep&RRL validity ranges (Clementini et al. 2023, Sect. 3.2).  A
# wrong period throws a star off the RR Lyrae locus in these planes, which is
# what selects the period there -- goodness of fit alone does not.
PHI21_RANGE = (3.0, 5.8)  # rad, for models with at least 2 harmonics
PHI31_RANGE = (0.6, 5.1)  # rad, for models with at least 3 harmonics
MIN_AMPLITUDE = 0.1  # mag, peak to peak
MIN_COVERAGE = 0.6
MAX_ORDER = 5
BIC_WINDOW = 30.0
RIDGE = 1e-10


@njit(cache=True, fastmath=True)
def _rss_fourier(phi, y, offs, order):
    """Unweighted residual sum for a Fourier series of the given order.

    Two passes per band: accumulate the normal equations without building the
    design matrix, solve, then accumulate residuals.  The design has 1 + 2*order
    columns, at most eleven, so the solve is cheap.
    """
    k = 1 + 2 * order
    G = np.zeros((k, k))
    b = np.zeros(k)
    col = np.zeros(k)
    total = 0.0
    for bi in range(offs.size - 1):
        s = offs[bi]
        e = offs[bi + 1]
        if e - s <= k:
            continue
        for a in range(k):
            b[a] = 0.0
            for c in range(k):
                G[a, c] = 0.0
        for i in range(s, e):
            p = 2.0 * np.pi * phi[i]
            col[0] = 1.0
            for h in range(1, order + 1):
                col[2 * h - 1] = np.cos(h * p)
                col[2 * h] = np.sin(h * p)
            yi = y[i]
            for a in range(k):
                ca = col[a]
                b[a] += ca * yi
                for c in range(a, k):
                    G[a, c] += ca * col[c]
        for a in range(k):
            G[a, a] += RIDGE
            for c in range(a):
                G[a, c] = G[c, a]
        beta = np.linalg.solve(G, b)
        for i in range(s, e):
            p = 2.0 * np.pi * phi[i]
            col[0] = 1.0
            for h in range(1, order + 1):
                col[2 * h - 1] = np.cos(h * p)
                col[2 * h] = np.sin(h * p)
            m = 0.0
            for a in range(k):
                m += beta[a] * col[a]
            r = y[i] - m
            total += r * r
    return total


def _fourier_params(phi, y, order):
    """Fourier parameters of one band: amplitude ratios and phase differences.

    Gaia validate an RR Lyrae period by where it falls in the phi21-P and
    phi31-P planes rather than by fit quality, so these are the quantities that
    actually do the selecting.
    """
    cols = [np.ones_like(phi)]
    for h in range(1, order + 1):
        cols.append(np.cos(2 * np.pi * h * phi))
        cols.append(np.sin(2 * np.pi * h * phi))
    X = np.column_stack(cols)
    try:
        beta = np.linalg.solve(X.T @ X + 1e-10 * np.eye(X.shape[1]), X.T @ y)
    except np.linalg.LinAlgError:
        return None
    A, psi = [], []
    for h in range(1, order + 1):
        a, b = beta[2 * h - 1], beta[2 * h]
        A.append(float(np.hypot(a, b)))
        psi.append(float(np.arctan2(-b, a)))
    grid = np.linspace(0, 1, 200)
    m = np.full_like(grid, beta[0])
    for h in range(1, order + 1):
        m = (
            m
            + beta[2 * h - 1] * np.cos(2 * np.pi * h * grid)
            + beta[2 * h] * np.sin(2 * np.pi * h * grid)
        )
    out = {
        "amp": float(m.max() - m.min()),
        "R21": np.nan,
        "R31": np.nan,
        "phi21": np.nan,
        "phi31": np.nan,
    }
    if order >= 2 and A[0] > 0:
        out["R21"] = A[1] / A[0]
        out["phi21"] = float((psi[1] - 2 * psi[0]) % (2 * np.pi))
    if order >= 3 and A[0] > 0:
        out["R31"] = A[2] / A[0]
        out["phi31"] = float((psi[2] - 3 * psi[0]) % (2 * np.pi))
    return out


def _on_rrl_locus(c):
    """Gaia validity checks; a period failing these is rejected outright."""
    if not np.isfinite(c.get("amp", np.nan)) or c["amp"] < MIN_AMPLITUDE:
        return False
    if c["order"] >= 3 and np.isfinite(c["phi31"]):
        if not (PHI31_RANGE[0] <= c["phi31"] <= PHI31_RANGE[1]):
            return False
    if c["order"] >= 2 and np.isfinite(c["phi21"]):
        if not (PHI21_RANGE[0] <= c["phi21"] <= PHI21_RANGE[1]):
            return False
    return True


def _bic(rss, npar, npts):
    if rss <= 0 or npts <= 0:
        return np.inf
    return npts * np.log(rss / npts) + npar * np.log(npts)


def _coverage(phi):
    return len(np.unique(np.clip((phi * 10).astype(int), 0, 9))) / 10.0


def _pack(t, y, band, period):
    phi_all = np.mod(t / period, 1.0)
    if _coverage(phi_all) < MIN_COVERAGE:
        return None
    order_idx, offs, tot = [], [0], 0
    for b in np.unique(band):
        k = np.flatnonzero(band == b)
        if k.size < MIN_PER_BAND:
            continue
        order_idx.append(k)
        tot += k.size
        offs.append(tot)
    if not order_idx:
        return None
    idx = np.concatenate(order_idx)
    return (
        np.ascontiguousarray(phi_all[idx]),
        np.ascontiguousarray(np.asarray(y, float)[idx]),
        np.asarray(offs, np.int64),
        int(tot),
        len(order_idx),
    )


def fit_period_models(t, y, band, period):
    """Fit Fourier series of every order at one trial period."""
    packed = _pack(t, y, band, period)
    if packed is None:
        return None
    phi, yy, offs, npts, nband = packed

    # reference band for the Fourier diagnostics: the best sampled one,
    # standing in for Gaia G
    iref = int(np.argmax(np.diff(offs)))

    chi_c = 0.0
    for bi in range(nband):
        seg = yy[offs[bi] : offs[bi + 1]]
        chi_c += float(np.sum((seg - seg.mean()) ** 2))

    cands = []
    for order in range(1, MAX_ORDER + 1):
        rss = float(_rss_fourier(phi, yy, offs, order))
        if not np.isfinite(rss) or rss <= 0:
            continue
        npar = (1 + 2 * order) * nband
        rec = {
            "name": f"FOURIER{order}",
            "order": order,
            "period": period,
            "chi2": rss,
            "bic": _bic(rss, npar, npts),
            "fvu": rss / chi_c if chi_c > 0 else np.nan,
        }
        ref = _fourier_params(
            phi[offs[iref] : offs[iref + 1]], yy[offs[iref] : offs[iref + 1]], order
        )
        rec.update(
            ref
            if ref
            else {
                "amp": np.nan,
                "R21": np.nan,
                "R31": np.nan,
                "phi21": np.nan,
                "phi31": np.nan,
            }
        )
        rec["on_locus"] = _on_rrl_locus(rec)
        cands.append(rec)
    return cands


def select(cands):
    """Lowest BIC wins.

    No model-rank prior here.  For eclipsing binaries Gaia deliberately prefers
    a two-eclipse interpretation, which resolves P against 2P; a pulsator has no
    equivalent ambiguity, so the fit is left to decide on its own.
    """
    cands = [c for c in cands if np.isfinite(c["bic"])]
    if not cands:
        return None
    # Gaia select on the Fourier locus first and use fit quality only to break
    # ties among survivors.  Fall back to plain BIC if nothing is on the locus,
    # flagging it so the caller knows the period was not validated.
    ok = [c for c in cands if c.get("on_locus")]
    validated = bool(ok)
    best = min(ok if ok else cands, key=lambda c: c["bic"])
    best["validated"] = validated
    best["n_cands"] = len(cands)
    best["n_on_locus"] = len(ok)
    return best


def global_ranking(fvu):
    if not np.isfinite(fvu) or fvu <= 0:
        return np.nan
    return 0.11 * (3.45 - np.log10(fvu))


# The post-selection stages live in a separate module; re-exported so a
# registry entry can name one object per class.
from .pulsating_sos import characterise, fourier_params  # noqa: E402,F401
