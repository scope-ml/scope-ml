#!/usr/bin/env python3
"""SOS Cep&RRL stages after period selection (Clementini et al. 2023).

pulsating.py picks a period from the stored candidates using the Fourier locus.
This adds the stages Gaia run afterwards, in their order:

  NonLinearFourierAnalysis   refine the period itself, not just choose one
  FourierDecomposition       R21, R31, phi21, phi31 on the reference band
  Mode identification        RRab against RRc from the period-amplitude plane
  SecondaryPeriodicities     double mode (RRd) search in the residuals
  Bootstrap errors           robust uncertainties by resampling
  Stellar parameters         [Fe/H] from phi31 for RRab

Two of their stages are deliberately absent: the MORO/ROFABO outlier operators,
and anything using RVS radial velocities, which Rubin does not have.

The refinement and the bootstrap are compiled with numba.  Between them they
evaluate the model some tens of thousands of times per star -- a grid of trial
periods, repeated for every resample -- which is far too many round trips
through the interpreter otherwise.  Everything else runs once per star and is
left in numpy.

Where a published calibration is needed the source is named in the code.  The
mode-separation boundary is the weakest part: the paper defines it graphically
(their Fig. 3) and gives only the segment Amp(G) <= 0.64 mag for P <= 0.34 d,
so the rest is a straight-line approximation and is flagged as such.
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


try:
    from astropy.timeseries import LombScargle

    HAVE_LS = True
except Exception:  # pragma: no cover
    HAVE_LS = False

N_BOOTSTRAP = 25  # Gaia use 100; fewer here, it only sets the error
RRD_MIN_EPOCHS = 40  # their double-mode module activates above this
RRD_MIN_RMS = 0.05  # mag, residual rms needed to look for a second period
RIDGE = 1e-10


@njit(cache=True, fastmath=True)
def _rss(t, y, offs, period, order):
    """Residual sum for a Fourier series at one period, summed over bands.

    Two passes per band: accumulate the normal equations without building the
    design matrix, solve, then accumulate residuals.
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
            ph = t[i] / period
            ph = ph - np.floor(ph)
            p = 2.0 * np.pi * ph
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
            ph = t[i] / period
            ph = ph - np.floor(ph)
            p = 2.0 * np.pi * ph
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


@njit(cache=True, fastmath=True)
def _refine(t, y, offs, period, order, frac, n, rounds):
    """Bracketed search over the period, narrowing each round.

    The Fourier coefficients are linear given the period, so only the period is
    searched here; that is equivalent to Gaia's joint Levenberg-Marquardt over
    period and coefficients, and cannot diverge.
    """
    best_p = period
    best_r = _rss(t, y, offs, period, order)
    half = frac * period
    step = 0.0
    rm = 0.0
    rp = 0.0
    for _ in range(rounds):
        lo = best_p - half
        hi = best_p + half
        if lo <= 0.0:
            lo = 1e-6
        step = (hi - lo) / (n - 1)
        for j in range(n):
            p = lo + step * j
            if p <= 0.0:
                continue
            r = _rss(t, y, offs, p, order)
            if r < best_r:
                best_r = r
                best_p = p
        half = half / (n // 4)
    # Parabolic interpolation through the three points around the minimum.
    # Without this the answer is always a grid node, so every bootstrap
    # resample returns an identical period and the error collapses to zero.
    if step > 0.0:
        rm = _rss(t, y, offs, best_p - step, order)
        rp = _rss(t, y, offs, best_p + step, order)
        den = rm - 2.0 * best_r + rp
        if den > 0.0:
            shift = 0.5 * (rm - rp) / den
            if -1.0 < shift < 1.0:
                best_p = best_p + shift * step
                best_r = _rss(t, y, offs, best_p, order)
    return best_p, best_r


@njit(cache=True, fastmath=True)
def _bootstrap(t, y, offs, period, order, n, seed):
    """Refined period for each resample, drawn within bands.

    Resampling inside each band keeps the band structure intact, so the fit
    always has every band it started with.
    """
    np.random.seed(seed)
    N = t.size
    out = np.empty(n)
    tb = np.empty(N)
    yb = np.empty(N)
    for it in range(n):
        for bi in range(offs.size - 1):
            s = offs[bi]
            e = offs[bi + 1]
            m = e - s
            for i in range(s, e):
                j = s + int(np.random.random() * m)
                if j >= e:
                    j = e - 1
                tb[i] = t[j]
                yb[i] = y[j]
        p, _ = _refine(tb, yb, offs, period, order, 0.01, 21, 2)
        out[it] = p
    return out


def _pack(t, y, band):
    """Sort by band and return the segment offsets the kernels expect."""
    order_idx, offs, tot = [], [0], 0
    for b in np.unique(band):
        k = np.flatnonzero(band == b)
        if k.size < 5:
            continue
        order_idx.append(k)
        tot += k.size
        offs.append(tot)
    if not order_idx:
        return None
    idx = np.concatenate(order_idx)
    return (
        np.ascontiguousarray(np.asarray(t, float)[idx]),
        np.ascontiguousarray(np.asarray(y, float)[idx]),
        np.asarray(offs, np.int64),
        idx,
    )


def _design(phi, order):
    cols = [np.ones_like(phi)]
    for h in range(1, order + 1):
        cols.append(np.cos(2 * np.pi * h * phi))
        cols.append(np.sin(2 * np.pi * h * phi))
    return np.column_stack(cols)


def fourier_params(t, y, band, period, order):
    """R21, R31, phi21, phi31 and amplitude on the best sampled band."""
    bands, counts = np.unique(band, return_counts=True)
    ref = bands[np.argmax(counts)]
    k = band == ref
    if k.sum() <= 2 * order + 1:
        return None
    phi = np.mod(np.asarray(t, float)[k] / period, 1.0)
    X = _design(phi, order)
    try:
        beta = np.linalg.solve(
            X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ np.asarray(y, float)[k]
        )
    except np.linalg.LinAlgError:
        return None
    A = [float(np.hypot(beta[2 * h - 1], beta[2 * h])) for h in range(1, order + 1)]
    psi = [
        float(np.arctan2(-beta[2 * h], beta[2 * h - 1])) for h in range(1, order + 1)
    ]
    g = np.linspace(0, 1, 400)
    m = np.full_like(g, beta[0])
    for h in range(1, order + 1):
        m += beta[2 * h - 1] * np.cos(2 * np.pi * h * g) + beta[2 * h] * np.sin(
            2 * np.pi * h * g
        )
    out = {
        "amp": float(m.max() - m.min()),
        "ref_band": str(ref),
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


def identify_mode(period, amp):
    """RRab against RRc from the period-amplitude plane.

    APPROXIMATE.  Clementini et al. define the boundary graphically (Fig. 3) and
    quote only the segment Amp(G) <= 0.64 mag for P <= 0.34 d.  The rest is a
    straight line through the gap between the two clumps in that figure, so
    treat borderline cases as unclassified rather than trusting the label.
    """
    if not np.isfinite(amp) or not np.isfinite(period):
        return "UNKNOWN"
    if period <= 0.34:
        return "RRc" if amp <= 0.64 else "RRab"
    if period > 0.5:
        return "RRab"
    boundary = 0.64 - 1.6 * (period - 0.34)
    return "RRc" if amp <= max(boundary, 0.15) else "RRab"


def secondary_period(t, y, band, period, order, pmin=0.15, pmax=1.5):
    """Search the residuals for a second periodicity, the RRd test."""
    if not HAVE_LS:
        return None
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    res = np.zeros_like(y)
    n_used = 0
    for b in np.unique(band):
        k = band == b
        if k.sum() <= 2 * order + 1:
            continue
        X = _design(np.mod(t[k] / period, 1.0), order)
        try:
            beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ y[k])
        except np.linalg.LinAlgError:
            continue
        res[k] = y[k] - X @ beta
        n_used += int(k.sum())
    m = res != 0
    rms = float(np.sqrt(np.mean(res[m] ** 2))) if n_used else np.nan
    if n_used < RRD_MIN_EPOCHS or not np.isfinite(rms) or rms < RRD_MIN_RMS:
        return {"p2": np.nan, "rms": rms, "activated": False}
    freq = np.linspace(1.0 / pmax, 1.0 / pmin, 20000)
    try:
        power = LombScargle(t[m], res[m]).power(freq)
    except Exception:
        return {"p2": np.nan, "rms": rms, "activated": False}
    p2 = float(1.0 / freq[int(np.argmax(power))])
    return {"p2": p2, "rms": rms, "activated": True, "p2_power": float(np.max(power))}


# Metallicity is deliberately NOT computed.  Gaia derive [Fe/H] from P and phi31
# using the RRab and RRc relations of Nemec et al. (2013), but Clementini et al.
# only cite that paper rather than reproducing the coefficients, and guessing a
# published calibration produces confident nonsense.  phi31 is reported here, so
# the relation can be applied later by whoever has the reference to hand.


def characterise(t, y, band, period, order, do_bootstrap=True, seed=0):
    """Run the whole post-selection chain on one star."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    band = np.asarray(band)
    packed = _pack(t, y, band)
    if packed is None:
        return None
    tp, yp, offs, _ = packed

    p_ref, rss = _refine(tp, yp, offs, float(period), int(order), 0.02, 41, 3)
    out = {
        "period_selected": float(period),
        "period_refined": float(p_ref),
        "order": int(order),
        "rss": float(rss),
    }

    fp = fourier_params(t, y, band, p_ref, order)
    out.update(
        fp
        if fp
        else {
            "amp": np.nan,
            "R21": np.nan,
            "R31": np.nan,
            "phi21": np.nan,
            "phi31": np.nan,
            "ref_band": "",
        }
    )

    out["mode"] = identify_mode(p_ref, out["amp"])

    sec = secondary_period(t, y, band, p_ref, order)
    if sec:
        out["p2"] = sec["p2"]
        out["resid_rms"] = sec["rms"]
        out["is_rrd"] = bool(
            sec["activated"]
            and np.isfinite(sec["p2"])
            and 0.70 < (p_ref / sec["p2"]) < 0.80
        )
    else:
        out["p2"] = np.nan
        out["resid_rms"] = np.nan
        out["is_rrd"] = False

    if do_bootstrap:
        ps = _bootstrap(tp, yp, offs, float(p_ref), int(order), N_BOOTSTRAP, seed)
        ps = ps[np.isfinite(ps)]
        out["period_err"] = (
            1.486 * float(np.median(np.abs(ps - np.median(ps)))) if ps.size else np.nan
        )
    else:
        out["period_err"] = np.nan
    out["phi31_err"] = np.nan
    return out
