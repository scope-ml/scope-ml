#!/usr/bin/env python3
"""Two-Gaussian eclipsing binary model, following Mowlavi et al. (2023).

Gaia fit the single G band; this fits all Rubin bands at once.  Eclipse timing
and duration are achromatic, so the Gaussian centres and widths are shared
across bands, while the depths are free per band because the two stars differ
in colour.

The residual evaluation and the small linear solve are compiled together with
numba: the cost is spread across some thousands of residual callbacks per
object rather than concentrated in one hotspot, which is the case compilation
addresses and micro-optimisation does not.

Falls back to pure numpy if numba is unavailable, so the module still imports.
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


NPHASE_BIN = 40
MIN_PER_BAND = 25
MAX_SIGMA = 0.4
MIN_SEP = 0.08
MIN_COVERAGE = 0.6
MIN_IN_ECLIPSE = 3
BIC_WINDOW = 30.0
#: above this the eclipse depths are not separately identifiable and are left
#: unreported rather than fabricated
MAX_CONDITION = 1e6
RIDGE = 1e-10

MODELS = (
    ("TWOGAUSSIANS", 6, 2, -1),
    ("TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1", 5, 2, 0),
    ("TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE2", 4, 2, 1),
    ("ONEGAUSSIAN", 3, 1, -1),
    ("ONEGAUSSIAN_WITH_ELLIPSOIDAL", 2, 1, 0),
    ("ELLIPSOIDAL", 1, 0, -2),  # -2: cosine, free centre
)


@njit(cache=True, fastmath=True)
def _resid_all(phi, y, offs, cs, ws, ng, mu, has_cos, res):
    """Residuals for every band, solving each band's linear part exactly.

    Two passes per band: the first accumulates the Gram matrix and right hand
    side without ever materialising the design matrix, the second writes the
    residuals.  k is at most four, so the solve is trivial.
    """
    k = 1 + ng + has_cos
    G = np.zeros((k, k))
    b = np.zeros(k)
    col = np.zeros(k)
    for bi in range(offs.size - 1):
        s = offs[bi]
        e = offs[bi + 1]
        for a in range(k):
            b[a] = 0.0
            for c in range(k):
                G[a, c] = 0.0
        for i in range(s, e):
            p = phi[i]
            col[0] = 1.0
            for g in range(ng):
                d = abs(p - cs[g])
                if d > 0.5:
                    d = 1.0 - d
                d = d / ws[g]
                col[1 + g] = np.exp(-0.5 * d * d)
            if has_cos == 1:
                col[k - 1] = np.cos(4.0 * np.pi * (p - mu))
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
            p = phi[i]
            col[0] = 1.0
            for g in range(ng):
                d = abs(p - cs[g])
                if d > 0.5:
                    d = 1.0 - d
                d = d / ws[g]
                col[1 + g] = np.exp(-0.5 * d * d)
            if has_cos == 1:
                col[k - 1] = np.cos(4.0 * np.pi * (p - mu))
            m = 0.0
            for a in range(k):
                m += beta[a] * col[a]
            res[i] = y[i] - m
    return res


@njit(cache=True, fastmath=True)
def _lm(phi, y, offs, p0, ng, cos_on, res, maxit):
    """Levenberg-Marquardt on the nonlinear parameters only.

    The linear parameters are eliminated inside _resid_all at every step, so
    this searches at most four dimensions: the eclipse centres and widths.
    """
    npar = p0.size
    p = p0.copy()
    cs = np.zeros(2)
    ws = np.ones(2) * 0.05
    lam = 1e-3

    def unpack(pp, cs, ws):
        if ng == 0:
            return pp[0]
        for g in range(ng):
            c = pp[2 * g] % 1.0
            if c < 0.0:
                c += 1.0
            cs[g] = c
            w = abs(pp[2 * g + 1])
            if w < 0.005:
                w = 0.005
            if w > MAX_SIGMA:
                w = MAX_SIGMA
            ws[g] = w
        if cos_on >= 0:
            return cs[cos_on]
        return 0.0

    has_cos = 1 if cos_on != -1 else 0
    mu = unpack(p, cs, ws)
    _resid_all(phi, y, offs, cs, ws, ng, mu, has_cos, res)
    f0 = 0.0
    for i in range(res.size):
        f0 += res[i] * res[i]

    J = np.zeros((res.size, npar))
    trial = np.zeros(npar)
    rt = np.zeros(res.size)
    for _ in range(maxit):
        for a in range(npar):  # forward differences
            h = 1e-5 * (abs(p[a]) + 1e-3)
            for c in range(npar):
                trial[c] = p[c]
            trial[a] = p[a] + h
            mu = unpack(trial, cs, ws)
            _resid_all(phi, y, offs, cs, ws, ng, mu, has_cos, rt)
            for i in range(res.size):
                J[i, a] = (rt[i] - res[i]) / h
        A = J.T @ J
        g = J.T @ res
        improved = False
        for _try in range(8):
            for a in range(npar):
                A[a, a] += lam * (A[a, a] + 1e-12)
            step = np.linalg.solve(A, -g)
            for a in range(npar):
                A[a, a] /= 1.0 + lam
                trial[a] = p[a] + step[a]
            mu = unpack(trial, cs, ws)
            _resid_all(phi, y, offs, cs, ws, ng, mu, has_cos, rt)
            f1 = 0.0
            for i in range(rt.size):
                f1 += rt[i] * rt[i]
            if f1 < f0:
                for a in range(npar):
                    p[a] = trial[a]
                for i in range(res.size):
                    res[i] = rt[i]
                if f0 - f1 < 1e-10 * (f0 + 1e-30):
                    f0 = f1
                    improved = False
                    break
                f0 = f1
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam = min(lam * 10.0, 1e9)
        if not improved:
            break
    return p, f0


def _wrapped_gauss(phi, c, w):
    d = np.abs(phi - c)
    d = np.minimum(d, 1.0 - d)
    return np.exp(-0.5 * (d / w) ** 2)


def _sep(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _bic(rss, npar, npts):
    if rss <= 0 or npts <= 0:
        return np.inf
    return npts * np.log(rss / npts) + npar * np.log(npts)


def _coverage(phi):
    return len(np.unique(np.clip((phi * 10).astype(int), 0, 9))) / 10.0


def _n_in_eclipse(phi_all, centre, sigma):
    half = min(5.6 * sigma, MAX_SIGMA) / 2.0
    d = np.abs(phi_all - centre)
    d = np.minimum(d, 1.0 - d)
    return int(np.sum(d <= half))


def _depths(phi, y, offs, band_names, centres, widths, n_gauss, cos_on):
    """Eclipse depths per band for the fitted geometry.

    The fit eliminates the linear parameters at every step and keeps only the
    residual, since that is all selection needs.  They are worth recovering
    once at the end: eclipse depth ratio and width are what separate the
    eclipsing subtypes, and depths in several bands constrain the temperature
    ratio of the two stars, which a single-band survey cannot measure.
    """
    if n_gauss == 0:
        return {}
    k = 1 + n_gauss + (1 if cos_on >= 0 else 0)
    mu = centres[cos_on] if cos_on >= 0 else None
    out = {}
    for bi, name in enumerate(band_names):
        s, e = int(offs[bi]), int(offs[bi + 1])
        if e - s <= k:
            continue
        cols = [np.ones(e - s)]
        for c, w in zip(centres, widths):
            cols.append(_wrapped_gauss(phi[s:e], c, w))
        if mu is not None:
            cols.append(np.cos(4 * np.pi * (phi[s:e] - mu)))
        X = np.column_stack(cols)
        # When two eclipses overlap, their Gaussian columns become nearly
        # collinear and the individual depths stop being identifiable: the fit
        # is fine, but the split of the total dip between the two components is
        # not determined.  Solving anyway returns huge cancelling coefficients
        # -- depths of 1e11 magnitudes.  Report nothing rather than a number
        # that looks like a measurement, and truncate small singular values for
        # the merely awkward cases.
        # A band only constrains a depth if it actually observed that eclipse.
        # _physical checks in-eclipse coverage across all bands together, so an
        # individual band can contribute no in-eclipse points at all, leaving
        # its Gaussian column nearly empty and its depth free to take any
        # value: that is where depths of thousands of magnitudes come from.
        covered = True
        for c, w in zip(centres, widths):
            off = np.abs(phi[s:e] - c)
            off = np.minimum(off, 1.0 - off)
            if int(np.sum(off <= 2.8 * w)) < MIN_IN_ECLIPSE:
                covered = False
                break
        if not covered:
            continue
        sv = np.linalg.svd(X, compute_uv=False)
        if sv[-1] <= 0 or sv[0] / sv[-1] > MAX_CONDITION:
            continue
        beta, *_ = np.linalg.lstsq(X, y[s:e], rcond=1e-8)
        out[name] = [float(v) for v in beta[1 : 1 + n_gauss]]
    return out


def _physical(n_gauss, cs, ws, phi_all):
    if n_gauss == 2 and _sep(cs[0], cs[1]) < MIN_SEP:
        return False
    if n_gauss == 1 and ws[0] > MAX_SIGMA:
        return False
    for c, w in zip(cs[:n_gauss], ws[:n_gauss]):
        if _n_in_eclipse(phi_all, c, w) < MIN_IN_ECLIPSE:
            return False
    return True


def fit_period_models(t, y, band, period):
    """Fit every geometry at one trial period; return the candidate list."""
    phi_all = np.mod(t / period, 1.0)
    if _coverage(phi_all) < MIN_COVERAGE:
        return None

    order, offs, tot, kept_bands = [], [0], 0, []
    for b in np.unique(band):
        k = np.flatnonzero(band == b)
        if k.size < MIN_PER_BAND:
            continue
        order.append(k)
        kept_bands.append(str(b))
        tot += k.size
        offs.append(tot)
    if not order:
        return None
    idx = np.concatenate(order)
    phi = np.ascontiguousarray(phi_all[idx])
    yy = np.ascontiguousarray(np.asarray(y, float)[idx])
    offs = np.asarray(offs, np.int64)
    npts = int(tot)
    nband = len(order)

    edges = np.linspace(0, 1, NPHASE_BIN + 1)
    bidx = np.clip(np.digitize(phi_all, edges) - 1, 0, NPHASE_BIN - 1)
    prof = np.array(
        [
            np.median(y[bidx == i]) if (bidx == i).any() else np.nan
            for i in range(NPHASE_BIN)
        ]
    )
    if np.all(np.isnan(prof)):
        return None
    rank = np.argsort(np.where(np.isnan(prof), -np.inf, prof))[::-1]
    c1 = (rank[0] + 0.5) / NPHASE_BIN
    c2 = (c1 + 0.5) % 1.0
    for j in rank[1:]:
        cand = (j + 0.5) / NPHASE_BIN
        if _sep(cand, c1) >= MIN_SEP:
            c2 = cand
            break

    chi_c = 0.0
    for bi in range(nband):
        seg = yy[offs[bi] : offs[bi + 1]]
        chi_c += float(np.sum((seg - seg.mean()) ** 2))

    cands = [
        {
            "name": "CONSTANT",
            "rank": 0,
            "chi2": chi_c,
            "period": period,
            "bic": _bic(chi_c, nband, npts),
            "info": {},
        }
    ]

    res = np.empty(npts)
    for name, mrank, ng, cos_on in MODELS:
        if ng == 0:
            p0 = np.array([c1])
            co = -2
        else:
            p0 = np.array([c1, 0.05] if ng == 1 else [c1, 0.05, c2, 0.05])
            co = cos_on
        try:
            p, rss = _lm(phi, yy, offs, p0, ng, co, res, 40)
        except Exception:
            continue
        if not np.isfinite(rss):
            continue
        if ng == 0:
            cs, ws = [], []
            ncol = 2 * nband
            npar = 1 + ncol
        else:
            cs = [float(p[2 * g] % 1.0) for g in range(ng)]
            ws = [
                float(min(max(abs(p[2 * g + 1]), 0.005), MAX_SIGMA)) for g in range(ng)
            ]
            if not _physical(ng, cs, ws, phi_all):
                continue
            ncol = (1 + ng + (1 if co >= 0 else 0)) * nband
            npar = 2 * ng + ncol
        cands.append(
            {
                "name": name,
                "rank": mrank,
                "chi2": float(rss),
                "period": period,
                "bic": _bic(rss, npar, npts),
                "info": {
                    "centres": cs,
                    "widths": ws,
                    "depths": _depths(phi, yy, offs, kept_bands, cs, ws, ng, co),
                    "separation": (_sep(cs[0], cs[1]) if ng == 2 else np.nan),
                },
            }
        )

    for c in cands:
        c["fvu"] = c["chi2"] / chi_c if chi_c > 0 else np.nan
    return cands


def select(cands):
    """Gaia's choice over models pooled from every trial period."""
    cands = [c for c in cands if np.isfinite(c["bic"])]
    if not cands:
        return None
    best_bic = min(c["bic"] for c in cands)
    within = [
        c
        for c in cands
        if c["bic"] <= best_bic + BIC_WINDOW and c["name"] != "CONSTANT"
    ]
    if not within:
        return None
    chosen = max(within, key=lambda c: (c["rank"], -c["bic"]))
    chosen["best_bic_overall"] = best_bic
    chosen["n_within"] = len(within)
    return chosen


def global_ranking(fvu):
    if not np.isfinite(fvu) or fvu <= 0:
        return np.nan
    return 0.11 * (3.45 - np.log10(fvu))
