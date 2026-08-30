#!/usr/bin/env python3
"""
Damped-random-walk (DRW / Ornstein-Uhlenbeck) features for AGN-like
stochastic variability.

Fits a celerite Gaussian process with a RealTerm (the DRW kernel) plus a
JitterTerm (extra white noise absorbing underestimated photometric errors)
by maximizing the marginal likelihood with L-BFGS-B. Standardized one-step-
ahead residuals under the fitted OU model provide an anomaly statistic:

    drw_tau          damping timescale (days)
    drw_sigma        asymptotic rms variability amplitude (mag)
    drw_max_z        max |standardized one-step-ahead residual|
    drw_fit_success  1 if the optimizer converged, else 0

An optional iterative sigma-clipping scheme (max_iter > 1) refits after
masking points with |z| > z_thr so that flares do not contaminate their own
baseline. The default is a single plain fit (max_iter = 1). Configurable via
config.yaml under feature_generation.drw.

Note: sigma here is the asymptotic (process) rms; the structure-function
amplitude commonly reported in the AGN literature is SF_inf = sqrt(2) * sigma.
"""

import numpy as np
from scipy.optimize import minimize

import celerite
from celerite import terms

DEFAULT_MIN_EPOCHS = 15
DEFAULT_MAX_ITER = 1
DEFAULT_Z_THR = 3.0

# Bounds on the (natural) log-parameters of the kernel
LOG_A_BOUNDS = (np.log(1e-8), np.log(1e2))  # process variance (mag^2)
LOG_C_BOUNDS = (np.log(1e-5), np.log(1e2))  # inverse timescale (1/day)
LOG_JITTER_BOUNDS = (np.log(1e-6), np.log(1e1))  # extra white noise (mag)

#: tau longer than this multiple of the baseline is treated as unresolved.
#: 1.0 is the weakest defensible cut - a timescale longer than the observing
#: window cannot be measured at all. The AGN literature is stricter (unbiased
#: tau recovery is usually quoted as needing baseline > ~10 tau), so 0.1 is the
#: conservative choice if you care about tau as a physical quantity rather than
#: as a nuisance parameter.
DEFAULT_TAU_MAX_BASELINE_FRAC = 1.0

DRW_FEATURE_KEYS = ['drw_tau', 'drw_sigma', 'drw_max_z', 'drw_fit_success']


def _fit_is_constrained(tau, sigma, t, tau_max_baseline_frac=None):
    """Did the data actually determine tau and sigma, or did the fit run to a bound?

    ``scipy``'s convergence flag says only that the minimiser terminated; it
    reports success just as happily when the optimum sits against a parameter
    bound. On a 257-day Rubin DP2 baseline that happens for most objects: tau
    piles up at both rails (0.01 d and 1e5 d) and sigma collapses onto its
    floor while the jitter term absorbs the variance. Those fits converged,
    but their parameters carry no information, and anything downstream that
    filters on ``drw_fit_success`` would otherwise treat them as trustworthy.

    Unconstrained when any of:

    * tau exceeds the baseline (the window cannot show the turnover),
    * tau falls below the median sampling interval (nothing resolves it),
    * sigma sits on its lower bound (the DRW term has no amplitude left).

    Note this is a statement about *identifiability*, not fit quality - a
    rail value is still meaningful as a limit, so tau and sigma are returned
    unchanged and only the flag is lowered.
    """
    if not (np.isfinite(tau) and np.isfinite(sigma)) or tau <= 0:
        return False

    frac = (
        DEFAULT_TAU_MAX_BASELINE_FRAC
        if tau_max_baseline_frac is None
        else tau_max_baseline_frac
    )

    baseline = float(t[-1] - t[0])
    if baseline <= 0 or tau > frac * baseline:
        return False

    if len(t) > 1:
        cadence = float(np.median(np.diff(t)))
        if cadence > 0 and tau < cadence:
            return False

    sigma_floor = np.sqrt(np.exp(LOG_A_BOUNDS[0]))
    if sigma <= sigma_floor * 1.01:
        return False

    return True


def _nan_stats():
    stats = {key: np.nan for key in DRW_FEATURE_KEYS}
    stats['drw_fit_success'] = 0
    return stats


def _neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    try:
        return -gp.log_likelihood(y)
    except celerite.solver.LinAlgError:
        return 1e25


def fit_drw(t, y, yerr):
    """Fit a DRW (RealTerm) + jitter GP to a single light curve.

    :param t: times (days), sorted ascending
    :param y: magnitudes
    :param yerr: magnitude errors

    :return: (tau, sigma, mu, jitter, success)
        tau: damping timescale (days)
        sigma: asymptotic rms amplitude (mag)
        mu: fitted mean magnitude
        jitter: fitted excess white-noise amplitude (mag)
        success: bool, optimizer convergence
    """
    var = np.var(y)
    baseline = t[-1] - t[0]

    log_a_init = np.log(max(var, 1e-8))
    log_c_init = np.log(5.0 / max(baseline, 1.0))
    log_jitter_init = np.log(max(np.median(yerr), 1e-4))

    kernel = terms.RealTerm(
        log_a=np.clip(log_a_init, *LOG_A_BOUNDS),
        log_c=np.clip(log_c_init, *LOG_C_BOUNDS),
        bounds=dict(log_a=LOG_A_BOUNDS, log_c=LOG_C_BOUNDS),
    ) + terms.JitterTerm(
        log_sigma=np.clip(log_jitter_init, *LOG_JITTER_BOUNDS),
        bounds=dict(log_sigma=LOG_JITTER_BOUNDS),
    )

    gp = celerite.GP(kernel, mean=np.median(y), fit_mean=True)
    gp.compute(t, yerr)

    # Numerical gradients: celerite's analytic gradient requires autograd
    r = minimize(
        _neg_log_like,
        gp.get_parameter_vector(),
        method='L-BFGS-B',
        bounds=gp.get_parameter_bounds(),
        args=(y, gp),
    )
    gp.set_parameter_vector(r.x)

    params = gp.get_parameter_dict()
    tau = 1.0 / np.exp(params['kernel:terms[0]:log_c'])
    sigma = np.sqrt(np.exp(params['kernel:terms[0]:log_a']))
    jitter = np.exp(params['kernel:terms[1]:log_sigma'])
    mu = params['mean:value']

    return tau, sigma, mu, jitter, bool(r.success)


def ou_resid(t, y, yerr, tau, sigma, mu, jitter=0.0):
    """Standardized one-step-ahead residuals under an OU model.

    The prediction for epoch i conditions on the previous observed epoch:
        m_i = mu + (y_{i-1} - mu) * exp(-dt/tau)
        v_i = sigma^2 * (1 - exp(-2 dt/tau))
        z_i = (y_i - m_i) / sqrt(v_i + yerr_i^2 + jitter^2)
    The first epoch is standardized against the stationary distribution.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    dt = np.diff(t)
    decay = np.exp(-dt / tau)

    m = np.empty_like(y)
    v = np.empty_like(y)
    m[0] = mu
    v[0] = sigma**2
    m[1:] = mu + (y[:-1] - mu) * decay
    v[1:] = sigma**2 * (1.0 - decay**2)

    return (y - m) / np.sqrt(v + yerr**2 + jitter**2)


def calc_drw_stats(
    tme,
    min_epochs=DEFAULT_MIN_EPOCHS,
    max_iter=DEFAULT_MAX_ITER,
    z_thr=DEFAULT_Z_THR,
    tau_max_baseline_frac=None,
    mask_regions=None,
):
    """Compute DRW features for a single (times, mags, magerrs) tuple.

    With max_iter = 1 (default) a single plain fit is performed. With
    max_iter > 1, points with |z| > z_thr are masked and the model refit,
    keeping flares out of their own baseline; residuals are always evaluated
    on all points.

    mask_regions is a sequence of (start_mjd, end_mjd) intervals to keep out of
    the fit - pdrs.flare_regions produces them. A flare is contiguous, so
    excluding the interval removes the rise and decay along with the peak,
    where the |z| > z_thr clipping above removes only the most deviant points
    and leaves the wings to inflate sigma. Residuals are still evaluated on all
    points, so a masked event remains visible in drw_max_z. The intervals are
    applied first and held for every iteration.

    Returns a dict with keys drw_tau, drw_sigma, drw_max_z, drw_fit_success.
    Values are NaN (and drw_fit_success = 0) when the light curve is too
    short or the fit fails.

    drw_fit_success means the optimiser converged AND the data constrained
    the parameters - see _fit_is_constrained. tau and sigma are reported
    even when the flag is 0, where a rail value should be read as a limit
    rather than an estimate.
    """
    t = np.asarray(tme[0], dtype=float)
    y = np.asarray(tme[1], dtype=float)
    yerr = np.asarray(tme[2], dtype=float)

    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    t, y, yerr = t[keep], y[keep], yerr[keep]

    if len(t) < min_epochs:
        return _nan_stats()

    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]

    # Merge exactly simultaneous epochs' handling: celerite requires strictly
    # increasing times
    unique = np.concatenate(([True], np.diff(t) > 0))
    t, y, yerr = t[unique], y[unique], yerr[unique]
    if len(t) < min_epochs:
        return _nan_stats()

    mask = np.zeros(len(t), dtype=bool)
    if mask_regions is not None:
        for start, end in mask_regions:
            mask |= (t >= start) & (t <= end)
        # Masking everything leaves nothing to fit; a curve that is entirely
        # flare is better described by an uncontaminated failure than by a fit
        # to its own event.
        if (~mask).sum() < min_epochs:
            mask = np.zeros(len(t), dtype=bool)
    fixed_mask = mask.copy()
    result = None

    for _ in range(max(1, int(max_iter))):
        ix = ~mask
        if ix.sum() < min_epochs:
            break

        try:
            tau, sigma, mu, jitter, success = fit_drw(t[ix], y[ix], yerr[ix])
        except Exception:
            break

        z = ou_resid(t, y, yerr, tau, sigma, mu, jitter)
        constrained = _fit_is_constrained(
            tau, sigma, t[ix], tau_max_baseline_frac=tau_max_baseline_frac
        )
        result = {
            'drw_tau': float(tau),
            'drw_sigma': float(sigma),
            'drw_max_z': float(np.max(np.abs(z))),
            'drw_fit_success': int(bool(success) and constrained),
        }

        new_mask = np.abs(z) > z_thr
        new_mask[0] = new_mask[-1] = False
        new_mask |= fixed_mask
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if result is None:
        return _nan_stats()

    return result


def _calc_drw_stats_pair(
    pair,
    min_epochs=DEFAULT_MIN_EPOCHS,
    max_iter=DEFAULT_MAX_ITER,
    z_thr=DEFAULT_Z_THR,
):
    """calc_drw_stats over a (tme, mask_regions) pair, for pool.map."""
    tme, mask_regions = pair
    return calc_drw_stats(
        tme,
        min_epochs=min_epochs,
        max_iter=max_iter,
        z_thr=z_thr,
        mask_regions=mask_regions,
    )


def calc_drw_stats_batch(
    tme_list,
    Ncore=1,
    min_epochs=DEFAULT_MIN_EPOCHS,
    max_iter=DEFAULT_MAX_ITER,
    z_thr=DEFAULT_Z_THR,
    mask_regions_list=None,
):
    """Compute DRW features for a list of (times, mags, magerrs) tuples,
    optionally in parallel across Ncore processes.

    mask_regions_list, when given, holds one sequence of (start, end) intervals
    per light curve - pdrs.flare_regions produces them - to keep out of that
    curve's fit. Pass None in a slot to fit that curve unmasked.
    """
    from functools import partial

    if mask_regions_list is None:
        mask_regions_list = [None] * len(tme_list)
    elif len(mask_regions_list) != len(tme_list):
        raise ValueError(
            'mask_regions_list has %d entries for %d light curves'
            % (len(mask_regions_list), len(tme_list))
        )

    func = partial(
        _calc_drw_stats_pair, min_epochs=min_epochs, max_iter=max_iter, z_thr=z_thr
    )
    pairs = list(zip(tme_list, mask_regions_list))

    if Ncore is None or Ncore <= 1 or len(tme_list) < 2:
        return [func(pair) for pair in pairs]

    import multiprocessing as mp

    with mp.Pool(min(Ncore, len(tme_list))) as pool:
        return pool.map(func, pairs)
