#!/usr/bin/env python3
"""
PDRS: peak-driven region segmentation for flare / high-activity features.

Adapted from https://github.com/zerozole/Peak_Driven_Region_Segmentation

Detects flare regions in a light curve by finding significant peaks in binned
flux, growing each peak into a region using a gradient-guided expansion,
merging regions across shallow saddles, and gating regions on their median
flux. Summarizes the detected regions as scalar features suitable for the
SCoPe feature pipeline:

    n_flares            number of distinct flare regions
    flare_significance  max over regions of (peak flux - median) / std
    flare_duty_cycle    fraction of the light curve baseline inside regions

Default hyperparameters are tuned for ZTF DR23 and are configurable via
config.yaml under feature_generation.pdrs.
"""

import numpy as np

DEFAULT_BIN_SIZE_DAYS = 3.0
DEFAULT_PEAK_THRESHOLD = 2.0
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_SADDLE_RATIO = 0.2
DEFAULT_REGION_THRESHOLD = 0.5
DEFAULT_MAX_GAP_DAYS = 60.0

PDRS_FEATURE_KEYS = ['n_flares', 'flare_significance', 'flare_duty_cycle']

#: Defaults for the fast (SNR) mode. Minute-scale bins, so the numbers look
#: nothing like the slow ones: bin 5 minutes, gap 90 minutes. Tuned on DP2 r
#: by injection-recovery over 1350 grid points on two independent host draws.
FAST_BIN_SIZE_DAYS = 5.0 / 1440.0
FAST_PEAK_THRESHOLD = 12.0
FAST_MAX_GAP_DAYS = 90.0 / 1440.0
FAST_SMOOTH_WINDOW = 3
FAST_REGION_THRESHOLD = 1.0

#: The fast mode emits its own columns rather than overwriting the slow ones -
#: they detect different things and are computed with different parameters.
#: n_fast_flares is positive for most real curves at minute bins, so
#: fast_flare_significance is the informative member of this set; see
#: calc_pdrs_stats.
FAST_PDRS_FEATURE_KEYS = [
    'n_fast_flares',
    'fast_flare_significance',
    'fast_flare_duty_cycle',
]

#: slow key -> fast key, so the renaming lives in exactly one place.
_FAST_KEY_MAP = dict(zip(PDRS_FEATURE_KEYS, FAST_PDRS_FEATURE_KEYS))


def _nan_stats():
    return {key: np.nan for key in PDRS_FEATURE_KEYS}


def get_uneven_gradient(mjd, flux, window):
    """
    O(N) gradient estimation for unevenly spaced data.
    Provides a smoothed 'trend' used for the expansion/descent logic.
    """
    n = len(flux)
    grad = np.zeros(n)
    half_w = window // 2
    mjd_ref = mjd - mjd[0]

    cs_x = np.cumsum(np.concatenate(([0], mjd_ref)))
    cs_y = np.cumsum(np.concatenate(([0], flux)))
    cs_xx = np.cumsum(np.concatenate(([0], mjd_ref**2)))
    cs_xy = np.cumsum(np.concatenate(([0], mjd_ref * flux)))

    for i in range(n):
        s = max(0, i - half_w)
        e = min(n, i + half_w + 1)
        w = e - s
        if w < 2:
            continue

        sum_x = cs_x[e] - cs_x[s]
        sum_y = cs_y[e] - cs_y[s]
        sum_xx = cs_xx[e] - cs_xx[s]
        sum_xy = cs_xy[e] - cs_xy[s]

        denom = w * sum_xx - sum_x**2
        grad[i] = (w * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0

    return grad


def bin_light_curve(mjd, flux, fluxerr, bin_size=DEFAULT_BIN_SIZE_DAYS):
    """O(N) vectorized light curve binning using np.bincount.

    Returns (binned_times, binned_flux, binned_fluxerr).
    """
    bin_edges = np.arange(mjd.min(), mjd.max() + bin_size, bin_size)
    bin_indices = np.digitize(mjd, bin_edges) - 1
    num_bins = len(bin_edges)

    wf = 1.0 / fluxerr**2
    counts = np.bincount(bin_indices, minlength=num_bins)
    sum_wf = np.bincount(bin_indices, weights=wf, minlength=num_bins)

    valid = (counts > 0) & (sum_wf > 0)

    b_t = np.bincount(bin_indices, weights=mjd, minlength=num_bins)[valid]
    b_t /= counts[valid]
    b_f = np.bincount(bin_indices, weights=flux * wf, minlength=num_bins)[valid]
    b_f /= sum_wf[valid]
    b_fe = np.sqrt(1.0 / sum_wf[valid])

    return b_t, b_f, b_fe


def detect_flares(
    mjd,
    flux,
    peak_threshold=DEFAULT_PEAK_THRESHOLD,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    saddle_ratio=DEFAULT_SADDLE_RATIO,
    region_threshold=DEFAULT_REGION_THRESHOLD,
    max_gap=DEFAULT_MAX_GAP_DAYS,
    err=None,
):
    """Peak-first flare detection on a (binned) flux light curve.

    The peak threshold, the region gate and the significance are all measured
    against a scale. With err=None that scale is the global scatter of the
    curve, which is the right denominator when a bin averages many
    measurements. Passing per-bin errors measures each bin against its own
    uncertainty instead - the right denominator when a bin holds a single
    exposure, i.e. minute-scale bins, where intrinsic variability has not had
    time to act and the measurement noise dominates.

    Returns a list of regions, each a dict with keys
    'start', 'end', 'peak_flux', 'significance'.
    """
    n_points = len(flux)
    if n_points == 0:
        return []

    median_f = np.median(flux)
    std_f = np.std(flux)

    if err is None:
        scale = np.full(n_points, std_f, dtype=float)
    else:
        scale = np.asarray(err, dtype=float)
        # A non-positive or non-finite error must never lower the bar.
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)

    flux_threshold = median_f + peak_threshold * scale

    # Calculate smoothed gradient for expansion boundaries
    grad = get_uneven_gradient(mjd, flux, window=smooth_window)

    # Find local maxima on raw flux (to preserve peak sensitivity)
    peaks = []
    if n_points >= 2 and flux[0] > flux[1]:
        peaks.append(0)
    for i in range(1, n_points - 1):
        if flux[i] > flux[i - 1] and flux[i] > flux[i + 1]:
            peaks.append(i)
    if n_points >= 2 and flux[-1] > flux[-2]:
        peaks.append(n_points - 1)

    peaks = [p for p in peaks if flux[p] > flux_threshold[p]]
    if not peaks:
        return []

    # BFS expansion using the smoothed gradient
    assignments = np.full(n_points, -1)
    frontiers = {p: {'left': p, 'right': p, 'active': True} for p in peaks}
    for p in peaks:
        assignments[p] = p

    any_active = True
    while any_active:
        any_active = False
        for p in peaks:
            if not frontiers[p]['active']:
                continue
            expanded = False

            # Expand LEFT: rise phase (moving backward in time). Stop if the
            # point dips below the global median or the gap exceeds max_gap.
            left = frontiers[p]['left']
            if (
                left > 0
                and assignments[left - 1] == -1
                and flux[left - 1] >= median_f
                and (mjd[left] - mjd[left - 1]) <= max_gap
            ):
                if flux[left - 1] < flux[left]:
                    assignments[left - 1] = p
                    frontiers[p]['left'] = left - 1
                    expanded = True
                elif grad[left] >= 0:
                    assignments[left - 1] = p
                    frontiers[p]['left'] = left - 1
                    expanded = True

            # Expand RIGHT: decay phase (moving forward in time)
            right = frontiers[p]['right']
            if (
                right < n_points - 1
                and assignments[right + 1] == -1
                and flux[right + 1] >= median_f
                and (mjd[right + 1] - mjd[right]) <= max_gap
            ):
                if flux[right + 1] < flux[right]:
                    assignments[right + 1] = p
                    frontiers[p]['right'] = right + 1
                    expanded = True
                elif grad[right] <= 0:
                    assignments[right + 1] = p
                    frontiers[p]['right'] = right + 1
                    expanded = True

            if not expanded:
                frontiers[p]['active'] = False
            else:
                any_active = True

    # Build raw clusters
    raw_clusters = []
    for p in peaks:
        left, right = frontiers[p]['left'], frontiers[p]['right']
        if (right - left + 1) >= min_cluster_size:
            raw_clusters.append(
                {'start_idx': left, 'end_idx': right, 'peak_flux': flux[p]}
            )

    if not raw_clusters:
        return []

    # Saddle-point merging
    merged_indices = [[raw_clusters[0]['start_idx'], raw_clusters[0]['end_idx']]]
    curr_peak_f = raw_clusters[0]['peak_flux']
    for i in range(1, len(raw_clusters)):
        prev_s, prev_e = merged_indices[-1]
        next_s, next_e = raw_clusters[i]['start_idx'], raw_clusters[i]['end_idx']
        next_peak_f = raw_clusters[i]['peak_flux']

        temporal_gap = mjd[next_s] - mjd[prev_e]
        no_data_between = next_s <= prev_e + 1

        if temporal_gap > max_gap:
            # Large data void - always separate
            merged_indices.append([next_s, next_e])
            curr_peak_f = next_peak_f
        elif no_data_between or next_s <= prev_e + 2:
            # Adjacent or overlapping - always merge
            merged_indices[-1][1] = next_e
            curr_peak_f = max(curr_peak_f, next_peak_f)
        else:
            # True saddle exists between clusters
            saddle_f = np.min(flux[prev_e + 1 : next_s])
            is_shallow = (saddle_f - median_f) > (
                saddle_ratio * (min(curr_peak_f, next_peak_f) - median_f)
            )
            if is_shallow:
                merged_indices[-1][1] = next_e
                curr_peak_f = max(curr_peak_f, next_peak_f)
            else:
                merged_indices.append([next_s, next_e])
                curr_peak_f = next_peak_f

    # Final gate: the region median must clear the global median by
    # region_threshold sigma
    support_threshold = median_f + region_threshold * scale
    flares = []
    for s, e in merged_indices:
        region_flux = flux[s : e + 1]
        if np.median(region_flux) >= np.median(support_threshold[s : e + 1]):
            k = s + int(np.argmax(region_flux))
            flares.append(
                {
                    'start': mjd[s],
                    'end': mjd[e],
                    'peak_flux': np.max(region_flux),
                    'significance': (flux[k] - median_f) / scale[k],
                }
            )

    return flares


def _prepare(tme, bin_size_days, min_cluster_size):
    """Clean, sort and bin a (times, mags, magerrs) tuple for detect_flares.

    Returns (t, b_t, b_f, b_fe), or None when the curve is too short. Shared by
    calc_pdrs_stats and flare_regions so both see exactly the same binning.
    """
    t = np.asarray(tme[0], dtype=float)
    mag = np.asarray(tme[1], dtype=float)
    magerr = np.asarray(tme[2], dtype=float)

    keep = np.isfinite(t) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    t, mag, magerr = t[keep], mag[keep], magerr[keep]

    if len(t) < max(4, min_cluster_size):
        return None

    order = np.argsort(t)
    t, mag, magerr = t[order], mag[order], magerr[order]

    flux = 10 ** (-0.4 * (mag - np.median(mag)))
    fluxerr = 0.4 * np.log(10) * flux * magerr

    b_t, b_f, b_fe = bin_light_curve(t, flux, fluxerr, bin_size=bin_size_days)

    if len(b_t) < max(4, min_cluster_size):
        return None

    return t, b_t, b_f, b_fe


def flare_regions(
    tme,
    bin_size_days=DEFAULT_BIN_SIZE_DAYS,
    peak_threshold=DEFAULT_PEAK_THRESHOLD,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    saddle_ratio=DEFAULT_SADDLE_RATIO,
    region_threshold=DEFAULT_REGION_THRESHOLD,
    max_gap_days=DEFAULT_MAX_GAP_DAYS,
    use_snr=False,
):
    """Flare intervals as a list of (start_mjd, end_mjd), earliest first.

    calc_pdrs_stats summarises the flares it finds and discards their extent.
    Callers that need the extent - keeping a flare out of a DRW fit, say - use
    this instead. The detection is identical; only the return value differs.
    """
    prep = _prepare(tme, bin_size_days, min_cluster_size)
    if prep is None:
        return []
    _, b_t, b_f, b_fe = prep

    flares = detect_flares(
        b_t,
        b_f,
        peak_threshold=peak_threshold,
        smooth_window=smooth_window,
        min_cluster_size=min_cluster_size,
        saddle_ratio=saddle_ratio,
        region_threshold=region_threshold,
        max_gap=max_gap_days,
        err=b_fe if use_snr else None,
    )
    return [(float(f['start']), float(f['end'])) for f in flares]


def calc_pdrs_stats(
    tme,
    bin_size_days=DEFAULT_BIN_SIZE_DAYS,
    peak_threshold=DEFAULT_PEAK_THRESHOLD,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    saddle_ratio=DEFAULT_SADDLE_RATIO,
    region_threshold=DEFAULT_REGION_THRESHOLD,
    max_gap_days=DEFAULT_MAX_GAP_DAYS,
    use_snr=False,
):
    """Compute PDRS flare features for a single (times, mags, magerrs) tuple.

    Magnitudes are converted to flux relative to the median magnitude, so a
    brightening (decreasing magnitude) appears as a flux increase.

    With use_snr the detection is measured against each bin's own uncertainty
    rather than against the scatter of the whole curve. That is the mode for
    minute-scale bins, where a bin holds a single exposure; see detect_flares.
    At those bin sizes n_flares is positive for most real light curves, so
    flare_significance - calibrated against the population, the way EVT treats
    drw_max_z - is the informative output rather than the count.

    Returns a dict with keys n_flares, flare_significance, flare_duty_cycle;
    all NaN if the light curve has too few usable points.
    """
    prep = _prepare(tme, bin_size_days, min_cluster_size)
    if prep is None:
        return _nan_stats()
    t, b_t, b_f, b_fe = prep

    flares = detect_flares(
        b_t,
        b_f,
        peak_threshold=peak_threshold,
        smooth_window=smooth_window,
        min_cluster_size=min_cluster_size,
        saddle_ratio=saddle_ratio,
        region_threshold=region_threshold,
        max_gap=max_gap_days,
        err=b_fe if use_snr else None,
    )

    baseline = t[-1] - t[0]
    if len(flares) == 0 or baseline <= 0:
        return {'n_flares': 0, 'flare_significance': 0.0, 'flare_duty_cycle': 0.0}

    total_flare_time = float(np.sum([f['end'] - f['start'] for f in flares]))

    return {
        'n_flares': len(flares),
        'flare_significance': float(np.max([f['significance'] for f in flares])),
        'flare_duty_cycle': min(total_flare_time / baseline, 1.0),
    }


def calc_fast_pdrs_stats(
    tme,
    bin_size_days=FAST_BIN_SIZE_DAYS,
    peak_threshold=FAST_PEAK_THRESHOLD,
    smooth_window=FAST_SMOOTH_WINDOW,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    saddle_ratio=DEFAULT_SADDLE_RATIO,
    region_threshold=FAST_REGION_THRESHOLD,
    max_gap_days=FAST_MAX_GAP_DAYS,
):
    """PDRS in fast (SNR) mode, returned under the fast feature names.

    Same detector as calc_pdrs_stats(use_snr=True); this only fixes the
    minute-scale defaults and renames the outputs, so a light curve can carry
    both the slow and the fast features without one overwriting the other.

    Returns a dict keyed by FAST_PDRS_FEATURE_KEYS.
    """
    stats = calc_pdrs_stats(
        tme,
        bin_size_days=bin_size_days,
        peak_threshold=peak_threshold,
        smooth_window=smooth_window,
        min_cluster_size=min_cluster_size,
        saddle_ratio=saddle_ratio,
        region_threshold=region_threshold,
        max_gap_days=max_gap_days,
        use_snr=True,
    )
    return {_FAST_KEY_MAP[k]: v for k, v in stats.items()}
