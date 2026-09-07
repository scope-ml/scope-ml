"""Light curve preparation, matching what feature generation does.

Refinement has to see the same photometry the period search saw.  If it does
not, it is choosing among candidate periods that were derived from different
data, and the comparison is not meaningful: fitting the raw photometry instead
of the cleaned version leaves flagged points and catastrophic outliers in, and
they pull the fit towards whichever trial period happens to accommodate them.

The steps here mirror the block in ``tools/generate_features_rubin.py`` (the
section that builds ``tme_collection``), in the same order and with the same
defaults.  That block is currently inline rather than a function; if it is ever
factored out, this module should call it instead of repeating it.
"""

import numpy as np

#: defaults from the period search config in generate_features_rubin.py
MIN_CADENCE_MINUTES = 5.0
MIN_N_LC_POINTS = 50
#: 5 sigma rather than 3: these are variable stars, and a tighter clip removes
#: real variability along with the outliers
SIGMA_CLIP = 5.0
MAD_TO_SIGMA = 1.4826


def prepare_lightcurve(
    entries,
    min_cadence_minutes=MIN_CADENCE_MINUTES,
    min_n_lc_points=MIN_N_LC_POINTS,
):
    """Clean one source's Kowalski-format light curve entries.

    Parameters
    ----------
    entries : list of dict
        The ``{"_id", "filter", "data"}`` dicts for a single source, one per
        band, as returned by ``scope.surveys.rubin`` and ``scope.surveys.fritz``.
    min_cadence_minutes : float
        Points closer together than this are dropped, keeping the first.
    min_n_lc_points : int
        Sources left with fewer points than this return None.

    Returns
    -------
    tuple of (times, mags, errors, bands) or None
    """
    times, mags, errors, bands = [], [], [], []
    for entry in entries:
        band = entry.get("filter")
        for point in entry.get("data", []):
            # catflags marks known-bad photometry; the period search never saw
            # these points, so the refinement must not either
            if point.get("catflags", 0) != 0:
                continue
            hjd, mag, magerr = point.get("hjd"), point.get("mag"), point.get("magerr")
            if hjd is None or mag is None:
                continue
            times.append(float(hjd))
            mags.append(float(mag))
            errors.append(float(magerr) if magerr is not None else np.nan)
            bands.append(band)
    if not times:
        return None

    times = np.asarray(times, float)
    mags = np.asarray(mags, float)
    errors = np.asarray(errors, float)
    bands = np.asarray(bands)

    order = np.argsort(times)
    times, mags, errors, bands = (
        times[order],
        mags[order],
        errors[order],
        bands[order],
    )

    keep = [0]
    threshold = min_cadence_minutes * 60.0 / 86400.0
    for i in range(1, len(times)):
        if times[i] - times[keep[-1]] >= threshold:
            keep.append(i)
    times, mags, errors, bands = (
        times[keep],
        mags[keep],
        errors[keep],
        bands[keep],
    )

    clip = np.ones(len(times), dtype=bool)
    for band in np.unique(bands):
        in_band = bands == band
        if in_band.sum() < 5:
            continue
        median = np.median(mags[in_band])
        sigma = np.median(np.abs(mags[in_band] - median)) * MAD_TO_SIGMA
        if sigma > 0:
            clip[in_band] &= np.abs(mags[in_band] - median) < SIGMA_CLIP * sigma
    times, mags, errors, bands = (
        times[clip],
        mags[clip],
        errors[clip],
        bands[clip],
    )

    if len(times) < min_n_lc_points:
        return None
    return times, mags, errors, bands


def prepare_all(lightcurves, band_names=None, **kwargs):
    """Group Kowalski-format dicts by source and clean each one.

    Returns {id: (times, mags, errors, bands)}, skipping sources that do not
    survive the cuts.
    """
    grouped = {}
    for entry in lightcurves or []:
        grouped.setdefault(int(entry["_id"]), []).append(entry)

    prepared = {}
    for identifier, entries in grouped.items():
        cleaned = prepare_lightcurve(entries, **kwargs)
        if cleaned is None:
            continue
        times, mags, errors, bands = cleaned
        if band_names is not None:
            # both surveys give integer filter ids; numpy integers hash equal
            # to python ints, so a plain lookup covers it
            bands = np.array([band_names.get(b, str(b)) for b in bands])
        else:
            bands = bands.astype(str)
        prepared[identifier] = (times, mags, errors, bands)
    return prepared
