import numpy as np
import warnings

import periodfind

# ---------------------------------------------------------------------------
# Algorithm name mapping
# ---------------------------------------------------------------------------

_ALGO_MAP = {"CE", "AOV", "LS", "FPW", "BLS", "MHF"}


def _normalize_algorithm(algorithm):
    """Parse algorithm string into (base_name, is_periodogram).

    Handles names like 'ECE', 'CE', 'EAOV_periodogram', 'LS', etc.
    Returns the canonical base name and whether periodogram output is requested.
    """
    algo_name = algorithm
    is_periodogram = "_periodogram" in algo_name
    algo_name = algo_name.replace("_periodogram", "")

    # Strip legacy "E" prefix (ECE -> CE, EAOV -> AOV, ELS -> LS)
    if algo_name.startswith("E") and algo_name[1:] in _ALGO_MAP:
        algo_name = algo_name[1:]

    return algo_name, is_periodogram


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _prepare_lightcurves(
    lightcurves, doSingleTimeSegment, return_errs=False, bands=None
):
    """Sort by time, subtract tmin, normalize mags, and cast to float32.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).
    doSingleTimeSegment : bool
        If True, build a common time grid across all lightcurves.
    return_errs : bool, default=False
        If True, also return the magnitude error arrays (for FPW).
    bands : list of ndarray or None
        Per-point band identifiers for each lightcurve.  When provided,
        per-band weighted means are subtracted before the [0,1]
        normalization, removing chromatic offsets from multi-band stacking.

    Returns
    -------
    time_stack : list of ndarray (float32)
    mag_stack : list of ndarray (float32)
    err_stack : list of ndarray (float32), only if return_errs=True
    """
    if doSingleTimeSegment:
        tt = np.empty((0, 1))
        for lightcurve in lightcurves:
            tt = np.unique(np.append(tt, lightcurve[0]))

    time_stack, mag_stack, err_stack = [], [], []
    for ii, lightcurve in enumerate(lightcurves):
        if doSingleTimeSegment:
            _, x_ind, y_ind = np.intersect1d(tt, lightcurve[0], return_indices=True)
            mag_array = 999 * np.ones(tt.shape)
            magerr_array = 999 * np.ones(tt.shape)
            mag_array[x_ind] = lightcurve[1][y_ind]
            magerr_array[x_ind] = lightcurve[2][y_ind]
            lightcurve = (tt, mag_array, magerr_array)
        else:
            idx = np.argsort(lightcurve[0])
            tmin = np.min(lightcurve[0])
            lightcurve = (
                lightcurve[0][idx] - tmin,
                lightcurve[1][idx],
                lightcurve[2][idx],
            )

        time_stack.append(np.asarray(lightcurve[0], dtype=np.float32))
        lc = np.asarray(lightcurve[1], dtype=np.float64)

        # Per-band mean subtraction: remove chromatic offsets before
        # normalizing so multi-band data aligns properly.
        if bands is not None:
            bb = bands[ii]
            # Apply same sort order as lightcurve if we sorted above
            if not doSingleTimeSegment:
                bb = bb[idx]
            for b in np.unique(bb):
                mask = bb == b
                w = 1.0 / np.maximum(
                    np.asarray(lightcurve[2], dtype=np.float64)[mask] ** 2,
                    1e-30,
                )
                wmean = np.average(lc[mask], weights=w)
                lc[mask] -= wmean
            # Shift so min is 0 (the subsequent [0,1] norm handles the rest)
            lc -= np.min(lc)

        lc_range = np.max(lc) - np.min(lc)
        if lc_range > 0:
            lc = lc / lc_range
        mag_stack.append(np.asarray(lc, dtype=np.float32))
        if return_errs:
            err_stack.append(np.asarray(lightcurve[2], dtype=np.float32))

    if return_errs:
        return time_stack, mag_stack, err_stack
    return time_stack, mag_stack


def _build_pdots(doUsePDot):
    """Build the array of period time-derivatives to test.

    Parameters
    ----------
    doUsePDot : bool
        If True, return a logarithmic grid of negative pdots plus zero.
        If False, return [0.0].

    Returns
    -------
    pdots_to_test : ndarray (float32)
    """
    if doUsePDot:
        num_pdots = 10
        max_pdot = 1e-10
        min_pdot = 1e-12
        pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
        pdots_to_test = np.append(0, pdots_to_test)
    else:
        pdots_to_test = np.array([0.0])
    return pdots_to_test.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_periods(
    algorithm,
    lightcurves,
    freqs,
    doGPU=False,
    doCPU=False,
    doRemoveTerrestrial=False,
    doUsePDot=False,
    doSingleTimeSegment=False,
    freqs_to_remove=None,
    phase_bins=20,
    mag_bins=10,
    Ncore=8,
    qmin=0.01,
    qmax=0.5,
    bands=None,
):

    if freqs_to_remove is not None:
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')

    # Normalize algorithm name
    algo_name, is_periodogram = _normalize_algorithm(algorithm)

    # -----------------------------------------------------------------------
    # Unified periodfind path for CE / AOV / LS (works on both CPU and GPU)
    # -----------------------------------------------------------------------
    if algo_name in _ALGO_MAP:
        device = 'gpu' if doGPU else 'cpu'
        needs_errs = algo_name in ("FPW", "BLS", "MHF")

        # Create algorithm via unified factory
        if algo_name == "CE":
            algo = periodfind.ConditionalEntropy(
                n_phase=phase_bins, n_mag=mag_bins, device=device
            )
        elif algo_name == "AOV":
            algo = periodfind.AOV(n_phase=phase_bins, device=device)
        elif algo_name == "LS":
            algo = periodfind.LombScargle(device=device)
        elif algo_name == "FPW":
            algo = periodfind.FPW(n_bins=phase_bins, device=device)
        elif algo_name == "BLS":
            algo = periodfind.BoxLeastSquares(
                n_bins=50, qmin=qmin, qmax=qmax, device=device
            )
        elif algo_name == "MHF":
            algo = periodfind.MultiHarmonicFourier(max_harmonics=3, device=device)

        # Prepare data
        # Per-band weighted means are subtracted for every algorithm.  Rubin
        # merges all six bands into one series before period finding, and the
        # chromatic offsets between them are far larger than the variability:
        # on a typical DP2 object the merged scatter is 0.985 mag against
        # 0.09-0.15 mag within any band, so ~85-90% of what an algorithm sees is
        # which filter was used.  This was previously applied to MHF alone,
        # which is why MHF appeared 4-7x more accurate than the rest.
        prep_bands = bands
        if needs_errs:
            time_stack, mag_stack, err_stack = _prepare_lightcurves(
                lightcurves,
                doSingleTimeSegment,
                return_errs=True,
                bands=prep_bands,
            )
        else:
            time_stack, mag_stack = _prepare_lightcurves(
                lightcurves,
                doSingleTimeSegment,
                bands=prep_bands,
            )
        periods = (1.0 / freqs).astype(np.float32)
        pdots_to_test = _build_pdots(doUsePDot)

        print("Number of lightcurves: %d" % len(time_stack))
        print("Number of frequency bins: %d" % len(freqs))
        print("Number of phase bins: %d" % phase_bins)
        print("Number of magnitude bins: %d" % mag_bins)

        output_mode = 'periodogram' if is_periodogram else 'stats'
        extra_kwargs = {"errs": err_stack} if needs_errs else {}
        data_out = algo.calc(
            time_stack,
            mag_stack,
            periods,
            pdots_to_test,
            output=output_mode,
            **extra_kwargs,
        )

        # Process results
        if not is_periodogram:
            # Stats output -> flat arrays
            periods_best = np.zeros((len(lightcurves), 1))
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            for ii, stat in enumerate(data_out):
                if np.isnan(stat.significance):
                    warnings.warn(
                        "Oops... significance  is nan... something went wrong"
                    )
                periods_best[ii] = stat.params[0]
                pdots[ii] = stat.params[1]
                significances[ii] = stat.significance

            pdots = pdots.flatten()
            periods_best = periods_best.flatten()
            significances = significances.flatten()
        else:
            # Periodogram output -> list of dicts
            periods_best = []
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            for ii, stat in enumerate(data_out):
                if algo_name == "CE":
                    significance = np.abs(
                        np.mean(stat.data) - np.min(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                else:
                    # AOV and LS use maxima
                    significance = np.abs(
                        np.mean(stat.data) - np.max(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmax(stat.data)]

                if np.isnan(significance):
                    warnings.warn(
                        "Oops... significance  is nan... something went wrong"
                    )

                periods_best.append({'period': period, 'data': stat.data})
                pdots[ii] = pdots_to_test[0]
                significances[ii] = significance

            pdots = pdots.flatten()
            significances = significances.flatten()

    return np.array(periods_best), np.array(significances), np.array(pdots)


def extract_top_n_periods(periodogram_results, freqs, n_top=8, n_chunks_multiplier=3):
    """Extract top N periods from periodogram using hybrid chunking.

    The grid is divided into ``n_chunks_multiplier * n_top`` chunks of equal
    width in *log period*.  The single best peak from each chunk is recorded,
    then all chunk-winners are ranked by significance and the top ``n_top``
    are returned.  This guarantees that the returned periods come from
    distinct regions of frequency space, avoiding the adjacent-bin
    duplication problem of the previous sort-and-take approach.

    Chunking in log period rather than in frequency matters whenever the
    frequency grid spans a wide dynamic range.  Equal-width frequency chunks
    are wildly unequal in period: over a 0.0084-288 c/d grid, 150 such chunks
    place every period longer than 0.52 d into the first chunk, capping the
    long-period share of the returned peaks at 1/150 no matter how strong
    those peaks are.

    Parameters
    ----------
    periodogram_results : list of dict
        Output from ``find_periods`` when called with a ``_periodogram``
        algorithm suffix.  Each dict has ``'period'`` (best float) and
        ``'data'`` (full periodogram array).
    freqs : ndarray
        The frequency grid used for period finding.
    n_top : int
        Number of top periods to return per source.
    n_chunks_multiplier : int
        The periodogram is split into ``n_chunks_multiplier * n_top``
        chunks.  Higher values give finer coverage but may split real
        peaks across chunk boundaries.  Default 3.

    Returns
    -------
    top_periods : ndarray, shape (n_sources, n_top)
        Periods in descending order of significance.
    top_significances : ndarray, shape (n_sources, n_top)
        Corresponding significance values.
    """
    periods_grid = 1.0 / freqs
    n_sources = len(periodogram_results)
    top_periods = np.full((n_sources, n_top), np.nan, dtype=np.float64)
    top_significances = np.full((n_sources, n_top), np.nan, dtype=np.float64)

    n_chunks = n_chunks_multiplier * n_top

    # Chunk boundaries in log period, computed once for all sources.
    order_p = np.argsort(periods_grid)
    p_sorted = periods_grid[order_p]
    edges = np.logspace(np.log10(p_sorted[0]), np.log10(p_sorted[-1]), n_chunks + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bounds = np.searchsorted(p_sorted, edges)

    for i, res in enumerate(periodogram_results):
        data = res['data'].flatten()
        n_freqs = len(data)
        if n_freqs == 0:
            continue

        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            continue

        # CE uses minima; others use maxima
        best_idx_min = np.argmin(data)
        best_period = res['period']
        use_minima = np.isclose(periods_grid[best_idx_min], best_period, rtol=1e-4)

        if n_freqs < n_chunks:
            # Fewer frequency bins than chunks — fall back to simple sort
            if use_minima:
                sorted_indices = np.argsort(data)
            else:
                sorted_indices = np.argsort(data)[::-1]
            sigs = np.abs(data[sorted_indices] - mean_val) / std_val
            n_fill = min(n_top, len(sorted_indices))
            top_periods[i, :n_fill] = periods_grid[sorted_indices[:n_fill]]
            top_significances[i, :n_fill] = sigs[:n_fill]
            continue

        # Find best peak in each log-period chunk.  Chunks holding no grid
        # points are skipped: at long periods the frequency grid is genuinely
        # sparse, and an empty chunk means there is no independent frequency
        # there to report.
        chunk_periods = []
        chunk_sigs = []

        for c in range(n_chunks):
            idx = order_p[bounds[c] : bounds[c + 1]]
            if idx.size == 0:
                continue
            chunk_data = data[idx]

            if use_minima:
                best_global = idx[np.argmin(chunk_data)]
            else:
                best_global = idx[np.argmax(chunk_data)]

            chunk_periods.append(periods_grid[best_global])
            chunk_sigs.append(abs(data[best_global] - mean_val) / std_val)

        chunk_periods = np.asarray(chunk_periods, dtype=np.float64)
        chunk_sigs = np.asarray(chunk_sigs, dtype=np.float64)

        # Rank by significance and keep top n_top
        order = np.argsort(chunk_sigs)[::-1]
        n_fill = min(n_top, len(order))
        top_periods[i, :n_fill] = chunk_periods[order[:n_fill]]
        top_significances[i, :n_fill] = chunk_sigs[order[:n_fill]]

    return top_periods, top_significances


def _period_match(pa_val, pb_val, tolerance=0.05, harmonics=None):
    """Check if two periods match within tolerance, allowing harmonics.

    Parameters
    ----------
    pa_val, pb_val : float
        Period values to compare.
    tolerance : float
        Fractional tolerance for matching (default 0.05 = 5%).
    harmonics : list of float or None
        Harmonic ratios to check.  ``None`` uses [1, 0.5, 2, 1/3, 3].

    Returns
    -------
    bool
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0, 1.0 / 3, 3.0]
    if np.isnan(pa_val) or np.isnan(pb_val) or pa_val <= 0 or pb_val <= 0:
        return False
    for h in harmonics:
        if abs(pa_val / (pb_val * h) - 1.0) < tolerance:
            return True
    return False


def compute_agreement_scores(
    period_dict,
    top_n_periods_dict,
    keep_id_list,
    period_algorithms,
    min_agree_period=0.007,
    harmonics=None,
    tolerance=0.05,
):
    """Compute cross-algorithm agreement scores for period finding.

    Multi-tier scoring with spurious-period filtering and expanded
    harmonics.  This was originally inlined in
    ``generate_features_rubin.py`` and is now shared across pipelines.

    Parameters
    ----------
    period_dict : dict
        ``{algorithm_key: 1-D array}`` of best periods per source,
        indexed in the same order as *keep_id_list*.
    top_n_periods_dict : dict
        ``{algorithm_key: 2-D array (n_sources, n_top)}`` of ranked
        periods per source.
    keep_id_list : list
        Source identifiers, one per row in the arrays above.
    period_algorithms : list of str
        Algorithm keys present in *period_dict* / *top_n_periods_dict*.
        Names like ``"ELS_ECE_EAOV"`` or ``"LS_CE_AOV"`` are kept
        verbatim; all others are shortened with ``algo.split('_')[0]``.
    min_agree_period : float
        Periods shorter than this (days) are treated as spurious and
        excluded from agreement checks.
    harmonics : list of float or None
        Harmonic ratios passed to :func:`_period_match`.
    tolerance : float
        Fractional tolerance passed to :func:`_period_match`.

    Returns
    -------
    dict
        ``{source_id: {feature_name: value, ...}}`` with keys:
        ``n_agree_pairs``, ``n_total_pairs``, ``agree_score``,
        ``agree_strict``, ``agree_weighted``, ``best_agree_period``,
        ``best_consensus_period``.
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0, 1.0 / 3, 3.0]

    NESTED_NAMES = {"ELS_ECE_EAOV", "LS_CE_AOV"}

    def _algo_display(algo):
        return algo if algo in NESTED_NAMES else algo.split('_')[0]

    # Build ordered list of (algo_key, display_name) for algorithms that
    # have top-N data.
    algo_pairs = []
    for algo in period_algorithms:
        if algo in top_n_periods_dict:
            algo_pairs.append((algo, _algo_display(algo)))

    results = {}
    for idx, _id in enumerate(keep_id_list):
        # Per-source lookup tables
        top_lists = {}
        best_period_per_algo = {}
        for algo_key, aname in algo_pairs:
            top_lists[aname] = top_n_periods_dict[algo_key][idx]
            best_period_per_algo[aname] = period_dict[algo_key][idx]

        anames = list(top_lists.keys())
        total_pairs = len(anames) * (len(anames) - 1) // 2

        # --- agree_strict: top-1 only, filtered, expanded harmonics ---
        n_strict = 0
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                p_a = best_period_per_algo[anames[ii]]
                p_b = best_period_per_algo[anames[jj]]
                if p_a >= min_agree_period and p_b >= min_agree_period:
                    if _period_match(p_a, p_b, tolerance, harmonics):
                        n_strict += 1

        # --- agree_score: top-N with filtering ---
        n_agree_pairs = 0
        best_agree_period = np.nan
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                ps_a = top_lists[anames[ii]]
                ps_b = top_lists[anames[jj]]
                matched = False
                for pa_val in ps_a:
                    if np.isnan(pa_val) or pa_val < min_agree_period:
                        continue
                    if matched:
                        break
                    for pb_val in ps_b:
                        if np.isnan(pb_val) or pb_val < min_agree_period:
                            continue
                        if _period_match(pa_val, pb_val, tolerance, harmonics):
                            matched = True
                            if np.isnan(best_agree_period):
                                best_agree_period = pa_val
                            break
                if matched:
                    n_agree_pairs += 1

        # --- agree_weighted: rank-weighted, weight = 1/(rank_i * rank_j) ---
        weighted_sum = 0.0
        weight_total = 0.0
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                ps_a = top_lists[anames[ii]]
                ps_b = top_lists[anames[jj]]
                pair_weight = 0.0
                for ri, pa_val in enumerate(ps_a):
                    if np.isnan(pa_val) or pa_val < min_agree_period:
                        continue
                    for rj, pb_val in enumerate(ps_b):
                        if np.isnan(pb_val) or pb_val < min_agree_period:
                            continue
                        if _period_match(pa_val, pb_val, tolerance, harmonics):
                            w = 1.0 / ((ri + 1) * (rj + 1))
                            if w > pair_weight:
                                pair_weight = w
                weighted_sum += pair_weight
                weight_total += 1.0

        # --- best_consensus_period: period with most weighted matches ---
        period_votes = {}
        for aname in anames:
            for rank, p in enumerate(top_lists[aname]):
                if np.isnan(p) or p < min_agree_period:
                    continue
                vote_weight = 0.0
                for other in anames:
                    if other == aname:
                        continue
                    for rj, p_other in enumerate(top_lists[other]):
                        if np.isnan(p_other) or p_other < min_agree_period:
                            continue
                        if _period_match(p, p_other, tolerance, harmonics):
                            vote_weight += 1.0 / ((rank + 1) * (rj + 1))
                            break
                matched_key = None
                for existing_p in period_votes:
                    if _period_match(p, existing_p, tolerance, harmonics):
                        matched_key = existing_p
                        break
                if matched_key is not None:
                    period_votes[matched_key] += vote_weight
                else:
                    period_votes[p] = vote_weight

        best_consensus = np.nan
        if period_votes:
            best_consensus = max(period_votes, key=period_votes.get)
            if period_votes[best_consensus] == 0:
                best_consensus = np.nan

        results[_id] = {
            'n_agree_pairs': n_agree_pairs,
            'n_total_pairs': total_pairs,
            'agree_score': (n_agree_pairs / total_pairs if total_pairs > 0 else 0.0),
            'agree_strict': (n_strict / total_pairs if total_pairs > 0 else 0.0),
            'agree_weighted': (
                weighted_sum / weight_total if weight_total > 0 else 0.0
            ),
            'best_agree_period': best_agree_period,
            'best_consensus_period': best_consensus,
        }

    return results


# ---------------------------------------------------------------------------
# Sidereal-alias family scoring
# ---------------------------------------------------------------------------

# Sidereal frequency in cycles/day (Earth's rotation period = 23.9345 hr)
_F_SIDEREAL_D = 24.0 / 23.9345  # ~1.00274 cycles/day


def _are_sidereal_aliases(p1_d, p2_d, harmonics=None, max_n=15, tol=0.03):
    """Check if two periods are sidereal aliases (optionally at harmonics).

    Two periods are related if their frequencies satisfy
    ``f1 = h * f2 + n * f_sidereal`` for some harmonic ratio *h* and
    integer *n* with ``|n| <= max_n``.

    Parameters
    ----------
    p1_d, p2_d : float
        Periods in days.
    harmonics : list of float or None
        Harmonic ratios to check.  Default ``[1, 0.5, 2]``.
    max_n : int
        Maximum sidereal alias order.
    tol : float
        Fractional tolerance on the sidereal-integer residual.

    Returns
    -------
    bool
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0]
    if np.isnan(p1_d) or np.isnan(p2_d) or p1_d <= 0 or p2_d <= 0:
        return False
    f1 = 1.0 / p1_d
    for h in harmonics:
        f2h = h / p2_d
        n_float = (f1 - f2h) / _F_SIDEREAL_D
        n_int = round(n_float)
        if abs(n_int) <= max_n and abs(n_float - n_int) < tol:
            return True
    return False


def compute_sidereal_family_scores(
    top_n_periods_dict,
    top_n_sigs_dict,
    keep_id_list,
    period_algorithms,
    f1_power_dict=None,
    period_dict=None,
    harmonics=None,
    max_n=15,
    tol=0.03,
):
    """Score periodicity via sidereal-alias family grouping.

    Groups all top-N periods across algorithms into families of
    sidereal aliases.  The dominant family's breadth across algorithms
    is a robust periodicity indicator even when individual algorithms
    return different aliases of the same true period.

    Parameters
    ----------
    top_n_periods_dict : dict
        ``{algo_key: 2-D array (n_sources, n_top)}`` of ranked periods.
    top_n_sigs_dict : dict
        ``{algo_key: 2-D array (n_sources, n_top)}`` of ranked
        significances, same shape as *top_n_periods_dict*.
    keep_id_list : list
        Source identifiers, one per row.
    period_algorithms : list of str
        Algorithm keys present in the dicts.
    f1_power_dict : dict or None
        ``{algo_key: 1-D array}`` of Fourier f1_power per source.
        Used to select the best algorithm for phase-folding.
    period_dict : dict or None
        ``{algo_key: 1-D array}`` of best (top-1) periods per source.
    harmonics : list of float or None
        Harmonic ratios for alias matching.  Default ``[1, 0.5, 2]``.
    max_n : int
        Maximum sidereal alias order.
    tol : float
        Fractional tolerance on the sidereal-integer residual.

    Returns
    -------
    dict
        ``{source_id: {feature_name: value, ...}}`` with keys:
        ``family_n_algos``, ``family_rank_score``,
        ``family_n_members``, ``family_n_total``,
        ``family_best_period``, ``family_best_algo``,
        ``family_best_f1_power``.
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0]

    NESTED_NAMES = {"ELS_ECE_EAOV", "LS_CE_AOV"}

    def _algo_display(algo):
        return algo if algo in NESTED_NAMES else algo.split('_')[0]

    algo_pairs = []
    for algo in period_algorithms:
        if algo in top_n_periods_dict:
            algo_pairs.append((algo, _algo_display(algo)))

    results = {}
    for idx, _id in enumerate(keep_id_list):
        # Collect all (algo_name, rank, period, sig) entries
        entries = []
        for algo_key, aname in algo_pairs:
            periods_row = top_n_periods_dict[algo_key][idx]
            sigs_row = top_n_sigs_dict[algo_key][idx]
            for rank, (p, s) in enumerate(zip(periods_row, sigs_row)):
                if not np.isnan(p) and p > 0:
                    entries.append((aname, rank, p, s))

        n_total = len(entries)

        if n_total == 0:
            results[_id] = {
                'family_n_algos': 0,
                'family_rank_score': 0.0,
                'family_n_members': 0,
                'family_n_total': 0,
                'family_best_period': np.nan,
                'family_best_algo': '',
                'family_best_f1_power': np.nan,
            }
            continue

        # Union-find to group sidereal aliases
        parent = list(range(n_total))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x, y):
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for i in range(n_total):
            for j in range(i + 1, n_total):
                if _are_sidereal_aliases(
                    entries[i][2],
                    entries[j][2],
                    harmonics=harmonics,
                    max_n=max_n,
                    tol=tol,
                ):
                    _union(i, j)

        # Collect families
        families = {}
        for i, (aname, rank, p, s) in enumerate(entries):
            root = _find(i)
            if root not in families:
                families[root] = []
            families[root].append((aname, rank, p, s))

        # Score each family
        best_family = None
        best_score = (-1, -1.0)  # (n_algos, rank_score)
        for fid, members in families.items():
            algo_set = set(m[0] for m in members)
            n_algos = len(algo_set)
            # Best rank per algorithm (0-indexed)
            best_rank_per_algo = {}
            for aname, rank, p, s in members:
                if aname not in best_rank_per_algo or rank < best_rank_per_algo[aname]:
                    best_rank_per_algo[aname] = rank
            rank_score = sum(1.0 / (r + 1) for r in best_rank_per_algo.values())
            score = (n_algos, rank_score)
            if score > best_score:
                best_score = score
                best_family = (n_algos, rank_score, len(members), members)

        n_algos, rank_score, n_members, members = best_family

        # Select best period: use algorithm with highest f1_power
        best_period = np.nan
        best_algo_name = ''
        best_f1 = np.nan
        if f1_power_dict is not None and period_dict is not None:
            for algo_key, aname in algo_pairs:
                # Check this algo is in the winning family
                if aname not in set(m[0] for m in members):
                    continue
                f1_val = f1_power_dict[algo_key][idx]
                if np.isnan(f1_val):
                    continue
                if np.isnan(best_f1) or f1_val > best_f1:
                    best_f1 = f1_val
                    best_algo_name = aname
                    best_period = period_dict[algo_key][idx]

        # Fallback: use highest-significance period in the family
        if np.isnan(best_period):
            best_sig = -1.0
            for aname, rank, p, s in members:
                if rank == 0 and s > best_sig:
                    best_sig = s
                    best_period = p
                    best_algo_name = aname

        results[_id] = {
            'family_n_algos': n_algos,
            'family_rank_score': rank_score,
            'family_n_members': n_members,
            'family_n_total': n_total,
            'family_best_period': best_period,
            'family_best_algo': best_algo_name,
            'family_best_f1_power': best_f1,
        }

    return results


def _ab_to_amp_phi(raw_features):
    """Convert raw Fourier (A,B) coefficients to (amplitude, phase) form.

    The Rust FourierDecomposition returns raw linear regression coefficients:
        [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]

    This converts harmonics to a physically interpretable representation:
        [power, BIC, offset, slope, amp1, phi1, relamp2, relphi2, ..., relamp5, relphi5]

    where:
        amp_n   = sqrt(A_n^2 + B_n^2)       (amplitude of n-th harmonic)
        phi_n   = arctan2(A_n, B_n)          (phase of n-th harmonic)
        relamp  = amp_n / amp_1              (relative amplitude, n >= 2)
        relphi  = (phi_n/n - phi_1) / (2pi/n) mod 1   (relative phase, n >= 2)

    This matches the convention in lcstats.AB2AmpPhi and the feature column
    names (f1_amp, f1_phi0, f1_relamp1, f1_relphi1, ...).

    Parameters
    ----------
    raw_features : ndarray, shape (n_curves, 14)

    Returns
    -------
    converted : ndarray, shape (n_curves, 14)
    """
    out = raw_features.copy()
    # Columns 4..13 are the (A,B) pairs for harmonics 1..5
    # Indices: 4=A1, 5=B1, 6=A2, 7=B2, 8=A3, 9=B3, 10=A4, 11=B4, 12=A5, 13=B5
    for k in range(5):
        col_a = 4 + 2 * k
        col_b = 5 + 2 * k
        a = out[:, col_a]
        b = out[:, col_b]
        amp = np.sqrt(a**2 + b**2)
        phi = np.arctan2(a, b)
        out[:, col_a] = amp
        out[:, col_b] = phi

    # Normalize amplitudes relative to fundamental (k >= 2)
    amp1 = out[:, 4].copy()
    amp1[amp1 == 0] = np.nan  # avoid division by zero
    for k in range(1, 5):
        col_a = 4 + 2 * (k)
        out[:, col_a] /= amp1

    # Compute relative phase differences (k >= 2)
    phi1 = out[:, 5].copy()
    for k in range(1, 5):
        n = k + 1  # harmonic number (2, 3, 4, 5)
        col_b = 5 + 2 * k
        out[:, col_b] = (out[:, col_b] / n - phi1) / (2.0 * np.pi / n) % 1

    return out


def compute_fourier_features(lightcurves, periods):
    """Compute Fourier decomposition features for a batch of light curves.

    Uses the periodfind Rust backend (weighted linear least-squares with
    BIC model selection) instead of the per-source scipy curve_fit loop.

    The Rust backend returns raw (A_n, B_n) Fourier coefficients.  These
    are converted to (amplitude, phase, relative_amplitude, relative_phase)
    form via ``_ab_to_amp_phi`` so that the output matches the feature
    column names (f1_amp, f1_phi0, f1_relamp1, f1_relphi1, ...).

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).  Raw magnitudes are used
        (no normalization to [0,1]).
    periods : array-like, shape (n_curves,)
        Best-fit period for each light curve.

    Returns
    -------
    features : ndarray, shape (n_curves, 14)
        Columns: [power, BIC, offset, slope, amp, phi0,
                  relamp1, relphi1, relamp2, relphi2,
                  relamp3, relphi3, relamp4, relphi4]
    """
    time_stack, mag_stack, err_stack = [], [], []
    for lightcurve in lightcurves:
        idx = np.argsort(lightcurve[0])
        time_stack.append(np.asarray(lightcurve[0][idx], dtype=np.float32))
        mag_stack.append(np.asarray(lightcurve[1][idx], dtype=np.float32))
        err_stack.append(np.asarray(lightcurve[2][idx], dtype=np.float32))

    periods_arr = np.asarray(periods, dtype=np.float32)

    fd = periodfind.FourierDecomposition(device='cpu')
    raw = fd.calc(time_stack, mag_stack, err_stack, periods_arr)
    return _ab_to_amp_phi(raw)


def compute_mhf_per_k_features(lightcurves, periods, max_harmonics=3, bands=None):
    """Compute per-harmonic MHF ΔBIC features at given periods.

    Evaluates the Multi-Harmonic Fourier model at each curve's best period
    and returns per-K ΔBIC values for morphology discrimination:
    - ΔBIC(K=3) >> ΔBIC(K=1) -> non-sinusoidal (sawtooth, eclipsing)
    - ΔBIC(K=3) ~ ΔBIC(K=1) -> sinusoidal (rotation, simple pulsation)

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).
    periods : array-like, shape (n_curves,)
        Best-fit period for each light curve.
    max_harmonics : int, default=3
        Maximum number of Fourier harmonics (1-5).
    bands : list of ndarray or None
        Per-point band identifiers for per-band mean subtraction.

    Returns
    -------
    features : ndarray, shape (n_curves, max_harmonics + 2)
        Each row: [ΔBIC_k0, ΔBIC_k1, ..., ΔBIC_kN, best_k]
    """
    time_stack, mag_stack, err_stack = _prepare_lightcurves(
        lightcurves,
        doSingleTimeSegment=False,
        return_errs=True,
        bands=bands,
    )

    periods_arr = np.asarray(periods, dtype=np.float32)

    mhf = periodfind.MultiHarmonicFourier(max_harmonics=max_harmonics, device='cpu')
    return mhf.calc_per_k(time_stack, mag_stack, err_stack, periods_arr)


def compute_dmdt_features(lightcurves, dmdt_ints):
    """Compute dm-dt histograms via Rust for a batch of light curves.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).
    dmdt_ints : dict
        Must contain 'dtints' and 'dmints' keys with bin edge arrays.

    Returns
    -------
    ndarray, shape (n_curves, n_dm_bins, n_dt_bins)
        L2-normalised dm-dt histograms.
    """
    time_stack, mag_stack = [], []
    for lc in lightcurves:
        time_stack.append(np.asarray(lc[0], dtype=np.float32))
        mag_stack.append(np.asarray(lc[1], dtype=np.float32))

    dt_edges = np.asarray(dmdt_ints['dtints'], dtype=np.float32)
    dm_edges = np.asarray(dmdt_ints['dmints'], dtype=np.float32)

    dd = periodfind.DmDt()
    return dd.calc(time_stack, mag_stack, dt_edges, dm_edges)


def compute_basic_stats(lightcurves):
    """Compute 22 basic statistics via Rust for a batch of light curves.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).

    Returns
    -------
    ndarray, shape (n_curves, 22)
    """
    time_stack, mag_stack, err_stack = [], [], []
    for lc in lightcurves:
        time_stack.append(np.asarray(lc[0], dtype=np.float32))
        mag_stack.append(np.asarray(lc[1], dtype=np.float32))
        err_stack.append(np.asarray(lc[2], dtype=np.float32))

    bs = periodfind.BasicStats()
    return bs.calc(time_stack, mag_stack, err_stack)


def remove_high_cadence_batch(tme_list, cadence_minutes=30.0):
    """Batch high-cadence removal via Rust.

    Parameters
    ----------
    tme_list : list of (times, mags, errs)
        Each element is a tuple of 1D arrays.
    cadence_minutes : float, default=30.0
        Minimum cadence in minutes.

    Returns
    -------
    list of (ndarray, ndarray, ndarray)
        Filtered (times, mags, errs) tuples.
    """
    times = [np.asarray(tme[0], dtype=np.float32) for tme in tme_list]
    mags = [np.asarray(tme[1], dtype=np.float32) for tme in tme_list]
    errs = [np.asarray(tme[2], dtype=np.float32) for tme in tme_list]

    return periodfind.remove_high_cadence(times, mags, errs, cadence_minutes)
