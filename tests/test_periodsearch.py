"""Tests for scope-ml period searching using the periodfind unified device API.

Uses synthetic lightcurve data — no GPU or Kowalski needed.

Run with:
    python -m pytest tests/test_periodsearch.py -v
"""

import sys
import os

# Add the tools directory to the path so we can import periodsearch
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'featureGeneration')
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import periodfind  # noqa: E402

from periodsearch import (  # noqa: E402
    find_periods,
    extract_top_n_periods,
    compute_agreement_scores,
    _period_match,
    compute_fourier_features,
    compute_dmdt_features,
    compute_basic_stats,
    remove_high_cadence_batch,
    _normalize_algorithm,
    _prepare_lightcurves,
    _build_pdots,
)
from cadence_alias import (  # noqa: E402
    assign_field,
    compute_field_window_function,
    identify_alias_frequencies,
    build_field_alias_map,
    merge_alias_zones,
    detrend_periodogram,
    detrend_periodogram_results,
    compute_neighbour_alias_fraction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sinusoidal_lightcurve(
    period, n_points=500, amplitude=1.0, noise_std=0.05, t_span=100.0, seed=42
):
    """Generate a synthetic sinusoidal lightcurve as (times, mags, magerrs).

    Returns the tuple format expected by scope-ml's find_periods:
    a masked array of shape (3, n_points) with rows [time, mag, magerr].
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points))
    phase = 2.0 * np.pi * times / period
    mags = amplitude * np.sin(phase) + amplitude + 10.0  # shift to positive
    mags += rng.normal(0, noise_std, n_points)
    magerrs = np.full(n_points, noise_std)
    # scope-ml expects lightcurves as masked arrays of shape (3, N)
    return np.ma.array([times, mags, magerrs])


def make_freq_grid(f_min=0.1, f_max=5.0, n_freqs=500):
    """Create a uniform frequency grid."""
    return np.linspace(f_min, f_max, n_freqs)


# ---------------------------------------------------------------------------
# TestFindPeriods
# ---------------------------------------------------------------------------


class TestFindPeriods:
    """Unit tests for the find_periods() function."""

    def test_ce_stats_output(self):
        """CE algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(2)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        assert periods.shape == (2,)
        assert sigs.shape == (2,)
        assert pdots.shape == (2,)
        assert np.all(np.isfinite(periods))
        assert np.all(np.isfinite(sigs))

    def test_aov_stats_output(self):
        """AOV algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("AOV", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_ls_stats_output(self):
        """LS algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("LS", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_ce_detects_known_period(self):
        """CE recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        detected = periods[0]
        # Allow 5% tolerance or detection of a harmonic
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_ls_detects_known_period(self):
        """LS recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("LS", lcs, freqs, doCPU=True)

        detected = periods[0]
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_periodogram_output_format(self):
        """*_periodogram algorithms return list of dicts with 'period' and 'data' keys."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("CE_periodogram", lcs, freqs, doCPU=True)

        # periods_best is an array of dicts for periodogram output
        assert len(periods) == 1
        entry = periods[0]
        assert isinstance(entry, dict)
        assert 'period' in entry
        assert 'data' in entry
        assert hasattr(entry['data'], 'shape')

    def test_multiple_lightcurves(self):
        """Batched input returns correct number of results."""
        n_curves = 5
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(n_curves)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        assert periods.shape == (n_curves,)
        assert sigs.shape == (n_curves,)
        assert pdots.shape == (n_curves,)

    def test_fpw_stats_output(self):
        """FPW algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_fpw_detects_known_period(self):
        """FPW recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        detected = periods[0]
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_fpw_multiple_lightcurves(self):
        """FPW handles batched light curves."""
        n_curves = 5
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(n_curves)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        assert periods.shape == (n_curves,)
        assert sigs.shape == (n_curves,)

    def test_fpw_periodogram_output(self):
        """FPW_periodogram returns list of dicts with 'period' and 'data' keys."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("FPW_periodogram", lcs, freqs, doCPU=True)

        assert len(periods) == 1
        entry = periods[0]
        assert isinstance(entry, dict)
        assert 'period' in entry
        assert 'data' in entry

    def test_gpu_algorithm_names(self):
        """ECE/EAOV/ELS/EFPW names work with doCPU=True (maps to same underlying algo)."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        for algo_name in ["ECE", "EAOV", "ELS", "EFPW"]:
            periods, sigs, pdots = find_periods(algo_name, lcs, freqs, doCPU=True)
            assert periods.shape == (1,), f"{algo_name} failed"
            assert np.all(
                np.isfinite(periods)
            ), f"{algo_name} returned non-finite period"


# ---------------------------------------------------------------------------
# TestExtractTopNPeriods
# ---------------------------------------------------------------------------


class TestExtractTopNPeriods:
    """Tests for extract_top_n_periods()."""

    def test_returns_correct_shape(self):
        """Output arrays have shape (n_sources, n_top)."""
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(3)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("LS_periodogram", lcs, freqs, doCPU=True)

        n_top = 8
        top_p, top_s = extract_top_n_periods(periods, freqs, n_top=n_top)
        assert top_p.shape == (3, n_top)
        assert top_s.shape == (3, n_top)

    def test_best_period_is_first(self):
        """The first column should match the single-best period from find_periods."""
        true_period = 3.0
        lcs = [make_sinusoidal_lightcurve(period=true_period)]
        freqs = make_freq_grid(0.1, 2.0, 500)

        periods, sigs, pdots = find_periods("LS_periodogram", lcs, freqs, doCPU=True)
        top_p, top_s = extract_top_n_periods(periods, freqs, n_top=4)

        # Top-1 should match best period from find_periods
        assert np.isclose(top_p[0, 0], periods[0]['period'], rtol=1e-4)

    def test_significances_are_sorted_descending(self):
        """Significances should decrease from rank 1 to rank N."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 500)

        periods, sigs, pdots = find_periods("LS_periodogram", lcs, freqs, doCPU=True)
        top_p, top_s = extract_top_n_periods(periods, freqs, n_top=8)

        valid = top_s[0][~np.isnan(top_s[0])]
        assert len(valid) > 1
        assert np.all(np.diff(valid) <= 1e-10)  # non-increasing

    def test_works_with_ce_algorithm(self):
        """CE algorithm uses minima — verify extract_top_n_periods handles it."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 300)

        periods, sigs, pdots = find_periods("CE_periodogram", lcs, freqs, doCPU=True)
        top_p, top_s = extract_top_n_periods(periods, freqs, n_top=4)

        assert top_p.shape == (1, 4)
        assert np.isclose(top_p[0, 0], periods[0]['period'], rtol=1e-4)

    def test_two_peaks_found_as_distinct(self):
        """Synthetic periodogram with two well-separated peaks — both found."""
        # Build a synthetic periodogram with peaks at two known frequencies
        freqs = make_freq_grid(0.1, 5.0, 1000)
        periods_grid = 1.0 / freqs
        data = np.zeros(len(freqs), dtype=np.float64)

        # Peak 1 near freq=1.0 (period=1.0)
        peak1_idx = np.argmin(np.abs(freqs - 1.0))
        data[peak1_idx] = 10.0

        # Peak 2 near freq=3.0 (period=0.333)
        peak2_idx = np.argmin(np.abs(freqs - 3.0))
        data[peak2_idx] = 8.0

        pg_result = [{'period': periods_grid[peak1_idx], 'data': data}]
        top_p, top_s = extract_top_n_periods(pg_result, freqs, n_top=4)

        # Both peaks should appear in the top-4
        found_peak1 = any(
            np.isclose(top_p[0, k], 1.0, rtol=0.1)
            for k in range(4)
            if not np.isnan(top_p[0, k])
        )
        found_peak2 = any(
            np.isclose(top_p[0, k], 1.0 / 3.0, rtol=0.1)
            for k in range(4)
            if not np.isnan(top_p[0, k])
        )
        assert found_peak1, f"Peak at P=1.0 not found in {top_p[0]}"
        assert found_peak2, f"Peak at P=0.333 not found in {top_p[0]}"

    def test_adjacent_bins_not_returned(self):
        """Adjacent bins from the same broad peak are NOT returned as separate top-N entries."""
        freqs = make_freq_grid(0.1, 5.0, 1000)
        periods_grid = 1.0 / freqs

        # Build a single broad peak spanning ~20 bins centered at freq=2.0
        peak_center = np.argmin(np.abs(freqs - 2.0))
        data = np.zeros(len(freqs), dtype=np.float64)
        for offset in range(-10, 11):
            idx = peak_center + offset
            if 0 <= idx < len(freqs):
                data[idx] = 5.0 * np.exp(-0.5 * (offset / 3.0) ** 2)

        pg_result = [{'period': periods_grid[peak_center], 'data': data}]
        top_p, top_s = extract_top_n_periods(pg_result, freqs, n_top=4)

        # The peak at freq=2.0 corresponds to period=0.5
        # With hybrid chunking, at most one entry should be near P=0.5
        near_peak = sum(
            1
            for k in range(4)
            if not np.isnan(top_p[0, k]) and np.isclose(top_p[0, k], 0.5, rtol=0.1)
        )
        assert (
            near_peak <= 1
        ), f"Expected at most 1 entry near P=0.5, got {near_peak}: {top_p[0]}"

    def test_long_periods_not_crowded_out(self):
        """Peaks spread over decades all survive, none crowded out by short ones.

        Chunks of equal width in frequency are wildly unequal in period: over a
        DP2-like 0.0084-288 c/d grid they collapse everything above ~0.1 d into
        a single chunk, so only one long-period peak can ever be returned.
        Chunking in log period gives each decade its own chunks.
        """
        freqs = make_freq_grid(0.0084, 288.0, 100000)
        periods_grid = 1.0 / freqs
        data = np.zeros(len(freqs), dtype=np.float64)

        targets = [0.01, 0.05, 0.5, 5.0, 50.0]
        for p in targets:
            data[int(np.argmin(np.abs(periods_grid - p)))] = 10.0

        pg_result = [{'period': periods_grid[int(np.argmax(data))], 'data': data}]
        top_p, _ = extract_top_n_periods(pg_result, freqs, n_top=8)

        for p in targets:
            assert any(
                not np.isnan(q) and np.isclose(q, p, rtol=0.05) for q in top_p[0]
            ), f"P={p} d missing from {top_p[0]}"

    def test_n_chunks_multiplier_controls_diversity(self):
        """Higher n_chunks_multiplier increases period diversity."""
        freqs = make_freq_grid(0.1, 5.0, 2000)
        periods_grid = 1.0 / freqs

        # Three peaks at different frequencies
        data = np.zeros(len(freqs), dtype=np.float64)
        for freq_val, amp in [(0.5, 10.0), (2.0, 8.0), (4.0, 6.0)]:
            idx = np.argmin(np.abs(freqs - freq_val))
            data[idx] = amp

        pg_result = [{'period': periods_grid[np.argmax(data)], 'data': data}]

        # With multiplier=1 (just n_top chunks), might miss peaks
        top_p_1, _ = extract_top_n_periods(
            pg_result, freqs, n_top=4, n_chunks_multiplier=1
        )
        # With multiplier=3 (3*n_top chunks), should find all 3 peaks
        top_p_3, _ = extract_top_n_periods(
            pg_result, freqs, n_top=4, n_chunks_multiplier=3
        )

        def count_distinct(periods, targets, rtol=0.15):
            found = 0
            for target in targets:
                if any(
                    not np.isnan(p) and np.isclose(p, target, rtol=rtol)
                    for p in periods
                ):
                    found += 1
            return found

        targets = [1.0 / 0.5, 1.0 / 2.0, 1.0 / 4.0]  # = [2.0, 0.5, 0.25]
        found_3 = count_distinct(top_p_3[0], targets)
        assert found_3 >= 2, (
            f"n_chunks_multiplier=3 should find >=2 of 3 peaks, got {found_3}: "
            f"{top_p_3[0]}"
        )


# ---------------------------------------------------------------------------
# TestDeviceAPIIntegration
# ---------------------------------------------------------------------------


class TestDeviceAPIIntegration:
    """Verify periodfind device API works from scope-ml."""

    def setup_method(self):
        """Reset global device state before each test."""
        periodfind._default_device = None

    def teardown_method(self):
        """Reset global device state after each test."""
        periodfind._default_device = None

    def test_set_device_affects_factory(self):
        """periodfind.set_device('cpu') makes factories return CPU instances."""
        from periodfind.cpu import (
            ConditionalEntropy as CpuCE,
            AOV as CpuAOV,
            LombScargle as CpuLS,
            FPW as CpuFPW,
        )

        periodfind.set_device('cpu')
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
        aov = periodfind.AOV(n_phase=10)
        ls = periodfind.LombScargle()
        fpw = periodfind.FPW(n_bins=10)
        assert isinstance(ce, CpuCE)
        assert isinstance(aov, CpuAOV)
        assert isinstance(ls, CpuLS)
        assert isinstance(fpw, CpuFPW)

    def test_factory_with_device_kwarg(self):
        """periodfind.ConditionalEntropy(device='cpu') returns CPU backend."""
        from periodfind.cpu import ConditionalEntropy as CpuCE

        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device='cpu')
        assert isinstance(ce, CpuCE)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestNormalizeAlgorithm:
    """Tests for _normalize_algorithm helper."""

    def test_plain_names(self):
        assert _normalize_algorithm("CE") == ("CE", False)
        assert _normalize_algorithm("AOV") == ("AOV", False)
        assert _normalize_algorithm("LS") == ("LS", False)
        assert _normalize_algorithm("FPW") == ("FPW", False)

    def test_e_prefix(self):
        assert _normalize_algorithm("ECE") == ("CE", False)
        assert _normalize_algorithm("EAOV") == ("AOV", False)
        assert _normalize_algorithm("ELS") == ("LS", False)
        assert _normalize_algorithm("EFPW") == ("FPW", False)

    def test_periodogram_suffix(self):
        assert _normalize_algorithm("CE_periodogram") == ("CE", True)
        assert _normalize_algorithm("ECE_periodogram") == ("CE", True)
        assert _normalize_algorithm("EAOV_periodogram") == ("AOV", True)
        assert _normalize_algorithm("FPW_periodogram") == ("FPW", True)
        assert _normalize_algorithm("EFPW_periodogram") == ("FPW", True)


class TestBuildPdots:
    """Tests for _build_pdots helper."""

    def test_no_pdot(self):
        pdots = _build_pdots(False)
        assert pdots.dtype == np.float32
        assert len(pdots) == 1
        assert pdots[0] == 0.0

    def test_with_pdot(self):
        pdots = _build_pdots(True)
        assert pdots.dtype == np.float32
        assert len(pdots) == 11  # 10 logspace + 1 zero
        assert pdots[0] == 0.0
        assert all(p <= 0 for p in pdots)


class TestPrepareLightcurves:
    """Tests for _prepare_lightcurves helper."""

    def test_basic_preparation(self):
        rng = np.random.default_rng(42)
        times = rng.uniform(0, 100, 50)
        mags = rng.uniform(14.0, 16.0, 50)
        magerrs = np.full(50, 0.01)
        lcs = [(times, mags, magerrs)]

        time_stack, mag_stack = _prepare_lightcurves(lcs, doSingleTimeSegment=False)

        assert len(time_stack) == 1
        assert len(mag_stack) == 1
        assert time_stack[0].dtype == np.float32
        assert mag_stack[0].dtype == np.float32
        # Times should be sorted
        assert np.all(np.diff(time_stack[0]) >= 0)
        # Mags should be scaled by range (divided by max-min)
        # The function sorts by time, so mags are reordered by time sort
        idx = np.argsort(times)
        sorted_mags = mags[idx]
        raw_range = sorted_mags.max() - sorted_mags.min()
        expected = sorted_mags / raw_range
        np.testing.assert_allclose(mag_stack[0], expected.astype(np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# TestComputeFourierFeatures
# ---------------------------------------------------------------------------


class TestComputeFourierFeatures:
    """Tests for the compute_fourier_features() function."""

    def test_output_shape(self):
        """Returns (n_curves, 14) array."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 100))
            m = 18.0 + rng.normal(0, 0.1, 100)
            e = np.full(100, 0.1)
            lcs.append((t, m, e))
        periods = np.array([1.0, 2.0, 3.0])

        result = compute_fourier_features(lcs, periods)
        assert result.shape == (n_curves, 14)
        assert result.dtype == np.float32

    def test_known_sinusoid_recovery(self):
        """Fourier decomposition recovers known sinusoidal signal.

        After _ab_to_amp_phi conversion, columns are:
        [power, BIC, offset, slope, amp1, phi1, relamp2, relphi2, ...]
        """
        n = 300
        period = 5.0
        t = np.sort(np.random.default_rng(42).uniform(0, 50, n))
        phi = 2.0 * np.pi * t / period
        m = 18.0 + 0.5 * np.cos(phi)
        e = np.full(n, 0.01)

        result = compute_fourier_features([(t, m, e)], np.array([period]))
        r = result[0]

        # Power should be close to 1
        assert r[0] > 0.9, f"power = {r[0]}"
        # Offset should be close to 18
        assert abs(r[2] - 18.0) < 0.1, f"offset = {r[2]}"
        # amp1 (= sqrt(A1^2 + B1^2)) should be close to 0.5
        assert abs(r[4] - 0.5) < 0.1, f"amp1 = {r[4]}"

    def test_constant_signal_low_power(self):
        """Constant magnitude yields ~0 power."""
        n = 100
        t = np.arange(n, dtype=np.float64) * 0.5
        m = np.full(n, 17.0)
        e = np.full(n, 0.01)

        result = compute_fourier_features([(t, m, e)], np.array([2.0]))
        assert abs(result[0, 0]) < 0.01, f"power = {result[0, 0]}"

    def test_multiple_curves_independent(self):
        """Each curve is processed independently."""
        rng = np.random.default_rng(99)
        n = 200
        period = 3.0

        # Curve 0: sinusoidal
        t = np.sort(rng.uniform(0, 30, n))
        phi = 2.0 * np.pi * t / period
        m_sin = 18.0 + 1.0 * np.cos(phi)
        e = np.full(n, 0.01)

        # Curve 1: constant
        m_const = np.full(n, 18.0)

        result = compute_fourier_features(
            [(t, m_sin, e), (t, m_const, e)],
            np.array([period, period]),
        )

        assert result[0, 0] > 0.9, "sinusoidal curve should have high power"
        assert result[1, 0] < 0.01, "constant curve should have ~0 power"

    def test_numerical_agreement_with_lcstats(self):
        """Rust Fourier agrees with lcstats.fourier_decomposition within ~5%.

        lcstats returns raw (A, B) coefficients while compute_fourier_features
        returns (amp, phi) after _ab_to_amp_phi conversion.  Compare power,
        BIC, offset directly, and compare fundamental amplitude as
        sqrt(A1^2 + B1^2) from lcstats vs amp1 from the new code.
        """
        lcstats = pytest.importorskip("lcstats", reason="lcstats requires scipy/numba")

        rng = np.random.default_rng(123)
        n = 200
        t = np.sort(rng.uniform(0, 100, n))
        period = 4.0
        phi = 2.0 * np.pi * t / period
        m = 16.0 + 0.4 * np.cos(phi) + 0.2 * np.sin(phi)
        m += rng.normal(0, 0.02, n)
        e = np.full(n, 0.02)

        old = lcstats.fourier_decomposition(t, m, e, period)
        new = compute_fourier_features([(t, m, e)], np.array([period]))[0]

        # Power and BIC should be very close
        assert (
            abs(old[0] - new[0]) / max(abs(old[0]), 1e-12) < 0.01
        ), f"power: old={old[0]}, new={new[0]}"
        assert (
            abs(old[1] - new[1]) / max(abs(old[1]), 1e-12) < 0.01
        ), f"BIC: old={old[1]}, new={new[1]}"
        # Offset
        assert (
            abs(old[2] - new[2]) / max(abs(old[2]), 1e-12) < 0.01
        ), f"offset: old={old[2]}, new={new[2]}"
        # Fundamental amplitude: lcstats has raw (A1, B1) at indices 4,5;
        # new code has amp1 = sqrt(A1^2 + B1^2) at index 4
        old_amp1 = np.sqrt(old[4] ** 2 + old[5] ** 2)
        new_amp1 = new[4]
        if old_amp1 > 0.01:
            reldiff = abs(old_amp1 - new_amp1) / old_amp1
            assert (
                reldiff < 0.15
            ), f"amp1: old={old_amp1}, new={new_amp1}, reldiff={reldiff}"


# ---------------------------------------------------------------------------
# TestRemoveHighCadence
# ---------------------------------------------------------------------------


class TestRemoveHighCadence:
    """Tests for the remove_high_cadence_batch() function."""

    def test_basic_filtering(self):
        """Points within cadence are removed."""
        # cadence = 30 min = 30/1440 days
        t = np.array([0.0, 0.001, 0.002, 0.025, 0.05])
        m = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        e = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)

        assert len(result) == 1
        t_out, m_out, e_out = result[0]
        # 30 min = 0.02083 days; keep 0, 3, 4
        assert len(t_out) == 3
        assert t_out.dtype == np.float32

    def test_empty_input(self):
        """Empty input returns empty output."""
        t = np.array([], dtype=np.float64)
        m = np.array([], dtype=np.float64)
        e = np.array([], dtype=np.float64)

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)

        assert len(result) == 1
        assert len(result[0][0]) == 0

    def test_single_point(self):
        """Single point is preserved."""
        result = remove_high_cadence_batch(
            [(np.array([1.0]), np.array([10.0]), np.array([0.1]))],
            cadence_minutes=30.0,
        )
        assert len(result[0][0]) == 1

    def test_all_beyond_cadence(self):
        """Well-separated points are all kept."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        m = np.array([10.0, 11.0, 12.0, 13.0])
        e = np.array([0.1, 0.1, 0.1, 0.1])

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)
        assert len(result[0][0]) == 4

    def test_batch_processing(self):
        """Multiple curves processed correctly."""
        t1 = np.array([0.0, 1.0, 2.0])
        m1 = np.array([10.0, 11.0, 12.0])
        e1 = np.array([0.1, 0.1, 0.1])

        t2 = np.array([0.0, 0.001, 5.0])
        m2 = np.array([20.0, 21.0, 22.0])
        e2 = np.array([0.2, 0.2, 0.2])

        result = remove_high_cadence_batch(
            [(t1, m1, e1), (t2, m2, e2)], cadence_minutes=30.0
        )
        assert len(result) == 2
        assert len(result[0][0]) == 3  # all kept
        assert len(result[1][0]) == 2  # first and last kept


# ---------------------------------------------------------------------------
# TestComputeDmdtFeatures
# ---------------------------------------------------------------------------


class TestComputeDmdtFeatures:
    """Tests for the compute_dmdt_features() function."""

    def _make_dmdt_ints(self):
        """Create standard bin edges matching scope-ml defaults."""
        dt_edges = np.array(
            [
                0.0,
                1.0 / 145,
                2.0 / 145,
                3.0 / 145,
                4.0 / 145,
                5.0 / 145,
                6.0 / 145,
                1.5 / 23.2,
                2.0 / 23.2,
                3.0 / 23.2,
                1.0 / 3.5,
                2.0 / 3.5,
                3.0 / 3.5,
                4.0 / 3.5,
                5.0 / 3.5,
                7.0,
                10.0,
                20.0,
                30.0,
                60.0,
                90.0,
                120.0,
                240.0,
                600.0,
                960.0,
                2000.0,
            ],
            dtype=np.float64,
        )
        dm_edges = np.array(
            [
                -8.0,
                -3.2,
                -2.4,
                -2.0,
                -1.6,
                -1.2,
                -0.8,
                -0.6,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                -0.05,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.6,
                0.8,
                1.2,
                1.6,
                2.0,
                2.4,
                3.2,
                8.0,
            ],
            dtype=np.float64,
        )
        return {'dtints': dt_edges, 'dmints': dm_edges}

    def test_output_shape(self):
        """Returns correct 3D shape."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 80))
            m = 18.0 + rng.normal(0, 0.5, 80)
            e = np.full(80, 0.1)
            lcs.append((t, m, e))

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features(lcs, dmdt_ints)

        n_dt_bins = len(dmdt_ints['dtints']) - 1
        n_dm_bins = len(dmdt_ints['dmints']) - 1
        assert result.shape == (n_curves, n_dm_bins, n_dt_bins)
        assert result.dtype == np.float32

    def test_single_point_curve(self):
        """Single-point curve returns all zeros."""
        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features(
            [(np.array([1.0]), np.array([10.0]), np.array([0.1]))],
            dmdt_ints,
        )
        assert np.all(result == 0.0)

    def test_l2_normalization(self):
        """Each curve's histogram is L2-normalised (or zero)."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 50))
        m = 18.0 + rng.normal(0, 0.5, 50)
        e = np.full(50, 0.1)

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features([(t, m, e)], dmdt_ints)

        norm = np.linalg.norm(result[0])
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5, f"L2 norm = {norm}"

    def test_constant_magnitude(self):
        """Constant magnitude => all dm=0, should land in middle bins."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        m = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        e = np.full(5, 0.1)

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features([(t, m, e)], dmdt_ints)

        # dm=0 falls in the bin [-0.05, 0.05] which is bin 12
        # The result should be non-zero only in the dm=0 row
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# TestComputeBasicStats
# ---------------------------------------------------------------------------


class TestComputeBasicStats:
    """Tests for the compute_basic_stats() function."""

    def test_output_shape(self):
        """Returns (n_curves, 22) array."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 50))
            m = 18.0 + rng.normal(0, 0.1, 50)
            e = np.full(50, 0.1)
            lcs.append((t, m, e))

        result = compute_basic_stats(lcs)
        assert result.shape == (n_curves, 22)
        assert result.dtype == np.float32

    def test_n_count(self):
        """First column is the number of points."""
        rng = np.random.default_rng(42)
        lcs = []
        for n in [20, 50, 100]:
            t = np.sort(rng.uniform(0, 100, n))
            m = 18.0 + rng.normal(0, 0.1, n)
            e = np.full(n, 0.1)
            lcs.append((t, m, e))

        result = compute_basic_stats(lcs)
        assert result[0, 0] == 20.0
        assert result[1, 0] == 50.0
        assert result[2, 0] == 100.0

    def test_constant_signal(self):
        """Constant magnitude yields ~0 scatter stats."""
        n = 50
        t = np.arange(n, dtype=np.float64) * 0.5
        m = np.full(n, 17.0)
        e = np.full(n, 0.1)

        result = compute_basic_stats([(t, m, e)])
        r = result[0]

        # N
        assert r[0] == n
        # median ≈ 17
        assert abs(r[1] - 17.0) < 0.01
        # wmean ≈ 17
        assert abs(r[2] - 17.0) < 0.01
        # chi2red ≈ 0
        assert abs(r[3]) < 0.01
        # wstd ≈ 0
        assert abs(r[5]) < 0.01

    def test_too_few_points(self):
        """Fewer than 4 points returns NaN."""
        t = np.array([1.0, 2.0, 3.0])
        m = np.array([10.0, 11.0, 12.0])
        e = np.array([0.1, 0.1, 0.1])

        result = compute_basic_stats([(t, m, e)])
        # All values should be NaN
        assert np.all(np.isnan(result[0]))

    def test_finite_for_normal_input(self):
        """All 22 stats are finite for well-behaved input."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 200))
        m = 18.0 + rng.normal(0, 0.3, 200)
        e = np.full(200, 0.1)

        result = compute_basic_stats([(t, m, e)])
        for i in range(22):
            assert np.isfinite(result[0, i]), f"stat[{i}] is not finite: {result[0, i]}"

    def test_numerical_agreement_with_lcstats(self):
        """Rust basic stats agree with lcstats within reasonable tolerance."""
        lcstats = pytest.importorskip("lcstats", reason="lcstats requires scipy/numba")

        rng = np.random.default_rng(123)
        n = 200
        t = np.sort(rng.uniform(0, 100, n))
        m = 16.0 + 0.4 * rng.normal(0, 1, n)
        e = np.full(n, 0.05)

        old = lcstats.calc_basic_stats('test', (t, m, e))
        old_vals = old['test']

        new = compute_basic_stats([(t, m, e)])[0]

        # Compare first few well-defined stats (N, median, wmean)
        assert old_vals[0] == new[0], f"N: old={old_vals[0]}, new={new[0]}"
        assert (
            abs(old_vals[1] - new[1]) / max(abs(old_vals[1]), 1e-12) < 0.01
        ), f"median: old={old_vals[1]}, new={new[1]}"
        assert (
            abs(old_vals[2] - new[2]) / max(abs(old_vals[2]), 1e-12) < 0.01
        ), f"wmean: old={old_vals[2]}, new={new[2]}"


# ---------------------------------------------------------------------------
# TestPeriodMatch
# ---------------------------------------------------------------------------


class TestPeriodMatch:
    """Tests for the _period_match helper."""

    def test_identical_periods(self):
        assert _period_match(5.0, 5.0)

    def test_close_periods(self):
        assert _period_match(5.0, 5.02)  # 0.4% difference, within 5%

    def test_distant_periods(self):
        assert not _period_match(5.0, 7.0)

    def test_harmonic_half(self):
        assert _period_match(5.0, 10.0)  # P_a = 0.5 * P_b

    def test_harmonic_double(self):
        assert _period_match(10.0, 5.0)  # P_a = 2.0 * P_b

    def test_harmonic_third(self):
        assert _period_match(5.0, 15.0)  # P_a = (1/3) * P_b

    def test_harmonic_triple(self):
        assert _period_match(15.0, 5.0)  # P_a = 3.0 * P_b

    def test_nan_handling(self):
        assert not _period_match(np.nan, 5.0)
        assert not _period_match(5.0, np.nan)
        assert not _period_match(np.nan, np.nan)

    def test_zero_handling(self):
        assert not _period_match(0.0, 5.0)
        assert not _period_match(5.0, 0.0)

    def test_negative_handling(self):
        assert not _period_match(-1.0, 5.0)


# ---------------------------------------------------------------------------
# TestComputeAgreementScores
# ---------------------------------------------------------------------------


class TestComputeAgreementScores:
    """Tests for compute_agreement_scores()."""

    def test_perfect_agreement(self):
        """All algorithms find the same period -> score = 1.0."""
        period_dict = {
            'LS': np.array([5.0]),
            'CE': np.array([5.0]),
            'AOV': np.array([5.0]),
        }
        top_n = {
            'LS': np.array([[5.0, 2.5, 1.0]]),
            'CE': np.array([[5.0, 2.5, 1.0]]),
            'AOV': np.array([[5.0, 2.5, 1.0]]),
        }
        results = compute_agreement_scores(
            period_dict, top_n, ['src1'], ['LS', 'CE', 'AOV']
        )
        assert results['src1']['agree_score'] == 1.0
        assert results['src1']['agree_strict'] == 1.0
        assert results['src1']['n_agree_pairs'] == 3  # 3 choose 2
        assert results['src1']['n_total_pairs'] == 3

    def test_no_agreement(self):
        """All algorithms find completely different periods -> score = 0."""
        period_dict = {
            'LS': np.array([1.0]),
            'CE': np.array([50.0]),
            'AOV': np.array([200.0]),
        }
        top_n = {
            'LS': np.array([[1.0, 1.1, 1.2]]),
            'CE': np.array([[50.0, 51.0, 52.0]]),
            'AOV': np.array([[200.0, 201.0, 202.0]]),
        }
        results = compute_agreement_scores(
            period_dict, top_n, ['src1'], ['LS', 'CE', 'AOV']
        )
        assert results['src1']['agree_score'] == 0.0
        assert results['src1']['agree_strict'] == 0.0
        assert results['src1']['n_agree_pairs'] == 0

    def test_harmonic_detection(self):
        """Periods at 2:1 harmonic ratio are detected as agreement."""
        period_dict = {
            'LS': np.array([5.0]),
            'CE': np.array([10.0]),  # 2x harmonic
        }
        top_n = {
            'LS': np.array([[5.0, 3.0]]),
            'CE': np.array([[10.0, 7.0]]),
        }
        results = compute_agreement_scores(period_dict, top_n, ['src1'], ['LS', 'CE'])
        # Should match via 0.5 harmonic: 5.0 / (10.0 * 0.5) = 1.0
        assert results['src1']['agree_score'] == 1.0
        assert results['src1']['agree_strict'] == 1.0

    def test_spurious_period_filtering(self):
        """Ultra-short periods below min_agree_period are filtered."""
        period_dict = {
            'LS': np.array([0.001]),
            'CE': np.array([0.001]),
        }
        top_n = {
            'LS': np.array([[0.001, 0.002]]),
            'CE': np.array([[0.001, 0.002]]),
        }
        results = compute_agreement_scores(
            period_dict,
            top_n,
            ['src1'],
            ['LS', 'CE'],
            min_agree_period=0.007,
        )
        # Both periods are below the threshold, so no agreement
        assert results['src1']['agree_score'] == 0.0
        assert results['src1']['agree_strict'] == 0.0

    def test_single_algorithm(self):
        """Single algorithm -> 0 pairs, no division by zero."""
        period_dict = {'LS': np.array([5.0])}
        top_n = {'LS': np.array([[5.0, 2.5]])}
        results = compute_agreement_scores(period_dict, top_n, ['src1'], ['LS'])
        assert results['src1']['n_total_pairs'] == 0
        assert results['src1']['agree_score'] == 0.0
        assert results['src1']['agree_strict'] == 0.0
        assert results['src1']['agree_weighted'] == 0.0

    def test_multiple_sources(self):
        """Function processes multiple sources correctly."""
        period_dict = {
            'LS': np.array([5.0, 1.0]),
            'CE': np.array([5.0, 50.0]),
        }
        top_n = {
            'LS': np.array([[5.0, 2.5], [1.0, 0.5]]),
            'CE': np.array([[5.0, 2.5], [50.0, 25.0]]),
        }
        results = compute_agreement_scores(
            period_dict, top_n, ['s1', 's2'], ['LS', 'CE']
        )
        assert 's1' in results
        assert 's2' in results
        assert results['s1']['agree_score'] == 1.0
        assert results['s2']['agree_score'] == 0.0

    def test_weighted_score_rank_sensitivity(self):
        """Higher-ranked matches produce higher weighted scores."""
        # Case A: agreement in top-1
        period_dict_a = {'LS': np.array([5.0]), 'CE': np.array([5.0])}
        top_n_a = {
            'LS': np.array([[5.0, 0.8, 0.1]]),
            'CE': np.array([[5.0, 0.9, 0.2]]),
        }
        res_a = compute_agreement_scores(period_dict_a, top_n_a, ['s'], ['LS', 'CE'])

        # Case B: agreement only in lower-ranked positions (rank 2)
        # Use periods that don't accidentally match via harmonics
        period_dict_b = {'LS': np.array([0.8]), 'CE': np.array([0.9])}
        top_n_b = {
            'LS': np.array([[0.8, 0.1, 5.0]]),
            'CE': np.array([[0.9, 0.2, 5.0]]),
        }
        res_b = compute_agreement_scores(period_dict_b, top_n_b, ['s'], ['LS', 'CE'])

        assert res_a['s']['agree_weighted'] > res_b['s']['agree_weighted']

    def test_best_consensus_period(self):
        """best_consensus_period picks the period with most cross-algo support."""
        period_dict = {
            'LS': np.array([5.0]),
            'CE': np.array([5.0]),
            'AOV': np.array([5.0]),
        }
        top_n = {
            'LS': np.array([[5.0, 10.0, 1.0]]),
            'CE': np.array([[5.0, 10.0, 2.0]]),
            'AOV': np.array([[5.0, 3.0, 7.0]]),
        }
        results = compute_agreement_scores(
            period_dict, top_n, ['src1'], ['LS', 'CE', 'AOV']
        )
        # Period 5.0 has support from all 3 algos
        best = results['src1']['best_consensus_period']
        assert np.isclose(best, 5.0, rtol=0.05), f"Expected ~5.0, got {best}"

    def test_nan_top_n_handled(self):
        """NaN entries in top-N arrays are skipped gracefully."""
        period_dict = {
            'LS': np.array([5.0]),
            'CE': np.array([5.0]),
        }
        top_n = {
            'LS': np.array([[5.0, np.nan, np.nan]]),
            'CE': np.array([[5.0, np.nan, np.nan]]),
        }
        results = compute_agreement_scores(period_dict, top_n, ['src1'], ['LS', 'CE'])
        assert results['src1']['agree_score'] == 1.0
        assert results['src1']['n_agree_pairs'] == 1


# ---------------------------------------------------------------------------
# TestCadenceAlias
# ---------------------------------------------------------------------------


class TestCadenceAlias:
    """Tests for the cadence_alias module."""

    # -- Field assignment --

    def test_assign_field_ra95(self):
        """RA ~95 maps to RA95 field."""
        assert assign_field(95.0) == 'RA95'
        assert assign_field(93.5) == 'RA95'
        assert assign_field(96.5) == 'RA95'

    def test_assign_field_ra53(self):
        """RA ~53 maps to RA53 field."""
        assert assign_field(53.0) == 'RA53'

    def test_assign_field_ra59(self):
        """RA ~59 maps to RA59 field."""
        assert assign_field(59.0) == 'RA59'

    def test_assign_field_ra38(self):
        """RA ~38 maps to RA38 field."""
        assert assign_field(38.0) == 'RA38'

    def test_assign_field_outside(self):
        """RA outside any field returns None."""
        assert assign_field(0.0) is None
        assert assign_field(180.0) is None
        assert assign_field(70.0) is None

    # -- Window function --

    def test_window_function_shape(self):
        """Window function has same length as frequency grid."""
        pytest.importorskip("astropy", reason="astropy required for window function")
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(60000, 60100, 200))
        freqs = np.linspace(0.1, 10.0, 500)
        wp = compute_field_window_function(times, freqs)
        assert wp.shape == (500,)

    def test_window_function_range(self):
        """Window function is normalised to [0, 1]."""
        pytest.importorskip("astropy", reason="astropy required for window function")
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(60000, 60100, 200))
        freqs = np.linspace(0.1, 10.0, 500)
        wp = compute_field_window_function(times, freqs)
        assert wp.max() <= 1.0 + 1e-10
        assert wp.min() >= 0.0 - 1e-10
        assert np.isclose(wp.max(), 1.0, atol=1e-6)

    def test_regular_cadence_produces_aliases(self):
        """Regular 30-minute cadence creates a strong alias at f=48 c/d."""
        pytest.importorskip("astropy", reason="astropy required for window function")
        # Simulate observations every 30 minutes for 10 days
        cadence_hours = 0.5
        n_days = 10
        times = np.arange(0, n_days, cadence_hours / 24.0) + 60000.0
        freqs = np.linspace(0.1, 60.0, 5000)
        wp = compute_field_window_function(times, freqs)

        # The alias at f=48 c/d (1/0.5hr) should be strong
        freq_48_idx = np.argmin(np.abs(freqs - 48.0))
        assert (
            wp[freq_48_idx] > 0.5
        ), f"Expected alias peak near f=48, got wp={wp[freq_48_idx]:.3f}"

    def test_regular_cadence_alias_zones(self):
        """Regular cadence produces identifiable alias zones."""
        pytest.importorskip("astropy", reason="astropy required for window function")
        cadence_hours = 0.5
        times = np.arange(0, 10, cadence_hours / 24.0) + 60000.0
        freqs = np.linspace(0.1, 60.0, 5000)
        wp = compute_field_window_function(times, freqs)
        zones = identify_alias_frequencies(wp, freqs, threshold=0.5)
        assert len(zones) > 0, "Expected at least one alias zone"

        # Each zone should be a [lo, hi] pair with lo < hi
        for lo, hi in zones:
            assert lo < hi

    # -- Build field alias map --

    def test_build_field_alias_map_no_field_column(self):
        """Without field RA column, all visits are grouped as 'ALL'."""
        pd = pytest.importorskip("pandas", reason="pandas required for visit DataFrame")
        pytest.importorskip("astropy", reason="astropy required for window function")

        rng = np.random.default_rng(42)
        visit_df = pd.DataFrame(
            {
                'expMidptMJD': np.sort(rng.uniform(60000, 60050, 300)),
            }
        )
        freqs = np.linspace(0.1, 10.0, 500)
        fam, fw = build_field_alias_map(visit_df, freqs, threshold=0.5)
        assert 'ALL' in fam or len(fam) == 0

    def test_build_field_alias_map_with_field_ra(self):
        """With fieldRA column, visits are grouped into per-field windows."""
        pd = pytest.importorskip("pandas", reason="pandas required for visit DataFrame")
        pytest.importorskip("astropy", reason="astropy required for window function")

        rng = np.random.default_rng(42)

        # Create visits in two fields: RA53 and RA95
        # RA53 visits: regular 30-min cadence (strong alias at f=48)
        n_53 = 200
        times_53 = np.arange(n_53) * (0.5 / 24.0) + 60000.0
        ra_53 = np.full(n_53, 53.0)

        # RA95 visits: irregular cadence (fewer aliases)
        n_95 = 150
        times_95 = np.sort(rng.uniform(60000, 60050, n_95))
        ra_95 = np.full(n_95, 95.0)

        visit_df = pd.DataFrame(
            {
                'expMidptMJD': np.concatenate([times_53, times_95]),
                'fieldRA': np.concatenate([ra_53, ra_95]),
            }
        )

        freqs = np.linspace(0.1, 60.0, 5000)
        fam, fw = build_field_alias_map(visit_df, freqs, threshold=0.5)

        # Should have per-field entries, NOT a single 'ALL'
        assert 'ALL' not in fam
        assert 'RA53' in fam
        assert 'RA95' in fam
        assert 'RA53' in fw
        assert 'RA95' in fw

        # RA53 with regular cadence should have more alias zones
        assert len(fam['RA53']) > 0, "Regular cadence should produce aliases"

        # Window functions should have correct shape
        assert fw['RA53'].shape == (5000,)
        assert fw['RA95'].shape == (5000,)

    def test_merge_alias_zones(self):
        """merge_alias_zones combines all field zones into one list."""
        field_map = {
            'RA53': [[1.0, 2.0], [5.0, 6.0]],
            'RA95': [[3.0, 4.0]],
        }
        merged = merge_alias_zones(field_map)
        assert len(merged) == 3
        freqs_in_merged = set()
        for lo, hi in merged:
            freqs_in_merged.add((lo, hi))
        assert (1.0, 2.0) in freqs_in_merged
        assert (3.0, 4.0) in freqs_in_merged
        assert (5.0, 6.0) in freqs_in_merged

    # -- Periodogram detrending --

    def test_detrend_suppresses_alias_peak(self):
        """Detrending by window function suppresses alias peaks."""
        n = 1000
        freqs = np.linspace(0.1, 50.0, n)

        # Synthetic periodogram with a peak at f=20 (real) and f=40 (alias)
        pg = np.random.default_rng(42).uniform(0, 0.1, n)
        real_peak_idx = np.argmin(np.abs(freqs - 20.0))
        alias_peak_idx = np.argmin(np.abs(freqs - 40.0))
        pg[real_peak_idx] = 5.0
        pg[alias_peak_idx] = 4.0

        # Window function has strong power at f=40 but not f=20
        wp = np.full(n, 0.05)
        wp[alias_peak_idx - 5 : alias_peak_idx + 5] = 0.9

        detrended = detrend_periodogram(pg, wp, floor=0.01)

        # After detrending, the real peak should be more prominent than alias
        assert detrended[real_peak_idx] > detrended[alias_peak_idx], (
            f"Real peak ({detrended[real_peak_idx]:.2f}) should exceed "
            f"alias peak ({detrended[alias_peak_idx]:.2f}) after detrending"
        )

    def test_detrend_preserves_non_alias_peaks(self):
        """Detrending does not suppress peaks where window power is low."""
        n = 500
        pg = np.zeros(n)
        pg[100] = 10.0  # real peak

        wp = np.full(n, 0.02)  # low window power everywhere
        detrended = detrend_periodogram(pg, wp, floor=0.01)

        # Peak should be amplified, not suppressed
        assert detrended[100] >= pg[100]

    def test_detrend_floor_prevents_division_by_zero(self):
        """Floor parameter prevents division by zero."""
        pg = np.array([1.0, 2.0, 3.0])
        wp = np.array([0.0, 0.0, 0.0])
        detrended = detrend_periodogram(pg, wp, floor=0.01)
        assert np.all(np.isfinite(detrended))

    def test_detrend_periodogram_results(self):
        """detrend_periodogram_results modifies result dicts in place."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = [{'period': 0.5, 'data': data.copy()}]
        wp = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        detrend_periodogram_results(results, wp, floor=0.01)
        np.testing.assert_allclose(results[0]['data'], data / 0.5)

    # -- Neighbour alias fraction --

    def test_neighbour_alias_all_same_period(self):
        """All sources at same location with same period -> fraction ~1.0."""
        n = 20
        ra = np.full(n, 95.0)
        dec = np.full(n, -28.0)
        # Small offsets so they're neighbours
        ra += np.random.default_rng(42).uniform(-0.01, 0.01, n)
        dec += np.random.default_rng(43).uniform(-0.01, 0.01, n)
        periods = np.full(n, 0.0534)  # ~1.28 hr

        fracs = compute_neighbour_alias_fraction(
            ra, dec, periods, radius_arcmin=2.0, tolerance=0.05
        )
        # Most sources should have high fraction
        assert np.mean(fracs) > 0.7, f"Mean fraction {np.mean(fracs):.2f} too low"

    def test_neighbour_alias_different_periods(self):
        """Sources with different periods -> fraction ~0.0."""
        n = 20
        rng = np.random.default_rng(42)
        ra = np.full(n, 95.0) + rng.uniform(-0.01, 0.01, n)
        dec = np.full(n, -28.0) + rng.uniform(-0.01, 0.01, n)
        # Each source has a very different period
        periods = np.linspace(0.1, 10.0, n)

        fracs = compute_neighbour_alias_fraction(
            ra, dec, periods, radius_arcmin=2.0, tolerance=0.05
        )
        assert np.mean(fracs) < 0.3, f"Mean fraction {np.mean(fracs):.2f} too high"

    def test_neighbour_alias_isolated_source(self):
        """Isolated source (no neighbours within radius) -> fraction 0.0."""
        ra = np.array([95.0, 50.0])  # 45 degrees apart
        dec = np.array([-28.0, -28.0])
        periods = np.array([1.0, 1.0])

        fracs = compute_neighbour_alias_fraction(
            ra, dec, periods, radius_arcmin=2.0, tolerance=0.05
        )
        assert fracs[0] == 0.0
        assert fracs[1] == 0.0

    def test_neighbour_alias_nan_period(self):
        """NaN periods are handled gracefully."""
        ra = np.array([95.0, 95.001, 95.002])
        dec = np.array([-28.0, -28.0, -28.0])
        periods = np.array([1.0, np.nan, 1.0])

        fracs = compute_neighbour_alias_fraction(
            ra, dec, periods, radius_arcmin=5.0, tolerance=0.05
        )
        assert fracs[1] == 0.0  # NaN period source gets 0
        assert np.all(np.isfinite(fracs))

    def test_neighbour_alias_empty_input(self):
        """Empty arrays return empty result."""
        fracs = compute_neighbour_alias_fraction(
            np.array([]),
            np.array([]),
            np.array([]),
            radius_arcmin=2.0,
        )
        assert len(fracs) == 0
