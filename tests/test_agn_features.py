"""Tests for the AGN/anomaly feature modules: PDRS flare segmentation,
DRW (damped random walk) fitting, and EVT calibration.

Uses synthetic lightcurve data — no GPU or Kowalski needed.

Run with:
    python -m pytest tests/test_agn_features.py -v
"""

import sys
import os

# Add the tools directories to the path so we can import the modules directly
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'featureGeneration')
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pdrs  # noqa: E402
import drw  # noqa: E402
import evt_calibration as evt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_flat_lightcurve(
    n_points=500, t_span=1000.0, mean_mag=17.0, noise_std=0.02, seed=42
):
    """Generate a constant-brightness lightcurve with Gaussian noise."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points))
    mags = mean_mag + rng.normal(0, noise_std, n_points)
    magerrs = np.full(n_points, noise_std)
    return times, mags, magerrs


def make_drw_lightcurve(
    tau=50.0,
    sigma=0.3,
    n_points=800,
    t_span=2000.0,
    mean_mag=17.0,
    err=0.02,
    seed=42,
):
    """Generate an exact OU-process lightcurve via the discrete recursion.

    x_{i+1} = x_i exp(-dt/tau) + N(0, sigma^2 (1 - exp(-2 dt/tau)))
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points))
    x = np.empty(n_points)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n_points):
        decay = np.exp(-(times[i] - times[i - 1]) / tau)
        x[i] = x[i - 1] * decay + rng.normal(0, sigma * np.sqrt(1 - decay**2))
    mags = mean_mag + x + rng.normal(0, err, n_points)
    magerrs = np.full(n_points, err)
    return times, mags, magerrs


def inject_flare(times, mags, t0, amplitude=1.0, rise=2.0, decay=7.0):
    """Inject a fast-rise exponential-decay flare (a brightening, so the
    magnitude decreases) into a lightcurve. Returns modified mags."""
    mags = mags.copy()
    profile = np.zeros_like(mags)
    before = times < t0
    after = ~before
    profile[before] = amplitude * np.exp((times[before] - t0) / rise)
    profile[after] = amplitude * np.exp(-(times[after] - t0) / decay)
    profile[np.abs(times - t0) > 10 * decay] = 0.0
    return mags - profile


# ---------------------------------------------------------------------------
# TestPDRS
# ---------------------------------------------------------------------------


class TestPDRS:
    """Tests for PDRS flare/high-activity segmentation."""

    def test_flat_lightcurve_only_weak_detections(self):
        # Pure noise can produce occasional low-significance regions (peaks
        # are defined relative to the noise sigma itself); what matters is
        # that any such detection stays weak and short
        tme = make_flat_lightcurve()
        stats = pdrs.calc_pdrs_stats(tme)
        assert stats['flare_significance'] < 4.0
        assert stats['flare_duty_cycle'] < 0.1

    def test_injected_flare_detected(self):
        times, mags, magerrs = make_flat_lightcurve(seed=1)
        flat_stats = pdrs.calc_pdrs_stats((times, mags, magerrs))

        mags = inject_flare(times, mags, t0=500.0, amplitude=1.5, decay=5.0)
        stats = pdrs.calc_pdrs_stats((times, mags, magerrs))

        assert stats['n_flares'] >= 1
        assert stats['flare_significance'] > 10.0
        assert stats['flare_significance'] > 3.0 * max(
            flat_stats['flare_significance'], 1.0
        )
        assert 0.0 < stats['flare_duty_cycle'] < 0.1

    def test_two_flares_detected_separately(self):
        # Constructed binned series: two identical bumps on a gently noisy
        # baseline, separated by a deep saddle, must remain two regions
        rng = np.random.default_rng(0)
        mjd = np.arange(40) * 3.0
        flux = 1.0 + rng.normal(0, 0.01, 40)
        bump = np.array([0.1, 0.3, 0.6, 0.3, 0.1])
        flux[8:13] += bump
        flux[28:33] += bump

        flares = pdrs.detect_flares(mjd, flux)
        assert len(flares) == 2
        assert flares[0]['start'] <= mjd[10] <= flares[0]['end']
        assert flares[1]['start'] <= mjd[30] <= flares[1]['end']

    def test_detected_region_overlaps_injection(self):
        times, mags, magerrs = make_flat_lightcurve(seed=3)
        t0 = 600.0
        mags = inject_flare(times, mags, t0=t0, amplitude=1.0, decay=10.0)

        flux = 10 ** (-0.4 * (mags - np.median(mags)))
        fluxerr = 0.4 * np.log(10) * flux * magerrs
        b_t, b_f, _ = pdrs.bin_light_curve(times, flux, fluxerr)
        flares = pdrs.detect_flares(b_t, b_f)

        assert len(flares) >= 1
        assert any(f['start'] - 10 <= t0 <= f['end'] + 10 for f in flares)

    def test_too_few_points_gives_nan(self):
        times = np.array([1.0, 2.0, 3.0])
        mags = np.array([17.0, 17.1, 17.0])
        magerrs = np.array([0.02, 0.02, 0.02])
        stats = pdrs.calc_pdrs_stats((times, mags, magerrs))
        assert all(np.isnan(v) for v in stats.values())

    def test_stat_keys_complete(self):
        tme = make_flat_lightcurve(n_points=100)
        stats = pdrs.calc_pdrs_stats(tme)
        assert set(stats.keys()) == set(pdrs.PDRS_FEATURE_KEYS)


# ---------------------------------------------------------------------------
# TestDRW
# ---------------------------------------------------------------------------


class TestDRW:
    """Tests for DRW fitting and residual-based anomaly statistics."""

    def test_recovers_known_parameters(self):
        tme = make_drw_lightcurve(tau=50.0, sigma=0.3, n_points=1000, seed=10)
        stats = drw.calc_drw_stats(tme)
        assert stats['drw_fit_success'] == 1
        # tau recovery is noisy even for well-sampled curves; require the
        # right order of magnitude
        assert 15.0 < stats['drw_tau'] < 200.0
        assert 0.15 < stats['drw_sigma'] < 0.6

    def test_quiet_drw_has_moderate_max_z(self):
        tme = make_drw_lightcurve(tau=50.0, sigma=0.3, seed=11)
        stats = drw.calc_drw_stats(tme)
        assert stats['drw_fit_success'] == 1
        assert np.isfinite(stats['drw_max_z'])
        assert stats['drw_max_z'] < 6.0

    def test_flare_raises_max_z(self):
        times, mags, magerrs = make_drw_lightcurve(tau=50.0, sigma=0.3, seed=12)
        quiet = drw.calc_drw_stats((times, mags, magerrs))

        flare_mags = inject_flare(times, mags, t0=1000.0, amplitude=1.5, decay=7.0)
        flared = drw.calc_drw_stats((times, flare_mags, magerrs))

        assert flared['drw_max_z'] > 2.0 * quiet['drw_max_z']
        assert flared['drw_max_z'] > 8.0

    def test_ou_resid_standardization(self):
        # With the true parameters, one-step-ahead residuals of a pure OU
        # process should be ~N(0, 1)
        tau, sigma, mean_mag, err = 50.0, 0.3, 17.0, 0.02
        times, mags, magerrs = make_drw_lightcurve(
            tau=tau, sigma=sigma, mean_mag=mean_mag, err=err, n_points=1000, seed=13
        )
        z = drw.ou_resid(times, mags, magerrs, tau, sigma, mean_mag)
        assert abs(np.std(z) - 1.0) < 0.2
        assert np.max(np.abs(z)) < 6.0

    def test_too_few_points_gives_nan(self):
        times, mags, magerrs = make_flat_lightcurve(n_points=10)
        stats = drw.calc_drw_stats((times, mags, magerrs))
        assert stats['drw_fit_success'] == 0
        assert np.isnan(stats['drw_tau'])
        assert np.isnan(stats['drw_max_z'])

    def test_constant_lightcurve_does_not_crash(self):
        times = np.linspace(0, 100, 50)
        mags = np.full(50, 17.0)
        magerrs = np.full(50, 0.02)
        stats = drw.calc_drw_stats((times, mags, magerrs))
        assert set(stats.keys()) == set(drw.DRW_FEATURE_KEYS)

    def test_iterative_masking_option(self):
        times, mags, magerrs = make_drw_lightcurve(tau=50.0, sigma=0.3, seed=14)
        mags = inject_flare(times, mags, t0=1000.0, amplitude=1.5, decay=7.0)
        stats = drw.calc_drw_stats((times, mags, magerrs), max_iter=3)
        assert stats['drw_fit_success'] == 1
        assert np.isfinite(stats['drw_max_z'])

    def test_batch_matches_single(self):
        tme1 = make_drw_lightcurve(seed=15)
        tme2 = make_flat_lightcurve(seed=16)
        batch = drw.calc_drw_stats_batch([tme1, tme2], Ncore=1)
        single = [drw.calc_drw_stats(tme1), drw.calc_drw_stats(tme2)]
        for b, s in zip(batch, single):
            for key in drw.DRW_FEATURE_KEYS:
                if np.isnan(s[key]):
                    assert np.isnan(b[key])
                else:
                    assert b[key] == s[key]

    # ---------------------------------------------------------------------------
    # TestEVT
    # ---------------------------------------------------------------------------

    def test_unconstrained_fit_is_not_flagged_success(self):
        """A fit that runs to a parameter bound must not report success.

        scipy reports convergence even when the optimum sits against a
        bound, so on a baseline too short to show the DRW turnover every
        fit came back drw_fit_success = 1 with tau pinned at a rail. The
        flag has to distinguish converged from constrained, or anything
        filtering on it downstream gets meaningless tau and sigma.
        """
        # A pure white-noise curve: there is no damping timescale to find,
        # so the fit has nothing to constrain tau with.
        rng = np.random.default_rng(1)
        n = 120
        times = np.sort(rng.uniform(0.0, 200.0, n))
        magerrs = np.full(n, 0.02)
        mags = 20.0 + rng.normal(0.0, 0.02, n)

        stats = drw.calc_drw_stats((times, mags, magerrs))

        baseline = times[-1] - times[0]
        constrained = drw._fit_is_constrained(
            stats['drw_tau'], stats['drw_sigma'], times
        )
        assert stats['drw_fit_success'] == int(constrained)
        if not constrained:
            # the parameters are still reported, as limits
            assert np.isfinite(stats['drw_tau'])
            assert np.isfinite(stats['drw_max_z'])

        # An explicitly out-of-range tau is never constrained
        assert not drw._fit_is_constrained(10.0 * baseline, 0.1, times)
        assert not drw._fit_is_constrained(1e-6, 0.1, times)
        assert not drw._fit_is_constrained(0.5 * baseline, 1e-8, times)
        # ... while a sane one is
        assert drw._fit_is_constrained(0.1 * baseline, 0.1, times)


class TestEVT:
    """Tests for the EVT (peaks-over-threshold / GPD) calibration."""

    @staticmethod
    def make_max_z_population(n=3000, seed=20):
        rng = np.random.default_rng(seed)
        # Heavy-ish tailed positive population resembling max |z| values
        return 2.0 + np.abs(rng.standard_normal(n)) + rng.exponential(0.5, n)

    def test_fit_gpd_basic(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals, pot_percentile=95.0)
        assert np.isfinite(calib['shape'])
        assert calib['scale'] > 0
        assert calib['nu'] == np.sum(vals > calib['u'])
        assert abs(calib['u'] - np.percentile(vals, 95.0)) < 1e-8

    def test_detection_threshold_above_u_and_monotonic(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals)
        thr_1pct = evt.detection_threshold(calib, p_target=0.01)
        thr_01pct = evt.detection_threshold(calib, p_target=0.001)
        assert thr_1pct > calib['u']
        assert thr_01pct > thr_1pct

    def test_exceedance_prob_decreasing(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals)
        z = np.array([2.0, 3.0, calib['u'] + 0.5, calib['u'] + 2.0])
        p = evt.exceedance_prob(z, calib)
        assert np.all(np.diff(p) < 0)
        assert np.all((p > 0) & (p <= 1))

    def test_anomaly_score_orders_sources(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals)
        thr = evt.detection_threshold(calib, p_target=0.01)
        scores = evt.anomaly_score(np.array([np.median(vals), thr + 1.0]), calib)
        assert scores[1] > scores[0]

    def test_anomaly_score_nan_passthrough(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals)
        scores = evt.anomaly_score(np.array([3.0, np.nan]), calib)
        assert np.isfinite(scores[0])
        assert np.isnan(scores[1])

    def test_calibration_json_serializable(self):
        vals = self.make_max_z_population()
        calib = evt.fit_gpd(vals)
        calib['detection_threshold'] = float(evt.detection_threshold(calib, 0.01))
        restored = json.loads(json.dumps(calib))
        assert restored['u'] == calib['u']

    def test_fit_gpd_rejects_small_samples(self):
        import pytest

        with pytest.raises(ValueError):
            evt.fit_gpd(np.ones(10))

    # ---------------------------------------------------------------------------
    # End-to-end: DRW max_z of a flaring source stands out after calibration
    # ---------------------------------------------------------------------------

    def test_detects_wide_and_long_layouts(self):
        """EVT must recognise the Rubin wide layout, not just the ZTF long one.

        #586 wrote evt_calibration.py against the ZTF feature tables: one row
        per source-band, a single drw_max_z column, and a filter column to
        group on. #587 emits one row per source with drw_max_z_<band>, so
        the tool matched no files and exited without scoring anything. Both
        layouts have to resolve, and the long path must keep grouping by
        filter.
        """
        wide_cols = [
            '_id',
            'n_g',
            'drw_tau_g',
            'drw_max_z_g',
            'drw_max_z_r',
            'drw_max_z_i',
            'n_flares_g',
        ]
        mapping, layout = evt.max_z_layout(wide_cols)
        assert layout == 'wide'
        assert mapping == {'g': 'drw_max_z_g', 'r': 'drw_max_z_r', 'i': 'drw_max_z_i'}

        long_cols = ['_id', 'filter', 'drw_tau', 'drw_max_z', 'n_flares']
        mapping, layout = evt.max_z_layout(long_cols)
        assert layout == 'long'
        assert mapping == {'all': 'drw_max_z'}

        mapping, layout = evt.max_z_layout(['_id', 'median', 'wstd'])
        assert layout is None
        assert mapping == {}

    def test_wide_layout_calibrates_each_band_separately(self):
        """A wide table must yield one calibration per band, not one pooled."""
        rng = np.random.default_rng(3)
        n = 400
        frame = pd.DataFrame(
            {
                '_id': np.arange(n),
                # deliberately different scales per band
                'drw_max_z_g': np.abs(rng.normal(0.0, 1.0, n)) + 3.0,
                'drw_max_z_r': np.abs(rng.normal(0.0, 2.0, n)) + 6.0,
            }
        )
        mapping, layout = evt.max_z_layout(frame.columns)
        assert layout == 'wide'

        calibs = {
            band: evt.fit_gpd(frame[col].to_numpy()) for band, col in mapping.items()
        }
        assert set(calibs) == {'g', 'r'}
        # the r band sits higher, so its POT threshold must too
        assert calibs['r']['u'] > calibs['g']['u']


class TestEndToEnd:
    def test_flaring_source_flagged_against_quiet_population(self):
        max_z_population = []
        for seed in range(40):
            tme = make_drw_lightcurve(
                tau=50.0, sigma=0.3, n_points=300, t_span=1500.0, seed=100 + seed
            )
            stats = drw.calc_drw_stats(tme)
            if np.isfinite(stats['drw_max_z']):
                max_z_population.append(stats['drw_max_z'])

        # Pad the population for a stable GPD fit using resampling with noise
        rng = np.random.default_rng(0)
        base = np.array(max_z_population)
        pop = np.concatenate([base, rng.choice(base, 500) + rng.normal(0, 0.1, 500)])

        calib = evt.fit_gpd(pop, pot_percentile=90.0)
        thr = evt.detection_threshold(calib, p_target=0.01)

        times, mags, magerrs = make_drw_lightcurve(
            tau=50.0, sigma=0.3, n_points=300, t_span=1500.0, seed=999
        )
        mags = inject_flare(times, mags, t0=700.0, amplitude=2.0, decay=7.0)
        flared = drw.calc_drw_stats((times, mags, magerrs))

        assert flared['drw_max_z'] > thr


def _variable_curve_with_dense_night(seed=7):
    """A realistic DP2-like curve: slow intrinsic variability sampled sparsely
    over 250 days, plus one night of dense intra-night sampling.

    The point is the ratio the real data has - wstd / median_magerr is about 16
    on DP2 - so the global scatter that stock PDRS thresholds on is set by the
    variability, not by the noise.
    """
    rng = np.random.default_rng(seed)

    # sparse epochs carrying the intrinsic wander
    t_sparse = np.sort(rng.uniform(0.0, 250.0, 200)) + 60000.0
    wander = np.cumsum(rng.standard_normal(200))
    wander = 0.25 * wander / np.std(wander)

    # one dense night, ~20 minutes of it, sitting on the same level
    t_night = np.sort(rng.uniform(0.0, 20.0 / 1440.0, 300)) + 60125.0
    level = np.interp(t_night, t_sparse, 19.0 + wander)

    t = np.concatenate([t_sparse, t_night])
    mag = np.concatenate([19.0 + wander, level]) + 0.01 * rng.standard_normal(500)
    magerr = np.full(500, 0.01)

    order = np.argsort(t)
    return t[order], mag[order], magerr[order], t_night


def test_use_snr_default_matches_stock():
    """err=None must reproduce the global-scatter behaviour exactly."""
    rng = np.random.default_rng(3)
    for trial in range(20):
        n = int(rng.integers(40, 200))
        t = np.sort(rng.uniform(0, 250, n))
        mag = 19 + 0.3 * rng.standard_normal(n)
        if trial % 2 == 0:
            t0 = rng.uniform(20, 200)
            mag -= rng.uniform(0.3, 1.5) * np.exp(-np.abs(t - t0) / 10.0)
        magerr = np.full(n, 0.03)
        a = pdrs.calc_pdrs_stats((t, mag, magerr), bin_size_days=1.0)
        b = pdrs.calc_pdrs_stats((t, mag, magerr), bin_size_days=1.0, use_snr=False)
        for k in pdrs.PDRS_FEATURE_KEYS:
            assert np.allclose(a[k], b[k], equal_nan=True)


def test_use_snr_finds_a_minute_scale_flare_that_stock_misses():
    """A 0.3 mag flare lasting minutes is well inside the intrinsic scatter of
    a variable source, so stock PDRS - which thresholds on the global scatter -
    does not see it. Against the per-bin measurement error it is a 30-sigma
    excursion, which is what use_snr measures."""
    t, mag, magerr, t_night = _variable_curve_with_dense_night()

    t0 = t_night[0] + 3.0 / 1440.0
    dur = 4.0 / 1440.0
    hit = (t >= t0) & (t <= t0 + 4 * dur)
    mag = mag.copy()
    mag[hit] -= 0.3 * np.exp(-(t[hit] - t0) / dur)

    # sanity: the curve is variability-dominated, like the real data
    assert np.std(mag) / np.median(magerr) > 10

    stock = pdrs.calc_pdrs_stats(
        (t, mag, magerr),
        bin_size_days=1.0,
        peak_threshold=2.0,
        min_cluster_size=3,
        max_gap_days=60.0,
        smooth_window=3,
        region_threshold=0.3,
    )
    fast = pdrs.calc_pdrs_stats(
        (t, mag, magerr),
        bin_size_days=2.0 / 1440.0,
        peak_threshold=8.0,
        min_cluster_size=3,
        max_gap_days=30.0 / 1440.0,
        smooth_window=3,
        region_threshold=1.0,
        use_snr=True,
    )
    assert stock['n_flares'] == 0
    assert fast['n_flares'] >= 1
    assert fast['flare_significance'] > 5.0


def test_flare_regions_bracket_the_event():
    """flare_regions returns the interval calc_pdrs_stats discards."""
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 200, 300)) + 60000.0
    mag = 19 + 0.02 * rng.standard_normal(300)
    t0 = 60100.0
    mag -= 1.0 * np.exp(-np.abs(t - t0) / 5.0)
    magerr = np.full(300, 0.02)

    params = dict(
        bin_size_days=1.0,
        peak_threshold=2.0,
        min_cluster_size=3,
        max_gap_days=60.0,
        smooth_window=3,
        region_threshold=0.3,
    )
    regions = pdrs.flare_regions((t, mag, magerr), **params)
    stats = pdrs.calc_pdrs_stats((t, mag, magerr), **params)

    assert len(regions) == stats['n_flares']
    assert any(start <= t0 <= end for start, end in regions)
    for start, end in regions:
        assert end >= start


def test_mask_regions_keeps_the_flare_out_of_the_drw_fit():
    """A flare inflates the fitted sigma; masking its interval recovers it,
    and the event is still visible in drw_max_z because residuals are
    evaluated on every point."""
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 250, 250)) + 60000.0
    mag = 19 + 0.05 * rng.standard_normal(250)
    magerr = np.full(250, 0.02)

    clean = drw.calc_drw_stats((t, mag, magerr))

    t0 = 60120.0
    flared = mag - 1.5 * np.exp(-np.abs(t - t0) / 4.0)
    dirty = drw.calc_drw_stats((t, flared, magerr))
    masked = drw.calc_drw_stats((t, flared, magerr), mask_regions=[(t0 - 20, t0 + 20)])

    assert dirty['drw_sigma'] > clean['drw_sigma']
    assert masked['drw_sigma'] < dirty['drw_sigma']
    assert np.isfinite(masked['drw_max_z'])


def test_mask_regions_covering_everything_falls_back():
    """Masking the whole curve would leave nothing to fit; the fit must still
    be attempted on all points rather than returning NaN."""
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0, 100, 60)) + 60000.0
    mag = 19 + 0.05 * rng.standard_normal(60)
    magerr = np.full(60, 0.02)
    out = drw.calc_drw_stats((t, mag, magerr), mask_regions=[(59000.0, 61000.0)])
    ref = drw.calc_drw_stats((t, mag, magerr))
    assert np.isfinite(out['drw_max_z'])
    assert np.allclose(out['drw_max_z'], ref['drw_max_z'], equal_nan=True)


def test_fast_and_slow_features_coexist():
    """The two modes emit disjoint key sets, so a light curve can carry both."""
    assert not set(pdrs.PDRS_FEATURE_KEYS) & set(pdrs.FAST_PDRS_FEATURE_KEYS)

    t, mag, magerr, t_night = _variable_curve_with_dense_night(seed=13)
    t0 = t_night[0] + 3.0 / 1440.0
    dur = 4.0 / 1440.0
    hit = (t >= t0) & (t <= t0 + 4 * dur)
    mag = mag.copy()
    mag[hit] -= 0.3 * np.exp(-(t[hit] - t0) / dur)

    slow = pdrs.calc_pdrs_stats((t, mag, magerr), bin_size_days=1.0)
    fast = pdrs.calc_fast_pdrs_stats((t, mag, magerr))

    assert set(slow) == set(pdrs.PDRS_FEATURE_KEYS)
    assert set(fast) == set(pdrs.FAST_PDRS_FEATURE_KEYS)

    merged = {**slow, **fast}
    assert len(merged) == len(pdrs.PDRS_FEATURE_KEYS) + len(pdrs.FAST_PDRS_FEATURE_KEYS)

    # the wrapper is the same detector, only renamed
    direct = pdrs.calc_pdrs_stats(
        (t, mag, magerr),
        bin_size_days=pdrs.FAST_BIN_SIZE_DAYS,
        peak_threshold=pdrs.FAST_PEAK_THRESHOLD,
        smooth_window=pdrs.FAST_SMOOTH_WINDOW,
        min_cluster_size=pdrs.DEFAULT_MIN_CLUSTER_SIZE,
        saddle_ratio=pdrs.DEFAULT_SADDLE_RATIO,
        region_threshold=pdrs.FAST_REGION_THRESHOLD,
        max_gap_days=pdrs.FAST_MAX_GAP_DAYS,
        use_snr=True,
    )
    for slow_key, fast_key in zip(pdrs.PDRS_FEATURE_KEYS, pdrs.FAST_PDRS_FEATURE_KEYS):
        assert np.allclose(direct[slow_key], fast[fast_key], equal_nan=True)


def test_short_curve_returns_fixed_fast_keys():
    """Too few points must still yield every fast key, so the column set does
    not depend on which sources happened to be measurable."""
    t = np.array([60000.0, 60000.1, 60000.2])
    out = pdrs.calc_fast_pdrs_stats((t, np.full(3, 19.0), np.full(3, 0.01)))
    assert set(out) == set(pdrs.FAST_PDRS_FEATURE_KEYS)
    assert all(np.isnan(v) for v in out.values())
