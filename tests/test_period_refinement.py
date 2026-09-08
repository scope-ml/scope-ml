"""Tests for class-specific period refinement.

Everything here runs on synthetic light curves, so no survey access is needed.
The cases are chosen around the failure that motivated the tool -- an eclipsing
binary reported at half its orbital period -- and around the ways a tool that
rewrites a published quantity could do harm quietly.
"""

import numpy as np
import pytest

from tools.periodRefinement import eclipsing, lookup, pulsating, supported
from tools.periodRefinement.preprocess import prepare_all, prepare_lightcurve
from tools.refine_periods import (
    build_parser,
    collect_trial_periods,
    load_lightcurves,
    refine_one,
)


RNG = np.random.default_rng(20260907)
BANDS = ("g", "r", "i", "z")


def _sample_times(n=800, baseline=257.0):
    return np.sort(RNG.uniform(0.0, baseline, n))


def _bands_for(n):
    return np.array([BANDS[i % len(BANDS)] for i in range(n)])


def _eclipsing_binary(
    period, depth_primary=0.40, depth_secondary=0.18, width=0.028, n=800, noise=0.01
):
    """Two unequal eclipses per orbit, so the curve repeats only every period.

    The secondary is deliberately much shallower than the primary: that is the
    whole signal the refinement has to find, and a model that ignores it will
    happily settle on half the period.
    """
    times = _sample_times(n)
    bands = _bands_for(n)
    phase = np.mod(times / period, 1.0)
    mags = np.zeros(n)
    for centre, depth in ((0.25, depth_primary), (0.75, depth_secondary)):
        offset = np.abs(phase - centre)
        offset = np.minimum(offset, 1.0 - offset)
        mags += depth * np.exp(-0.5 * (offset / width) ** 2)
    mags += RNG.normal(0.0, noise, n)
    return times, mags, bands


def _sawtooth(period, n=700, noise=0.02):
    """A pulsator: fast rise, slow decline, several harmonics."""
    times = _sample_times(n)
    bands = _bands_for(n)
    phase = 2 * np.pi * np.mod(times / period, 1.0)
    mags = (
        0.50 * np.sin(phase)
        + 0.22 * np.sin(2 * phase + 1.0)
        + 0.09 * np.sin(3 * phase + 2.0)
        + RNG.normal(0.0, noise, n)
    )
    return times, mags, bands


def _select_over(model, times, mags, bands, trials):
    return refine_one(model, times, mags, bands, trials)


class TestEclipsing:
    def test_recovers_orbital_period_from_half(self):
        """The case the tool exists for: offered P/2 and P, it must choose P."""
        period = 0.62
        times, mags, bands = _eclipsing_binary(period)
        chosen = _select_over(lookup("ECL"), times, mags, bands, [period / 2, period])
        assert chosen is not None
        assert chosen["period"] == pytest.approx(period, rel=0.02)

    def test_leaves_a_correct_period_alone(self):
        """Offered only the true period and unrelated ones, it must not double.

        Guards the failure mode of every binary "P or 2P" test we discarded:
        preferring the longer period regardless of the data.
        """
        period = 0.62
        times, mags, bands = _eclipsing_binary(period)
        chosen = _select_over(
            lookup("ECL"), times, mags, bands, [period, 0.31, 1.07, 0.44]
        )
        assert chosen is not None
        assert chosen["period"] == pytest.approx(period, rel=0.02)

    def test_returns_the_shape_parameters_not_just_the_period(self):
        """Depths, widths and separation are what the classifier can use.

        The fit eliminates the linear parameters at every step, so they have to
        be recovered deliberately at the end; without this they are computed
        thousands of times and thrown away.
        """
        period = 0.62
        times, mags, bands = _eclipsing_binary(period)
        chosen = _select_over(lookup("ECL"), times, mags, bands, [period])
        info = chosen["info"]
        assert len(info["centres"]) == 2
        assert len(info["widths"]) == 2
        assert info["depths"], "per-band depths were not returned"
        for depths in info["depths"].values():
            assert len(depths) == 2
        # the secondary is the shallower of the two by construction
        deeper = [max(abs(d[0]), abs(d[1])) for d in info["depths"].values()]
        shallower = [min(abs(d[0]), abs(d[1])) for d in info["depths"].values()]
        assert np.mean(shallower) < np.mean(deeper)
        assert 0.0 < info["separation"] <= 0.5
        # near-collinear Gaussian columns must not produce absurd depths
        for depths in info["depths"].values():
            assert all(abs(d) < 5.0 for d in depths)

    def test_prefers_two_eclipses_when_they_differ(self):
        period = 0.62
        times, mags, bands = _eclipsing_binary(period)
        chosen = _select_over(lookup("ECL"), times, mags, bands, [period])
        assert chosen is not None
        assert chosen["name"].startswith("TWOGAUSSIANS")

    def test_rejects_poor_phase_coverage(self):
        """A period that leaves most of the phase empty must not be fitted.

        Silence is the right answer here; returning a period from a fit with
        holes in it is how a wrong value reaches a catalogue.
        """
        times = np.sort(RNG.uniform(0.0, 1.0, 200))  # one night only
        mags = RNG.normal(0.0, 0.01, 200)
        bands = _bands_for(200)
        assert eclipsing.fit_period_models(times, mags, bands, 40.0) is None

    def test_rejects_too_few_points_per_band(self):
        times = _sample_times(12)
        mags = RNG.normal(0.0, 0.01, 12)
        bands = _bands_for(12)
        assert eclipsing.fit_period_models(times, mags, bands, 0.5) is None

    def test_unweighted_bic_does_not_favour_the_simplest_model(self):
        """Regression test for the information criterion.

        The fit is unweighted, so BIC must estimate the noise from the
        residuals.  Keeping the weighted form leaves the penalty dominating the
        residual term, and the fewest-parameter model wins every time whatever
        the data look like.
        """
        npts = 900
        strong = eclipsing._bic(1.0, 8, npts)
        weak = eclipsing._bic(20.0, 4, npts)
        assert strong < weak


class TestPulsating:
    def test_recovers_period_among_aliases(self):
        period = 0.53
        times, mags, bands = _sawtooth(period)
        chosen = _select_over(
            lookup("RR"), times, mags, bands, [period, period / 2, 0.31, 1.02]
        )
        assert chosen is not None
        assert chosen["period"] == pytest.approx(period, rel=0.02)

    def test_fourier_phases_are_finite_and_bounded(self):
        period = 0.53
        times, mags, bands = _sawtooth(period)
        params = (
            pulsating.fourier_params(times, mags, bands, period, 3)
            if hasattr(pulsating, "fourier_params")
            else None
        )
        if params is None:  # module layout guard
            pytest.skip("fourier_params not exposed")
        assert 0.0 <= params["phi21"] <= 2 * np.pi
        assert 0.0 <= params["phi31"] <= 2 * np.pi
        assert params["amp"] > 0.0

    def test_refinement_beats_the_candidate_it_started_from(self):
        """Selection can only return a candidate; refinement must do better."""
        period = 0.53
        times, mags, bands = _sawtooth(period, noise=0.01)
        offered = period * 1.004  # 0.4 per cent off
        refined = pulsating.characterise(
            times, mags, bands, offered, 3, do_bootstrap=False
        )
        assert refined is not None
        assert abs(refined["period_refined"] - period) < abs(offered - period)

    def test_bootstrap_error_is_not_identically_zero(self):
        """Regression test: a grid-only refinement gives every resample the
        same answer, and the quoted uncertainty collapses to exactly zero."""
        period = 0.53
        times, mags, bands = _sawtooth(period)
        refined = pulsating.characterise(
            times, mags, bands, period, 3, do_bootstrap=True
        )
        assert refined["period_err"] > 0.0


class TestRegistry:
    def test_unregistered_class_has_no_model(self):
        assert lookup("LPV") is None
        assert lookup("AGN") is None
        assert lookup(None) is None
        assert lookup("") is None

    def test_lookup_is_case_insensitive(self):
        assert lookup("ecl") is lookup("ECL")

    def test_every_registered_class_is_usable(self):
        for name in supported():
            model = lookup(name)
            assert callable(model.fit_period_models)
            assert callable(model.select)
            assert callable(model.global_ranking)
            low, high = model.period_range
            assert 0.0 < low < high

    def test_rr_range_admits_the_long_period_tail(self):
        """Gaia cut RR Lyrae at 0.2-1.0 d, and our validation set is their
        catalogue, so adopting that window could never be shown to be wrong.
        Ours is wider on purpose."""
        low, high = lookup("RR").period_range
        assert low <= 0.15 and high >= 1.2


class TestTrialPeriods:
    def _row(self):
        row = {}
        for i in range(1, 21):
            for algorithm in ("LS", "CE", "AOV", "FPW", "MHF"):
                row[f"period_{i}_{algorithm}"] = 0.1 * i
        return row

    def test_respects_the_class_period_range(self):
        trials = collect_trial_periods(self._row(), 20, 0.15, 1.1)
        assert trials
        assert all(0.15 <= p <= 1.1 for p in trials)

    def test_deduplicates_across_algorithms(self):
        trials = collect_trial_periods(self._row(), 20, 0.0, 10.0)
        assert len(trials) == len(set(round(p, 6) for p in trials))

    def test_extra_candidates_are_range_checked(self):
        trials = collect_trial_periods(
            self._row(), 20, 0.15, 1.1, extra=(0.62, 99.0, np.nan, None)
        )
        assert any(abs(p - 0.62) < 1e-9 for p in trials)
        assert all(p <= 1.1 for p in trials)


class TestPreprocessing:
    """The refinement must see the same photometry the period search saw."""

    @staticmethod
    def _entry(band, points):
        return {"_id": 7, "filter": band, "data": points}

    @staticmethod
    def _points(n, start=0.0, mag=18.0, step=1.0, catflags=0):
        return [
            {
                "hjd": start + i * step,
                "mag": mag,
                "magerr": 0.01,
                "catflags": catflags,
            }
            for i in range(n)
        ]

    def test_groups_bands_into_one_source(self):
        payload = [
            self._entry("g", self._points(40)),
            self._entry("r", self._points(40, start=0.5)),
        ]
        prepared = prepare_all(payload, min_n_lc_points=10)
        times, mags, errors, bands = prepared[7]
        assert len(times) == len(mags) == len(errors) == len(bands) == 80
        assert set(bands) == {"g", "r"}

    def test_drops_flagged_points(self):
        """catflags marks known-bad photometry, which the period search skipped."""
        payload = [
            self._entry("g", self._points(30)),
            self._entry("g", self._points(30, start=100.0, catflags=1)),
        ]
        times, _, _, _ = prepare_all(payload, min_n_lc_points=10)[7]
        assert len(times) == 30

    def test_thins_high_cadence_duplicates(self):
        """Points closer than the cadence limit collapse to the first."""
        dense = [
            {"hjd": i * 1e-5, "mag": 18.0, "magerr": 0.01, "catflags": 0}
            for i in range(50)
        ]
        prepared = prepare_lightcurve(
            [self._entry("g", dense)], min_cadence_minutes=5.0, min_n_lc_points=1
        )
        assert prepared is not None
        assert len(prepared[0]) == 1

    def test_clips_catastrophic_outliers(self):
        points = self._points(60)
        # real scatter, or the MAD is zero and clipping is skipped by design
        for i, point in enumerate(points):
            point["mag"] = 18.0 + 0.02 * ((i % 7) - 3)
        points[0]["mag"] = 40.0  # a wild outlier, not real variability
        times, mags, _, _ = prepare_all([self._entry("g", points)], min_n_lc_points=10)[
            7
        ]
        assert 40.0 not in mags
        assert len(times) == 59

    def test_keeps_everything_when_the_scatter_is_zero(self):
        """A constant light curve has MAD zero, so no clip is possible.

        Guarding the sigma > 0 branch: without it every point sits more than
        five sigma from the median and the whole source is thrown away.
        """
        points = self._points(30)
        times, _, _, _ = prepare_all([self._entry("g", points)], min_n_lc_points=10)[7]
        assert len(times) == 30

    def test_times_come_back_sorted(self):
        points = self._points(30)
        points.reverse()
        times, _, _, _ = prepare_all([self._entry("g", points)], min_n_lc_points=10)[7]
        assert np.all(np.diff(times) >= 0)

    def test_too_few_points_is_dropped_not_returned(self):
        assert prepare_all([self._entry("g", self._points(3))]) == {}

    def test_empty_input_is_not_an_error(self):
        assert prepare_all([]) == {}
        assert prepare_all(None) == {}


# The light curve fetch had no test at all, which is how a ZTF call with two
# missing required arguments, and a config that was never read, both shipped.


def test_ztf_survey_fails_with_an_explanation():
    with pytest.raises(NotImplementedError, match="Kowalski"):
        load_lightcurves([1, 2], "ztf")


def test_unknown_survey_is_rejected():
    with pytest.raises(ValueError, match="unknown survey"):
        load_lightcurves([1, 2], "hsc")


def test_parser_offers_only_implemented_surveys():
    args = build_parser().parse_args(
        ["--features", "f.parquet", "--output", "o.parquet"]
    )
    assert args.survey == "rubin"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--features", "f.parquet", "--output", "o.parquet", "--survey", "ztf"]
        )


def test_configured_data_path_reaches_the_rubin_client(monkeypatch):
    """A data path set the documented way must not be ignored."""
    import scope.surveys.rubin as rubin_module

    seen = {}

    class FakeClient:
        def get_lightcurves(self, identifiers):
            seen["ids"] = list(identifiers)
            return []

    def fake_make_rubin_client(config=None, use_dia=False, release=None):
        seen["config"] = config
        seen["use_dia"] = use_dia
        seen["release"] = release
        return FakeClient()

    monkeypatch.setattr(rubin_module, "make_rubin_client", fake_make_rubin_client)
    out = load_lightcurves([7], "rubin", config={"dp2_data_path": "/data/dp2.parquet"})
    assert out == {}
    assert seen["config"] == {"dp2_data_path": "/data/dp2.parquet"}
    assert seen["use_dia"] is True
    assert seen["ids"] == [7]


def test_missing_config_is_not_fatal(monkeypatch):
    """Running from a job script with no checkout must still work."""
    import tools.refine_periods as refine

    def raise_missing():
        raise FileNotFoundError("config.yaml")

    monkeypatch.setattr(refine, "parse_load_config", raise_missing)
    assert refine.load_config_if_any() == {}


def test_main_hands_the_configured_path_to_the_fetch(tmp_path, monkeypatch):
    """The config bug was in main(), not in the fetch it calls."""
    import pandas as pd

    import tools.refine_periods as refine

    features = tmp_path / "features.parquet"
    pd.DataFrame({"_id": [1], "best_agree_period": [0.5], "class": ["ECL"]}).to_parquet(
        features, index=False
    )

    seen = {}

    def fake_load_lightcurves(identifiers, survey, config=None, **cleaning):
        seen["config"] = config
        seen["cleaning"] = cleaning
        return {}

    monkeypatch.setattr(refine, "load_lightcurves", fake_load_lightcurves)
    monkeypatch.setattr(
        refine,
        "parse_load_config",
        lambda: {"rubin": {"dp2_data_path": "/data/dp2.parquet"}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "refine-periods",
            "--features",
            str(features),
            "--output",
            str(tmp_path / "out.parquet"),
            "--min-cadence-minutes",
            "0",
        ],
    )

    refine.main()

    assert seen["config"] == {"dp2_data_path": "/data/dp2.parquet"}
    assert seen["cleaning"]["min_cadence_minutes"] == 0
