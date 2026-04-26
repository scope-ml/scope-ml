#!/usr/bin/env python
"""
Tests for scope.surveys.wise module.

Unit tests (no network required) cover band mapping, quality cuts,
Kowalski-format conversion, and the local-parquet client. Integration
tests that hit IRSA TAP are skipped unless WISE_RUN_IRSA=1.
"""

import os
import numpy as np
import pandas as pd
import pytest

from scope.surveys.wise import (
    DEFAULT_BAND_MAP,
    DEFAULT_QUALITY,
    IRSA_TAP_URL,
    JD_MINUS_MJD,
    WISELocalClient,
    _catflags_from_quality,
    _format_as_kowalski,
    _mjd_to_hjd,
    _quality_mask,
    make_wise_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_detections():
    """Synthetic NEOWISE-R detections mimicking the real table schema."""
    n = 6
    return pd.DataFrame({
        "source_name": ["J000000.00+000000.0"] * 3 + ["J010000.00+100000.0"] * 3,
        "ra": [0.0, 0.0, 0.0, 15.0, 15.0, 15.0],
        "dec": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        "mjd": np.array([56700.0, 56880.0, 57060.0, 56700.5, 56880.5, 57060.5]),
        "w1mpro": np.array([14.0, 14.1, 13.9, 12.0, 12.2, 12.1]),
        "w1sigmpro": np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03]),
        "w2mpro": np.array([13.5, 13.6, 13.4, 11.5, 11.7, 11.6]),
        "w2sigmpro": np.array([0.06, 0.06, 0.06, 0.04, 0.04, 0.04]),
        "cc_flags": ["0000", "0000", "h000", "0000", "0000", "0000"],
        "ph_qual": ["AA", "AA", "AA", "AA", "AB", "BA"],
        "qi_fact": [1.0, 1.0, 1.0, 0.6, 0.8, 0.3],
        "moon_masked": ["00", "00", "00", "00", "00", "00"],
        "saa_sep": [20.0, 20.0, 20.0, 50.0, 50.0, 50.0],
    })


@pytest.fixture
def local_parquet(tmp_path, sample_detections):
    """Write sample_detections to a combined-style parquet for WISELocalClient tests."""
    p = tmp_path / "nep_lcs.parquet"
    sample_detections.to_parquet(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Unit tests: module-level constants & helpers
# ---------------------------------------------------------------------------
class TestConstants:
    def test_band_map_non_overlapping_with_ztf(self):
        """WISE band IDs must not collide with ZTF (1=g,2=r,3=i) or Rubin (0..5)."""
        ztf_rubin_ids = {0, 1, 2, 3, 4, 5}
        for band, fid in DEFAULT_BAND_MAP.items():
            assert fid not in ztf_rubin_ids, f"{band} id {fid} clashes with ZTF/Rubin"

    def test_band_map_members(self):
        assert set(DEFAULT_BAND_MAP.keys()) == {"W1", "W2", "W3", "W4"}

    def test_jd_mjd_offset(self):
        assert JD_MINUS_MJD == 2400000.5

    def test_irsa_tap_url(self):
        assert "irsa.ipac" in IRSA_TAP_URL


class TestMjdToHjd:
    def test_scalar(self):
        assert _mjd_to_hjd(np.array([0.0]))[0] == 2400000.5

    def test_array(self):
        mjd = np.array([56700.0, 58000.0, 60000.0])
        hjd = _mjd_to_hjd(mjd)
        np.testing.assert_allclose(hjd - mjd, JD_MINUS_MJD)


# ---------------------------------------------------------------------------
# Quality mask + catflags
# ---------------------------------------------------------------------------
class TestQualityMask:
    def test_clean_rows_kept(self, sample_detections):
        mask = _quality_mask(sample_detections)
        # Row 0-1 clean, row 2 fails cc_flags, rows 3-5 have varying qi_fact and ph_qual
        assert mask.iloc[0] and mask.iloc[1]

    def test_bad_cc_flags_rejected(self, sample_detections):
        mask = _quality_mask(sample_detections)
        assert not mask.iloc[2]  # 'h000' cc_flags

    def test_low_qi_fact_rejected(self, sample_detections):
        mask = _quality_mask(sample_detections)
        assert not mask.iloc[5]  # qi_fact=0.3 < 0.5

    def test_custom_quality_overrides(self, sample_detections):
        # Loosen the cut — row 5 (qi_fact 0.3) now passes
        q = {"qi_fact_min": 0.1}
        mask = _quality_mask(sample_detections, q)
        assert mask.iloc[5]

    def test_missing_columns_ignored(self):
        # A DataFrame missing some optional quality columns should not crash
        df = pd.DataFrame({"w1mpro": [14.0], "mjd": [56700.0]})
        mask = _quality_mask(df)
        assert mask.all()


class TestCatflagsFromQuality:
    def test_clean_row_zero(self, sample_detections):
        cf = _catflags_from_quality(sample_detections)
        assert cf[0] == 0  # clean row

    def test_bad_ccflags_sets_bit0(self, sample_detections):
        cf = _catflags_from_quality(sample_detections)
        # row 2 has 'h000' → bit 0 set
        assert cf[2] & (1 << 0)

    def test_low_qifact_sets_bit2(self, sample_detections):
        cf = _catflags_from_quality(sample_detections)
        # row 5 has qi_fact 0.3 → bit 2 set
        assert cf[5] & (1 << 2)


# ---------------------------------------------------------------------------
# Kowalski-format conversion
# ---------------------------------------------------------------------------
class TestFormatAsKowalski:
    def test_output_structure(self, sample_detections):
        out = _format_as_kowalski(sample_detections)
        # 2 sources × 2 bands (W1, W2) = 4 band-LCs
        assert len(out) == 4
        for entry in out:
            assert set(entry.keys()) >= {"_id", "filter", "data", "source_id", "band_name"}

    def test_filter_ids_match_band_map(self, sample_detections):
        out = _format_as_kowalski(sample_detections)
        filter_ids = {entry["band_name"]: entry["filter"] for entry in out}
        assert filter_ids["W1"] == DEFAULT_BAND_MAP["W1"]
        assert filter_ids["W2"] == DEFAULT_BAND_MAP["W2"]

    def test_data_row_keys(self, sample_detections):
        out = _format_as_kowalski(sample_detections)
        data_row = out[0]["data"][0]
        assert set(data_row.keys()) == {"hjd", "mag", "magerr", "catflags"}

    def test_hjd_is_mjd_plus_offset(self, sample_detections):
        out = _format_as_kowalski(sample_detections)
        first_entry = [e for e in out if e["band_name"] == "W1"][0]
        mjd_expected = sample_detections[
            sample_detections["source_name"] == first_entry["source_id"]
        ]["mjd"].iloc[0]
        assert np.isclose(first_entry["data"][0]["hjd"], mjd_expected + JD_MINUS_MJD)

    def test_nan_magnitudes_dropped(self):
        df = pd.DataFrame({
            "source_name": ["s1"] * 3,
            "ra": [0.0] * 3, "dec": [0.0] * 3, "mjd": [1.0, 2.0, 3.0],
            "w1mpro": [14.0, np.nan, 14.1],
            "w1sigmpro": [0.05, 0.05, 0.05],
            "w2mpro": [13.5, 13.6, 13.4],
            "w2sigmpro": [0.06, 0.06, 0.06],
        })
        out = _format_as_kowalski(df, bands=("W1",))
        # NaN magnitude dropped; 2 points survive
        assert len(out) == 1 and len(out[0]["data"]) == 2

    def test_custom_band_subset(self, sample_detections):
        out = _format_as_kowalski(sample_detections, bands=("W1",))
        bands_emitted = {e["band_name"] for e in out}
        assert bands_emitted == {"W1"}


# ---------------------------------------------------------------------------
# WISELocalClient
# ---------------------------------------------------------------------------
class TestWISELocalClient:
    def test_rejects_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            WISELocalClient("/nonexistent/path.parquet")

    def test_combined_parquet_loads(self, local_parquet):
        client = WISELocalClient(local_parquet)
        assert client._is_combined is True

    def test_get_objects_by_cone(self, local_parquet):
        client = WISELocalClient(local_parquet)
        # Cone at (0, 0) with 5 arcsec radius: matches J000000.00+000000.0
        hits = client.get_objects_by_cone(ra=0.0, dec=0.0, radius_arcsec=5.0)
        assert len(hits) == 1
        assert hits.iloc[0]["source_name"] == "J000000.00+000000.0"

    def test_cone_no_match(self, local_parquet):
        client = WISELocalClient(local_parquet)
        hits = client.get_objects_by_cone(ra=100.0, dec=-80.0, radius_arcsec=1.0)
        assert len(hits) == 0

    def test_get_lightcurves(self, local_parquet):
        client = WISELocalClient(local_parquet)
        lcs = client.get_lightcurves(["J000000.00+000000.0"])
        # 2 clean rows for this source × 2 bands = 4 band-LCs maximum;
        # row 2 dropped by cc_flags cut → 2 kept per band
        per_band = {lc["band_name"]: lc for lc in lcs}
        assert set(per_band.keys()) == {"W1", "W2"}
        assert len(per_band["W1"]["data"]) == 2

    def test_get_lightcurves_for_cone_roundtrip(self, local_parquet):
        client = WISELocalClient(local_parquet)
        lcs = client.get_lightcurves_for_cone(ra=0.0, dec=0.0, radius_arcsec=5.0)
        assert len(lcs) > 0
        # Source id should match the one in the cone
        assert all(lc["source_id"] == "J000000.00+000000.0" for lc in lcs)


# ---------------------------------------------------------------------------
# make_wise_client factory
# ---------------------------------------------------------------------------
class TestMakeWiseClient:
    def test_local_path_returns_local_client(self, local_parquet):
        config = {"wise": {"data_path": local_parquet}}
        client = make_wise_client(config)
        assert isinstance(client, WISELocalClient)

    def test_env_var_local_path(self, monkeypatch, local_parquet):
        monkeypatch.setenv("WISE_DATA_PATH", local_parquet)
        client = make_wise_client({})
        assert isinstance(client, WISELocalClient)

    def test_missing_local_path_warns(self):
        config = {"wise": {"data_path": "/nonexistent/path.parquet"}}
        with pytest.warns(UserWarning, match="does not exist"):
            make_wise_client(config)

    def test_no_config_falls_through_to_tap_or_none(self):
        """Without local path and without pyvo this must return None gracefully."""
        # We can't easily unload pyvo, so just assert the call doesn't crash
        # and returns either a TAP client or None.
        from scope.surveys.wise import WISETAPClient, HAS_PYVO
        client = make_wise_client(None)
        if HAS_PYVO:
            assert isinstance(client, WISETAPClient)
        else:
            assert client is None


# ---------------------------------------------------------------------------
# Integration (optional, off by default)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("WISE_RUN_IRSA") != "1",
    reason="IRSA TAP integration test (set WISE_RUN_IRSA=1 to enable)",
)
class TestIntegrationIRSA:
    """Live TAP queries against IRSA. Minutes of runtime; only on demand."""

    def test_cone_query(self):
        from scope.surveys.wise import WISETAPClient
        client = WISETAPClient()
        df = client.get_objects_by_cone(ra=270.0, dec=66.5, radius_arcsec=10.0)
        assert len(df) > 0
        assert {"source_name", "ra", "dec"}.issubset(df.columns)
