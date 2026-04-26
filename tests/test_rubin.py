#!/usr/bin/env python
"""
Tests for scope.rubin module.

Unit tests (no network required) test flux conversion, band mapping,
and format conversion. Integration tests require a RUBIN_TAP_TOKEN
environment variable.
"""

import os
import numpy as np
import pytest

from scope.surveys.rubin import (
    _flux_to_mag,
    _format_as_kowalski,
    DEFAULT_BAND_MAP,
    NANOJANSKY_ZP,
    RubinLocalClient,
    make_rubin_client,
)


class TestFluxToMag:
    """Unit tests for nanoJansky flux to AB magnitude conversion."""

    def test_known_conversion(self):
        """1 nJy should give mag = 31.4 (the AB zero point for nJy)."""
        mag, magerr = _flux_to_mag(1.0, 0.1)
        assert np.isclose(mag, NANOJANSKY_ZP, atol=1e-10)

    def test_known_bright_source(self):
        """10^6 nJy ≈ 1 mJy → mag ≈ 16.4"""
        mag, magerr = _flux_to_mag(1e6, 1e4)
        expected = -2.5 * np.log10(1e6) + NANOJANSKY_ZP  # = 16.4
        assert np.isclose(mag, expected, atol=1e-10)

    def test_error_propagation(self):
        """Check that magerr = (2.5/ln10) * (flux_err/flux)."""
        flux, flux_err = 1000.0, 50.0
        mag, magerr = _flux_to_mag(flux, flux_err)
        expected_magerr = (2.5 / np.log(10)) * (flux_err / flux)
        assert np.isclose(magerr, expected_magerr, atol=1e-10)

    def test_array_input(self):
        """Should handle numpy arrays."""
        fluxes = np.array([100.0, 1000.0, 10000.0])
        errors = np.array([10.0, 100.0, 1000.0])
        mags, magerrs = _flux_to_mag(fluxes, errors)
        assert mags.shape == (3,)
        assert magerrs.shape == (3,)
        # Brighter flux = smaller magnitude
        assert mags[0] > mags[1] > mags[2]

    def test_single_float(self):
        """Should handle single float input."""
        mag, magerr = _flux_to_mag(500.0, 25.0)
        assert isinstance(float(mag), float)
        assert isinstance(float(magerr), float)

    def test_symmetric_error_ratio(self):
        """All sources with 10% flux error should have same magerr."""
        fluxes = np.array([100.0, 1000.0, 10000.0])
        errors = fluxes * 0.1
        _, magerrs = _flux_to_mag(fluxes, errors)
        # All should have the same magerr since err/flux = 0.1 for all
        assert np.allclose(magerrs, magerrs[0], atol=1e-10)


class TestBandMap:
    """Unit tests for band name to filter ID mapping."""

    def test_default_band_map_has_all_bands(self):
        """Default band map should have u, g, r, i, z, y."""
        expected = {"u", "g", "r", "i", "z", "y"}
        assert set(DEFAULT_BAND_MAP.keys()) == expected

    def test_ztf_compatible_bands(self):
        """g=1, r=2, i=3 should match ZTF filter IDs."""
        assert DEFAULT_BAND_MAP["g"] == 1
        assert DEFAULT_BAND_MAP["r"] == 2
        assert DEFAULT_BAND_MAP["i"] == 3

    def test_rubin_only_bands(self):
        """u=0, z=4, y=5 are Rubin-only bands."""
        assert DEFAULT_BAND_MAP["u"] == 0
        assert DEFAULT_BAND_MAP["z"] == 4
        assert DEFAULT_BAND_MAP["y"] == 5


class TestFormatAsKowalski:
    """Unit tests for TAP result → Kowalski format conversion."""

    def _make_rows(self, n=5, objectId=12345, band="g"):
        """Helper to create mock TAP result rows."""
        rows = []
        for i in range(n):
            rows.append(
                {
                    "objectId": objectId,
                    "band": band,
                    "psfFlux": 1000.0 + i * 10,
                    "psfFluxErr": 50.0 + i,
                    "expMidptMJD": 60000.0 + i,
                    "pixelFlags": 0,
                }
            )
        return rows

    def test_basic_conversion(self):
        """Should produce a list of dicts with _id, filter, data keys."""
        rows = self._make_rows()
        result = _format_as_kowalski(rows)
        assert len(result) == 1
        lc = result[0]
        assert "_id" in lc
        assert "filter" in lc
        assert "data" in lc
        assert lc["_id"] == 12345
        assert lc["filter"] == 1  # g band

    def test_data_point_keys(self):
        """Each data point should have hjd, mag, magerr, catflags."""
        rows = self._make_rows(n=3)
        result = _format_as_kowalski(rows)
        for point in result[0]["data"]:
            assert "hjd" in point
            assert "mag" in point
            assert "magerr" in point
            assert "catflags" in point

    def test_mjd_passthrough(self):
        """MJD values should be stored in the 'hjd' field."""
        rows = [
            {
                "objectId": 1,
                "band": "r",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60123.456,
                "pixelFlags": 0,
            }
        ]
        result = _format_as_kowalski(rows)
        assert np.isclose(result[0]["data"][0]["hjd"], 60123.456)

    def test_pixel_flags_to_catflags(self):
        """pixelFlags should be mapped to catflags."""
        rows = [
            {
                "objectId": 1,
                "band": "r",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60000.0,
                "pixelFlags": 42,
            }
        ]
        result = _format_as_kowalski(rows)
        assert result[0]["data"][0]["catflags"] == 42

    def test_multiple_bands_grouped(self):
        """Rows with different bands should produce separate entries."""
        rows_g = self._make_rows(n=3, objectId=1, band="g")
        rows_r = self._make_rows(n=4, objectId=1, band="r")
        result = _format_as_kowalski(rows_g + rows_r)
        assert len(result) == 2
        filters = {lc["filter"] for lc in result}
        assert filters == {1, 2}  # g=1, r=2

    def test_multiple_objects(self):
        """Rows with different objectIds should produce separate entries."""
        rows1 = self._make_rows(n=3, objectId=100, band="g")
        rows2 = self._make_rows(n=3, objectId=200, band="g")
        result = _format_as_kowalski(rows1 + rows2)
        assert len(result) == 2
        ids = {lc["_id"] for lc in result}
        assert ids == {100, 200}

    def test_negative_flux_filtered(self):
        """Rows with non-positive flux should be excluded."""
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": -100.0,
                "psfFluxErr": 10.0,
                "expMidptMJD": 60000.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 0.0,
                "psfFluxErr": 10.0,
                "expMidptMJD": 60001.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60002.0,
                "pixelFlags": 0,
            },
        ]
        result = _format_as_kowalski(rows)
        # Only the last row with positive flux should remain
        assert len(result) == 1
        assert len(result[0]["data"]) == 1

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _format_as_kowalski([])
        assert result == []

    def test_custom_band_map(self):
        """Should use a custom band map if provided."""
        custom_map = {"g": 10, "r": 20}
        rows = self._make_rows(n=2, band="g")
        result = _format_as_kowalski(rows, band_map=custom_map)
        assert result[0]["filter"] == 10

    def test_unknown_band_returns_negative_one(self):
        """Unknown band should map to filter ID -1."""
        rows = self._make_rows(n=2, band="x")
        result = _format_as_kowalski(rows)
        assert result[0]["filter"] == -1

    def test_magnitude_values_reasonable(self):
        """Converted magnitudes should be in a reasonable range."""
        rows = self._make_rows(n=10)
        result = _format_as_kowalski(rows)
        for point in result[0]["data"]:
            # Typical astronomical magnitudes are 0-30
            assert 0 < point["mag"] < 40
            assert point["magerr"] > 0


class TestMJDHandling:
    """Tests for MJD timestamp handling."""

    def test_mjd_preserved_exactly(self):
        """MJD should be stored exactly as-is (no HJD conversion)."""
        mjd = 60457.123456789
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": mjd,
                "pixelFlags": 0,
            }
        ]
        result = _format_as_kowalski(rows)
        assert result[0]["data"][0]["hjd"] == mjd

    def test_time_ordering_not_enforced(self):
        """Format conversion should not reorder points (that's sort_lightcurve's job)."""
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60002.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60001.0,
                "pixelFlags": 0,
            },
        ]
        result = _format_as_kowalski(rows)
        times = [p["hjd"] for p in result[0]["data"]]
        # Should preserve original order
        assert times[0] == 60002.0
        assert times[1] == 60001.0


# --- Integration tests (require RUBIN_TAP_TOKEN) ---

# DP1 footprint: Extended Chandra Deep Field South (ECDFS)
# Center ~(53.1, -28.1), ~9.6 sq deg
DP1_TEST_RA = 53.1
DP1_TEST_DEC = -28.1


@pytest.mark.skipif(
    os.environ.get("RUBIN_TAP_TOKEN") is None,
    reason="Integration test requires RUBIN_TAP_TOKEN environment variable",
)
class TestIntegrationRubinTAP:
    """Integration tests that connect to the real Rubin TAP service."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from scope.surveys.rubin import RubinTAPClient

        self.client = RubinTAPClient(
            tap_url="https://data.lsst.cloud/api/tap",
            token=os.environ["RUBIN_TAP_TOKEN"],
        )

    def test_cone_search(self):
        """Cone search should return objects as a dict."""
        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=10,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        assert isinstance(objects, dict)
        if len(objects) > 0:
            first = next(iter(objects.values()))
            assert "coord_ra" in first
            assert "coord_dec" in first

    def test_lightcurve_retrieval(self):
        """Should retrieve lightcurves for discovered objects."""
        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=5,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        if len(objects) == 0:
            pytest.skip("No objects found in cone search")

        objectids = list(objects.keys())[:3]
        lcs = self.client.get_lightcurves(objectids, bands=["g", "r"])
        assert isinstance(lcs, list)
        if len(lcs) > 0:
            lc = lcs[0]
            assert "_id" in lc
            assert "filter" in lc
            assert "data" in lc
            assert len(lc["data"]) > 0
            point = lc["data"][0]
            assert "hjd" in point
            assert "mag" in point
            assert "magerr" in point
            assert "catflags" in point

    def test_pipeline_compatibility(self):
        """Lightcurve format should be compatible with sort_lightcurve."""
        from scope.utils import sort_lightcurve

        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=5,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        if len(objects) == 0:
            pytest.skip("No objects found in cone search")

        objectids = list(objects.keys())[:1]
        lcs = self.client.get_lightcurves(objectids)
        if len(lcs) == 0:
            pytest.skip("No lightcurves retrieved")

        lc = lcs[0]
        tme = np.array([[p["hjd"], p["mag"], p["magerr"]] for p in lc["data"]])
        t, m, e = tme.T
        t_sorted, m_sorted, e_sorted = sort_lightcurve(t, m, e)

        # Should be monotonically increasing
        assert np.all(np.diff(t_sorted) >= 0)
        assert len(t_sorted) == len(t)


# --- RubinLocalClient unit tests ---


def _create_synthetic_parquet(tmp_path):
    """Create synthetic parquet files for testing RubinLocalClient."""
    import pandas as pd

    # Object table: 5 objects, 4 isolated + 1 not
    # Place them near (ra=53.1, dec=-28.1)
    obj_df = pd.DataFrame(
        {
            "objectId": [1001, 1002, 1003, 1004, 1005],
            "coord_ra": [53.10, 53.1001, 53.1002, 53.10, 53.20],
            "coord_dec": [-28.10, -28.1001, -28.1002, -28.10, -28.10],
            "detect_isIsolated": [True, True, True, False, True],
        }
    )
    obj_df.to_parquet(tmp_path / "Object.parquet.gzip", compression="gzip")

    # Visit table: 3 visits
    visit_df = pd.DataFrame(
        {
            "visit": [100, 200, 300],
            "expMidptMJD": [60000.0, 60001.0, 60002.0],
        }
    )
    visit_df.to_parquet(tmp_path / "Visit.parquet.gzip", compression="gzip")

    # ForcedSource table: lightcurve data for objects 1001-1003
    fs_rows = []
    for oid in [1001, 1002, 1003]:
        for visit_id, band in [(100, "g"), (200, "g"), (300, "r")]:
            fs_rows.append(
                {
                    "objectId": oid,
                    "visit": visit_id,
                    "band": band,
                    "psfFlux": 1000.0 + oid * 0.1,
                    "psfFluxErr": 50.0,
                    "pixelFlags_bad": False,
                    "pixelFlags_cr": False,
                    "pixelFlags_edge": False,
                    "pixelFlags_saturated": False,
                    "pixelFlags_suspect": False,
                }
            )
    # Add a negative flux row to test filtering
    fs_rows.append(
        {
            "objectId": 1001,
            "visit": 100,
            "band": "i",
            "psfFlux": -100.0,
            "psfFluxErr": 10.0,
            "pixelFlags_bad": False,
            "pixelFlags_cr": True,
            "pixelFlags_edge": False,
            "pixelFlags_saturated": False,
            "pixelFlags_suspect": False,
        }
    )
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_parquet(tmp_path / "ForcedSource.parquet.gzip", compression="gzip")


class TestRubinLocalClient:
    """Unit tests for RubinLocalClient using synthetic parquet files."""

    @pytest.fixture(autouse=True)
    def setup_data(self, tmp_path):
        _create_synthetic_parquet(tmp_path)
        self.data_path = tmp_path
        self.client = RubinLocalClient(data_path=str(tmp_path))

    def test_cone_search_returns_dict(self):
        """Cone search should return a dict of objects."""
        objects = self.client.get_objects_by_cone(53.1, -28.1, 30.0)
        assert isinstance(objects, dict)
        assert len(objects) > 0

    def test_cone_search_isolated_only(self):
        """Cone search should only return isolated objects."""
        # Object 1004 is not isolated, should not appear
        objects = self.client.get_objects_by_cone(53.1, -28.1, 60.0)
        assert 1004 not in objects

    def test_cone_search_structure(self):
        """Each object should have coord_ra and coord_dec."""
        objects = self.client.get_objects_by_cone(53.1, -28.1, 30.0)
        if len(objects) > 0:
            first = next(iter(objects.values()))
            assert "coord_ra" in first
            assert "coord_dec" in first

    def test_cone_search_limit(self):
        """Limit should cap the number of returned objects."""
        objects = self.client.get_objects_by_cone(53.1, -28.1, 60.0, limit=2)
        assert len(objects) <= 2

    def test_cone_search_far_away_returns_empty(self):
        """Searching far from the data should return no objects."""
        objects = self.client.get_objects_by_cone(180.0, 45.0, 10.0)
        assert len(objects) == 0

    def test_get_lightcurves_returns_kowalski_format(self):
        """Lightcurves should be in Kowalski format."""
        lcs = self.client.get_lightcurves([1001])
        assert isinstance(lcs, list)
        assert len(lcs) > 0
        lc = lcs[0]
        assert "_id" in lc
        assert "filter" in lc
        assert "data" in lc
        assert len(lc["data"]) > 0
        point = lc["data"][0]
        assert "hjd" in point
        assert "mag" in point
        assert "magerr" in point
        assert "catflags" in point

    def test_get_lightcurves_band_filter(self):
        """Band filtering should restrict results."""
        lcs_g = self.client.get_lightcurves([1001], bands=["g"])
        lcs_r = self.client.get_lightcurves([1001], bands=["r"])
        lcs_all = self.client.get_lightcurves([1001])
        # g-only should have 2 data points, r-only should have 1
        g_points = sum(len(lc["data"]) for lc in lcs_g)
        r_points = sum(len(lc["data"]) for lc in lcs_r)
        all_points = sum(len(lc["data"]) for lc in lcs_all)
        assert g_points == 2
        assert r_points == 1
        assert all_points == g_points + r_points

    def test_get_lightcurves_empty_ids(self):
        """Empty ID list should return empty result."""
        lcs = self.client.get_lightcurves([])
        assert lcs == []

    def test_get_lightcurves_unknown_id(self):
        """Unknown object ID should return empty result."""
        lcs = self.client.get_lightcurves([999999])
        assert lcs == []

    def test_negative_flux_filtered(self):
        """Negative flux rows should be excluded from lightcurves."""
        lcs = self.client.get_lightcurves([1001], bands=["i"])
        # The only i-band row has negative flux, so no data
        assert len(lcs) == 0

    def test_get_lightcurves_for_cone(self):
        """Convenience method should return objects and lightcurves."""
        objects, lcs = self.client.get_lightcurves_for_cone(
            53.1, -28.1, 30.0, bands=["g", "r"]
        )
        assert isinstance(objects, dict)
        assert isinstance(lcs, list)
        if len(objects) > 0:
            assert len(lcs) > 0

    def test_missing_file_raises_error(self, tmp_path):
        """Missing parquet file should raise FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            RubinLocalClient(data_path=str(empty_dir))


class TestMakeRubinClient:
    """Tests for the make_rubin_client factory function."""

    def test_local_client_from_config(self, tmp_path):
        """Factory should return RubinLocalClient when data_path is set."""
        _create_synthetic_parquet(tmp_path)
        client = make_rubin_client({"data_path": str(tmp_path)})
        assert isinstance(client, RubinLocalClient)

    def test_env_var_override(self, tmp_path, monkeypatch):
        """RUBIN_DATA_PATH env var should trigger local client."""
        _create_synthetic_parquet(tmp_path)
        monkeypatch.setenv("RUBIN_DATA_PATH", str(tmp_path))
        client = make_rubin_client({})
        assert isinstance(client, RubinLocalClient)

    def test_config_data_path_takes_precedence(self, tmp_path, monkeypatch):
        """Config data_path should take precedence over env var."""
        _create_synthetic_parquet(tmp_path)
        monkeypatch.setenv("RUBIN_DATA_PATH", "/nonexistent")
        client = make_rubin_client({"data_path": str(tmp_path)})
        assert isinstance(client, RubinLocalClient)
        assert client.data_path == str(tmp_path)


# --- Golden reference test using real DP1 data sample ---
# This fixture uses a tiny slice of real DP1 data (3 objects, 3 visits,
# 9 ForcedSource rows) so the local client can be validated against known
# outputs without needing the full dataset or TAP access.


def _create_dp1_reference_parquet(tmp_path):
    """Create parquet files from a real DP1 data sample.

    The data below was extracted from the full DP1 parquet files.  It
    contains 3 isolated objects, 1 non-isolated object, 3 visits, and 9
    ForcedSource rows (including 2 with negative flux that should be
    filtered out).
    """
    import pandas as pd

    # Object table
    obj_df = pd.DataFrame(
        [
            {
                "objectId": 579574500513809549,
                "coord_ra": 6.299577852095733,
                "coord_dec": -72.65306479894076,
                "detect_isIsolated": True,
            },
            {
                "objectId": 579574500513809560,
                "coord_ra": 6.2976843024706275,
                "coord_dec": -72.65539119051866,
                "detect_isIsolated": True,
            },
            {
                "objectId": 579574569233285963,
                "coord_ra": 6.291714462729156,
                "coord_dec": -72.65061271307519,
                "detect_isIsolated": True,
            },
            {
                "objectId": 579574500513809614,
                "coord_ra": 6.299677852095733,
                "coord_dec": -72.65296479894076,
                "detect_isIsolated": False,
            },
        ]
    )
    obj_df.to_parquet(tmp_path / "Object.parquet.gzip", compression="gzip")

    # Visit table
    visit_df = pd.DataFrame(
        [
            {"visit": 2024113000109, "expMidptMJD": 60645.05975877311},
            {"visit": 2024120600027, "expMidptMJD": 60651.05833247099},
            {"visit": 2024120600053, "expMidptMJD": 60651.07542607068},
        ]
    )
    visit_df.to_parquet(tmp_path / "Visit.parquet.gzip", compression="gzip")

    # ForcedSource table (includes 2 rows with negative psfFlux)
    fs_df = pd.DataFrame(
        [
            {
                "objectId": 579574569233285963,
                "visit": 2024120600053,
                "band": "g",
                "psfFlux": 301.9809875488281,
                "psfFluxErr": 209.0,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574569233285963,
                "visit": 2024120600027,
                "band": "r",
                "psfFlux": 604.708984375,
                "psfFluxErr": 251.61500549316406,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574569233285963,
                "visit": 2024113000109,
                "band": "r",
                "psfFlux": -469.27099609375,
                "psfFluxErr": 282.5690002441406,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809560,
                "visit": 2024120600027,
                "band": "r",
                "psfFlux": 279.75299072265625,
                "psfFluxErr": 248.8820037841797,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809560,
                "visit": 2024120600053,
                "band": "g",
                "psfFlux": 814.885986328125,
                "psfFluxErr": 210.5500030517578,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809560,
                "visit": 2024113000109,
                "band": "r",
                "psfFlux": 455.89599609375,
                "psfFluxErr": 288.19500732421875,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809549,
                "visit": 2024120600027,
                "band": "r",
                "psfFlux": 1219.18994140625,
                "psfFluxErr": 250.33200073242188,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809549,
                "visit": 2024120600053,
                "band": "g",
                "psfFlux": 729.5900268554688,
                "psfFluxErr": 210.42100524902344,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
            {
                "objectId": 579574500513809549,
                "visit": 2024113000109,
                "band": "r",
                "psfFlux": -121.57099914550781,
                "psfFluxErr": 285.89801025390625,
                "pixelFlags_bad": False,
                "pixelFlags_cr": False,
                "pixelFlags_edge": False,
                "pixelFlags_saturated": False,
                "pixelFlags_suspect": False,
            },
        ]
    )
    fs_df.to_parquet(tmp_path / "ForcedSource.parquet.gzip", compression="gzip")


class TestGoldenReferenceDP1:
    """Validate RubinLocalClient against known outputs from real DP1 data.

    The reference data was extracted from the full DP1 parquet files.
    Expected magnitudes/errors were computed independently via
    ``_format_as_kowalski`` applied to the raw TAP-equivalent rows.
    This ensures the local client reproduces the same numbers that the
    TAP client would return for the same underlying data.
    """

    @pytest.fixture(autouse=True)
    def setup_data(self, tmp_path):
        _create_dp1_reference_parquet(tmp_path)
        self.client = RubinLocalClient(data_path=str(tmp_path))

    # -- Cone search --

    def test_cone_search_returns_all_isolated(self):
        """A wide cone should return all 3 isolated objects."""
        objects = self.client.get_objects_by_cone(6.296, -72.653, 60.0)
        assert len(objects) == 3
        assert 579574500513809549 in objects
        assert 579574500513809560 in objects
        assert 579574569233285963 in objects

    def test_cone_search_excludes_non_isolated(self):
        """Non-isolated object 579574500513809614 must not appear."""
        objects = self.client.get_objects_by_cone(6.296, -72.653, 60.0)
        assert 579574500513809614 not in objects

    def test_cone_search_coordinates_match(self):
        """Returned coordinates should match the Object table values."""
        objects = self.client.get_objects_by_cone(6.296, -72.653, 60.0)
        assert np.isclose(
            objects[579574500513809549]["coord_ra"], 6.299577852095733, atol=1e-8
        )
        assert np.isclose(
            objects[579574500513809549]["coord_dec"], -72.65306479894076, atol=1e-8
        )

    # -- Lightcurve retrieval --

    def test_lightcurve_count(self):
        """3 objects x 2 bands = up to 6 (objectId, band) entries.

        Two objects have a negative-flux r-band row that gets filtered,
        leaving 7 valid rows across 6 (objectId, band) combinations.
        """
        oids = [579574500513809549, 579574500513809560, 579574569233285963]
        lcs = self.client.get_lightcurves(oids, bands=["g", "r"])
        assert len(lcs) == 6

    def test_negative_flux_filtered(self):
        """Objects 579574500513809549 and 579574569233285963 each have a
        negative-flux r-band row that must be excluded."""
        lcs = self.client.get_lightcurves([579574500513809549], bands=["r"])
        # Only 1 valid r-band row (visit 2024120600027), the other is negative
        assert len(lcs) == 1
        assert len(lcs[0]["data"]) == 1

    def test_magnitude_values(self):
        """Spot-check computed magnitudes against independently derived values.

        Reference: objectId 579574500513809549, g-band, psfFlux=729.59 nJy
        Expected mag = -2.5*log10(729.59) + 31.4 = 24.24230...
        """
        lcs = self.client.get_lightcurves([579574500513809549], bands=["g"])
        assert len(lcs) == 1
        point = lcs[0]["data"][0]
        expected_mag = -2.5 * np.log10(729.5900268554688) + 31.4
        expected_magerr = (2.5 / np.log(10)) * (210.42100524902344 / 729.5900268554688)
        assert np.isclose(point["mag"], expected_mag, atol=1e-6)
        assert np.isclose(point["magerr"], expected_magerr, atol=1e-6)

    def test_mjd_values(self):
        """MJD should match the Visit table's expMidptMJD exactly."""
        lcs = self.client.get_lightcurves([579574500513809549], bands=["g"])
        # The g-band row uses visit 2024120600053 → expMidptMJD 60651.07542607068
        assert np.isclose(lcs[0]["data"][0]["hjd"], 60651.07542607068, atol=1e-8)

    def test_pixel_flags_zero(self):
        """All reference rows have no flags set, so catflags should be 0."""
        oids = [579574500513809549, 579574500513809560, 579574569233285963]
        lcs = self.client.get_lightcurves(oids)
        for lc in lcs:
            for point in lc["data"]:
                assert point["catflags"] == 0

    def test_band_filter_ids(self):
        """g-band entries should have filter=1, r-band filter=2."""
        oids = [579574500513809549]
        lcs_g = self.client.get_lightcurves(oids, bands=["g"])
        lcs_r = self.client.get_lightcurves(oids, bands=["r"])
        assert lcs_g[0]["filter"] == 1
        assert lcs_r[0]["filter"] == 2

    def test_multi_row_object(self):
        """Object 579574500513809560 has 2 valid r-band rows (visits
        2024120600027 and 2024113000109)."""
        lcs = self.client.get_lightcurves([579574500513809560], bands=["r"])
        assert len(lcs) == 1
        assert len(lcs[0]["data"]) == 2

    def test_full_pipeline_round_trip(self):
        """Cone search → lightcurves → sort should work end-to-end."""
        from scope.utils import sort_lightcurve

        objects, lcs = self.client.get_lightcurves_for_cone(
            6.296, -72.653, 60.0, bands=["g", "r"]
        )
        assert len(objects) == 3
        assert len(lcs) == 6

        # Pick the multi-row entry and sort it
        multi = [lc for lc in lcs if len(lc["data"]) > 1]
        assert len(multi) > 0
        lc = multi[0]
        tme = np.array([[p["hjd"], p["mag"], p["magerr"]] for p in lc["data"]])
        t, m, e = tme.T
        t_sorted, m_sorted, e_sorted = sort_lightcurve(t, m, e)
        assert np.all(np.diff(t_sorted) >= 0)
        assert len(t_sorted) == len(t)

    def test_matches_format_as_kowalski_directly(self):
        """Local client output should exactly match _format_as_kowalski
        applied to the same rows, ensuring parity with the TAP path."""
        # Manually replicate what get_lightcurves does
        oids = [579574500513809549, 579574500513809560, 579574569233285963]

        # Get result from client
        local_result = self.client.get_lightcurves(oids, bands=["g", "r"])

        # Build the same rows the TAP query would return
        # (positive flux only, with pixelFlags sum and expMidptMJD joined)
        tap_rows = [
            {
                "objectId": 579574569233285963,
                "band": "g",
                "psfFlux": 301.9809875488281,
                "psfFluxErr": 209.0,
                "expMidptMJD": 60651.07542607068,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574569233285963,
                "band": "r",
                "psfFlux": 604.708984375,
                "psfFluxErr": 251.61500549316406,
                "expMidptMJD": 60651.05833247099,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574500513809560,
                "band": "r",
                "psfFlux": 279.75299072265625,
                "psfFluxErr": 248.8820037841797,
                "expMidptMJD": 60651.05833247099,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574500513809560,
                "band": "g",
                "psfFlux": 814.885986328125,
                "psfFluxErr": 210.5500030517578,
                "expMidptMJD": 60651.07542607068,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574500513809560,
                "band": "r",
                "psfFlux": 455.89599609375,
                "psfFluxErr": 288.19500732421875,
                "expMidptMJD": 60645.05975877311,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574500513809549,
                "band": "r",
                "psfFlux": 1219.18994140625,
                "psfFluxErr": 250.33200073242188,
                "expMidptMJD": 60651.05833247099,
                "pixelFlags": 0,
            },
            {
                "objectId": 579574500513809549,
                "band": "g",
                "psfFlux": 729.5900268554688,
                "psfFluxErr": 210.42100524902344,
                "expMidptMJD": 60651.07542607068,
                "pixelFlags": 0,
            },
        ]
        tap_result = _format_as_kowalski(tap_rows)

        # Compare: same number of entries, same data points per entry
        assert len(local_result) == len(tap_result)

        # Build lookup by (_id, filter)
        local_by_key = {(lc["_id"], lc["filter"]): lc for lc in local_result}
        tap_by_key = {(lc["_id"], lc["filter"]): lc for lc in tap_result}

        assert set(local_by_key.keys()) == set(tap_by_key.keys())

        for key in tap_by_key:
            local_data = sorted(local_by_key[key]["data"], key=lambda p: p["hjd"])
            tap_data = sorted(tap_by_key[key]["data"], key=lambda p: p["hjd"])
            assert len(local_data) == len(tap_data), f"Mismatch for {key}"
            for lp, tp in zip(local_data, tap_data):
                assert np.isclose(lp["hjd"], tp["hjd"], atol=1e-8)
                assert np.isclose(lp["mag"], tp["mag"], atol=1e-6)
                assert np.isclose(lp["magerr"], tp["magerr"], atol=1e-6)
                assert lp["catflags"] == tp["catflags"]


# --- Integration tests for local parquet backend ---

DP1_LOCAL_PATH = "/fred/oz480/jfreebur/DP1/"

# Test position for local data integration tests.
# The local parquet files may cover a different sky region than the TAP
# test position above, so we use coordinates where ForcedSource data exists.
DP1_LOCAL_TEST_RA = 38.215
DP1_LOCAL_TEST_DEC = 6.459


@pytest.mark.skipif(
    not os.path.isdir(DP1_LOCAL_PATH),
    reason=f"Integration test requires local data at {DP1_LOCAL_PATH}",
)
class TestIntegrationRubinLocal:
    """Integration tests using real DP1 parquet files."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = RubinLocalClient(data_path=DP1_LOCAL_PATH)

    def test_cone_search(self):
        """Cone search should return objects from real data."""
        objects = self.client.get_objects_by_cone(
            ra=DP1_LOCAL_TEST_RA,
            dec=DP1_LOCAL_TEST_DEC,
            radius_arcsec=30.0,
            limit=10,
        )
        assert isinstance(objects, dict)
        assert len(objects) > 0
        first = next(iter(objects.values()))
        assert "coord_ra" in first
        assert "coord_dec" in first

    def test_lightcurve_retrieval(self):
        """Should retrieve lightcurves for discovered objects."""
        objects = self.client.get_objects_by_cone(
            ra=DP1_LOCAL_TEST_RA,
            dec=DP1_LOCAL_TEST_DEC,
            radius_arcsec=30.0,
            limit=5,
        )
        assert len(objects) > 0

        objectids = list(objects.keys())[:3]
        lcs = self.client.get_lightcurves(objectids, bands=["g", "r"])
        assert isinstance(lcs, list)
        assert len(lcs) > 0
        lc = lcs[0]
        assert "_id" in lc
        assert "filter" in lc
        assert "data" in lc
        assert len(lc["data"]) > 0
        point = lc["data"][0]
        assert "hjd" in point
        assert "mag" in point
        assert "magerr" in point
        assert "catflags" in point

    def test_pipeline_compatibility(self):
        """Lightcurve format should be compatible with sort_lightcurve."""
        from scope.utils import sort_lightcurve

        objects = self.client.get_objects_by_cone(
            ra=DP1_LOCAL_TEST_RA,
            dec=DP1_LOCAL_TEST_DEC,
            radius_arcsec=30.0,
            limit=5,
        )
        assert len(objects) > 0

        objectids = list(objects.keys())[:1]
        lcs = self.client.get_lightcurves(objectids)
        assert len(lcs) > 0

        lc = lcs[0]
        tme = np.array([[p["hjd"], p["mag"], p["magerr"]] for p in lc["data"]])
        t, m, e = tme.T
        t_sorted, m_sorted, e_sorted = sort_lightcurve(t, m, e)

        # Should be monotonically increasing
        assert np.all(np.diff(t_sorted) >= 0)
        assert len(t_sorted) == len(t)
