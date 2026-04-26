#!/usr/bin/env python
"""IRSA TAP client for NEOWISE-R single-exposure light curves.

Provides access to the NEOWISE Reactivation single-exposure point-source
catalog (``neowiser_p1bs_psd``) via the IRSA TAP API, returning light curves
in the same Kowalski-style dict format consumed by the scope-ml feature
generation pipeline.

Source positions can come from:
  * :class:`WISETAPClient.get_objects_by_cone` — small-area AllWISE/CatWISE query
  * an external source list (``ra``, ``dec`` arrays)
  * :class:`WISETAPClient.get_lightcurves_by_band` — bulk NEP-style pull keyed
    on ecliptic-latitude/longitude cells (no upload, no source-list bias).

WISE filter IDs follow scope-ml convention:
  * W1 = 10, W2 = 11, W3 = 12, W4 = 13 (distinct from ZTF g=1/r=2/i=3 and
    Rubin u=0..y=5 to keep all bands unambiguous inside the same catalog).

The class structure mirrors :mod:`scope.surveys.rubin` so the two can be used
interchangeably downstream.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import pyvo  # type: ignore

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

# Filter IDs (distinct from ZTF {g=1, r=2, i=3} and Rubin {u=0..y=5}).
DEFAULT_BAND_MAP: Dict[str, int] = {"W1": 10, "W2": 11, "W3": 12, "W4": 13}

# Default quality cuts for the NEOWISE-R single-exposure table.
#   cc_flags      artifact contamination flags, per band — '0' = clean
#   ph_qual       photometric quality: A best, B, C (no D/U) per band
#   qi_fact       frame quality-index (0–1), > 0.5 → usable
#   moon_masked   moon-contaminated frame flag, '0' = clean
#   saa_sep       deg from SAA, > 5 recommended
DEFAULT_QUALITY = {
    "cc_flags": "0000",
    "ph_qual_allowed": ("AA", "AB", "BA", "AC", "CA"),
    "qi_fact_min": 0.5,
    "moon_masked": "00",
    "saa_sep_min": 5.0,
    "w1snr_min": 3.0,
}

IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
JD_MINUS_MJD = 2400000.5  # HJD = MJD + 2400000.5 (approx., heliocentric correction small at |β|>85°)

# ---------------------------------------------------------------------------
# WISE-specific period-search aliases
# ---------------------------------------------------------------------------
# NEOWISE visits each non-polar sky spot every ~180 days, which creates a
# dense family of false peaks in any periodogram. The fundamental cadence
# period is T = 180 d (half-year revisit); the full alias family is
#
#     P_alias = 180 × (k / n)     for small integers k, n
#
# covering both sub-harmonics (k=1) and rational mixed harmonics. On top
# of that the sidereal year (~365.25 d) contributes its own beat.
# See diagnose_749143901351063040.py for the motivating case where our
# pipeline locked onto 176.8 d (≈ 180 × 49/50) and 65.4 d (≈ 180 × 3/8.25)
# for a source with true period 309.55 d.
#
# Values are cycles/day (frequency), ready to feed into
# ``periodsearch.find_periods(freqs_to_remove=…)``. Scalars are expanded
# to ±1% windows internally; pass ``(freq, tol)`` for wider masks.

WISE_CADENCE_FUNDAMENTAL_DAYS = 180.0
WISE_SIDEREAL_YEAR_DAYS = 365.25


def _generate_rational_aliases(
    fundamental_days: float,
    max_numerator: int = 3,
    max_denominator: int = 8,
    p_min_days: float = 15.0,
    p_max_days: float = 700.0,
) -> tuple:
    """Return a sorted tuple of P = fundamental × k/n for small integer k, n,
    clipped to ``[p_min_days, p_max_days]``."""
    periods = set()
    for k in range(1, max_numerator + 1):
        for n in range(1, max_denominator + 1):
            p = fundamental_days * k / n
            if p_min_days <= p <= p_max_days:
                periods.add(round(p, 3))
    return tuple(sorted(periods, reverse=True))


WISE_CADENCE_ALIAS_PERIODS_DAYS = tuple(
    list(
        _generate_rational_aliases(
            WISE_CADENCE_FUNDAMENTAL_DAYS,
            max_numerator=3,
            max_denominator=8,
            p_min_days=15.0,
            p_max_days=700.0,
        )
    )
    + [WISE_SIDEREAL_YEAR_DAYS]
)
"""Tuple of WISE cadence alias periods (in days).

Spans the 180 × k/n rational family (k ≤ 3, n ≤ 8, 15 d ≤ P ≤ 700 d)
plus the sidereal year. Typical entries: 540, 360, 270, 180, 135, 120,
108, 90, 77.14, 72, 67.5, 60, 51.43, 45, 36, 30, 25.71, 22.5, 18, 365.25.
"""

WISE_CADENCE_ALIAS_FREQS = tuple(1.0 / p for p in WISE_CADENCE_ALIAS_PERIODS_DAYS)
"""Frequency (cycles/day) of WISE cadence aliases, ready to feed into
``periodsearch.find_periods(freqs_to_remove=WISE_CADENCE_ALIAS_FREQS)``."""


def _mjd_to_hjd(mjd: np.ndarray) -> np.ndarray:
    """Convert MJD to HJD (Kowalski/SCoPe convention). Heliocentric correction
    is ≲ 8 min which we absorb into the LC noise — exact conversion would
    require source RA/Dec + time-of-year."""
    return mjd + JD_MINUS_MJD


def _quality_mask(df, quality: Dict[str, Any] = None) -> np.ndarray:
    """Return a boolean mask for standard NEOWISE-R quality cuts."""
    q = {**DEFAULT_QUALITY, **(quality or {})}
    mask = np.ones(len(df), dtype=bool)
    if "cc_flags" in df.columns:
        mask &= df["cc_flags"].astype(str) == q["cc_flags"]
    if "ph_qual" in df.columns:
        mask &= df["ph_qual"].isin(q["ph_qual_allowed"])
    if "qi_fact" in df.columns:
        mask &= df["qi_fact"] > q["qi_fact_min"]
    if "moon_masked" in df.columns:
        mask &= df["moon_masked"].astype(str) == q["moon_masked"]
    if "saa_sep" in df.columns:
        mask &= df["saa_sep"] > q["saa_sep_min"]
    return mask


def _catflags_from_quality(df) -> np.ndarray:
    """Map NEOWISE-R quality fields to a single integer ``catflags``
    field matching the ZTF convention (0 = clean, else flagged). Sums
    bitwise contributions so a caller can still decompose the reasons."""
    cf = np.zeros(len(df), dtype=np.int32)
    if "cc_flags" in df.columns:
        cf |= np.where(df["cc_flags"].astype(str) == "0000", 0, 1 << 0)
    if "ph_qual" in df.columns:
        cf |= np.where(df["ph_qual"].isin(("AA", "AB", "BA")), 0, 1 << 1)
    if "qi_fact" in df.columns:
        cf |= np.where(df["qi_fact"] > 0.5, 0, 1 << 2)
    if "moon_masked" in df.columns:
        cf |= np.where(df["moon_masked"].astype(str) == "00", 0, 1 << 3)
    return cf


def _format_as_kowalski(
    rows,
    id_col: str = "source_name",
    band_map: Optional[Dict[str, int]] = None,
    bands: Sequence[str] = ("W1", "W2"),
) -> List[Dict[str, Any]]:
    """Convert a pandas DataFrame of NEOWISE-R detections into Kowalski-format
    light-curve dicts.

    Parameters
    ----------
    rows : pandas.DataFrame
        Detection rows with at minimum ``id_col``, ``mjd``, and for each band
        ``{band}mpro`` and ``{band}sigmpro`` columns.
    id_col : str, default 'source_name'
        Column used to group detections into per-source light curves.
    band_map : dict, optional
        Band name -> integer filter ID.
    bands : sequence of str, default ('W1', 'W2')
        Which bands to emit.

    Returns
    -------
    list of dict
        One entry per (source, band) with keys ``_id``, ``filter``, ``data``,
        where ``data`` is a list of ``{hjd, mag, magerr, catflags}`` rows.
    """
    if band_map is None:
        band_map = DEFAULT_BAND_MAP

    catflags = _catflags_from_quality(rows)
    hjd = _mjd_to_hjd(rows["mjd"].values)

    out: List[Dict[str, Any]] = []
    for src_id, idx in rows.groupby(id_col).groups.items():
        idx = np.asarray(idx)
        for band in bands:
            magcol = f"{band.lower()}mpro"
            errcol = f"{band.lower()}sigmpro"
            if magcol not in rows.columns:
                continue
            band_mag = rows[magcol].values[idx]
            band_err = rows[errcol].values[idx]
            band_hjd = hjd[idx]
            band_cfl = catflags[idx]
            valid = np.isfinite(band_mag) & np.isfinite(band_err)
            if not valid.any():
                continue
            data = [
                {
                    "hjd": float(band_hjd[k]),
                    "mag": float(band_mag[k]),
                    "magerr": float(band_err[k]),
                    "catflags": int(band_cfl[k]),
                }
                for k in np.where(valid)[0]
            ]
            out.append(
                {
                    "_id": f"{src_id}_{band}",
                    "filter": band_map[band],
                    "data": data,
                    "source_id": src_id,
                    "band_name": band,
                }
            )
    return out


class WISETAPClient:
    """IRSA TAP client for NEOWISE-R single-exposure light curves.

    Parameters
    ----------
    tap_url : str
        TAP service URL. Defaults to IRSA's public endpoint.
    band_map : dict, optional
        WISE band name -> integer filter ID.
    quality : dict, optional
        Overrides for default quality cuts. See :data:`DEFAULT_QUALITY`.
    """

    def __init__(
        self,
        tap_url: str = IRSA_TAP_URL,
        band_map: Optional[Dict[str, int]] = None,
        quality: Optional[Dict[str, Any]] = None,
    ):
        if not HAS_PYVO:
            raise ImportError("pyvo is required for WISETAPClient")
        self.tap_url = tap_url
        self.tap = pyvo.dal.TAPService(tap_url)
        self.band_map = band_map if band_map is not None else dict(DEFAULT_BAND_MAP)
        self.quality = quality

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _submit_async(
        self,
        adql: str,
        uploads: Optional[dict] = None,
        poll: float = 15.0,
        timeout_sec: float = 14400.0,
    ):
        """Submit ADQL as an async TAP job, poll, return the fetched table as a
        pandas DataFrame. Retries a handful of times on known flaky responses
        (base-URL-only returns from IRSA)."""
        # IRSA sometimes returns just the base URL instead of the job-specific URL.
        # Retry up to 5 times on that pathology.
        job = None
        for _ in range(5):
            job = (
                self.tap.submit_job(adql, uploads=uploads)
                if uploads
                else self.tap.submit_job(adql)
            )
            url = job.url
            if "/async/" in url and not url.rstrip("/").endswith("/async"):
                break
            try:
                job.delete()
            except Exception:
                pass
            time.sleep(10)
        else:
            raise RuntimeError(
                f"IRSA TAP never returned a job-specific URL (last: {job.url!r})"
            )

        job.run()
        t0 = time.time()
        p = poll
        while job.phase in ("QUEUED", "EXECUTING", "PENDING"):
            time.sleep(p)
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(
                    f"TAP job still {job.phase} after {timeout_sec:.0f}s"
                )
            p = min(p * 1.2, 60.0)
        if job.phase != "COMPLETED":
            raise RuntimeError(f"TAP job failed with phase {job.phase}")
        tbl = job.fetch_result().to_table().to_pandas()
        tbl.columns = [c.lower() for c in tbl.columns]
        return tbl

    # ------------------------------------------------------------------
    # Public interface (mirrors scope.surveys.rubin)
    # ------------------------------------------------------------------
    def get_objects_by_cone(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        limit: int = 10000,
        source_catalog: str = "catwise_2020",
    ):
        """Return CatWISE2020 sources within a cone.

        Uses CatWISE2020 because its astrometry includes NEOWISE-R data
        through 2018 and has cleaner NEP-region positions than AllWISE.
        """
        adql = f"""
        SELECT TOP {int(limit)} source_name, ra, dec, elat, elon,
               w1mpro_pm, w2mpro_pm, w1snr_pm, w2snr_pm, pmra, pmdec
        FROM {source_catalog}
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})) = 1
        """
        return self._submit_async(adql)

    def get_lightcurves_by_position(
        self,
        ra: Sequence[float],
        dec: Sequence[float],
        match_radius_arcsec: float = 1.5,
        bands: Sequence[str] = ("W1", "W2"),
        ids: Optional[Sequence] = None,
    ):
        """Return Kowalski-format LCs for a list of (ra, dec) positions.

        Cross-matches each position to NEOWISE-R single-exposure detections
        within ``match_radius_arcsec`` and formats as LCs.
        """
        import pandas as pd  # lazy import
        from astropy.table import Table

        if ids is None:
            ids = [f"pos_{i}" for i in range(len(ra))]
        df = pd.DataFrame({"source_name": ids, "ra": ra, "dec": dec})
        tbl = Table.from_pandas(df)
        adql = f"""
        SELECT u.source_name, n.ra, n.dec, n.mjd,
               n.w1mpro, n.w1sigmpro, n.w2mpro, n.w2sigmpro,
               n.cc_flags, n.qi_fact, n.saa_sep, n.moon_masked, n.ph_qual, n.w1snr
        FROM TAP_UPLOAD.usercat AS u, neowiser_p1bs_psd AS n
        WHERE CONTAINS(POINT('ICRS', n.ra, n.dec),
                       CIRCLE('ICRS', u.ra, u.dec, {match_radius_arcsec/3600})) = 1
        """
        rows = self._submit_async(adql, uploads={"usercat": tbl})
        # Apply quality cuts, then format
        mask = _quality_mask(rows, self.quality)
        return _format_as_kowalski(
            rows[mask], id_col="source_name", band_map=self.band_map, bands=bands
        )

    def get_lightcurves_for_cone(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        bands: Sequence[str] = ("W1", "W2"),
        limit: int = 10000,
    ):
        """Convenience: find all CatWISE2020 sources in a cone and fetch their
        NEOWISE-R light curves in one call."""
        catwise = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(catwise) == 0:
            return []
        return self.get_lightcurves_by_position(
            ra=catwise["ra"].values,
            dec=catwise["dec"].values,
            ids=catwise["source_name"].values,
            bands=bands,
        )

    def get_detections_by_ecliptic_cell(
        self,
        elat_lo: float,
        elat_hi: float,
        elon_lo: float,
        elon_hi: float,
        w1snr_min: float = 3.0,
    ):
        """Bulk-fetch raw NEOWISE-R detections in an ecliptic (lat, lon) cell.

        Used for the NEP |β|>85° scans where we want position-blind coverage
        (no source list, so no CatWISE selection bias).
        """
        adql = f"""
        SELECT ra, dec, elat, elon, mjd,
               w1mpro, w1sigmpro, w2mpro, w2sigmpro,
               cc_flags, qi_fact, saa_sep, moon_masked, ph_qual, w1snr, w2snr
        FROM neowiser_p1bs_psd
        WHERE elat BETWEEN {elat_lo} AND {elat_hi}
          AND elon BETWEEN {elon_lo} AND {elon_hi}
          AND cc_flags = '0000'
          AND ph_qual IN ('AA','AB','BA','AC','CA')
          AND qi_fact > 0.5
          AND moon_masked = '00'
          AND w1snr > {w1snr_min}
        """
        return self._submit_async(adql)


class WISELocalClient:
    """Local-parquet client for NEOWISE-R light curves.

    Intended for compute nodes without internet access. Reads from a directory
    of NEOWISE-R single-exposure parquets produced by :class:`WISETAPClient`
    (or equivalent bulk-pull tooling). Presents the same ``get_lightcurves_*``
    interface as :class:`WISETAPClient` so downstream code is interchangeable.

    Expected inputs (in priority order):
      1. A single combined parquet (``--combined``): sorted by ``source_name``
         and produced by ``combine_matched.py`` style tooling.
      2. A directory of per-cell parquets (``--cells-dir``).
      3. Both — combined is preferred.

    Parameters
    ----------
    data_path : str
        Either a combined parquet file or a directory containing cell parquets
        (``cell_*.parquet``). Directory form requires a separate CatWISE2020
        source-list parquet with (source_name, ra, dec) for cross-matching.
    catwise_source_list : str, optional
        Path to a parquet with CatWISE2020 positions. Only required when
        ``data_path`` is a directory of un-matched cell parquets.
    band_map : dict, optional
    """

    def __init__(
        self,
        data_path: str,
        catwise_source_list: Optional[str] = None,
        band_map: Optional[Dict[str, int]] = None,
    ):
        self.data_path = data_path
        self.catwise_source_list = catwise_source_list
        self.band_map = band_map if band_map is not None else dict(DEFAULT_BAND_MAP)
        self._is_combined = os.path.isfile(data_path) and data_path.endswith(".parquet")
        if not self._is_combined and not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"data_path {data_path!r} is neither a parquet file nor a directory"
            )
        self._sources_df = None  # lazy-loaded CatWISE source list

    # ------------------------------------------------------------------
    def _load_sources(self):
        import pandas as pd  # lazy import

        if self._sources_df is not None:
            return self._sources_df
        if self._is_combined:
            # A combined matched parquet already has source_name per-detection;
            # derive a unique-source catalog from it.
            df = pd.read_parquet(self.data_path, columns=["source_name", "ra", "dec"])
            self._sources_df = df.drop_duplicates("source_name").reset_index(drop=True)
        else:
            if self.catwise_source_list is None:
                raise ValueError(
                    "catwise_source_list is required when data_path is a directory"
                )
            self._sources_df = pd.read_parquet(
                self.catwise_source_list, columns=["source_name", "ra", "dec"]
            )
        return self._sources_df

    # ------------------------------------------------------------------
    def get_objects_by_cone(
        self, ra: float, dec: float, radius_arcsec: float, limit: int = 10000
    ):
        """Return CatWISE-like rows within a cone, from the local source list."""
        import pandas as pd  # noqa: F401 — triggers pandas availability error cleanly if missing

        sources = self._load_sources()
        # Small-angle approximation: OK for |β|>85 where everything is near the pole,
        # but use proper angular separation for correctness.
        ra_s = np.asarray(sources["ra"])
        dec_s = np.asarray(sources["dec"])
        cos_d = np.cos(np.radians(dec_s))
        dra = (ra_s - ra) * cos_d
        ddec = dec_s - dec
        sep_arcsec = np.sqrt(dra * dra + ddec * ddec) * 3600
        mask = sep_arcsec < radius_arcsec
        hits = sources[mask].copy()
        hits["sep_arcsec"] = sep_arcsec[mask]
        return hits.sort_values("sep_arcsec").head(limit)

    # ------------------------------------------------------------------
    def get_lightcurves(
        self,
        source_ids: Sequence[str],
        bands: Sequence[str] = ("W1", "W2"),
    ):
        """Return Kowalski-format LCs for a list of CatWISE source names."""
        import pyarrow as pa  # lazy
        import pyarrow.compute as pc
        import pyarrow.dataset as pds

        source_ids = list(source_ids)
        if self._is_combined:
            ds = pds.dataset(self.data_path, format="parquet")
        else:
            files = sorted(
                f
                for f in os.listdir(self.data_path)
                if f.startswith("cell_") and f.endswith(".parquet")
            )
            if not files:
                raise FileNotFoundError(f"No cell_*.parquet under {self.data_path}")
            ds = pds.dataset(
                [os.path.join(self.data_path, f) for f in files], format="parquet"
            )

        sn_type = ds.schema.field("source_name").type
        pa_set = pa.array(source_ids, type=sn_type)
        filt = pc.is_in(pds.field("source_name"), value_set=pa_set)
        cols = [
            "source_name",
            "mjd",
            "w1mpro",
            "w1sigmpro",
            "w2mpro",
            "w2sigmpro",
            "cc_flags",
            "ph_qual",
            "qi_fact",
            "moon_masked",
        ]
        # Drop columns that don't exist in this particular schema
        cols = [c for c in cols if c in ds.schema.names]
        tbl = ds.to_table(columns=cols, filter=filt)
        df = tbl.to_pandas()
        mask = _quality_mask(df)
        return _format_as_kowalski(
            df[mask], id_col="source_name", band_map=self.band_map, bands=bands
        )

    def get_lightcurves_for_cone(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        bands: Sequence[str] = ("W1", "W2"),
        limit: int = 10000,
    ):
        hits = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(hits) == 0:
            return []
        return self.get_lightcurves(hits["source_name"].values, bands=bands)


def make_wise_client(
    config: Optional[Dict[str, Any]] = None,
    *,
    prefer_local: bool = False,
):
    """Factory: return an appropriate WISE client based on config + environment.

    Resolution order:
      1. ``config['wise']['data_path']`` set (or env ``WISE_DATA_PATH``) → return
         :class:`WISELocalClient` reading from local cache. This is the only
         option on offline compute nodes.
      2. Otherwise, return :class:`WISETAPClient` if pyvo is available.
      3. Otherwise, return ``None``.

    Parameters
    ----------
    config : dict, optional
        Full scope-ml config; expects ``config['wise']`` sub-tree with
        ``data_path`` (for local) and/or ``tap_url``, ``band_map``, ``quality``.
    prefer_local : bool, default False
        Force local-first even if both are available.
    """
    wise_cfg = (config or {}).get("wise", {})
    local_path = wise_cfg.get("data_path") or os.environ.get("WISE_DATA_PATH")
    catwise_list = wise_cfg.get("catwise_source_list") or os.environ.get(
        "WISE_CATWISE_LIST"
    )

    # Local-cache path wins on offline nodes
    if local_path:
        if not os.path.exists(local_path):
            warnings.warn(f"WISE local cache {local_path!r} does not exist")
        else:
            return WISELocalClient(
                data_path=local_path,
                catwise_source_list=catwise_list,
                band_map=wise_cfg.get("band_map"),
            )

    if prefer_local:
        # User explicitly asked for local but we didn't find it
        warnings.warn("prefer_local=True but no WISE local cache configured")
        return None

    # Fall through to TAP
    if not HAS_PYVO:
        warnings.warn(
            "pyvo not installed and no WISE local cache — WISE access unavailable"
        )
        return None
    return WISETAPClient(
        tap_url=wise_cfg.get("tap_url", IRSA_TAP_URL),
        band_map=wise_cfg.get("band_map"),
        quality=wise_cfg.get("quality"),
    )
