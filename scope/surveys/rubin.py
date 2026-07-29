#!/usr/bin/env python
"""
Rubin LSST TAP client for scope-ml.

Provides access to Rubin Data Preview (DP1) and Early Data Preview 2
(EDP2/DP2) data via the TAP API or local parquet exports, converting
forced photometry lightcurves into the same format consumed by the
scope-ml feature generation pipeline.
"""

import glob
import os
import numpy as np
import warnings

try:
    import pyvo
    import pyvo.auth

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

# Band name -> integer filter ID mapping
# g=1, r=2, i=3 match ZTF filter IDs; u, z, y are Rubin-only
DEFAULT_BAND_MAP = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

# AB magnitude zero point for nanoJansky: mag = -2.5 * log10(flux_nJy) + 31.4
NANOJANSKY_ZP = 31.4


def _flux_to_mag(flux_nJy, flux_err_nJy):
    """
    Convert flux in nanoJansky to AB magnitude and magnitude error.

    Parameters
    ----------
    flux_nJy : float or array-like
        Flux in nanoJansky. Must be > 0.
    flux_err_nJy : float or array-like
        Flux uncertainty in nanoJansky. Must be > 0.

    Returns
    -------
    mag : float or ndarray
        AB magnitude
    magerr : float or ndarray
        Magnitude uncertainty
    """
    flux_nJy = np.asarray(flux_nJy, dtype=float)
    flux_err_nJy = np.asarray(flux_err_nJy, dtype=float)

    mag = -2.5 * np.log10(flux_nJy) + NANOJANSKY_ZP
    magerr = (2.5 / np.log(10)) * (flux_err_nJy / flux_nJy)

    return mag, magerr


def _format_as_kowalski(rows, band_map=None):
    """
    Convert TAP query result rows into Kowalski-format lightcurve dicts.

    The output matches the format returned by ``get_lightcurves_via_ids``
    in ``scope.fritz``:
        [{'_id': objectId, 'filter': band_int, 'data': [
            {'hjd': mjd, 'mag': m, 'magerr': e, 'catflags': flags}, ...
        ]}, ...]

    One entry per (objectId, band) combination is returned.

    Parameters
    ----------
    rows : list of dict
        Each dict must have keys: objectId, band, psfFlux, psfFluxErr,
        expMidptMJD, pixelFlags.
    band_map : dict, optional
        Mapping from band name (str) to integer filter ID.

    Returns
    -------
    list of dict
        Kowalski-format lightcurve dicts.
    """
    if band_map is None:
        band_map = DEFAULT_BAND_MAP

    # Group by (objectId, band)
    grouped = {}
    for row in rows:
        key = (int(row["objectId"]), str(row["band"]))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    result = []
    for (obj_id, band), points in grouped.items():
        filt = band_map.get(band, -1)
        data = []
        for p in points:
            flux = float(p["psfFlux"])
            flux_err = float(p["psfFluxErr"])
            if flux <= 0 or flux_err <= 0:
                continue
            mag, magerr = _flux_to_mag(flux, flux_err)
            data.append(
                {
                    "hjd": float(p["expMidptMJD"]),
                    "mag": float(mag),
                    "magerr": float(magerr),
                    "catflags": int(p["pixelFlags"]),
                }
            )
        if len(data) > 0:
            result.append({"_id": obj_id, "filter": filt, "data": data})

    return result


def _format_frame_as_kowalski(
    df,
    band_map=None,
    id_col="objectId",
    band_col="band",
    time_col="expMidptMJD",
    flux_col="psfFlux",
    flux_err_col="psfFluxErr",
    flag_col="pixelFlags",
):
    """
    Vectorized equivalent of :func:`_format_as_kowalski` for DataFrames.

    Produces the same output structure, but converts fluxes with numpy
    array operations instead of a Python loop over rows. Points within
    each (objectId, band) group are sorted by time.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns named by ``id_col``, ``band_col``,
        ``time_col``, ``flux_col``, ``flux_err_col`` and ``flag_col``.
    band_map : dict, optional
        Mapping from band name (str) to integer filter ID.

    Returns
    -------
    list of dict
        Kowalski-format lightcurve dicts.
    """
    if band_map is None:
        band_map = DEFAULT_BAND_MAP

    if len(df) == 0:
        return []

    flux = df[flux_col].values.astype(np.float64)
    flux_err = df[flux_err_col].values.astype(np.float64)
    good = (flux > 0) & (flux_err > 0)
    if not good.any():
        return []

    obj_ids = df[id_col].values[good]
    bands = df[band_col].values[good]
    times = df[time_col].values[good].astype(np.float64)
    flags = df[flag_col].values[good].astype(np.int64)
    mag, magerr = _flux_to_mag(flux[good], flux_err[good])

    # Sort by (objectId, band, time) so each group is contiguous and
    # time-ordered; np.lexsort keys are applied last-to-first.
    order = np.lexsort((times, bands, obj_ids))
    obj_ids = obj_ids[order]
    bands = bands[order]
    times = times[order]
    flags = flags[order]
    mag = mag[order]
    magerr = magerr[order]

    # Group boundaries wherever (objectId, band) changes
    if len(obj_ids) == 1:
        starts = np.array([0])
    else:
        changed = (obj_ids[1:] != obj_ids[:-1]) | (bands[1:] != bands[:-1])
        starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    ends = np.concatenate((starts[1:], [len(obj_ids)]))

    result = []
    for start, end in zip(starts, ends):
        data = [
            {
                "hjd": float(times[k]),
                "mag": float(mag[k]),
                "magerr": float(magerr[k]),
                "catflags": int(flags[k]),
            }
            for k in range(start, end)
        ]
        result.append(
            {
                "_id": int(obj_ids[start]),
                "filter": band_map.get(str(bands[start]), -1),
                "data": data,
            }
        )

    return result


class RubinTAPClient:
    """
    Client for querying Rubin LSST data via the TAP API.

    Parameters
    ----------
    tap_url : str
        URL of the TAP service endpoint.
    token : str
        Authentication token for the TAP service.
    timeout : int, optional
        Query timeout in seconds (default 300).
    band_map : dict, optional
        Mapping from band name to integer filter ID.
    config : dict, optional
        Rubin config section. If provided, overrides tap_url/token/timeout
        unless those are explicitly given.
    """

    def __init__(
        self,
        tap_url=None,
        token=None,
        timeout=None,
        band_map=None,
        config=None,
    ):
        if not HAS_PYVO:
            raise ImportError(
                "pyvo is required for Rubin TAP access. "
                "Install it with: pip install pyvo"
            )

        # Resolve from config if provided
        if config is not None:
            tap_url = tap_url or config.get(
                "tap_url", "https://data.lsst.cloud/api/tap"
            )
            token = token or config.get("token")
            timeout = timeout if timeout is not None else config.get("timeout", 300)
            band_map = band_map or config.get("band_map", DEFAULT_BAND_MAP)

        if token is None:
            raise ValueError(
                "Rubin TAP token is required. Set it in config.yaml under "
                "rubin.token or via the RUBIN_TAP_TOKEN environment variable."
            )

        self.tap_url = tap_url
        self.token = token
        self.timeout = timeout or 300
        self.band_map = band_map or DEFAULT_BAND_MAP

        # Tables (configurable via config)
        if config is not None and "tables" in config:
            tables = config["tables"]
            self.table_object = tables.get("object", "dp1.Object")
            self.table_forced_source = tables.get("forced_source", "dp1.ForcedSource")
            self.table_visit = tables.get("visit", "dp1.Visit")
        else:
            self.table_object = "dp1.Object"
            self.table_forced_source = "dp1.ForcedSource"
            self.table_visit = "dp1.Visit"

        # Set up authenticated TAP service via Bearer token
        import requests

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {self.token}"
        self.service = pyvo.dal.TAPService(self.tap_url, session=session)

    def get_objects_by_cone(self, ra, dec, radius_arcsec, limit=10000):
        """
        Find objects within a cone search region.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        limit : int, optional
            Maximum number of objects to return (default 10000).

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        radius_deg = radius_arcsec / 3600.0

        top_clause = f"TOP {limit}" if limit is not None else ""
        query = f"""
        SELECT {top_clause} objectId, coord_ra, coord_dec
        FROM {self.table_object}
        WHERE CONTAINS(
            POINT('ICRS', coord_ra, coord_dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        AND detect_isIsolated = 1
        """

        result = self.service.search(query)
        objects = {}
        for row in result:
            oid = int(row["objectId"])
            objects[oid] = {
                "coord_ra": float(row["coord_ra"]),
                "coord_dec": float(row["coord_dec"]),
            }
        return objects

    def get_lightcurves(self, objectids, bands=None, batch_size=1000):
        """
        Retrieve forced photometry lightcurves for a list of object IDs.

        Parameters
        ----------
        objectids : list of int
            Object IDs to query.
        bands : list of str, optional
            Band names to filter (e.g., ['g', 'r']). If None, all bands.
        batch_size : int, optional
            Number of IDs per query batch (default 1000).

        Returns
        -------
        list of dict
            Kowalski-format lightcurve dicts.
        """
        if len(objectids) == 0:
            return []

        all_rows = []
        n_batches = (len(objectids) + batch_size - 1) // batch_size

        # Batch the queries
        for i in range(0, len(objectids), batch_size):
            batch_num = i // batch_size + 1
            if batch_num % 10 == 1 or batch_num == n_batches:
                print(
                    f"  TAP batch {batch_num}/{n_batches} "
                    f"({len(all_rows):,} rows so far)"
                )
            batch = objectids[i : i + batch_size]
            id_list = ", ".join(str(oid) for oid in batch)

            band_clause = ""
            if bands is not None and len(bands) > 0:
                band_values = ", ".join(f"'{b}'" for b in bands)
                band_clause = f"AND fs.band IN ({band_values})"

            query = f"""
            SELECT fs.objectId, fs.band, fs.psfFlux, fs.psfFluxErr,
                   v.expMidptMJD,
                   (fs.pixelFlags_bad + fs.pixelFlags_cr
                    + fs.pixelFlags_edge + fs.pixelFlags_saturated
                    + fs.pixelFlags_suspect) AS pixelFlags
            FROM {self.table_forced_source} AS fs
            JOIN {self.table_visit} AS v
                ON fs.visit = v.visit
            WHERE fs.objectId IN ({id_list})
            AND fs.psfFlux > 0
            AND fs.psfFluxErr > 0
            {band_clause}
            """

            try:
                # Use async TAP for large queries
                if len(batch) > 100:
                    job = self.service.submit_job(query)
                    job.run()
                    job.wait(
                        phases=["COMPLETED", "ERROR", "ABORTED"], timeout=self.timeout
                    )
                    if job.phase == "COMPLETED":
                        result = job.fetch_result()
                    else:
                        warnings.warn(
                            f"TAP async job failed with phase: {job.phase}. "
                            f"Skipping batch starting at index {i}."
                        )
                        continue
                else:
                    result = self.service.search(query)

                for row in result:
                    all_rows.append(
                        {
                            "objectId": row["objectId"],
                            "band": row["band"],
                            "psfFlux": row["psfFlux"],
                            "psfFluxErr": row["psfFluxErr"],
                            "expMidptMJD": row["expMidptMJD"],
                            "pixelFlags": row["pixelFlags"],
                        }
                    )

            except Exception as e:
                warnings.warn(f"TAP query failed for batch starting at index {i}: {e}")
                continue

        return _format_as_kowalski(all_rows, band_map=self.band_map)

    def get_lightcurves_for_cone(self, ra, dec, radius_arcsec, bands=None, limit=10000):
        """
        Convenience method: cone search + lightcurve retrieval in one call.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        bands : list of str, optional
            Band names to filter.
        limit : int, optional
            Maximum number of objects from cone search.

        Returns
        -------
        objects : dict
            Object metadata from cone search.
        lightcurves : list of dict
            Kowalski-format lightcurve dicts.
        """
        objects = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(objects) == 0:
            return objects, []

        objectids = list(objects.keys())
        lightcurves = self.get_lightcurves(objectids, bands=bands)
        return objects, lightcurves


class RubinLocalClient:
    """
    Client for querying Rubin LSST data from local parquet files.

    Drop-in replacement for RubinTAPClient that reads from a local
    directory containing Object.parquet.gzip, ForcedSource.parquet.gzip,
    and Visit.parquet.gzip files.

    Parameters
    ----------
    data_path : str
        Path to directory containing the parquet files.
    band_map : dict, optional
        Mapping from band name to integer filter ID.
    """

    def __init__(self, data_path, band_map=None):
        import pandas as pd

        self.data_path = data_path
        self.band_map = band_map or DEFAULT_BAND_MAP

        # Validate that the required files exist
        required_files = [
            "Object.parquet.gzip",
            "ForcedSource.parquet.gzip",
            "Visit.parquet.gzip",
        ]
        for fname in required_files:
            fpath = os.path.join(data_path, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Required file not found: {fpath}. "
                    f"The data_path directory must contain: {required_files}"
                )

        # Visit table is small (~203 KB), load eagerly
        visit_path = os.path.join(data_path, "Visit.parquet.gzip")
        self._visit_df = pd.read_parquet(visit_path, columns=["visit", "expMidptMJD"])

        # Lazy-loaded caches
        self._obj_ids = None
        self._obj_ra_rad = None
        self._obj_dec_rad = None
        self._obj_ra_deg = None
        self._obj_dec_deg = None
        self._obj_isolated = None

        self._fs_df = None

    def _load_objects(self):
        """Lazy-load the Object table and precompute radians."""
        if self._obj_ids is not None:
            return

        import pandas as pd

        obj_path = os.path.join(self.data_path, "Object.parquet.gzip")
        df = pd.read_parquet(
            obj_path,
            columns=["objectId", "coord_ra", "coord_dec", "detect_isIsolated"],
        )

        self._obj_ids = df["objectId"].values
        self._obj_ra_deg = df["coord_ra"].values.astype(np.float64)
        self._obj_dec_deg = df["coord_dec"].values.astype(np.float64)
        self._obj_isolated = df["detect_isIsolated"].values

        # Precompute radians for angular_separation
        self._obj_ra_rad = np.deg2rad(self._obj_ra_deg)
        self._obj_dec_rad = np.deg2rad(self._obj_dec_deg)

    def _load_forced_sources(self):
        """Lazy-load the ForcedSource table, joined with Visit on visit."""
        if self._fs_df is not None:
            return

        import pandas as pd

        fs_path = os.path.join(self.data_path, "ForcedSource.parquet.gzip")
        fs_df = pd.read_parquet(
            fs_path,
            columns=[
                "objectId",
                "visit",
                "band",
                "psfFlux",
                "psfFluxErr",
                "pixelFlags_bad",
                "pixelFlags_cr",
                "pixelFlags_edge",
                "pixelFlags_saturated",
                "pixelFlags_suspect",
            ],
        )

        # Join with Visit to get expMidptMJD
        fs_df = fs_df.merge(self._visit_df, on="visit", how="inner")

        # Compute pixelFlags as sum of boolean columns
        flag_cols = [
            "pixelFlags_bad",
            "pixelFlags_cr",
            "pixelFlags_edge",
            "pixelFlags_saturated",
            "pixelFlags_suspect",
        ]
        fs_df["pixelFlags"] = sum(fs_df[c].astype(int) for c in flag_cols)

        # Sort by objectId for searchsorted lookups
        fs_df.sort_values("objectId", inplace=True)
        fs_df.reset_index(drop=True, inplace=True)

        # Keep only the columns we need
        self._fs_df = fs_df[
            ["objectId", "band", "psfFlux", "psfFluxErr", "expMidptMJD", "pixelFlags"]
        ]
        self._fs_object_ids = self._fs_df["objectId"].values

    def get_objects_by_cone(self, ra, dec, radius_arcsec, limit=10000):
        """
        Find isolated objects within a cone search region.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        limit : int, optional
            Maximum number of objects to return (default 10000).

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        from astropy.coordinates import angular_separation

        self._load_objects()

        radius_rad = np.deg2rad(radius_arcsec / 3600.0)
        center_ra_rad = np.deg2rad(ra)
        center_dec_rad = np.deg2rad(dec)

        # Filter to isolated objects first
        isolated_mask = self._obj_isolated == True  # noqa: E712

        obj_ra_rad = self._obj_ra_rad[isolated_mask]
        obj_dec_rad = self._obj_dec_rad[isolated_mask]

        # Vectorized angular separation
        sep = angular_separation(center_ra_rad, center_dec_rad, obj_ra_rad, obj_dec_rad)

        cone_mask = sep <= radius_rad
        obj_ids = self._obj_ids[isolated_mask][cone_mask]
        obj_ra = self._obj_ra_deg[isolated_mask][cone_mask]
        obj_dec = self._obj_dec_deg[isolated_mask][cone_mask]

        # Apply limit
        if limit is not None and len(obj_ids) > limit:
            obj_ids = obj_ids[:limit]
            obj_ra = obj_ra[:limit]
            obj_dec = obj_dec[:limit]

        objects = {}
        for i in range(len(obj_ids)):
            objects[int(obj_ids[i])] = {
                "coord_ra": float(obj_ra[i]),
                "coord_dec": float(obj_dec[i]),
            }

        return objects

    def get_lightcurves(self, objectids, bands=None, batch_size=1000):
        """
        Retrieve forced photometry lightcurves for a list of object IDs.

        Parameters
        ----------
        objectids : list of int
            Object IDs to query.
        bands : list of str, optional
            Band names to filter (e.g., ['g', 'r']). If None, all bands.
        batch_size : int, optional
            Not used for local backend (kept for API compatibility).

        Returns
        -------
        list of dict
            Kowalski-format lightcurve dicts.
        """
        if len(objectids) == 0:
            return []

        self._load_forced_sources()

        # Use searchsorted for efficient lookup
        oid_set = set(int(oid) for oid in objectids)
        left_indices = np.searchsorted(self._fs_object_ids, list(oid_set), side="left")
        right_indices = np.searchsorted(
            self._fs_object_ids, list(oid_set), side="right"
        )

        # Gather row indices
        idx_list = []
        for left, right in zip(left_indices, right_indices):
            if left < right:
                idx_list.append(np.arange(left, right))

        if len(idx_list) == 0:
            return []

        all_idx = np.concatenate(idx_list)
        subset = self._fs_df.iloc[all_idx]

        # Filter by bands if specified
        if bands is not None and len(bands) > 0:
            subset = subset[subset["band"].isin(bands)]

        # Filter positive flux
        subset = subset[(subset["psfFlux"] > 0) & (subset["psfFluxErr"] > 0)]

        if len(subset) == 0:
            return []

        # Convert to list of dicts for _format_as_kowalski
        rows = subset.to_dict("records")
        return _format_as_kowalski(rows, band_map=self.band_map)

    def get_lightcurves_for_cone(self, ra, dec, radius_arcsec, bands=None, limit=10000):
        """
        Convenience method: cone search + lightcurve retrieval in one call.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        bands : list of str, optional
            Band names to filter.
        limit : int, optional
            Maximum number of objects from cone search.

        Returns
        -------
        objects : dict
            Object metadata from cone search.
        lightcurves : list of dict
            Kowalski-format lightcurve dicts.
        """
        objects = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(objects) == 0:
            return objects, []

        objectids = list(objects.keys())
        lightcurves = self.get_lightcurves(objectids, bands=bands)
        return objects, lightcurves


class RubinLocalDiaClient:
    """
    Client for querying Rubin LSST DIA data from local parquet files.

    Drop-in replacement for RubinLocalClient that reads from DiaObject
    and ForcedSourceOnDiaObject tables instead of Object and ForcedSource.
    Covers all DP1 fields with no isolation restriction.

    Parameters
    ----------
    data_path : str
        Path to directory containing the parquet files.
    band_map : dict, optional
        Mapping from band name to integer filter ID.
    """

    def __init__(self, data_path, band_map=None):
        import pandas as pd

        self.data_path = data_path
        self.band_map = band_map or DEFAULT_BAND_MAP

        # Validate that the required files exist
        required_files = [
            "DiaObject.parquet.gzip",
            "ForcedSourceOnDiaObject.parquet.gzip",
            "Visit.parquet.gzip",
        ]
        for fname in required_files:
            fpath = os.path.join(data_path, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Required file not found: {fpath}. "
                    f"The data_path directory must contain: {required_files}"
                )

        # Visit table is small (~203 KB), load eagerly
        visit_path = os.path.join(data_path, "Visit.parquet.gzip")
        self._visit_df = pd.read_parquet(visit_path, columns=["visit", "expMidptMJD"])

        # Lazy-loaded caches
        self._obj_ids = None
        self._obj_ra_rad = None
        self._obj_dec_rad = None
        self._obj_ra_deg = None
        self._obj_dec_deg = None

        self._fs_df = None

    def _load_objects(self):
        """Lazy-load the DiaObject table and precompute radians."""
        if self._obj_ids is not None:
            return

        import pandas as pd

        obj_path = os.path.join(self.data_path, "DiaObject.parquet.gzip")
        df = pd.read_parquet(
            obj_path,
            columns=["diaObjectId", "ra", "dec"],
        )

        self._obj_ids = df["diaObjectId"].values
        self._obj_ra_deg = df["ra"].values.astype(np.float64)
        self._obj_dec_deg = df["dec"].values.astype(np.float64)

        # Precompute radians for angular_separation
        self._obj_ra_rad = np.deg2rad(self._obj_ra_deg)
        self._obj_dec_rad = np.deg2rad(self._obj_dec_deg)

    def _load_forced_sources(self):
        """Lazy-load the ForcedSourceOnDiaObject table, joined with Visit."""
        if self._fs_df is not None:
            return

        import pandas as pd

        fs_path = os.path.join(self.data_path, "ForcedSourceOnDiaObject.parquet.gzip")
        fs_df = pd.read_parquet(
            fs_path,
            columns=[
                "diaObjectId",
                "visit",
                "band",
                "psfFlux",
                "psfFluxErr",
                "pixelFlags_bad",
                "pixelFlags_cr",
                "pixelFlags_edge",
                "pixelFlags_saturated",
                "pixelFlags_suspect",
            ],
        )

        # Rename diaObjectId -> objectId for compatibility with _format_as_kowalski
        fs_df.rename(columns={"diaObjectId": "objectId"}, inplace=True)

        # Join with Visit to get expMidptMJD
        fs_df = fs_df.merge(self._visit_df, on="visit", how="inner")

        # Compute pixelFlags as sum of boolean columns
        flag_cols = [
            "pixelFlags_bad",
            "pixelFlags_cr",
            "pixelFlags_edge",
            "pixelFlags_saturated",
            "pixelFlags_suspect",
        ]
        fs_df["pixelFlags"] = sum(fs_df[c].astype(int) for c in flag_cols)

        # Sort by objectId for searchsorted lookups
        fs_df.sort_values("objectId", inplace=True)
        fs_df.reset_index(drop=True, inplace=True)

        # Keep only the columns we need
        self._fs_df = fs_df[
            [
                "objectId",
                "band",
                "psfFlux",
                "psfFluxErr",
                "expMidptMJD",
                "pixelFlags",
            ]
        ]
        self._fs_object_ids = self._fs_df["objectId"].values

    def get_objects_by_cone(self, ra, dec, radius_arcsec, limit=10000):
        """
        Find DIA objects within a cone search region (no isolation filter).

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        limit : int, optional
            Maximum number of objects to return (default 10000).

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        from astropy.coordinates import angular_separation

        self._load_objects()

        radius_rad = np.deg2rad(radius_arcsec / 3600.0)
        center_ra_rad = np.deg2rad(ra)
        center_dec_rad = np.deg2rad(dec)

        sep = angular_separation(
            center_ra_rad, center_dec_rad, self._obj_ra_rad, self._obj_dec_rad
        )

        cone_mask = sep <= radius_rad
        obj_ids = self._obj_ids[cone_mask]
        obj_ra = self._obj_ra_deg[cone_mask]
        obj_dec = self._obj_dec_deg[cone_mask]

        if limit is not None and len(obj_ids) > limit:
            obj_ids = obj_ids[:limit]
            obj_ra = obj_ra[:limit]
            obj_dec = obj_dec[:limit]

        objects = {}
        for i in range(len(obj_ids)):
            objects[int(obj_ids[i])] = {
                "coord_ra": float(obj_ra[i]),
                "coord_dec": float(obj_dec[i]),
            }

        return objects

    def get_lightcurves(self, objectids, bands=None, batch_size=1000):
        """
        Retrieve forced photometry lightcurves for a list of DIA object IDs.

        Parameters
        ----------
        objectids : list of int
            DIA object IDs to query.
        bands : list of str, optional
            Band names to filter (e.g., ['g', 'r']). If None, all bands.
        batch_size : int, optional
            Not used for local backend (kept for API compatibility).

        Returns
        -------
        list of dict
            Kowalski-format lightcurve dicts.
        """
        if len(objectids) == 0:
            return []

        self._load_forced_sources()

        # Use searchsorted for efficient lookup
        oid_set = set(int(oid) for oid in objectids)
        left_indices = np.searchsorted(self._fs_object_ids, list(oid_set), side="left")
        right_indices = np.searchsorted(
            self._fs_object_ids, list(oid_set), side="right"
        )

        # Gather row indices
        idx_list = []
        for left, right in zip(left_indices, right_indices):
            if left < right:
                idx_list.append(np.arange(left, right))

        if len(idx_list) == 0:
            return []

        all_idx = np.concatenate(idx_list)
        subset = self._fs_df.iloc[all_idx]

        # Filter by bands if specified
        if bands is not None and len(bands) > 0:
            subset = subset[subset["band"].isin(bands)]

        # Filter positive flux
        subset = subset[(subset["psfFlux"] > 0) & (subset["psfFluxErr"] > 0)]

        if len(subset) == 0:
            return []

        # Convert to list of dicts for _format_as_kowalski
        rows = subset.to_dict("records")
        return _format_as_kowalski(rows, band_map=self.band_map)

    def get_visit_positions(self):
        """Return visit DataFrame enriched with representative field RA.

        Determines the approximate sky position of each visit by sampling
        one DIA object observed in that visit and looking up its RA from
        the DiaObject table.

        Returns
        -------
        pandas.DataFrame
            Copy of the visit table with an added ``fieldRA`` column.
        """
        import pandas as pd

        self._load_objects()
        self._load_forced_sources()

        visit_sample = self._fs_df[["expMidptMJD", "objectId"]].drop_duplicates(
            "expMidptMJD"
        )

        obj_ra_lookup = pd.Series(self._obj_ra_deg, index=self._obj_ids)
        visit_sample = visit_sample.copy()
        visit_sample["fieldRA"] = visit_sample["objectId"].map(obj_ra_lookup)

        result = self._visit_df.merge(
            visit_sample[["expMidptMJD", "fieldRA"]],
            on="expMidptMJD",
            how="left",
        )

        return result

    def get_lightcurves_for_cone(self, ra, dec, radius_arcsec, bands=None, limit=10000):
        """
        Convenience method: cone search + lightcurve retrieval in one call.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        bands : list of str, optional
            Band names to filter.
        limit : int, optional
            Maximum number of objects from cone search.

        Returns
        -------
        objects : dict
            Object metadata from cone search.
        lightcurves : list of dict
            Kowalski-format lightcurve dicts.
        """
        objects = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(objects) == 0:
            return objects, []

        objectids = list(objects.keys())
        lightcurves = self.get_lightcurves(objectids, bands=bands)
        return objects, lightcurves


class RubinLocalDP2Client:
    """
    Client for Rubin EDP2/DP2 DIA forced photometry from a local parquet export.

    Unlike DP1, which is distributed here as one parquet file per table,
    the EDP2 export is a single pre-joined table produced by a TAP query of
    the form::

        SELECT f.*, d.ra, d.dec
        FROM dp2.ForcedSourceOnDiaObject AS f,
             dp2.DiaObject AS d, dp2.DiaSource AS s
        WHERE f.diaObjectId = d.diaObjectId AND d.diaObjectId = s.diaObjectId
          AND ... (variability / quality cuts on s)

    Two consequences are handled here:

    1. The three-way join has no ``DISTINCT``, so every forced-source row is
       repeated once per DiaSource row that passed the cuts (~9x for the
       reference export). Rows are de-duplicated on
       ``(diaObjectId, visit, detector)``.
    2. ``ForcedSourceOnDiaObject`` carries no timestamp, only ``visit``. A
       separate ``dp2.Visit`` export is required to map visit -> MJD; fetch
       one with ``tools/fetch_rubin_visits.py``.

    Parameters
    ----------
    data_path : str
        Path to the pre-joined parquet file, or to a directory containing
        it (the first ``*.parquet`` / ``*.parquet.gzip`` file that is not a
        visit table is used).
    visit_path : str, optional
        Path to the visit table (parquet with ``visit`` and ``expMidptMJD``
        columns). If None, looks for ``Visit.parquet``/``Visit.parquet.gzip``
        next to ``data_path``, then in the ``RUBIN_VISIT_PATH`` environment
        variable.
    band_map : dict, optional
        Mapping from band name to integer filter ID.
    flux_column : str, optional
        Which forced photometry to convert to magnitudes: ``'psfFlux'``
        (direct-image forced PSF flux, the default) or ``'psfDiffFlux'``
        (difference-image flux). Difference fluxes are frequently negative
        and those epochs are dropped, so ``'psfFlux'`` is normally what you
        want for period searching.
    """

    #: Pixel flags summed into ``catflags``; matches the DP1 clients.
    PIXEL_FLAG_COLUMNS = [
        "pixelFlags_bad",
        "pixelFlags_cr",
        "pixelFlags_edge",
        "pixelFlags_saturated",
        "pixelFlags_suspect",
    ]

    #: Per-measurement flags also folded into ``catflags`` (DP2 only).
    EXTRA_FLAG_COLUMNS = ["invalidPsfFlag"]

    FLUX_ERR_COLUMNS = {
        "psfFlux": ("psfFluxErr", "psfFlux_flag"),
        "psfDiffFlux": ("psfDiffFluxErr", "psfDiffFlux_flag"),
    }

    def __init__(
        self,
        data_path,
        visit_path=None,
        band_map=None,
        flux_column="psfFlux",
        dedupe=True,
    ):
        import pandas as pd

        if flux_column not in self.FLUX_ERR_COLUMNS:
            raise ValueError(
                f"flux_column must be one of {sorted(self.FLUX_ERR_COLUMNS)}, "
                f"got {flux_column!r}"
            )

        self.data_path = self._resolve_data_file(data_path)
        self.band_map = band_map or DEFAULT_BAND_MAP
        self.flux_column = flux_column
        self.flux_err_column, self.flux_flag_column = self.FLUX_ERR_COLUMNS[flux_column]
        self.dedupe = dedupe

        self.visit_path = self._resolve_visit_file(self.data_path, visit_path)
        self._visit_df = pd.read_parquet(
            self.visit_path, columns=["visit", "expMidptMJD"]
        )
        self._visit_df = self._visit_df.drop_duplicates("visit")

        # Lazy-loaded caches
        self._obj_ids = None
        self._obj_ra_rad = None
        self._obj_dec_rad = None
        self._obj_ra_deg = None
        self._obj_dec_deg = None

        self._fs_df = None
        self._fs_object_ids = None

    @staticmethod
    def _resolve_data_file(data_path):
        """Accept either the parquet file itself or its parent directory."""
        if os.path.isfile(data_path):
            return data_path

        if os.path.isdir(data_path):
            candidates = sorted(
                p
                for pattern in ("*.parquet", "*.parquet.gzip")
                for p in glob.glob(os.path.join(data_path, pattern))
                if not os.path.basename(p).lower().startswith("visit")
            )
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(
                    f"{data_path} contains multiple parquet files "
                    f"({[os.path.basename(c) for c in candidates]}); "
                    f"pass the DP2 forced-source file explicitly."
                )

        raise FileNotFoundError(
            f"No DP2 forced-source parquet file found at {data_path}."
        )

    @staticmethod
    def _resolve_visit_file(data_file, visit_path):
        """Locate the visit table needed to convert visit IDs to MJD."""
        if visit_path is not None:
            if not os.path.isfile(visit_path):
                raise FileNotFoundError(f"Visit table not found: {visit_path}")
            return visit_path

        data_dir = os.path.dirname(os.path.abspath(data_file))
        for fname in ("Visit.parquet", "Visit.parquet.gzip"):
            candidate = os.path.join(data_dir, fname)
            if os.path.isfile(candidate):
                return candidate

        env_path = os.environ.get("RUBIN_VISIT_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path

        raise FileNotFoundError(
            "The DP2 forced-source table has no timestamp column, so a visit "
            "table is required to map visit -> expMidptMJD. Looked for "
            f"Visit.parquet next to {data_file} and in $RUBIN_VISIT_PATH. "
            "Create one with: python tools/fetch_rubin_visits.py "
            "--release dp2 --output <dir>/Visit.parquet"
        )

    def _load_objects(self):
        """Lazy-load per-object coordinates from the pre-joined table."""
        if self._obj_ids is not None:
            return

        import pandas as pd

        df = pd.read_parquet(self.data_path, columns=["diaObjectId", "ra", "dec"])
        df = df.drop_duplicates("diaObjectId")
        df = df.sort_values("diaObjectId")

        self._obj_ids = df["diaObjectId"].values
        self._obj_ra_deg = df["ra"].values.astype(np.float64)
        self._obj_dec_deg = df["dec"].values.astype(np.float64)

        self._obj_ra_rad = np.deg2rad(self._obj_ra_deg)
        self._obj_dec_rad = np.deg2rad(self._obj_dec_deg)

    def _load_forced_sources(self):
        """Lazy-load the forced-source table, de-duplicated and time-stamped."""
        if self._fs_df is not None:
            return

        import pandas as pd

        columns = (
            [
                "diaObjectId",
                "visit",
                "detector",
                "band",
                self.flux_column,
                self.flux_err_column,
                self.flux_flag_column,
            ]
            + self.PIXEL_FLAG_COLUMNS
            + self.EXTRA_FLAG_COLUMNS
        )
        fs_df = pd.read_parquet(self.data_path, columns=columns)

        # The source query joins ForcedSourceOnDiaObject against DiaSource
        # without DISTINCT, duplicating each epoch once per matching source.
        if self.dedupe:
            fs_df = fs_df.drop_duplicates(subset=["diaObjectId", "visit", "detector"])

        # Rename diaObjectId -> objectId for compatibility with the DP1 clients
        fs_df = fs_df.rename(columns={"diaObjectId": "objectId"})

        # Join with the visit table to get expMidptMJD
        fs_df = fs_df.merge(self._visit_df, on="visit", how="inner")

        flag_cols = (
            self.PIXEL_FLAG_COLUMNS + self.EXTRA_FLAG_COLUMNS + [self.flux_flag_column]
        )
        fs_df["pixelFlags"] = sum(fs_df[c].astype(np.int64) for c in flag_cols)

        # Sort by objectId for searchsorted lookups
        fs_df = fs_df.sort_values("objectId").reset_index(drop=True)

        self._fs_df = fs_df[
            [
                "objectId",
                "band",
                self.flux_column,
                self.flux_err_column,
                "expMidptMJD",
                "pixelFlags",
            ]
        ]
        self._fs_object_ids = self._fs_df["objectId"].values

    def get_all_objects(self):
        """
        Return every object in the export.

        The DP2 export is already a variability-candidate selection, so
        iterating all of it is a common entry point.

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        self._load_objects()
        return {
            int(self._obj_ids[i]): {
                "coord_ra": float(self._obj_ra_deg[i]),
                "coord_dec": float(self._obj_dec_deg[i]),
            }
            for i in range(len(self._obj_ids))
        }

    def get_objects_by_cone(self, ra, dec, radius_arcsec, limit=10000):
        """
        Find DIA objects within a cone search region.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        limit : int, optional
            Maximum number of objects to return (default 10000).

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        from astropy.coordinates import angular_separation

        self._load_objects()

        radius_rad = np.deg2rad(radius_arcsec / 3600.0)
        sep = angular_separation(
            np.deg2rad(ra),
            np.deg2rad(dec),
            self._obj_ra_rad,
            self._obj_dec_rad,
        )

        cone_mask = sep <= radius_rad
        obj_ids = self._obj_ids[cone_mask]
        obj_ra = self._obj_ra_deg[cone_mask]
        obj_dec = self._obj_dec_deg[cone_mask]

        if limit is not None and len(obj_ids) > limit:
            obj_ids = obj_ids[:limit]
            obj_ra = obj_ra[:limit]
            obj_dec = obj_dec[:limit]

        objects = {}
        for i in range(len(obj_ids)):
            objects[int(obj_ids[i])] = {
                "coord_ra": float(obj_ra[i]),
                "coord_dec": float(obj_dec[i]),
            }

        return objects

    def get_lightcurves(self, objectids, bands=None, batch_size=1000):
        """
        Retrieve forced photometry lightcurves for a list of DIA object IDs.

        Parameters
        ----------
        objectids : list of int
            DIA object IDs to query.
        bands : list of str, optional
            Band names to filter (e.g., ['g', 'r']). If None, all bands.
        batch_size : int, optional
            Not used for local backend (kept for API compatibility).

        Returns
        -------
        list of dict
            Kowalski-format lightcurve dicts, one per (objectId, band).
        """
        if len(objectids) == 0:
            return []

        self._load_forced_sources()

        oid_list = sorted({int(oid) for oid in objectids})
        left_indices = np.searchsorted(self._fs_object_ids, oid_list, side="left")
        right_indices = np.searchsorted(self._fs_object_ids, oid_list, side="right")

        idx_list = [
            np.arange(left, right)
            for left, right in zip(left_indices, right_indices)
            if left < right
        ]
        if len(idx_list) == 0:
            return []

        subset = self._fs_df.iloc[np.concatenate(idx_list)]

        if bands is not None and len(bands) > 0:
            subset = subset[subset["band"].isin(bands)]

        if len(subset) == 0:
            return []

        return _format_frame_as_kowalski(
            subset,
            band_map=self.band_map,
            flux_col=self.flux_column,
            flux_err_col=self.flux_err_column,
        )

    def get_visit_positions(self):
        """
        Return the visit table used to time-stamp the forced sources.

        Returns
        -------
        pandas.DataFrame
            The visit table with ``visit`` and ``expMidptMJD`` columns, plus
            whatever else the export carried (e.g. ``ra``, ``dec``, ``band``).
        """
        import pandas as pd

        return pd.read_parquet(self.visit_path)

    def get_lightcurves_for_cone(self, ra, dec, radius_arcsec, bands=None, limit=10000):
        """
        Convenience method: cone search + lightcurve retrieval in one call.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        bands : list of str, optional
            Band names to filter.
        limit : int, optional
            Maximum number of objects from cone search.

        Returns
        -------
        objects : dict
            Object metadata from cone search.
        lightcurves : list of dict
            Kowalski-format lightcurve dicts.
        """
        objects = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(objects) == 0:
            return objects, []

        objectids = list(objects.keys())
        lightcurves = self.get_lightcurves(objectids, bands=bands)
        return objects, lightcurves


def make_rubin_client(config=None, use_dia=False, release=None):
    """
    Factory function to create the appropriate Rubin client.

    Returns a RubinLocalClient (or RubinLocalDiaClient if use_dia=True,
    or RubinLocalDP2Client for the DP2 release) if a data_path is
    configured (via config dict or RUBIN_DATA_PATH environment variable),
    otherwise returns a RubinTAPClient.

    Parameters
    ----------
    config : dict, optional
        Rubin config section (e.g., from config['rubin']).
    use_dia : bool, optional
        If True, use DiaObject + ForcedSourceOnDiaObject tables instead
        of Object + ForcedSource. Default False. Ignored for DP2, whose
        local export is DIA forced photometry by construction.
    release : str, optional
        Data release to read: ``'dp1'`` (default) or ``'dp2'``. If None,
        falls back to ``config['release']`` then the ``RUBIN_RELEASE``
        environment variable. DP2 reads ``config['dp2_data_path']`` (or
        ``RUBIN_DP2_DATA_PATH``) if set, so one config can hold both
        releases; otherwise it falls back to ``data_path``.

    Returns
    -------
    RubinTAPClient, RubinLocalClient, RubinLocalDiaClient, or
    RubinLocalDP2Client
    """
    if config is None:
        config = {}

    if release is None:
        release = config.get("release") or os.environ.get("RUBIN_RELEASE") or "dp1"
    release = str(release).lower()

    if release not in ("dp1", "dp2"):
        raise ValueError(f"Unknown Rubin release {release!r}; expected 'dp1' or 'dp2'.")

    data_path = config.get("data_path") or os.environ.get("RUBIN_DATA_PATH")
    if release == "dp2":
        # dp2_data_path lets a single config hold both releases, since the
        # DP1 and DP2 exports have entirely different layouts.
        data_path = (
            config.get("dp2_data_path")
            or os.environ.get("RUBIN_DP2_DATA_PATH")
            or data_path
        )

    if data_path:
        band_map = config.get("band_map", DEFAULT_BAND_MAP)
        if release == "dp2":
            return RubinLocalDP2Client(
                data_path=data_path,
                visit_path=config.get("visit_path")
                or os.environ.get("RUBIN_VISIT_PATH"),
                band_map=band_map,
                flux_column=config.get("flux_column", "psfFlux"),
            )
        if use_dia:
            return RubinLocalDiaClient(data_path=data_path, band_map=band_map)
        return RubinLocalClient(data_path=data_path, band_map=band_map)
    else:
        if release == "dp2":
            raise ValueError(
                "The DP2 backend requires a local export; set rubin.dp2_data_path "
                "(or RUBIN_DP2_DATA_PATH) to the DP2 forced-source parquet file."
            )
        return RubinTAPClient(config=config)
