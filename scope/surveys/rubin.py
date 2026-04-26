#!/usr/bin/env python
"""
Rubin LSST TAP client for scope-ml.

Provides access to Rubin Data Preview (DP1) data via the TAP API,
converting forced photometry lightcurves into the same format consumed
by the scope-ml feature generation pipeline.
"""

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
        self._visit_df = pd.read_parquet(
            visit_path, columns=["visit", "expMidptMJD"]
        )

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

        fs_path = os.path.join(
            self.data_path, "ForcedSourceOnDiaObject.parquet.gzip"
        )
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
        left_indices = np.searchsorted(
            self._fs_object_ids, list(oid_set), side="left"
        )
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

    def get_lightcurves_for_cone(
        self, ra, dec, radius_arcsec, bands=None, limit=10000
    ):
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


def make_rubin_client(config=None, use_dia=False):
    """
    Factory function to create the appropriate Rubin client.

    Returns a RubinLocalClient (or RubinLocalDiaClient if use_dia=True)
    if a data_path is configured (via config dict or RUBIN_DATA_PATH
    environment variable), otherwise returns a RubinTAPClient.

    Parameters
    ----------
    config : dict, optional
        Rubin config section (e.g., from config['rubin']).
    use_dia : bool, optional
        If True, use DiaObject + ForcedSourceOnDiaObject tables instead
        of Object + ForcedSource. Default False.

    Returns
    -------
    RubinTAPClient, RubinLocalClient, or RubinLocalDiaClient
    """
    if config is None:
        config = {}

    data_path = config.get("data_path") or os.environ.get("RUBIN_DATA_PATH")

    if data_path:
        band_map = config.get("band_map", DEFAULT_BAND_MAP)
        if use_dia:
            return RubinLocalDiaClient(data_path=data_path, band_map=band_map)
        return RubinLocalClient(data_path=data_path, band_map=band_map)
    else:
        return RubinTAPClient(config=config)
