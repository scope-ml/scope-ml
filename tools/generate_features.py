#!/usr/bin/env python
import scope
import argparse
import pathlib
import os

try:
    from tools.get_quad_ids import get_ids_loop, get_field_ids
except Exception:
    get_ids_loop = None
    get_field_ids = None

from scope.surveys.fritz import get_lightcurves_via_ids
from scope.utils import (
    TychoBVfromGaia,
    exclude_radius,
    removeHighCadence,
    write_parquet,
    sort_lightcurve,
    read_parquet,
    read_hdf,
    split_dict,
    parse_load_config,
)
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
from datetime import datetime
from tools.featureGeneration import (
    periodsearch,
    alertstats,
    external_xmatch,
    pdrs,
    drw,
)
import warnings
from cesium.featurize import time_series, featurize_single_ts
import json
from scipy.stats import circmean
import time
import pickle as _pickle

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

source_catalog = config['kowalski']['collections']['sources']
alerts_catalog = config['kowalski']['collections']['alerts']
gaia_catalog = config['kowalski']['collections']['gaia']
dmdt_ints = config['feature_generation']['dmdt_ints']
ext_catalog_info = config['feature_generation']['external_catalog_features']
cesium_feature_list = config['feature_generation']['cesium_features']
period_algorithms = config['feature_generation']['period_algorithms']
path_to_features = config['feature_generation']['path_to_features']
pdrs_params = config['feature_generation'].get('pdrs', {})
drw_params = config['feature_generation'].get('drw', {})

if path_to_features is not None:
    BASE_DIR = pathlib.Path(path_to_features)

kowalski_instances = None
try:
    from penquins import Kowalski

    # use tokens specified as env vars (if exist)
    kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
    gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
    melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

    if kowalski_token_env is not None:
        config["kowalski"]["hosts"]["kowalski"]["token"] = kowalski_token_env
    if gloria_token_env is not None:
        config["kowalski"]["hosts"]["gloria"]["token"] = gloria_token_env
    if melman_token_env is not None:
        config["kowalski"]["hosts"]["melman"]["token"] = melman_token_env

    timeout = config['kowalski']['timeout']

    hosts = [
        x
        for x in config['kowalski']['hosts']
        if config['kowalski']['hosts'][x]['token'] is not None
    ]
    instances = {
        host: {
            'protocol': config['kowalski']['protocol'],
            'port': config['kowalski']['port'],
            'host': f'{host}.caltech.edu',
            'token': config['kowalski']['hosts'][host]['token'],
        }
        for host in hosts
    }

    kowalski_instances = Kowalski(timeout=timeout, instances=instances)
except Exception:
    pass


def drop_close_bright_stars(
    id_dct: dict,
    catalog: str = gaia_catalog,
    query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    doSpecificIDs: bool = False,
    limit: int = 10000,
    Ncore: int = 8,
    save: bool = False,
    save_directory: str = 'generated_features',
    save_filename: str = 'specific_ids_dropCloseSources.json',
):
    """
    Use Gaia to identify and drop sources that are too close to bright stars

    :param id_dct: one quadrant's worth of id-coordinate pairs (dict)
    :param catalog: name of catalog to use [currently only supports Gaia catalogs] (str)
    :param query_radius_arcsec: size of cone search radius to search for bright stars.
        Default is 300 corresponding with approximate maximum from A. Drake's exclusion radius (float)
    :param xmatch_radius_arcsec: size of cone within which to match a queried source with an input source.
        If any sources from the query fall within this cone, the closest one will be matched to the input source and dropped from further bright-star considerations (float)
    :param doSpecificIDs: if set, query for specific ZTF IDs instead of a single quadrant (bool)
    :param limit: if doSpecificIDs is set, max number of sources to be queries in one batch (int)
    :param Ncore: if doSpecificIDs is set, number of cores over which to parallelize queries (int)
    :param save: if set, save sources passing the close source analysis (bool)
    :param save_directory: directory within BASE_DIR to save sources (str)
    :param save_filename: filename to use when saving sources (str)

    :return id_dct_keep: dictionary containing subset of input sources far enough away from bright stars
    """

    ids = [x for x in id_dct]
    coords = np.array(
        [x['radec_geojson']['coordinates'] for x in id_dct.values()]
    ).transpose()

    sources_ra = coords[0] + 180.0
    sources_dec = coords[1]
    SC = SkyCoord(sources_ra, sources_dec, unit=[u.deg, u.deg])

    ctr_ra = circmean(sources_ra * np.pi / 180.0) * 180.0 / np.pi
    ctr_dec = np.mean(sources_dec)
    ctr_SC = SkyCoord(ctr_ra, ctr_dec, unit=[u.deg, u.deg])

    max_cone_radius = (
        np.max(SC.separation(ctr_SC)).to(u.arcsec) + query_radius_arcsec * u.arcsec
    ).value

    id_dct_keep = id_dct.copy()

    gaia_results_dct = {}

    if not doSpecificIDs:
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "radec": {'quad': [ctr_ra, ctr_dec]},
                    "cone_search_radius": max_cone_radius,
                    "cone_search_unit": 'arcsec',
                },
                "catalogs": {
                    catalog: {
                        # Select sources brighter than G magnitude 13:
                        # -Conversion to Tycho mags only good for G < 13
                        # -Need for exclusion radius only for stars with B <~ 13
                        # -For most stars, if G > 13, B > 13
                        "filter": {"phot_g_mean_mag": {"$lt": 13.0}},
                        "projection": {
                            "phot_g_mean_mag": 1,
                            "bp_rp": 1,
                            "coordinates.radec_geojson.coordinates": 1,
                        },
                    }
                },
                "filter": {},
            },
        }

        responses = kowalski_instances.query(query)
        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    gaia_results = response.get("data")
                    gaia_results_dct.update(gaia_results[catalog])

            print('Identifying sources too close to bright stars...')

            # Loop over each id to compare with query results
            query_result = np.array(gaia_results_dct['quad'])

    else:
        n_sources = len(id_dct)
        if n_sources % limit != 0:
            n_iterations = n_sources // limit + 1
        else:
            n_iterations = n_sources // limit

        print(f'Querying {catalog} catalog in batches...')

        for i in range(0, n_iterations):
            print(f"Iteration {i+1} of {n_iterations}...")
            id_slice = [x for x in id_dct.keys()][
                i * limit : min(n_sources, (i + 1) * limit)
            ]

            radec_geojson = np.array(
                [id_dct[x]['radec_geojson']['coordinates'] for x in id_slice]
            ).transpose()

            # Need to add 180 -> no negative RAs
            radec_geojson[0, :] += 180.0
            radec_dict = dict(zip(id_slice, radec_geojson.transpose().tolist()))

            if Ncore > 1:
                # Split dictionary for parallel querying
                radec_split_list = [lst for lst in split_dict(radec_dict, Ncore)]
                queries = [
                    {
                        "query_type": "cone_search",
                        "query": {
                            "object_coordinates": {
                                "radec": dct,
                                "cone_search_radius": query_radius_arcsec,
                                "cone_search_unit": 'arcsec',
                            },
                            "catalogs": {
                                catalog: {
                                    # Select sources brighter than G magnitude 13:
                                    # -Conversion to Tycho mags only good for G < 13
                                    # -Need for exclusion radius only for stars with B <~ 13
                                    # -For most stars, if G > 13, B > 13
                                    "filter": {"phot_g_mean_mag": {"$lt": 13.0}},
                                    "projection": {
                                        "phot_g_mean_mag": 1,
                                        "bp_rp": 1,
                                        "coordinates.radec_geojson.coordinates": 1,
                                    },
                                }
                            },
                            "filter": {},
                        },
                    }
                    for dct in radec_split_list
                ]

                responses = kowalski_instances.query(
                    queries=queries, use_batch_query=True, max_n_threads=Ncore
                )
                for name in responses.keys():
                    if len(responses[name]) > 0:
                        response_list = responses[name]
                        for response in response_list:
                            if response.get("status", "error") == "success":
                                gaia_results = response.get('data').get(catalog)
                                gaia_results_dct.update(gaia_results)

            else:
                # Get Gaia EDR3 ID, G mag, BP-RP, and coordinates
                query = {
                    "query_type": "cone_search",
                    "query": {
                        "object_coordinates": {
                            # "radec": dict(zip(id_slice, radec_geojson.transpose().tolist())),
                            "radec": radec_dict,
                            "cone_search_radius": query_radius_arcsec,
                            "cone_search_unit": 'arcsec',
                        },
                        "catalogs": {
                            catalog: {
                                # Select sources brighter than G magnitude 13:
                                # -Conversion to Tycho mags only good for G < 13
                                # -Need for exclusion radius only for stars with B <~ 13
                                # -For most stars, if G > 13, B > 13
                                "filter": {"phot_g_mean_mag": {"$lt": 13.0}},
                                "projection": {
                                    "phot_g_mean_mag": 1,
                                    "bp_rp": 1,
                                    "coordinates.radec_geojson.coordinates": 1,
                                },
                            }
                        },
                        "filter": {},
                    },
                }
                responses = kowalski_instances.query(query)
                for name in responses.keys():
                    if len(responses[name]) > 0:
                        response = responses[name]
                        if response.get("status", "error") == "success":
                            gaia_results = response.get('data').get(catalog)
                            gaia_results_dct.update(gaia_results)
        query_result = gaia_results_dct

    if len(query_result) > 0:
        if not doSpecificIDs:
            source_coords = np.array(
                [x['radec_geojson']['coordinates'] for x in id_dct.values()]
            )
            source_coords[:, 0] += 180.0
            Source_Coords = SkyCoord(source_coords, unit=[u.deg, u.deg])
            lon1 = Source_Coords.spherical.lon.to(u.rad).value
            lat1 = Source_Coords.spherical.lat.to(u.rad).value

            query_coords = np.array(
                [x['coordinates']['radec_geojson']['coordinates'] for x in query_result]
            )
            query_coords[:, 0] += 180.0
            Query_Coords = SkyCoord(query_coords, unit=[u.deg, u.deg])
            lon2 = Query_Coords.spherical.lon.to(u.rad).value
            lat2 = Query_Coords.spherical.lat.to(u.rad).value

            for i, id in enumerate(ids):
                # Reset bright source dictionary for each iteration
                single_result = query_result.copy()

                Coords = np.copy(Query_Coords)

                val = id_dct[id]
                ra_geojson, dec_geojson = val['radec_geojson']['coordinates']

                # Compute separations in radians and convert, using 1 rad = 206265 arcsec
                # ~10x faster than SkyCoord.separation() if lon/lat is calculated out of loop
                all_separations = (
                    angular_separation(lon1[i], lat1[i], lon2, lat2) * 206265
                )
                within_range = all_separations < query_radius_arcsec

                if np.sum(within_range) > 0:
                    all_separations = all_separations[within_range]
                    Coords = Coords[within_range]
                    single_result = query_result[within_range].tolist()

                    # Identify closest source to input
                    drop_source = np.argmin(all_separations)

                    # If closest source is within specified radius, treat it as the input source and drop it from further consideration
                    xmatch_source = {}
                    if all_separations[drop_source] < xmatch_radius_arcsec:
                        xmatch_source = single_result.pop(drop_source)
                        Coords = np.delete(Coords, drop_source)

                    update_id_dict(
                        id_dct_keep,
                        id,
                        xmatch_source,
                        ra_geojson,
                        dec_geojson,
                        single_result,
                        Coords,
                    )
        else:
            # Loop over each id to compare with query results
            count = 0
            for id in ids:
                count += 1
                if count % limit == 0:
                    print(f"{count} done")
                if count == len(ids):
                    print(f"{count} done")

                val = id_dct[id]

                ra_geojson, dec_geojson = val['radec_geojson']['coordinates']

                single_result = gaia_results_dct[str(id)]
                if len(single_result) > 0:
                    coords = np.array(
                        [
                            x['coordinates']['radec_geojson']['coordinates']
                            for x in single_result
                        ]
                    )
                    coords[:, 0] += 180.0

                    # SkyCoord object for query results
                    Coords = SkyCoord(coords, unit=['deg', 'deg'])
                    # SkyCoord object for input source
                    coord = SkyCoord(
                        ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg']
                    )

                    all_separations = Coords.separation(coord)
                    # Identify closest source to input
                    drop_source = np.argmin(all_separations)

                    # If closest source is within specified radius, treat it as the input source and drop it from further consideration
                    xmatch_source = {}
                    if all_separations[drop_source] < xmatch_radius_arcsec * u.arcsec:
                        xmatch_source = single_result.pop(drop_source)
                        Coords = np.delete(Coords, drop_source)

                    update_id_dict(
                        id_dct_keep,
                        id,
                        xmatch_source,
                        ra_geojson,
                        dec_geojson,
                        single_result,
                        Coords,
                    )
    else:
        id_dct_keep = id_dct

    if save:
        os.makedirs(BASE_DIR / save_directory, exist_ok=True)
        with open(str(BASE_DIR / save_directory / save_filename), 'w') as f:
            json.dump(id_dct_keep, f)

    print(f"Dropped {len(id_dct) - len(id_dct_keep)} sources.")
    return id_dct_keep


def update_id_dict(
    id_dct_keep, id, xmatch_source, ra_geojson, dec_geojson, single_result, Coords
):
    """
    Helper function called by drop_close_bright_stars
    """
    # If possible, use all-Gaia coordinates for next step
    if len(xmatch_source) > 0:
        xmatch_ra, xmatch_dec = xmatch_source['coordinates']['radec_geojson'][
            'coordinates'
        ]
        xmatch_ra += 180.0
        xmatch_coord = SkyCoord(xmatch_ra, xmatch_dec, unit=['deg', 'deg'])
    else:
        xmatch_coord = SkyCoord(ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg'])

    # Use mapping from Gaia -> Tycho to set exclusion radius for each source
    for idx, source in enumerate(single_result):
        try:
            B, V = TychoBVfromGaia(source['phot_g_mean_mag'], source['bp_rp'])
            excl_radius = exclude_radius(B, V)
        except KeyError:
            # Not all Gaia sources have BP-RP
            excl_radius = 0.0
        if excl_radius > 0.0:
            sep = xmatch_coord.separation(Coords[idx])
            if excl_radius * u.arcsec > sep.to(u.arcsec):
                # If there is a bright star that's too close, drop from returned dict
                id_dct_keep.pop(id)
                break


def _save_period_checkpoint(
    checkpoint_dir,
    batch_idx,
    all_periods,
    all_significances,
    all_pdots,
    all_top_n_periods,
    all_top_n_sigs,
):
    """Save period-finding progress after a completed batch."""
    ckpt_path = os.path.join(checkpoint_dir, 'period_checkpoint.pkl')
    tmp_path = ckpt_path + '.tmp'
    state = {
        'completed_batch': batch_idx,
        'all_periods': {
            k: np.array(v) if isinstance(v, np.ndarray) else v
            for k, v in all_periods.items()
        },
        'all_significances': {
            k: np.array(v) if isinstance(v, np.ndarray) else v
            for k, v in all_significances.items()
        },
        'all_pdots': {
            k: np.array(v) if isinstance(v, np.ndarray) else v
            for k, v in all_pdots.items()
        },
        'all_top_n_periods': {k: list(v) for k, v in all_top_n_periods.items()},
        'all_top_n_sigs': {k: list(v) for k, v in all_top_n_sigs.items()},
    }
    with open(tmp_path, 'wb') as f:
        _pickle.dump(state, f, protocol=4)
    os.replace(tmp_path, ckpt_path)
    print(f'  [checkpoint] Saved after batch {batch_idx + 1}', flush=True)


def _load_period_checkpoint(checkpoint_dir):
    """Load period-finding checkpoint if it exists. Returns dict or None."""
    ckpt_path = os.path.join(checkpoint_dir, 'period_checkpoint.pkl')
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path, 'rb') as f:
            state = _pickle.load(f)
        print(
            f'  [checkpoint] Resuming from batch {state["completed_batch"] + 2}',
            flush=True,
        )
        return state
    except Exception as e:
        print(f'  [checkpoint] Failed to load checkpoint: {e}', flush=True)
        return None


def generate_features(
    source_catalog: str = source_catalog,
    alerts_catalog: str = alerts_catalog,
    gaia_catalog: str = gaia_catalog,
    bright_star_query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    limit: int = 10000,
    period_algorithms: dict = period_algorithms,
    period_batch_size: int = 1000,
    doCPU: bool = False,
    doGPU: bool = False,
    samples_per_peak: int = 10,
    doScaleMinPeriod: bool = False,
    doRemoveTerrestrial: bool = False,
    Ncore: int = 8,
    field: int = 296,
    ccd: int = 1,
    quad: int = 1,
    min_n_lc_points: int = 50,
    min_cadence_minutes: float = 30.0,
    dirname: str = 'generated_features',
    filename: str = 'gen_features',
    doCesium: bool = False,
    doFlareDetection: bool = False,
    doDRW: bool = False,
    doNotSave: bool = False,
    stop_early: bool = False,
    doQuadrantFile: bool = False,
    quadrant_file: str = 'slurm.dat',
    quadrant_index: int = 0,
    doSpecificIDs: bool = False,
    skipCloseSources: bool = False,
    top_n_periods: int = 10,
    max_freq: float = 48.0,
    fg_dataset: str = None,
    max_timestamp_hjd: float = None,
    local_lightcurves=None,
    local_alert_stats=None,
    local_xmatch=None,
    checkpoint_dir=None,
    qmin=0.01,
    qmax=0.5,
):
    """
    Generate features for ZTF light curves

    :param source_catalog*: name of Kowalski catalog containing ZTF sources (str)
    :param alerts_catalog*: name of Kowalski catalog containing ZTF alerts (str)
    :param gaia_catalog*: name of Kowalski catalog containing Gaia data (str)
    :param bright_star_query_radius_arcsec: maximum angular distance from ZTF sources to query nearby bright stars in Gaia (float)
    :param xmatch_radius_arcsec: maximum angular distance from ZTF sources to match external catalog sources (float)
    :param limit: maximum number of sources to process in batch queries / statistics calculations (int)
    :param period_algorithms*: dictionary containing names of period algorithms to run. Normally specified in config - if specified here, should be a (list)
    :param period_batch_size: maximum number of sources to simultaneously perform period finding (int)
    :param doCPU: flag to run config-specified CPU period algorithms (bool)
    :param doGPU: flag to run config-specified GPU period algorithms (bool)
    :param samples_per_peak: number of samples per periodogram peak (int)
    :param doScaleMinPeriod: for period finding, scale min period based on min_cadence_minutes [otherwise, min P = 3 min] (bool)
    :param doRemoveTerrestrial: remove terrestrial frequencies from period-finding analysis (bool)
    :param Ncore: number of CPU cores to parallelize queries (int)
    :param field: ZTF field to run (int)
    :param ccd: ZTF ccd to run (int)
    :param quad: ZTF quadrant to run (int)
    :param min_n_lc_points: minimum number of points required to generate features for a light curve (int)
    :param min_cadence_minutes: minimum cadence between light curve points. Higher-cadence data are dropped except for the first point in the sequence (float)
    :param dirname: name of generated feature directory (str)
    :param filename: prefix of each feature filename (str)
    :param doCesium: flag to compute config-specified cesium features in addition to default list (bool)
    :param doFlareDetection: flag to compute PDRS flare/high-activity segmentation features (bool)
    :param doDRW: flag to fit damped-random-walk models and compute DRW anomaly features (bool)
    :param doNotSave: flag to avoid saving generated features (bool)
    :param stop_early: flag to stop feature generation before entire quadrant is run. Pair with --limit to run small-scale tests (bool)
    :param doQuadrantFile: flag to use a generated file containing [jobID, field, ccd, quad] columns instead of specifying --field, --ccd and --quad (bool)
    :param quadrant_file: name of quadrant file in the generated_features/slurm directory or equivalent (str)
    :param quadrant_index: number of job in quadrant file to run (int)
    :param doSpecificIDs: flag to perform feature generation for ztf_id column in config-specified file (bool)
    :param skipCloseSources: flag to skip removal of sources too close to bright stars via Gaia (bool)
    :param top_n_periods: number of (E)LS, (E)CE periods to pass to (E)AOV if using (E)LS_(E)CE_(E)AOV algorithm (int)
    :param max_freq: maximum frequency [1 / days] to use for period finding. Overridden by --doScaleMinPeriod (float)
    :param fg_dataset*: path to parquet, hdf5 or csv file containing specific sources for feature generation (str)
    :param max_timestamp_hjd*: maximum timestamp of queried light curves, HJD (float)
    :param local_lightcurves: pre-fetched light curve list, same format as get_lightcurves_via_ids return (list or None)
    :param local_alert_stats: dict {source_id: {n_ztf_alerts, mean_ztf_alert_braai}} or None
    :param local_xmatch: dict {source_id: {catalog__field: value, ...}} or None

    :return feature_df: dataframe containing generated features

    * - specified in config.yaml
    """
    t0 = time.time()

    # Get code version and current date/time for metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    # Select period algorithms from config based on CPU or GPU specification
    if isinstance(period_algorithms, dict):
        if doCPU:
            period_algorithms = period_algorithms['CPU']
        elif doGPU:
            period_algorithms = period_algorithms['GPU']

    if not doSpecificIDs:
        # Code supporting parallelization across fields/ccds/quads
        slurmDir = os.path.join(str(BASE_DIR / dirname), 'slurm')
        if doQuadrantFile:
            names = ["job_number", "field", "ccd", "quadrant"]
            df_original = pd.read_csv(
                os.path.join(slurmDir, quadrant_file),
                header=None,
                delimiter=' ',
                names=names,
            )
            row = df_original.iloc[quadrant_index, :]
            field, ccd, quad = int(row["field"]), int(row["ccd"]), int(row["quadrant"])

        print(f'Running field {field}, CCD {ccd}, Quadrant {quad}...')
        print('Getting IDs...')
        _, lst = get_ids_loop(
            get_field_ids,
            catalog=source_catalog,
            kowalski_instances=kowalski_instances,
            limit=limit,
            field=field,
            ccd_range=ccd,
            quad_range=quad,
            minobs=min_n_lc_points,
            save=False,
            get_coords=True,
            stop_early=stop_early,
        )

        if not skipCloseSources:
            # Each index of lst corresponds to a different ccd/quad combo
            feature_gen_source_dict = drop_close_bright_stars(
                lst[0],
                catalog=gaia_catalog,
                query_radius_arcsec=bright_star_query_radius_arcsec,
                xmatch_radius_arcsec=xmatch_radius_arcsec,
                limit=limit,
                Ncore=Ncore,
            )
        else:
            feature_gen_source_dict = lst[0]
    else:
        if not skipCloseSources:
            # Read ztf_id column from csv, hdf5 or parquet file specified in config entry/kwargs
            if fg_dataset is None:
                fg_dataset = config['feature_generation']['dataset']

            fg_sources_path = str(BASE_DIR / fg_dataset)

            if fg_sources_path.endswith('.parquet'):
                fg_sources = read_parquet(fg_sources_path)
            elif fg_sources_path.endswith('.h5'):
                fg_sources = read_hdf(fg_sources_path)
            elif fg_sources_path.endswith('.csv'):
                fg_sources = pd.read_csv(fg_sources_path)
            else:
                raise ValueError(
                    "Sources must be stored in .parquet, .h5 or .csv format."
                )

            try:
                ztf_ids = fg_sources['ztf_id'].values.tolist()
                coordinates = fg_sources['coordinates'].values.tolist()

            except KeyError:
                raise KeyError(
                    'Columns "ztf_id" and "coordinates" must be included in source dataset.'
                )

            hasFritzNames = False
            try:
                fritz_names = fg_sources['fritz_name'].values.tolist()
                hasFritzNames = True
            except KeyError:
                warnings.warn('No obj_id column found in dataset.')

        else:
            if fg_dataset is not None:
                # Load source list directly from the provided dataset
                fg_sources_path = str(BASE_DIR / fg_dataset)

                if fg_sources_path.endswith('.parquet'):
                    fg_sources_df = read_parquet(fg_sources_path)
                elif fg_sources_path.endswith('.h5'):
                    fg_sources_df = read_hdf(fg_sources_path)
                elif fg_sources_path.endswith('.csv'):
                    fg_sources_df = pd.read_csv(fg_sources_path)
                else:
                    raise ValueError(
                        "Sources must be stored in .parquet, .h5 or .csv format."
                    )

                try:
                    ztf_ids = fg_sources_df['ztf_id'].values.tolist()
                    coordinates = fg_sources_df['coordinates'].values.tolist()
                except KeyError:
                    raise KeyError(
                        'Columns "ztf_id" and "coordinates" must be included in source dataset.'
                    )

                n_fg_sources = len(ztf_ids)
                if stop_early:
                    n_fg_sources = limit

                feature_gen_source_dict = {}
                for i in range(n_fg_sources):
                    coord = coordinates[i]
                    if isinstance(coord, dict):
                        feature_gen_source_dict[ztf_ids[i]] = {
                            'radec_geojson': {
                                'coordinates': coord['radec_geojson']['coordinates']
                            }
                        }
                    else:
                        feature_gen_source_dict[ztf_ids[i]] = {
                            'radec_geojson': {'coordinates': coord}
                        }

                print(
                    f'Loaded {len(feature_gen_source_dict)} sources from {fg_dataset} (skipping Gaia filtering).'
                )
            else:
                # Load pre-saved dataset if Gaia analysis already complete
                fg_sources_config = config['feature_generation']['ids_skipGaia']
                fg_sources_path = str(BASE_DIR / dirname / fg_sources_config)

                if fg_sources_path.endswith('.json'):
                    with open(fg_sources_path, 'r') as f:
                        fg_sources = json.load(f)
                else:
                    raise ValueError("Sources must be stored in .json format.")

                n_fg_sources = len(fg_sources)
                if stop_early:
                    n_fg_sources = limit

                feature_gen_source_dict = {
                    int(k): fg_sources[k] for k in list(fg_sources)[:n_fg_sources]
                }

        if not skipCloseSources:
            n_fg_sources = len(fg_sources)
            if stop_early:
                n_fg_sources = min(n_fg_sources, limit)
            # Create list with same structure as query results
            if hasFritzNames:
                dct_for_lst = {
                    ztf_ids[i]: {
                        'radec_geojson': {
                            'coordinates': coordinates[i]['radec_geojson'][
                                'coordinates'
                            ].tolist()
                        },
                        'fritz_name': fritz_names[i],
                    }
                    for i in range(n_fg_sources)
                }
            else:
                dct_for_lst = {
                    ztf_ids[i]: {
                        'radec_geojson': {
                            'coordinates': coordinates[i]['radec_geojson'][
                                'coordinates'
                            ].tolist()
                        }
                    }
                    for i in range(n_fg_sources)
                }

            lst = [dct_for_lst]
            print(f'Loaded ZTF IDs for {len(lst[0])} sources.')

            # Each index of lst corresponds to a different ccd/quad combo
            feature_gen_source_dict = drop_close_bright_stars(
                lst[0],
                catalog=gaia_catalog,
                query_radius_arcsec=bright_star_query_radius_arcsec,
                xmatch_radius_arcsec=xmatch_radius_arcsec,
                doSpecificIDs=True,
                limit=limit,
                Ncore=Ncore,
                save=not doNotSave,
                save_directory=dirname,
            )

    print('Getting lightcurves...', flush=True)
    feature_gen_ids = [x for x in feature_gen_source_dict.keys()]

    if local_lightcurves is not None:
        lcs = local_lightcurves
    elif kowalski_instances is not None:
        # For small source lists, shrink LC query limit until batching occurs
        lc_limit = limit
        if len(feature_gen_source_dict) < limit:
            lc_limit = int(np.ceil(len(feature_gen_source_dict) / Ncore))

        lcs = get_lightcurves_via_ids(
            kowalski_instances=kowalski_instances,
            ids=feature_gen_ids,
            catalog=source_catalog,
            limit_per_query=lc_limit,
            Ncore=Ncore,
            get_basic_data=True,
            max_timestamp_hjd=max_timestamp_hjd,
        )
    else:
        raise ConnectionError("No Kowalski connection and no local light curve cache.")

    # Remake feature_gen_source_dict if some light curves are missing
    lc_ids = [lc['_id'] for lc in lcs]
    if len(lc_ids) != len(feature_gen_ids):
        feature_gen_source_dict = {x: feature_gen_source_dict[x] for x in lc_ids}

    feature_dict = feature_gen_source_dict.copy()
    print('Analyzing lightcuves and computing basic features...', flush=True)
    # Start by dropping flagged points
    count = 0
    baseline = 0
    keep_id_list = []
    tme_collection = []
    tme_dict = {}
    for idx, lc in enumerate(lcs):
        count += 1
        if (idx + 1) % limit == 0:
            print(f"{count} done")
        if count == len(lcs):
            print(f"{count} done")

        _id = lc['_id']
        lc_unflagged = [x for x in lc['data'] if x['catflags'] == 0]
        flt = lc['filter']

        tme = [[x['hjd'], x['mag'], x['magerr']] for x in lc_unflagged]
        try:
            tme_arr = np.array(tme)
            t, m, e = tme_arr.transpose()

            # Ensure light curves are monotonically increasing in time
            t, m, e = sort_lightcurve(t, m, e)

            # Remove all but the first of each group of high-cadence points
            tt, mm, ee = removeHighCadence(t, m, e, cadence_minutes=min_cadence_minutes)

            # Discard sources lacking minimum number of points
            if len(tt) < min_n_lc_points:
                feature_dict.pop(_id)
            else:
                keep_id_list += [_id]
                if doCesium:
                    cesium_TS = time_series.TimeSeries(tt, mm, ee)

                # Determine largest time baseline over loop
                new_baseline = max(tt) - min(tt)
                if new_baseline > baseline:
                    baseline = new_baseline

                new_tme_arr = np.array([tt, mm, ee])
                tme_collection += [new_tme_arr]
                tme_dict[_id] = {}
                tme_dict[_id]['tme'] = new_tme_arr

                # Add basic info
                feature_dict[_id]['ra'] = (
                    feature_gen_source_dict[_id]['radec_geojson']['coordinates'][0]
                    + 180.0
                )
                feature_dict[_id]['dec'] = feature_gen_source_dict[_id][
                    'radec_geojson'
                ]['coordinates'][1]
                feature_dict[_id]['field'] = field
                feature_dict[_id]['ccd'] = ccd
                feature_dict[_id]['quad'] = quad
                feature_dict[_id]['filter'] = flt

                if doCesium:
                    cesium_features = featurize_single_ts(
                        cesium_TS, cesium_feature_list
                    )
                    cesium_features_dict = (
                        cesium_features.reset_index()
                        .drop('channel', axis=1)
                        .set_index('feature')
                        .to_dict()[0]
                    )
                    feature_dict[_id].update(cesium_features_dict)

        except ValueError:
            feature_dict.pop(_id)
            if _id in keep_id_list:
                keep_id_list.remove(_id)
                tme_dict.pop(_id)

    id_list_bs = list(tme_dict.keys())
    lightcurves_bs = [tme_dict[_id]['tme'] for _id in id_list_bs]
    basic_stats_arr = periodsearch.compute_basic_stats(lightcurves_bs)

    stat_keys = [
        'n',
        'median',
        'wmean',
        'chi2red',
        'roms',
        'wstd',
        'norm_peak_to_peak_amp',
        'norm_excess_var',
        'median_abs_dev',
        'iqr',
        'i60r',
        'i70r',
        'i80r',
        'i90r',
        'skew',
        'smallkurt',
        'inv_vonneumannratio',
        'welch_i',
        'stetson_j',
        'stetson_k',
        'ad',
        'sw',
    ]
    for idx, _id in enumerate(id_list_bs):
        for si, key in enumerate(stat_keys):
            feature_dict[_id][key] = float(basic_stats_arr[idx, si])

    if doFlareDetection:
        print(f'Computing PDRS flare features for {len(id_list_bs)} sources...')
        for _id in id_list_bs:
            feature_dict[_id].update(
                pdrs.calc_pdrs_stats(tme_dict[_id]['tme'], **pdrs_params)
            )

    if doDRW:
        print(f'Computing DRW features for {len(id_list_bs)} sources...')
        drw_stats_list = drw.calc_drw_stats_batch(
            [tme_dict[_id]['tme'] for _id in id_list_bs],
            Ncore=Ncore,
            **drw_params,
        )
        for _id, drw_stats in zip(id_list_bs, drw_stats_list):
            feature_dict[_id].update(drw_stats)

    if baseline > 0:
        # Define frequency grid using largest LC time baseline
        if doScaleMinPeriod:
            fmin, fmax = 2 / baseline, 1 / (
                2 * min_cadence_minutes / 1440
            )  # Nyquist frequency given minimum cadence converted to days
        else:
            fmin, fmax = 2 / baseline, max_freq

        df = 1.0 / (samples_per_peak * baseline)
        nf = int(np.ceil((fmax - fmin) / df))
        freqs = fmin + df * np.arange(nf)

        # Define terrestrial frequencies to remove
        if doRemoveTerrestrial:
            freqs_to_remove = [
                [0.0025, 0.003],  # 1y
                [0.00125, 0.0015],  # 2 y
                [0.000833, 0.001],  # 3 y
                [0.000625, 0.00075],  # 4 y
                [0.0005, 0.0006],  # 5 y
                [0.005, 0.006],  # 0.5 y
                [3e-2, 4e-2],  # 30 d
                [3.95, 4.05],  # 0.25 d
                [2.95, 3.05],  # 0.33 d
                [1.95, 2.05],  # 0.5 d
                [0.95, 1.05],  # 1 d
                [0.48, 0.52],  # 2 d
                [0.32, 0.34],  # 3 d
            ]
        else:
            freqs_to_remove = None

        # Create separate grid with terrestrial freqs removed
        freqs_copy = freqs.copy()
        if freqs_to_remove is not None:
            for pair in freqs_to_remove:
                idx = np.where((freqs_copy < pair[0]) | (freqs_copy > pair[1]))[0]
                freqs_copy = freqs_copy[idx]
        freqs_no_terrestrial = freqs_copy

        # Continue with periodsearch/periodfind
        period_dict = {}
        significance_dict = {}
        pdot_dict = {}
        do_nested_algorithms = False
        do_top_n = False
        top_n_periods_dict = {}
        if doCPU or doGPU:
            if doCPU and doGPU:
                raise KeyError('Please set only one of --doCPU or --doGPU.')

            if 'ELS_ECE_EAOV' in period_algorithms:
                period_algorithms = [
                    'ELS_periodogram',
                    'ECE_periodogram',
                    'EAOV_periodogram',
                ]
                do_nested_algorithms = True
                warnings.warn(
                    'Performing nested ELS/ECE -> EAOV period search. Other algorithms in config will be ignored.'
                )
            elif 'LS_CE_AOV' in period_algorithms:
                period_algorithms = [
                    'LS_periodogram',
                    'CE_periodogram',
                    'AOV_periodogram',
                ]
                do_nested_algorithms = True
                warnings.warn(
                    'Performing nested ELS/ECE -> EAOV period search. Other algorithms in config will be ignored.'
                )

            # When top_n_periods > 1, request periodogram output for all
            # algorithms so we can extract the top-N peaks.
            do_top_n = top_n_periods > 1 and not do_nested_algorithms
            if do_top_n:
                pa_run = [
                    a if '_periodogram' in a else a + '_periodogram'
                    for a in period_algorithms
                ]
            else:
                pa_run = list(period_algorithms)

            n_sources = len(feature_dict)
            if n_sources % period_batch_size != 0:
                n_iterations = n_sources // period_batch_size + 1
            else:
                n_iterations = n_sources // period_batch_size

            all_periods = {algorithm: [] for algorithm in period_algorithms}
            all_significances = {algorithm: [] for algorithm in period_algorithms}
            all_pdots = {algorithm: [] for algorithm in period_algorithms}
            # top-N storage: keyed by original algo name
            all_top_n_periods = (
                {algorithm: [] for algorithm in period_algorithms} if do_top_n else {}
            )
            all_top_n_sigs = (
                {algorithm: [] for algorithm in period_algorithms} if do_top_n else {}
            )

            if do_nested_algorithms:
                if doGPU:
                    nested_key = 'ELS_ECE_EAOV'
                elif doCPU:
                    nested_key = 'ELS_ECE_EAOV'
                    # Note: name of CPU period algorithms is currently the same as for GPU algorithms
                    # nested_key = 'LS_CE_AOV'

                # Additional entry for nested algorithm
                all_periods[nested_key] = []
                all_significances[nested_key] = []
                all_pdots[nested_key] = []

            print(
                f'Running {len(period_algorithms)} period algorithms for {len(feature_dict)} sources in batches of {period_batch_size}...'
            )
            if do_top_n:
                print(f'Saving top {top_n_periods} periods per algorithm.')

            # --- Checkpoint resume ---
            start_batch = 0
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt = _load_period_checkpoint(checkpoint_dir)
                if ckpt is not None:
                    start_batch = ckpt['completed_batch'] + 1
                    all_periods = ckpt['all_periods']
                    all_significances = ckpt['all_significances']
                    all_pdots = ckpt['all_pdots']
                    all_top_n_periods = ckpt.get('all_top_n_periods', {})
                    all_top_n_sigs = ckpt.get('all_top_n_sigs', {})
                    print(
                        f'  Skipping batches 1-{start_batch} (already completed)',
                        flush=True,
                    )

            for i in range(start_batch, n_iterations):
                print(f"Iteration {i+1} of {n_iterations}...", flush=True)

                for algorithm, algo_run in zip(period_algorithms, pa_run):
                    print(f'Running {algo_run} algorithm:', flush=True)
                    # Iterate over algorithms
                    periods, significances, pdots = periodsearch.find_periods(
                        algo_run,
                        tme_collection[
                            i
                            * period_batch_size : min(
                                n_sources, (i + 1) * period_batch_size
                            )
                        ],
                        freqs,
                        doGPU=doGPU,
                        doCPU=doCPU,
                        doRemoveTerrestrial=doRemoveTerrestrial,
                        doUsePDot=False,
                        doSingleTimeSegment=False,
                        freqs_to_remove=freqs_to_remove,
                        phase_bins=20,
                        mag_bins=10,
                        Ncore=Ncore,
                        qmin=qmin,
                        qmax=qmax,
                    )

                    if not do_nested_algorithms:
                        if do_top_n:
                            # periods is list of dicts with 'period' and 'data'
                            p_vals = np.array([p['period'] for p in periods])
                            all_periods[algorithm] = np.concatenate(
                                [all_periods[algorithm], p_vals]
                            )
                            top_p, top_s = periodsearch.extract_top_n_periods(
                                periods,
                                freqs_no_terrestrial if doRemoveTerrestrial else freqs,
                                n_top=top_n_periods,
                            )
                            all_top_n_periods[algorithm].append(top_p)
                            all_top_n_sigs[algorithm].append(top_s)
                        else:
                            all_periods[algorithm] = np.concatenate(
                                [all_periods[algorithm], periods]
                            )

                    else:
                        p_vals = [p['period'] for p in periods]
                        p_stats = [p['data'] for p in periods]
                        all_periods[algorithm] = np.concatenate(
                            [all_periods[algorithm], p_vals]
                        )
                        if (algorithm == 'ELS_periodogram') | (
                            algorithm == 'LS_periodogram'
                        ):
                            # Maximum statistic is best for ELS/LS; select top N
                            topN_significance_indices_ELS = [
                                np.argsort(ps.flatten())[::-1][:top_n_periods]
                                for ps in p_stats
                            ]
                        elif (algorithm == 'ECE_periodogram') | (
                            algorithm == 'CE_periodogram'
                        ):
                            # Minimum statistic is best for ECE/CE; select top N
                            topN_significance_indices_ECE = [
                                np.argsort(ps.flatten())[:top_n_periods]
                                for ps in p_stats
                            ]
                        elif (algorithm == 'EAOV_periodogram') | (
                            algorithm == 'AOV_periodogram'
                        ):
                            ELS_ECE_top_indices = np.concatenate(
                                [
                                    topN_significance_indices_ELS,
                                    topN_significance_indices_ECE,
                                ],
                                axis=1,
                            )
                            ELS_ECE_top_indices = [
                                np.unique(x) for x in ELS_ECE_top_indices
                            ]

                            best_index_of_indices = [
                                np.argmax(p_stats[i][ELS_ECE_top_indices[i]])
                                for i in range(len(p_stats))
                            ]

                            best_indices = [
                                ELS_ECE_top_indices[i][best_index_of_indices[i]]
                                for i in range(len(ELS_ECE_top_indices))
                            ]

                            all_periods[nested_key] = np.concatenate(
                                [
                                    all_periods[nested_key],
                                    1 / freqs_no_terrestrial[best_indices],
                                ]
                            )

                            all_significances[nested_key] = np.concatenate(
                                [
                                    all_significances[nested_key],
                                    [
                                        p_stats[i].flatten()[best_indices[i]]
                                        for i in range(len(best_indices))
                                    ],
                                ]
                            )

                            all_pdots[nested_key] = np.concatenate(
                                [
                                    all_pdots[nested_key],
                                    pdots,
                                ]
                            )

                    all_significances[algorithm] = np.concatenate(
                        [all_significances[algorithm], significances]
                    )
                    all_pdots[algorithm] = np.concatenate([all_pdots[algorithm], pdots])

                # --- Save checkpoint after each batch ---
                if checkpoint_dir is not None:
                    _save_period_checkpoint(
                        checkpoint_dir,
                        i,
                        all_periods,
                        all_significances,
                        all_pdots,
                        all_top_n_periods,
                        all_top_n_sigs,
                    )

            # --- Clean up checkpoint on successful completion ---
            if checkpoint_dir is not None:
                ckpt_path = os.path.join(checkpoint_dir, 'period_checkpoint.pkl')
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(
                        '  [checkpoint] Period finding complete, checkpoint removed.',
                        flush=True,
                    )

            period_dict = all_periods
            significance_dict = all_significances
            pdot_dict = all_pdots

            # Concatenate top-N arrays across batches
            if do_top_n:
                top_n_periods_dict = {}
                top_n_sigs_dict = {}
                for algorithm in period_algorithms:
                    if all_top_n_periods.get(algorithm):
                        top_n_periods_dict[algorithm] = np.concatenate(
                            all_top_n_periods[algorithm], axis=0
                        )
                        top_n_sigs_dict[algorithm] = np.concatenate(
                            all_top_n_sigs[algorithm], axis=0
                        )

            if do_nested_algorithms:
                period_algorithms += [nested_key]

        else:
            warnings.warn("Skipping period finding; setting all periods to 1.0 d.")
            # Default periods 1.0 d
            period_algorithms = ['Ones']
            period_dict['Ones'] = np.ones(len(tme_collection))
            significance_dict['Ones'] = np.ones(len(tme_collection))
            pdot_dict['Ones'] = np.ones(len(tme_collection))

        for algorithm in period_algorithms:
            if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
                algorithm_name = algorithm.split('_')[0]
            else:
                algorithm_name = algorithm

            for idx, _id in enumerate(keep_id_list):

                period = period_dict[algorithm][idx]
                significance = significance_dict[algorithm][idx]
                pdot = pdot_dict[algorithm][idx]

                tme_dict[_id][f'period_{algorithm_name}'] = period
                tme_dict[_id][f'significance_{algorithm_name}'] = significance
                tme_dict[_id][f'pdot_{algorithm_name}'] = pdot

                feature_dict[_id][f'period_{algorithm_name}'] = period
                feature_dict[_id][f'significance_{algorithm_name}'] = significance
                feature_dict[_id][f'pdot_{algorithm_name}'] = pdot

        # --- Top-N periods and cross-algorithm agreement ---
        if do_top_n and top_n_periods_dict:
            print(f'Storing top {top_n_periods} periods per algorithm...', flush=True)
            for algorithm in period_algorithms:
                if algorithm not in top_n_periods_dict:
                    continue
                if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
                    algorithm_name = algorithm.split('_')[0]
                else:
                    algorithm_name = algorithm
                top_p = top_n_periods_dict[algorithm]
                top_s = top_n_sigs_dict[algorithm]
                for idx, _id in enumerate(keep_id_list):
                    for rank in range(top_n_periods):
                        feature_dict[_id][f'period_{rank+1}_{algorithm_name}'] = float(
                            top_p[idx, rank]
                        )
                        feature_dict[_id][f'significance_{rank+1}_{algorithm_name}'] = (
                            float(top_s[idx, rank])
                        )

            # Cross-algorithm agreement scores
            print('Computing cross-algorithm agreement scores...')
            agree_results = periodsearch.compute_agreement_scores(
                period_dict,
                top_n_periods_dict,
                keep_id_list,
                period_algorithms,
            )
            for _id, scores in agree_results.items():
                feature_dict[_id].update(scores)

        print(
            f'Computing Fourier stats for {len(period_dict)} algorithms...', flush=True
        )
        id_list = list(tme_dict.keys())
        lightcurves_ordered = [tme_dict[_id]['tme'] for _id in id_list]

        fourier_names = [
            'f1_power',
            'f1_BIC',
            'f1_a',
            'f1_b',
            'f1_amp',
            'f1_phi0',
            'f1_relamp1',
            'f1_relphi1',
            'f1_relamp2',
            'f1_relphi2',
            'f1_relamp3',
            'f1_relphi3',
            'f1_relamp4',
            'f1_relphi4',
        ]

        for algorithm in period_algorithms:
            if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
                algorithm_name = algorithm.split('_')[0]
            else:
                algorithm_name = algorithm
            print(f'- Algorithm: {algorithm}')

            periods_for_algo = np.array(
                [tme_dict[_id][f'period_{algorithm_name}'] for _id in id_list],
                dtype=np.float32,
            )
            fourier_features = periodsearch.compute_fourier_features(
                lightcurves_ordered, periods_for_algo
            )

            for idx, _id in enumerate(id_list):
                for i, name in enumerate(fourier_names):
                    feature_dict[_id][f'{name}_{algorithm_name}'] = float(
                        fourier_features[idx, i]
                    )

        print('Computing dmdt histograms...', flush=True)
        id_list_dmdt = list(tme_dict.keys())
        lightcurves_dmdt = [tme_dict[_id]['tme'] for _id in id_list_dmdt]
        dmdt_arr = periodsearch.compute_dmdt_features(lightcurves_dmdt, dmdt_ints)

        for idx, _id in enumerate(id_list_dmdt):
            feature_dict[_id]['dmdt'] = dmdt_arr[idx].tolist()

        # Get ZTF alert stats
        if local_alert_stats is not None:
            for _id in feature_dict.keys():
                stats = local_alert_stats.get(
                    _id, {'n_ztf_alerts': 0, 'mean_ztf_alert_braai': 0.0}
                )
                feature_dict[_id]['n_ztf_alerts'] = stats['n_ztf_alerts']
                feature_dict[_id]['mean_ztf_alert_braai'] = stats[
                    'mean_ztf_alert_braai'
                ]
        else:
            alert_stats_dct = alertstats.get_ztf_alert_stats(
                feature_dict,
                kowalski_instances,
                catalog=alerts_catalog,
                radius_arcsec=xmatch_radius_arcsec,
                limit=limit,
                Ncore=Ncore,
            )
            for _id in feature_dict.keys():
                feature_dict[_id]['n_ztf_alerts'] = alert_stats_dct[_id]['n_ztf_alerts']
                feature_dict[_id]['mean_ztf_alert_braai'] = alert_stats_dct[_id][
                    'mean_ztf_alert_braai'
                ]

        # Add crossmatches to Gaia, AllWISE and PS1 (by default, see config.yaml)
        if local_xmatch is not None:
            for _id in feature_dict.keys():
                xm = local_xmatch.get(_id, {})
                feature_dict[_id].update(xm)
        elif kowalski_instances is not None:
            feature_dict = external_xmatch.xmatch(
                feature_dict,
                kowalski_instances,
                ext_catalog_info,
                radius_arcsec=xmatch_radius_arcsec,
                limit=limit,
                Ncore=Ncore,
            )
        else:
            warnings.warn(
                "Skipping external cross-matches (Kowalski not available and no local cache)."
            )
        feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

        # Rename index column to '_id' and reset index
        feature_df.index.set_names('_id', inplace=True)
        feature_df.reset_index(inplace=True)

        # Convert various _id datatypes to Int64
        colnames = [x for x in feature_df.columns]
        for col in colnames:
            if '_id' in col:
                feature_df[col] = feature_df[col].astype("Int64")

    else:
        # If baseline is still zero, then no light curves met the selection criteria
        # Generate an empty DF instead so this field/ccd/quad is treated as done
        print('No light curves meet selection criteria.')
        feature_df = pd.DataFrame()

    utcnow = datetime.utcnow()
    end_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    # Add metadata
    feature_df.attrs['scope_code_version'] = code_version
    feature_df.attrs['feature_generation_start_dateTime_utc'] = start_dt
    feature_df.attrs['feature_generation_end_dateTime_utc'] = end_dt
    feature_df.attrs['ZTF_source_catalog'] = source_catalog
    feature_df.attrs['ZTF_alerts_catalog'] = alerts_catalog
    feature_df.attrs['Gaia_catalog'] = gaia_catalog

    # Write results
    if not doNotSave:
        if not doSpecificIDs:
            filename += f"_field_{field}_ccd_{ccd}_quad_{quad}"
            filename += '.parquet'
            dirpath = BASE_DIR / dirname / f"field_{field}"
            os.makedirs(dirpath, exist_ok=True)

            source_count = len(feature_df)
            meta_dct = {}
            meta_dct["(field, ccd, quad)"] = {
                f"({field}, {ccd}, {quad})": {
                    "minobs": min_n_lc_points,
                    "start_time_utc": start_dt,
                    "end_time_utc": end_dt,
                    "ZTF_source_catalog": source_catalog,
                    "ZTF_alerts_catalog": alerts_catalog,
                    "Gaia_catalog": gaia_catalog,
                    "total": source_count,
                }
            }
        else:
            filename += "_specific_ids"
            filename += '.parquet'
            dirpath = BASE_DIR / dirname / "specific_ids"
            os.makedirs(dirpath, exist_ok=True)

            source_count = len(feature_df)
            meta_dct = {}
            meta_dct["specific_ids"] = {
                "specific_ids": {
                    "minobs": min_n_lc_points,
                    "start_time_utc": start_dt,
                    "end_time_utc": end_dt,
                    "ZTF_source_catalog": source_catalog,
                    "ZTF_alerts_catalog": alerts_catalog,
                    "Gaia_catalog": gaia_catalog,
                    "total": source_count,
                }
            }

        meta_filename = BASE_DIR / dirname / "meta.json"

        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                dct = json.load(f)
                if not doSpecificIDs:
                    dct["(field, ccd, quad)"].update(meta_dct["(field, ccd, quad)"])
                else:
                    dct["specific_ids"] = meta_dct["specific_ids"]
                meta_dct = dct

        with open(meta_filename, 'w') as f:
            try:
                json.dump(meta_dct, f)
            except Exception as e:
                print("error dumping to json, message: ", e)

        filepath = dirpath / filename
        write_parquet(feature_df, str(filepath))
        print(f"Wrote features for {source_count} sources.")

    else:
        print(f"Generated features for {len(feature_df)} sources.")

    t1 = time.time()
    print(f"Finished running in {t1 - t0} seconds.")

    return feature_df


def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)
    parser = argparse.ArgumentParser(add_help=add_help)

    parser.add_argument(
        "--source-catalog",
        default=source_catalog,
        help="name of source collection on Kowalski",
    )
    parser.add_argument(
        "--alerts-catalog",
        default=alerts_catalog,
        help="name of alerts collection on Kowalski",
    )
    parser.add_argument(
        "--gaia-catalog",
        default=gaia_catalog,
        help="name of Gaia collection on Kowalski",
    )
    parser.add_argument(
        "--bright-star-query-radius-arcsec",
        type=float,
        default=300.0,
        help="size of cone search radius to search for bright stars",
    )
    parser.add_argument(
        "--xmatch-radius-arcsec",
        type=float,
        default=2.0,
        help="cone radius for all crossmatches",
    )
    parser.add_argument(
        "--query-size-limit",
        type=int,
        default=10000,
        help="sources per query limit for large Kowalski queries",
    )
    parser.add_argument(
        "--period-algorithms",
        nargs='+',
        default=period_algorithms,
        help="to override config, list algorithms to use for period-finding with periodsearch.py",
    )
    parser.add_argument(
        "--period-batch-size",
        type=int,
        default=1000,
        help="batch size for GPU-accelerated period algorithms",
    )
    parser.add_argument(
        "--doCPU",
        action='store_true',
        default=False,
        help="if set, run period-finding algorithm on CPU",
    )
    parser.add_argument(
        "--doGPU",
        action='store_true',
        default=False,
        help="if set, use GPU-accelerated period algorithm",
    )
    parser.add_argument(
        "--samples-per-peak",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--doScaleMinPeriod",
        action='store_true',
        default=False,
        help="if set, scale min period using --min-cadence-minutes",
    )
    parser.add_argument(
        "--doRemoveTerrestrial",
        action='store_true',
        default=False,
        help="if set, remove terrestrial frequencies from period analysis",
    )
    parser.add_argument(
        "--Ncore",
        default=8,
        type=int,
        help="number of cores for parallel period finding",
    )
    parser.add_argument(
        "--field",
        type=int,
        default=296,
        help="if not -doAllFields, ZTF field to run on",
    )
    parser.add_argument(
        "--ccd", type=int, default=1, help="if not -doAllCCDs, ZTF ccd to run on"
    )
    parser.add_argument(
        "--quad", type=int, default=1, help="if not -doAllQuads, ZTF field to run on"
    )
    parser.add_argument(
        "--min-n-lc-points",
        type=int,
        default=50,
        help="minimum number of unflagged light curve points to run feature generation",
    )
    parser.add_argument(
        "--min-cadence-minutes",
        type=float,
        default=30.0,
        help="minimum cadence (in minutes) between light curve points. For groups of points closer together than this value, only the first will be kept.",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default='generated_features',
        help="Directory name for generated features",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='gen_features',
        help="Prefix for generated feature file",
    )
    parser.add_argument(
        "--doCesium",
        action='store_true',
        default=False,
        help="if set, use Cesium to generate additional features specified in config",
    )
    parser.add_argument(
        "--doFlareDetection",
        action='store_true',
        default=False,
        help="if set, compute PDRS flare/high-activity segmentation features",
    )
    parser.add_argument(
        "--doDRW",
        action='store_true',
        default=False,
        help="if set, fit damped-random-walk models and compute DRW anomaly features",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        default=False,
        help="if set, do not save features",
    )
    parser.add_argument(
        "--stop-early",
        action='store_true',
        default=False,
        help="if set, stop when number of sources reaches --query-size-limit. Helpful for testing on small samples.",
    )
    parser.add_argument("--doQuadrantFile", action="store_true", default=False)
    parser.add_argument("--quadrant-file", default="slurm.dat")
    parser.add_argument("--quadrant-index", default=0, type=int)
    parser.add_argument(
        "--doSpecificIDs",
        action='store_true',
        default=False,
        help="if set, perform feature generation for ztf_id column in config-specified file",
    )
    parser.add_argument(
        "--skipCloseSources",
        action='store_true',
        default=False,
        help="if set, skip removal of sources too close to bright stars via Gaia. May be useful if input data has previously been analyzed in this way.",
    )
    parser.add_argument(
        "--top-n-periods",
        type=int,
        default=10,
        help="number of (E)LS, (E)CE periods to pass to (E)AOV if using (E)LS_(E)CE_(E)AOV algorithm",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=48.0,
        help="maximum frequency [1 / days] to use for period finding. Overridden by --doScaleMinPeriod",
    )
    parser.add_argument(
        "--fg-dataset",
        type=str,
        default=None,
        help="path to parquet, hdf5 or csv file containing specific sources for feature generation",
    )
    parser.add_argument(
        "--max-timestamp-hjd",
        type=float,
        help="maximum timestamp for queried light curves (HJD)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="directory for period-finding checkpoints (enables resume on restart)",
    )
    parser.add_argument(
        "--qmin",
        type=float,
        default=0.01,
        help="BLS minimum transit/eclipse duration as fraction of period (default 0.01)",
    )
    parser.add_argument(
        "--qmax",
        type=float,
        default=0.5,
        help="BLS maximum transit/eclipse duration as fraction of period (default 0.5)",
    )
    return parser


def main():

    parser = get_parser()
    args, _ = parser.parse_known_args()

    # call generate_features
    generate_features(
        source_catalog=args.source_catalog,
        alerts_catalog=args.alerts_catalog,
        gaia_catalog=args.gaia_catalog,
        bright_star_query_radius_arcsec=args.bright_star_query_radius_arcsec,
        xmatch_radius_arcsec=args.xmatch_radius_arcsec,
        limit=args.query_size_limit,
        period_algorithms=args.period_algorithms,
        period_batch_size=args.period_batch_size,
        doCPU=args.doCPU,
        doGPU=args.doGPU,
        samples_per_peak=args.samples_per_peak,
        doScaleMinPeriod=args.doScaleMinPeriod,
        doRemoveTerrestrial=args.doRemoveTerrestrial,
        Ncore=args.Ncore,
        field=args.field,
        ccd=args.ccd,
        quad=args.quad,
        min_n_lc_points=args.min_n_lc_points,
        min_cadence_minutes=args.min_cadence_minutes,
        dirname=args.dirname,
        filename=args.filename,
        doCesium=args.doCesium,
        doFlareDetection=args.doFlareDetection,
        doDRW=args.doDRW,
        doNotSave=args.doNotSave,
        stop_early=args.stop_early,
        doQuadrantFile=args.doQuadrantFile,
        quadrant_file=args.quadrant_file,
        quadrant_index=args.quadrant_index,
        doSpecificIDs=args.doSpecificIDs,
        skipCloseSources=args.skipCloseSources,
        top_n_periods=args.top_n_periods,
        max_freq=args.max_freq,
        fg_dataset=args.fg_dataset,
        max_timestamp_hjd=args.max_timestamp_hjd,
        checkpoint_dir=args.checkpoint_dir,
        qmin=args.qmin,
        qmax=args.qmax,
    )
