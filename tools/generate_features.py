import scope
import argparse
import pathlib
import os
from tools.get_quad_ids import get_ids_loop, get_field_ids
from scope.fritz import get_lightcurves_via_ids
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
from penquins import Kowalski
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
from datetime import datetime
from tools.featureGeneration import lcstats, periodsearch, alertstats, external_xmatch
import warnings
from cesium.featurize import time_series, featurize_single_ts
import json
from joblib import Parallel, delayed
from scipy.stats import circmean
import time
from concurrent.futures import ProcessPoolExecutor


BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

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

source_catalog = config['kowalski']['collections']['sources']
alerts_catalog = config['kowalski']['collections']['alerts']
gaia_catalog = config['kowalski']['collections']['gaia']
dmdt_ints = config['feature_generation']['dmdt_ints']
ext_catalog_info = config['feature_generation']['external_catalog_features']
cesium_feature_list = config['feature_generation']['cesium_features']
period_algorithms = config['feature_generation']['period_algorithms']
path_to_features = config['feature_generation']['path_to_features']

if path_to_features is not None:
    BASE_DIR = pathlib.Path(path_to_features)

kowalski_instances = Kowalski(timeout=timeout, instances=instances)


def period_finding_func(alg, tme_collection, freqs, gpu_id='0',top_n_periods=50, **kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    periods, significances, pdots = periodsearch.find_periods(alg, tme_collection, freqs, kwargs)
    p_stats = np.array([p['data'].flatten() for p in periods])
    p_vals = np.array([p['period'] for p in periods])
    del periods
    print(f'Deleted periods:{alg}')
    match alg:
        case 'ELS_periodogram' | 'LS_periodogram':
            # Maximum statistic is best for ELS/LS; select top N
            top_inds = top_n_ind_nosort(p_stats,bottom=False,k=top_n_periods)
            #Clear Memory
            del p_stats
            print(f'Deleted p_stats:{alg}')
            p_stats=None
            # periods from arg sort
        case 'ECE_periodogram' | 'CE_periodogram':
            # Minimum statistic is best for ECE/CE; select top N
            top_inds =  top_n_ind_nosort(p_stats,bottom=True,k=top_n_periods)
            #Clear Memory
            del p_stats
            print(f'Deleted p_stats:{alg}')
            p_stats=None
        case 'EAOV_periodogram' | 'AOV_periodogram':
            #Dont need to collect inds
            top_inds=None
    return pdots, significances, p_vals, top_inds, p_stats

def top_n_ind_nosort(array,k=50,bottom=True):
    if bottom: # get the bottom k inds
        unsorted_inds=np.argpartition(array,k,axis=1)[:,0:k]
        unsorted_subset=np.take_along_axis(array, unsorted_inds, axis=1)
        sorted_inds=np.take_along_axis(unsorted_inds,np.argsort(unsorted_subset,axis=1),axis=1)
        return sorted_inds
    else: #get the top k inds
        unsorted_inds=np.argpartition(array,-1*k,axis=1)[:,-1*k:]
        unsorted_subset=np.take_along_axis(array, unsorted_inds, axis=1)
        sorted_inds=np.take_along_axis(unsorted_inds,np.argsort(unsorted_subset,axis=1)[:,::-1],axis=1)
        return sorted_inds

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
            print(f"Iteration {i + 1} of {n_iterations}...")
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
    doNotSave: bool = False,
    stop_early: bool = False,
    doQuadrantFile: bool = False,
    quadrant_file: str = 'slurm.dat',
    quadrant_index: int = 0,
    doSpecificIDs: bool = False,
    skipCloseSources: bool = False,
    top_n_periods: int = 50,
    max_freq: float = 48.0,
    fg_dataset: str = None,
    max_timestamp_hjd: float = None,
    min_timestamp_hjd: float = 0.0,
    num_gpu: int = 1,
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
    :param num_gpu: number of gpus to use in periodfinding

    :return feature_df: dataframe containing generated features

    * - specified in config.yaml
    """
    t0 = time.time()
    # define times
    total_dmdt = -1
    total_alerts = -1
    total_basic_features = -1
    total_dmdt = -1
    total_drop_close = -1
    total_foureir = -1
    total_get_ids = -1
    total_get_lc = -1
    total_period = -1
    total_write = -1
    total_xmatch = -1

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
        t_get_ids = time.time()
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
        total_get_ids = time.time() - t_get_ids
        print(f'Time to get IDs: {total_get_ids}')
        if not skipCloseSources:
            # Each index of lst corresponds to a different ccd/quad combo
            t_drop_close = time.time()
            feature_gen_source_dict = drop_close_bright_stars(
                lst[0],
                catalog=gaia_catalog,
                query_radius_arcsec=bright_star_query_radius_arcsec,
                xmatch_radius_arcsec=xmatch_radius_arcsec,
                limit=limit,
                Ncore=Ncore,
            )
            total_drop_close = time.time() - t_drop_close
            print(f'Time to drop close stars: {total_drop_close}')
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

        if not skipCloseSources:
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
            t_drop_close
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
            print(f'Time to drop close stars: {time.time() - t_drop_close}')

        else:
            feature_gen_source_dict = {
                int(k): fg_sources[k] for k in list(fg_sources)[:n_fg_sources]
            }

    print('Getting lightcurves...')
    # For small source lists, shrink LC query limit until batching occurs
    lc_limit = limit
    if len(feature_gen_source_dict) < limit:
        lc_limit = int(np.ceil(len(feature_gen_source_dict) / Ncore))

    feature_gen_ids = [x for x in feature_gen_source_dict.keys()]
    t_get_lc = time.time()
    lcs = get_lightcurves_via_ids(
        kowalski_instances=kowalski_instances,
        ids=feature_gen_ids,
        catalog=source_catalog,
        limit_per_query=lc_limit,
        Ncore=Ncore,
        get_basic_data=True,
        max_timestamp_hjd=max_timestamp_hjd,
        min_timestamp_hjd=min_timestamp_hjd,
    )
    total_get_lc = time.time() - t_get_lc
    print(f'Time to get lcs: {total_get_lc}')
    # Remake feature_gen_source_dict if some light curves are missing
    lc_ids = [lc['_id'] for lc in lcs]
    if len(lc_ids) != len(feature_gen_ids):
        feature_gen_source_dict = {x: feature_gen_source_dict[x] for x in lc_ids}

    feature_dict = feature_gen_source_dict.copy()

    # Get ZTF alert stats
    t_alert = time.time()
    alert_stats_dct = alertstats.get_ztf_alert_stats(
        feature_dict,
        kowalski_instances,
        catalog=alerts_catalog,
        radius_arcsec=xmatch_radius_arcsec,
        limit=limit,
        Ncore=Ncore,
    )
    total_alerts = time.time() - t_alert
    print(f'Time to get alerts: {total_alerts}')
    for _id in feature_dict.keys():
        feature_dict[_id]['n_ztf_alerts'] = alert_stats_dct[_id]['n_ztf_alerts']
        feature_dict[_id]['mean_ztf_alert_braai'] = alert_stats_dct[_id][
            'mean_ztf_alert_braai'
        ]

    # Add crossmatches to Gaia, AllWISE and PS1 (by default, see config.yaml)
    t_xmatch = time.time()
    feature_dict = external_xmatch.xmatch(
        feature_dict,
        kowalski_instances,
        ext_catalog_info,
        radius_arcsec=xmatch_radius_arcsec,
        limit=limit,
        Ncore=Ncore,
    )
    total_xmatch = time.time() - t_xmatch
    print(f'Time to xmatch: {total_xmatch}')

    print('Analyzing lightcuves and computing basic features...')
    t_basic_features = time.time()
    # Start by dropping flagged points
    count = 0
    baseline = 0
    max_lc_size = 0  # used this value to assign batch size
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
                # determine the largest size of a lc
                if len(tt) > max_lc_size:
                    max_lc_size = len(tt)

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
    # not sure this is being used the best way, want to max out Ncore but not kill kowalski
    # write using concurent futures?
    basicStats = Parallel(n_jobs=Ncore)(
        delayed(lcstats.calc_basic_stats)(id, vals['tme'])
        for id, vals in tme_dict.items()
    )

    for statline in basicStats:
        _id = [x for x in statline.keys()][0]
        statvals = [x for x in statline.values()][0]

        feature_dict[_id]['n'] = statvals[0]
        feature_dict[_id]['median'] = statvals[1]
        feature_dict[_id]['wmean'] = statvals[2]
        feature_dict[_id]['chi2red'] = statvals[3]
        feature_dict[_id]['roms'] = statvals[4]
        feature_dict[_id]['wstd'] = statvals[5]
        feature_dict[_id]['norm_peak_to_peak_amp'] = statvals[6]
        feature_dict[_id]['norm_excess_var'] = statvals[7]
        feature_dict[_id]['median_abs_dev'] = statvals[8]
        feature_dict[_id]['iqr'] = statvals[9]
        feature_dict[_id]['i60r'] = statvals[10]
        feature_dict[_id]['i70r'] = statvals[11]
        feature_dict[_id]['i80r'] = statvals[12]
        feature_dict[_id]['i90r'] = statvals[13]
        feature_dict[_id]['skew'] = statvals[14]
        feature_dict[_id]['smallkurt'] = statvals[15]
        feature_dict[_id]['inv_vonneumannratio'] = statvals[16]
        feature_dict[_id]['welch_i'] = statvals[17]
        feature_dict[_id]['stetson_j'] = statvals[18]
        feature_dict[_id]['stetson_k'] = statvals[19]
        feature_dict[_id]['ad'] = statvals[20]
        feature_dict[_id]['sw'] = statvals[21]
    total_basic_features = time.time() - t_basic_features
    print(f"Basic stats time :{total_basic_features}")
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
        print(f'Number of freq {nf}')
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

            n_sources = len(feature_dict)
            if n_sources % period_batch_size != 0:
                n_iterations = n_sources // period_batch_size + 1
            else:
                n_iterations = n_sources // period_batch_size

            all_periods = {algorithm: [] for algorithm in period_algorithms}
            all_significances = {algorithm: [] for algorithm in period_algorithms}
            all_pdots = {algorithm: [] for algorithm in period_algorithms}

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
            time_period = time.time()
            pp = ProcessPoolExecutor(max_workers=3)
            LS_fs=[]
            CE_fs=[]
            AoV_fs=[]
            for i in range(0, n_iterations): #Submit the AOV and CE jobs, D:kinda of broken for when you dont use all 3 algs
                f_LS = pp.submit(
                    period_finding_func,
                    'ELS_periodogram',
                    tme_collection[
                        i
                        * period_batch_size : min(
                            n_sources, (i + 1) * period_batch_size
                        )
                    ],
                    freqs_no_terrestrial,
                    doGPU=doGPU,
                    doCPU=doCPU,
                    doRemoveTerrestrial=doRemoveTerrestrial,
                    doUsePDot=False,
                    doSingleTimeSegment=False,
                    freqs_to_remove=freqs_to_remove,
                    phase_bins=20,
                    mag_bins=10,
                    Ncore=Ncore,
                    gpu_id=str(0),
                    top_n_periods=top_n_periods,
                )
                LS_fs.append(f_LS)
                f_CE = pp.submit(
                    period_finding_func,
                    'ECE_periodogram',
                    tme_collection[
                        i
                        * period_batch_size : min(
                            n_sources, (i + 1) * period_batch_size
                        )
                    ],
                    freqs_no_terrestrial,
                    doGPU=doGPU,
                    doCPU=doCPU,
                    doRemoveTerrestrial=doRemoveTerrestrial,
                    doUsePDot=False,
                    doSingleTimeSegment=False,
                    freqs_to_remove=freqs_to_remove,
                    phase_bins=20,
                    mag_bins=10,
                    Ncore=Ncore,
                    gpu_id=str(0),
                    top_n_periods=top_n_periods,
                )
                CE_fs.append(f_CE)
            #limit the number of live AoV jobs (they require more memory)
            aov_pp=ProcessPoolExecutor(max_workers=3) #D:seperate pool to let AOV jump the queue
            completed_jobs=0
            while completed_jobs < n_iterations:
                print(f'Running aov jobs:{len(AoV_fs)}')
                if len(AoV_fs) <= 8 and len(AoV_fs)+completed_jobs < n_iterations: #D; 8 running for now to be safe
                    batch_index=len(AoV_fs)+completed_jobs
                    AoV_f=aov_pp.submit(
                    period_finding_func,
                    'EAOV_periodogram',
                    tme_collection[
                        batch_index
                        * period_batch_size : min(
                            n_sources, (batch_index + 1) * period_batch_size
                        )
                    ],
                    freqs_no_terrestrial,
                    doGPU=doGPU,
                    doCPU=doCPU,
                    doRemoveTerrestrial=doRemoveTerrestrial,
                    doUsePDot=False,
                    doSingleTimeSegment=False,
                    freqs_to_remove=freqs_to_remove,
                    phase_bins=20,
                    mag_bins=10,
                    Ncore=Ncore,
                    gpu_id=str(0),
                    top_n_periods=top_n_periods,
                    )
                    AoV_fs.append(AoV_f)
                else: #Clear out one AoV
                    print(f'Computing batch {completed_jobs}')
                    #remove first element
                    temp_LS=LS_fs.pop(0) 
                    temp_CE=CE_fs.pop(0)
                    temp_AoV=AoV_fs.pop(0)
                    completed_jobs +=1
                    #collect result
                    topN_significance_indices_ELS = []
                    topN_significance_indices_ECE = []
                    for f,algorithm in zip([temp_LS,temp_CE,temp_AoV],period_algorithms):
                        print(algorithm)
                        pdots, significances, p_vals, top_inds, p_stats = f.result()
                        # save data into final dict
                        all_periods[algorithm] = np.concatenate([all_periods[algorithm], p_vals])
                        all_pdots[algorithm] = np.concatenate([all_pdots[algorithm], pdots])
                        all_significances[algorithm] = np.concatenate([all_significances[algorithm], significances])
                        if (
                            do_nested_algorithms
                        ):  # logic to sort the top N for the tie breaking AOV run
                            match algorithm:
                                case 'ELS_periodogram' | 'LS_periodogram':
                                    topN_significance_indices_ELS = top_inds
                                case 'ECE_periodogram' | 'CE_periodogram':
                                    topN_significance_indices_ECE =  top_inds
                                case (
                                    'EAOV_periodogram' | 'AOV_periodogram'
                                ):  # D: AOV must come LAST design flaw for now...
                                    # combine top indices from other algs and get unique ones
                                    top_combined_significance_indices = np.concatenate(
                                        [topN_significance_indices_ELS,
                                        topN_significance_indices_ECE],
                                        axis=1,
                                    )
                                    # get single best AoV
                                    top_combined_significance_indices=[
                                    np.unique(x) for x in top_combined_significance_indices
                                ]
                                    # select only INDEX of best score from AOV of the top indices from LS and CE
                                    best_index_of_indices = [
                                        np.argmax(
                                            p_stats[j][top_combined_significance_indices[j]]
                                        )
                                        for j in range(len(p_stats))
                                    ]
                                    # Get the index on the periods
                                    best_indices = [
                                        top_combined_significance_indices[j][
                                            best_index_of_indices[j]
                                        ]
                                        for j in range(
                                            len(top_combined_significance_indices)
                                        )
                                    ]

                                    # Concat best single period for whole batch in order
                                    all_periods[nested_key] = np.concatenate(
                                        [
                                            all_periods[nested_key],
                                            1 / freqs_no_terrestrial[best_indices],
                                        ]
                                    )
                                    # Concat associated significance in order
                                    all_significances[nested_key] = np.concatenate(
                                        [
                                            all_significances[nested_key],
                                            [
                                                p_stats[j].flatten()[best_indices[j]]
                                                for j in range(len(best_indices))
                                            ],
                                        ]
                                    )
                                    # D: Pdots is wrong here but SCoPe was trained like this so I leave it wrong. Also Pdot is not used
                                    all_pdots[nested_key] = np.concatenate(
                                        [
                                            all_pdots[nested_key],
                                            pdots,
                                        ]
                                    )
                                case _:
                                    print(
                                        f'{algorithm} not recognized, How did the code get this far?'
                                    )
                    del temp_LS
                    del temp_CE
                    del temp_AoV
            pp.shutdown()
            aov_pp.shutdown()
            period_dict = all_periods
            significance_dict = all_significances
            pdot_dict = all_pdots
            #print(type(all_periods['ELS_periodogram']))
            #print(all_periods['ELS_periodogram'].shape)
            #print(all_periods['ELS_periodogram'])

            if do_nested_algorithms:
                period_algorithms += [nested_key]
            total_period = time.time() - time_period
            print(f'Time to do only period finding {total_period}')
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
        print(f'Computing Fourier stats for {len(period_dict)} algorithms...')
        #print(len(tme_dict))
        #print(len(tme_dict.keys()))
        #print(tme_dict.keys())
        time_fourier = time.time()
        for algorithm in period_algorithms:
            if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
                algorithm_name = algorithm.split('_')[0]
            else:
                algorithm_name = algorithm
            print(f'- Algorithm: {algorithm}')
            # Does this do computations in parallel the best?
            fourierStats = Parallel(n_jobs=Ncore)(
                delayed(lcstats.calc_fourier_stats)(
                    id, vals['tme'], vals[f'period_{algorithm_name}']
                )
                for id, vals in tme_dict.items()
            )

            for statline in fourierStats:
                _id = [x for x in statline.keys()][0]
                statvals = [x for x in statline.values()][0]

                feature_dict[_id][f'f1_power_{algorithm_name}'] = statvals[0]
                feature_dict[_id][f'f1_BIC_{algorithm_name}'] = statvals[1]
                feature_dict[_id][f'f1_a_{algorithm_name}'] = statvals[2]
                feature_dict[_id][f'f1_b_{algorithm_name}'] = statvals[3]
                feature_dict[_id][f'f1_amp_{algorithm_name}'] = statvals[4]
                feature_dict[_id][f'f1_phi0_{algorithm_name}'] = statvals[5]
                feature_dict[_id][f'f1_relamp1_{algorithm_name}'] = statvals[6]
                feature_dict[_id][f'f1_relphi1_{algorithm_name}'] = statvals[7]
                feature_dict[_id][f'f1_relamp2_{algorithm_name}'] = statvals[8]
                feature_dict[_id][f'f1_relphi2_{algorithm_name}'] = statvals[9]
                feature_dict[_id][f'f1_relamp3_{algorithm_name}'] = statvals[10]
                feature_dict[_id][f'f1_relphi3_{algorithm_name}'] = statvals[11]
                feature_dict[_id][f'f1_relamp4_{algorithm_name}'] = statvals[12]
                feature_dict[_id][f'f1_relphi4_{algorithm_name}'] = statvals[13]
        total_foureir = time.time() - time_fourier
        print(f"Time for Foureir: {total_foureir}")
        print('Computing dmdt histograms...')
        # check parrel
        time_dmdt = time.time()
        dmdt = Parallel(n_jobs=Ncore)(
            delayed(lcstats.compute_dmdt)(id, vals['tme'], dmdt_ints)
            for id, vals in tme_dict.items()
        )

        for dmdtline in dmdt:
            _id = [x for x in dmdtline.keys()][0]
            dmdtvals = [x for x in dmdtline.values()][0]
            feature_dict[_id]['dmdt'] = dmdtvals.tolist()

        feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')
        total_dmdt = time.time() - time_dmdt
        print(f'dmdt time : {total_dmdt}')
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
    time_write = time.time()
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
    total_write = time.time() - time_write
    print(f'Write time: {total_write}')
    t1 = time.time()
    print(f"Finished running in {t1 - t0} seconds.")
    print('Section  time break down')
    print(f'total_alerts : {total_alerts}')
    print(f'total_basic_features : {total_basic_features}')
    print(f'total_dmdt : {total_dmdt}')
    print(f'total_drop_close : {total_drop_close}')
    print(f'total_foureir : {total_foureir}')
    print(f'total_get_ids : {total_get_ids}')
    print(f'total_get_lc : {total_get_lc}')
    print(f'total_period : {total_period}')
    print(f'total_write : {total_write}')
    print(f'total_xmatch : {total_xmatch}')

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
        default=50,
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
        "--min-timestamp-hjd",
        type=float,
        help="min-imum timestamp for queried light curves (HJD)",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        help="Number of gpus to use in periodfinding",
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
        num_gpu=args.num_gpu,
    )
