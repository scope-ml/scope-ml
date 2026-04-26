import urllib
import requests
import time
from typing import Optional, Mapping
import numpy as np
import pandas as pd
from requests.exceptions import InvalidJSONError, JSONDecodeError
from urllib3.exceptions import ProtocolError
from scope.utils import parse_load_config

# define the baseurl and set the fritz token to connect

config = parse_load_config()

BASE_URL = f"{config['fritz']['protocol']}://{config['fritz']['host']}/"
MAX_ATTEMPTS = config['fritz']['max_attempts']
SLEEP_TIME = config['fritz']['sleep_time']
fritz_token = config['fritz']['token']
default_catalog = config['kowalski']['collections'].get('sources')


def api(
    method: str,
    endpoint: str,
    data: Optional[Mapping] = None,
    token: str = fritz_token,
    base_url: str = BASE_URL,
    max_attempts: int = MAX_ATTEMPTS,
    sleep_time: int = SLEEP_TIME,
):
    method = method.upper()
    headers = {"Authorization": f"token {token}"}
    kwargs = {
        "method": method,
        "url": urllib.parse.urljoin(base_url, endpoint),
        "headers": headers,
    }
    if method not in ("GET", "HEAD"):
        kwargs["json"] = data
    elif method == "GET":
        kwargs["params"] = data

    for attempt in range(max_attempts):
        try:
            response = requests.request(**kwargs)
            break
        except (
            InvalidJSONError,
            ConnectionError,
            ProtocolError,
            OSError,
            JSONDecodeError,
        ):
            print(f'Error - Retrying (attempt {attempt+1}).')
            time.sleep(sleep_time)
            continue

    return response


def radec_to_iau_name(ra: float, dec: float, prefix: str = "ZTFJ"):
    """Transform R.A./Decl. in degrees to IAU-style hexadecimal designations."""
    if not 0.0 <= ra < 360.0:
        raise ValueError("Bad RA value in degrees")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("Bad Dec value in degrees")

    ra_h = np.floor(ra * 12.0 / 180.0)
    ra_m = np.floor((ra * 12.0 / 180.0 - ra_h) * 60.0)
    ra_s = ((ra * 12.0 / 180.0 - ra_h) * 60.0 - ra_m) * 60.0

    dec_d = np.floor(abs(dec)) * np.sign(dec)
    dec_m = np.floor(np.abs(dec - dec_d) * 60.0)
    dec_s = np.abs(np.abs(dec - dec_d) * 60.0 - dec_m) * 60.0

    hms = f"{ra_h:02.0f}{ra_m:02.0f}{ra_s:05.2f}"
    dms = f"{dec_d:+03.0f}{dec_m:02.0f}{dec_s:04.1f}"

    return prefix + hms + dms


def get_lightcurves_via_coords(
    kowalski_instances,
    ra,
    dec,
    radius=2.0,
    catalog=default_catalog,
    program_id_selector=list([1, 2, 3]),
    limit_per_query=1000,
    Ncore=1,
    get_basic_data=False,
    max_timestamp_hjd=None,
):

    if catalog is None:
        raise ValueError(
            'No catalog specified. Please add one to config.yaml under kowalski: collectons: sources:'
        )

    if max_timestamp_hjd is None:
        max_timestamp_hjd = config['kowalski']['max_timestamp_hjd']

    light_curve_ids = []
    query = {
        "query_type": "near",
        "query": {
            "max_distance": radius,
            "distance_units": "arcsec",
            "radec": {"query_coords": [ra, dec]},
            "catalogs": {
                catalog: {
                    "filter": {},
                    "projection": {"_id": 1},
                }
            },
        },
        "kwargs": {
            "max_time_ms": 10000,
            "limit": 1000,
        },
    }

    responses = kowalski_instances.query(query=query)
    for name in responses.keys():
        if len(responses[name]) > 0:
            response = responses[name]
            if response.get("status", "error") == "success":
                lc_ids = [
                    item["_id"]
                    for item in response.get("data")[catalog]["query_coords"]
                ]
                light_curve_ids += lc_ids

    if len(light_curve_ids) == 0:
        return None
    else:
        print("Found %d lightcurves" % len(light_curve_ids))

    return get_lightcurves_via_ids(
        kowalski_instances,
        light_curve_ids,
        catalog,
        program_id_selector=program_id_selector,
        limit_per_query=limit_per_query,
        Ncore=Ncore,
        get_basic_data=get_basic_data,
        max_timestamp_hjd=max_timestamp_hjd,
    )


def get_lightcurves_via_ids(
    kowalski_instances,
    ids,
    catalog,
    program_id_selector=list([1, 2, 3]),
    limit_per_query=1000,
    Ncore=1,
    get_ids_coords_only=False,
    get_basic_data=False,
    max_timestamp_hjd=None,
):

    if max_timestamp_hjd is None:
        max_timestamp_hjd = config['kowalski']['max_timestamp_hjd']

    itr = 0
    lcs = []
    Nsources = len(ids)

    if get_ids_coords_only:
        # Only retrieve ids and coordinates
        projection = {
            "_id": 1,
            "coordinates.radec_geojson.coordinates": 1,
        }
    elif get_basic_data:
        # Only retrive basic data (esp. for feature generation)
        projection = {
            "_id": 1,
            "filter": 1,
            "data.hjd": 1,
            "data.mag": 1,
            "data.magerr": 1,
            "data.catflags": 1,
        }
    else:
        projection = {
            "_id": 1,
            "ra": 1,
            "dec": 1,
            "filter": 1,
            "meanmag": 1,
            "vonneumannratio": 1,
            "refchi": 1,
            "refmag": 1,
            "refmagerr": 1,
            "iqr": 1,
            "data": 1,
        }

    while True:
        Nqueries = int(np.ceil(Nsources / limit_per_query))

        time_filter = {"$gt": 0.0}
        if max_timestamp_hjd is not None:
            time_filter["$lte"] = max_timestamp_hjd

        queries = [
            {
                "query_type": "find",
                "query": {
                    "catalog": catalog,
                    "filter": {
                        "_id": {
                            "$in": ids[i * limit_per_query : (i + 1) * limit_per_query]
                        },
                        "data.programid": {
                            "$in": program_id_selector,
                        },
                        "data.hjd": time_filter,
                    },
                    "projection": projection,
                },
            }
            for i in range(itr, itr + min(Nqueries, Ncore))
        ]

        responses = kowalski_instances.query(
            queries=queries, use_batch_query=True, max_n_threads=Ncore
        )
        Nsources -= len(queries) * limit_per_query

        for name in responses.keys():
            if len(responses[name]) > 0:
                response_list = responses[name]
                for response in response_list:
                    if response.get("status", "error") == "success":
                        light_curves = response.get("data")
                        lcs += light_curves

        if Nsources <= 0:
            print(f'{len(ids)} done')
            break
        itr += len(queries)
        if (itr * limit_per_query) % limit_per_query == 0:
            print(itr * limit_per_query, "done")

    return lcs


def make_photometry(light_curves: list, drop_flagged: bool = False):
    """
    Make a pandas.DataFrame with photometry
    :param light_curves: list of photometric time series
    :param drop_flagged: drop data points with catflags!=0
    :return:
    """
    dfs = []
    for light_curve in light_curves:
        if len(light_curve["data"]):
            df = pd.DataFrame.from_records(light_curve["data"])
            df["fid"] = light_curve["filter"]
            dfs.append(df)

    df_light_curve = pd.concat(dfs, ignore_index=True, sort=False)

    ztf_filters = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    df_light_curve["ztf_filter"] = df_light_curve["fid"].apply(lambda x: ztf_filters[x])
    df_light_curve["magsys"] = "ab"
    df_light_curve["zp"] = 23.9
    df_light_curve["mjd"] = df_light_curve["hjd"] - 2400000.5

    df_light_curve["mjd"] = df_light_curve["mjd"].apply(lambda x: np.float64(x))
    df_light_curve["mag"] = df_light_curve["mag"].apply(lambda x: np.float32(x))
    df_light_curve["magerr"] = df_light_curve["magerr"].apply(lambda x: np.float32(x))

    # filter out flagged data:
    if drop_flagged:
        mask_not_flagged = df_light_curve["catflags"] == 0
        df_light_curve = df_light_curve.loc[mask_not_flagged]

    return df_light_curve


def save_newsource(
    kowalski_instances,
    group_ids,
    ra,
    dec,
    radius=2.0,
    obj_id=None,
    post_source=True,
    period=None,
    return_id=False,
    return_phot=False,
    skip_phot=False,
    instrument_id=1,
):

    # get the lightcurves
    light_curves = get_lightcurves_via_coords(kowalski_instances, ra, dec, radius)

    # generate position-based name if obj_id not set
    newsource = False
    if obj_id is None:
        newsource = True
        if light_curves is not None:
            ra_mean = float(
                np.mean(
                    [
                        light_curve["ra"]
                        for light_curve in light_curves
                        if light_curve.get("ra") is not None
                    ]
                )
            )
            dec_mean = float(
                np.mean(
                    [
                        light_curve["dec"]
                        for light_curve in light_curves
                        if light_curve.get("dec") is not None
                    ]
                )
            )

        else:
            print("No lightcurves found. Skipping source.")
            return None

        obj_id = radec_to_iau_name(ra_mean, dec_mean, prefix="ZTFJ")

    else:
        ra_mean, dec_mean = ra, dec

    # get photometry; drop flagged/nan data
    df_photometry = make_photometry(light_curves, drop_flagged=True)
    df_photometry = (
        df_photometry.dropna().drop_duplicates('uexpid').reset_index(drop=True)
    )

    photometry = {
        "obj_id": obj_id,
        "instrument_id": instrument_id,
        "mjd": df_photometry["mjd"].tolist(),
        "mag": df_photometry["mag"].tolist(),
        "magerr": df_photometry["magerr"].tolist(),
        "limiting_mag": df_photometry["zp"].tolist(),
        "magsys": df_photometry["magsys"].tolist(),
        "filter": df_photometry["ztf_filter"].tolist(),
        "ra": df_photometry["ra"].tolist(),
        "dec": df_photometry["dec"].tolist(),
        "group_ids": group_ids,
    }

    if len(photometry.get("mag", ())) == 0:
        print('No unflagged photometry available. Skipping source.')
        return None

    # post new source to Fritz
    if newsource or post_source:
        post_source_data = {
            "id": obj_id,
            "ra": ra_mean,
            "dec": dec_mean,
            "group_ids": group_ids,
            "origin": "Fritz",
        }

        response = api(
            "POST",
            "/api/sources",
            post_source_data,
            max_attempts=MAX_ATTEMPTS,
        )

        if response.json()["status"] == "error":
            print(f"Failed to save {obj_id} as a Source")
            return None

    # post photometry
    if not skip_phot:
        print("Uploading photometry for %s" % obj_id)
        response = api("PUT", "/api/photometry", photometry, max_attempts=MAX_ATTEMPTS)
        if response.json()["status"] == "error":
            print('Failed to post photometry to Fritz')
            print(response.json())
            return None

    if period is not None:
        response_anotations = api(
            'GET', 'api/sources/%s/annotations' % obj_id, max_attempts=MAX_ATTEMPTS
        )

        annotations_data = response_anotations.json().get('data')

        has_period_annotation = False
        for annotation in annotations_data:
            if annotation['origin'] == 'uploadedperiod':
                has_period_annotation = True

        if not has_period_annotation:
            # upload the period if it is provided and there is not already a period annotation
            data = {
                "origin": "uploadedperiod",
                "group_ids": group_ids,
                "data": {'period': period},
            }
            response = api(
                "POST",
                "api/sources/%s/annotations" % obj_id,
                data=data,
                max_attempts=MAX_ATTEMPTS,
            )

    if return_id & return_phot:
        return obj_id, photometry
    elif return_id:
        return obj_id
    elif return_phot:
        return photometry
    else:
        return None


def get_highscoring_objects(
    G,
    otype='vnv',
    condition="$or",
    limit=0.9,
    limit_dnn=None,
    limit_xgb=None,
):

    if limit_dnn is None:
        limit_dnn = limit
    if limit_xgb is None:
        limit_xgb = limit

    # example
    q = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_source_classifications_DR16",
            "filter": {
                condition: [
                    {'%s_xgb' % otype: {'$gt': limit_xgb}},
                    {'%s_dnn' % otype: {'$gt': limit_dnn}},
                ],
            },
            "projection": {},
        },
        "kwargs": {},
    }

    r = G.query(q)

    return pd.DataFrame(r['data'])


def get_stats(G, ids):
    qs = [
        {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_source_features_DR16",
                "filter": {'_id': i},
                "projection": {},
            },
            "kwargs": {},
        }
        for i in ids
    ]
    rs = G.batch_query(qs, n_treads=32)

    return pd.DataFrame([s['data'][0] for s in rs])
