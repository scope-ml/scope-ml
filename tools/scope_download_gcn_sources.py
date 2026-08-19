#!/usr/bin/env python
import argparse
import pathlib
import os
from penquins import Kowalski
from datetime import datetime, timedelta
import numpy as np
import json
from scope.surveys.fritz import api
from tools.get_quad_ids import get_cone_ids
from scope.utils import write_parquet, parse_load_config

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()
NUM_PER_PAGE = 100

# EM+GW is group 1544
# Recommendation: query all groups when downloading, then upload all classifications to single group (e.g. EM+GW)

# use tokens specified as env vars (if exist)
kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

# Set up Kowalski instance connection
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

source_catalog = config['kowalski']['collections']['sources']


def download_gcn_sources(
    dateobs: str,
    group_ids: list = [],
    days_range: float = 7.0,
    radius_arcsec: float = 0.5,
    save_filename: str = 'fritzDownload/specific_ids_GCN_sources',
):
    """
    Download sources for a GCN event from Fritz (with intent to generate features/run inference on these sources)

    :param dateobs: unique dateObs of GCN event (str)
    :param group_ids: group ids to query sources [all if not specified] (list)
    :param days_range: max days past event to search for sources (float)
    :param radius_arcsec: radius [arcsec] around new sources to search for existing ZTF sources (float)
    :param save_filename: filename to save source ids/coordinates (str)

    """

    # Colons can confuse the file system; replace them for saving
    save_dateobs = dateobs.replace(':', '-')

    dateobs_datetime = datetime.strptime(dateobs, '%Y-%m-%dT%H:%M:%S')
    endDate_datetime = dateobs_datetime + timedelta(days=days_range)
    endDate = endDate_datetime.strftime("%Y-%m-%dT%H:%M:%S")

    data = {
        'group_ids': group_ids,
        'localizationDateobs': dateobs,
        'startDate': dateobs,
        'endDate': endDate,
        'numPerPage': NUM_PER_PAGE,
    }
    response = api('GET', '/api/sources', data=data)
    data = response.json().get('data')
    status = response.json().get('status')
    message = response.json().get('message')

    if status == 'error':
        print("Error during query: ", message)
        return None

    # Determine number of pages
    allMatches = data.get('totalMatches')
    nPerPage = data.get('numPerPage')
    pages = int(np.ceil(allMatches / nPerPage))

    gcn_source_dct = {}

    # iterate over all pages in results
    if (allMatches is not None) & (allMatches > 0):
        print(f'Downloading {allMatches} sources...')
        for pageNum in range(1, pages + 1):
            print(f'Page {pageNum} of {pages}...')
            page_response = api(
                "GET",
                '/api/sources',
                {
                    "group_ids": group_ids,
                    'localizationDateobs': dateobs,
                    'startDate': dateobs,
                    'endDate': endDate,
                    'numPerPage': NUM_PER_PAGE,
                    'pageNumber': pageNum,
                },  # page numbers start at 1
            )
            page_data = page_response.json().get('data')

            for src in page_data['sources']:
                id = src['id']
                ra = src['ra']
                dec = src['dec']
                gcn_source_dct[id] = {'gcn_source_coordinates': [ra, dec]}

    else:
        if allMatches is None:
            print('Check if query successfully completed. Status: ', status)
            return None
        elif allMatches == 0:
            print('No sources found.')
            return None

    print('Finding ZTF sources around GCN source coords...')

    obj_id_list = [x for x in gcn_source_dct.keys()]
    ra_list = [x['gcn_source_coordinates'][0] for x in gcn_source_dct.values()]
    dec_list = [x['gcn_source_coordinates'][1] for x in gcn_source_dct.values()]

    ids = get_cone_ids(
        obj_id_list=obj_id_list,
        ra_list=ra_list,
        dec_list=dec_list,
        catalog=source_catalog,
        kowalski_instances=kowalski_instances,
        max_distance=radius_arcsec,
        distance_units='arcsec',
        get_coords=True,
    )

    if len(ids) > 0:
        ids = ids.drop_duplicates('_id').reset_index(drop=True)
        ids.rename({'_id': 'ztf_id', 'obj_id': 'fritz_name'}, axis=1, inplace=True)

        write_parquet(ids, str(BASE_DIR / f"{save_filename}.{save_dateobs}.parquet"))

        coord_col = ids['coordinates']
        ids['radec_geojson'] = [row['radec_geojson'] for row in coord_col]
        ids.drop('coordinates', axis=1, inplace=True)

        ids = ids.set_index('ztf_id').to_dict(orient='index')
        with open(str(BASE_DIR / f"{save_filename}.{save_dateobs}.json"), 'w') as f:
            json.dump(ids, f)
    else:
        print('No associated ZTF sources found.')
        ids = None

    return ids


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dateobs", type=str, help="unique dateObs of GCN event")
    parser.add_argument(
        "--group-ids",
        type=int,
        nargs='+',
        help="group ids to query sources (all if not specified)",
    )
    parser.add_argument(
        "--days-range",
        type=float,
        default=7.0,
        help="max days past event to search for sources",
    )
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=0.5,
        help="radius around new sources to search for existing ZTF sources",
    )
    parser.add_argument(
        "--save-filename",
        type=str,
        default='fritzDownload/specific_ids_GCN_sources',
        help="filename to save source ids/coordinates",
    )

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    download_gcn_sources(
        dateobs=args.dateobs,
        group_ids=args.group_ids,
        days_range=args.days_range,
        radius_arcsec=args.radius_arcsec,
        save_filename=args.save_filename,
    )
