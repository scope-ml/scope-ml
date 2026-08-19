import pandas as pd
from penquins import Kowalski
import numpy as np
from scope.surveys.fritz import get_highscoring_objects, get_stats
import argparse
import csv
import json

# This code is deprecated and will not be updated. It should eventually be removed.


def upload(data):
    import requests

    def api(method, endpoint, data=None):
        response = requests.request(method, endpoint, json=data, headers=headers)
        return response

    headers = {"Authorization": f"token {args.token}"}

    for index, row in data.iterrows():
        i = row['_id']

        # upload
        json = {"catalog": "ZTF_sources_20210401", "group_ids": [args.id]}
        json['light_curve_ids'] = [int(i)]
        response = requests.post(
            url='https://fritz.science/api/archive',
            json=json,
            headers=headers,
        )

        # get objid
        try:
            obj_id = response.json()['data']['obj_id']
        except Exception:
            continue

        # annotate
        url = 'https://fritz.science/api/sources/%s/annotations' % obj_id
        json = {"origin": "ML_DR5_d15_v1", "group_ids": [args.id]}

        Gaia = {
            'Plx': row['Gaia_EDR3__parallax'],
            'Mag_G': row['Gaia_EDR3__phot_g_mean_mag'],
            'Mag_Bp': row['Gaia_EDR3__phot_bp_mean_mag'],
            'Mag_Rp': row['Gaia_EDR3__phot_rp_mean_mag'],
        }

        json['data'] = {
            'period': row['period'],
            'vnv': {
                'vnv_dnn': np.round(row['vnv_dnn'], 3),
                'vnv_xgb': np.round(row['vnv_xgb'], 3),
            },
            'Gaia': Gaia,
        }

        response = requests.post(
            url=url,
            json=json,
            headers=headers,
        )


if __name__ == "__main__":
    # load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-id", type=int, default=1, help="group id on Fritz")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    args = parser.parse_args()
    with open(args.file) as f:
        data = csv.reader(f)
        trainingset = pd.DataFrame(data)

    # setup connection to gloria to get the lightcurves
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    G = Kowalski(**secrets['gloria'], verbose=False)

    # get scores and data and combine
    scores = get_highscoring_objects(G, otype='vnv', condition='$or')

    index = scores.index
    condition = ((scores["vnv_dnn"] > 0.95) & (scores['vnv_xgb'] <= 0.1)) | (
        (scores["vnv_dnn"] <= 0.1) & (scores['vnv_xgb'] > 0.95)
    )
    disagreements = index[condition]
    disagreeing_scores = scores.iloc[disagreements, :]

    stats = get_stats(G, disagreeing_scores['_id'])
    data = pd.merge(disagreeing_scores, stats, left_on='_id', right_on='_id')
    data['train'] = np.isin(data['_id'], trainingset['ztf_id'])
    sample = data[~data['train']]

    # upload disagreeing objects to target group id on Fritz
    upload(sample)
