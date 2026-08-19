#!/usr/bin/env python
import argparse
import pathlib
import os
from datetime import datetime
import pandas as pd
from penquins import Kowalski
from scope.surveys.fritz import get_lightcurves_via_ids, radec_to_iau_name
from tools.get_quad_ids import get_cone_ids
from scope.utils import (
    read_parquet,
    read_hdf,
    write_parquet,
    write_hdf,
    parse_load_config,
)
from tools import generate_features, inference


BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

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
source_catalog = config['kowalski']['collections']['sources']

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


def run_scope_local(
    path_dataset,
    **kwargs,
):
    currentDate = datetime.utcnow()
    current_dt = currentDate.strftime("%Y-%m-%dT%H-%M-%S")

    cone_radius_arcsec = kwargs.get("cone_radius_arcsec", 2.0)
    save_sources_filepath = kwargs.get("save_sources_filepath", "sources.parquet")
    algorithms = kwargs.get("algorithms", [])

    for algorithm in algorithms:
        if algorithm in ['XGB', 'xgb', 'XGBoost', 'xgboost', 'XGBOOST']:
            algorithm = 'xgb'
        elif algorithm in ['dnn', 'DNN', 'nn', 'NN']:
            algorithm = 'dnn'
    group_names = kwargs.get("group_names", [])
    alg_grp_dict = dict(zip(algorithms, group_names))

    # Read file located in base scope directory unless fully-qualified path is given
    if path_dataset[0] != '/':
        sources_path = str(BASE_DIR / path_dataset)
    else:
        sources_path = path_dataset

    # Currently supporting parquet, hdf5 and csv file formats
    if sources_path.endswith('.parquet'):
        sources = read_parquet(sources_path)
    elif sources_path.endswith('.h5'):
        sources = read_hdf(sources_path)
    elif sources_path.endswith('.csv'):
        sources = pd.read_csv(sources_path)
    else:
        raise ValueError("Sources must be stored in .parquet, .h5 or .csv format.")

    cols = sources.columns
    ra_vals = []
    dec_vals = []
    ztf_id_vals = []
    obj_id_vals = []

    # Search for common RA, Dec, ztf_id and obj_id column names
    for ra_name in ['ra', 'RA', 'R.A.', 'r.a.', 'radeg', 'ra_deg']:
        if ra_name in cols:
            ra_vals = sources[ra_name].values.tolist()
            sources.rename({ra_name: "ra"}, axis=1, inplace=True)
            break
    for dec_name in ['dec', 'DEC', 'Dec.', 'Decl.', 'dedeg', 'de_deg']:
        if dec_name in cols:
            dec_vals = sources[dec_name].values.tolist()
            sources.rename({dec_name: "dec"}, axis=1, inplace=True)
            break
    for ztf_id_name in ['ztf_id', 'ztfid', 'ZTFID', '_id']:
        if ztf_id_name in cols:
            ztf_id_vals = sources[ztf_id_name].values.tolist()
            break
    for obj_id_name in ['obj_id', 'OBJ_ID', 'object_id', 'objectID', 'name']:
        if obj_id_name in cols:
            print(f"Found object ID column: {obj_id_name}")
            print()
            obj_id_vals = sources[obj_id_name].values.tolist()
            sources.rename({obj_id_name: "obj_id"}, axis=1, inplace=True)
            break

    if not obj_id_vals:
        if ra_vals and dec_vals:
            print("No object ID column found. Generating coordinate-based names.")
            obj_id_vals = [
                radec_to_iau_name(ra_vals[i], dec_vals[i]) for i in range(len(sources))
            ]
        elif ztf_id_vals:
            print("No object ID column found. Using ZTF IDs for names.")
            obj_id_vals = [f"ZTF_{x}" for x in ztf_id_vals]
        else:
            raise ValueError(
                "Please provide either ra/dec or ZTF light curve ID columns."
            )
        sources["obj_id"] = obj_id_vals

    if not ztf_id_vals:
        print("Querying ZTF light curve IDs using coordinates...")
        print()

        # Must provide RA/Dec
        if (not ra_vals) or (not dec_vals):
            raise ValueError(
                "Could not find RA and/or Dec columns, and ZTF light curve IDs not provided."
            )

        queried_ids = get_cone_ids(
            obj_id_list=obj_id_vals,
            ra_list=ra_vals,
            dec_list=dec_vals,
            max_distance=cone_radius_arcsec,
            get_coords=True,
        )
        queried_ids.rename({"_id": "ztf_id"}, axis=1, inplace=True)
        merge_on = "obj_id"
    else:
        print("Querying ZTF light curve IDs using input IDs...")
        print()
        queried_ids = get_lightcurves_via_ids(
            kowalski_instances=kowalski_instances,
            ids=ztf_id_vals,
            catalog=source_catalog,
            get_ids_coords_only=True,
        )

        queried_ids = pd.DataFrame.from_dict(queried_ids)
        queried_ids.rename({"_id": "ztf_id"}, axis=1, inplace=True)
        merge_on = "ztf_id"

        if len(ztf_id_vals) != len(queried_ids):
            print(
                "Warning: discrepancies between length of input and queried IDs (possibly due to data release differences)."
            )
            print(
                "Recommended usage is to avoid using inputting a ztf_id column and use ra/dec coordinates instead."
            )
            print()

    sources = pd.merge(sources, queried_ids, on=merge_on)

    print(
        "Renaming object ID column to 'fritz_name' and dropping ra/dec columns for compatibility with feature generation script."
    )
    print()
    sources.rename({"obj_id": "fritz_name"}, axis=1, inplace=True)
    sources.drop(['ra', 'dec'], axis=1, inplace=True, errors="ignore")

    # Save sources
    if save_sources_filepath[0] == '/':
        save_sources_filepath = save_sources_filepath
    else:
        save_sources_filepath = str(BASE_DIR / save_sources_filepath)

    if save_sources_filepath.endswith(".parquet"):
        write_parquet(sources, save_sources_filepath)
    elif save_sources_filepath.endswith(".h5"):
        write_hdf(sources, save_sources_filepath)
    elif save_sources_filepath.endswith(".csv"):
        sources.to_csv(save_sources_filepath, index=False)
    else:
        raise ValueError("Must save sources in .parquet, .h5 or .csv format.")

    fg_dirname = f"{kwargs.get('dirname')}_{current_dt}"
    fg_filename = f"{kwargs.get('filename')}_{current_dt}"
    inference_field = f"{current_dt}_specific_ids"

    # Generate features
    print("Running feature generation...")
    generate_features.generate_features(
        source_catalog=kwargs.get("source_catalog"),
        alerts_catalog=kwargs.get("alerts_catalog"),
        gaia_catalog=kwargs.get("gaia_catalog"),
        bright_star_query_radius_arcsec=kwargs.get("bright_star_query_radius_arcsec"),
        xmatch_radius_arcsec=kwargs.get("xmatch_radius_arcsec"),
        limit=kwargs.get("query_size_limit"),
        period_algorithms=kwargs.get("period_algorithms"),
        period_batch_size=kwargs.get("period_batch_size"),
        doCPU=kwargs.get("doCPU"),
        doGPU=kwargs.get("doGPU"),
        samples_per_peak=kwargs.get("samples_per_peak"),
        doScaleMinPeriod=kwargs.get("doScaleMinPeriod"),
        doRemoveTerrestrial=kwargs.get("doRemoveTerrestrial"),
        Ncore=kwargs.get("Ncore"),
        field=kwargs.get("field"),
        ccd=kwargs.get("ccd"),
        quad=kwargs.get("quad"),
        min_n_lc_points=kwargs.get("min_n_lc_points"),
        min_cadence_minutes=kwargs.get("min_cadence_minutes"),
        dirname=fg_dirname,
        filename=fg_filename,
        doCesium=kwargs.get("doCesium"),
        doNotSave=kwargs.get("doNotSave"),
        stop_early=kwargs.get("stop_early"),
        doQuadrantFile=kwargs.get("doQuadrantFile"),
        quadrant_file=kwargs.get("quadrant_file"),
        quadrant_index=kwargs.get("quadrant_index"),
        doSpecificIDs=True,
        skipCloseSources=kwargs.get("skipCloseSources"),
        top_n_periods=kwargs.get("top_n_periods"),
        max_freq=kwargs.get("max_freq"),
        fg_dataset=save_sources_filepath,
        max_timestamp_hjd=kwargs.get("max_timestamp_hjd"),
    )

    for key in alg_grp_dict.keys():
        paths_models = []
        model_class_names = []
        alg = key
        grp = alg_grp_dict[alg]

        group_path = BASE_DIR / f'models_{alg}' / grp
        gen = os.walk(group_path)
        model_tags = [tag[1] for tag in gen]
        model_tags = model_tags[0]

        # Identify trained models for given algorithm
        if alg == 'dnn':
            for tag in model_tags:
                tag_file_gen = (group_path / tag).glob('*.h5')
                most_recent_file = max(
                    [file for file in tag_file_gen], key=os.path.getctime
                ).name
                paths_models.append(
                    f'{str(BASE_DIR)}/models_{alg}/{grp}/{tag}/{most_recent_file}'
                )
                model_class_names.append(f'{tag}')
        elif alg == 'xgb':
            for tag in model_tags:
                tag_file_gen = (group_path / tag).glob('*.json')
                most_recent_file = max(
                    [file for file in tag_file_gen], key=os.path.getctime
                ).name
                paths_models.append(
                    f'{str(BASE_DIR)}/models_{alg}/{grp}/{tag}/{most_recent_file}'
                )
                model_class_names.append(f'{tag}')

        # Run inference using trained models
        inference.run_inference(
            paths_models=paths_models,
            model_class_names=model_class_names,
            field=inference_field,
            whole_field=True,
            flag_ids=kwargs.get("flag_ids"),
            xgb_model=(alg == 'xgb'),
            verbose=kwargs.get("verbose"),
            time_run=kwargs.get("time_run"),
            write_csv=kwargs.get("write_csv"),
            float_convert_types=kwargs.get("float_convert_types"),
            feature_stats=kwargs.get("feature_stats"),
            scale_features=kwargs.get("scale_features"),
            trainingSet=kwargs.get("trainingSet"),
            feature_directory=fg_dirname,
            feature_file_prefix=kwargs.get("feature_file_prefix"),
            period_suffix=kwargs.get("period_suffix"),
            no_write_metadata=kwargs.get("no_write_metadata"),
            batch_size=kwargs.get("batch_size"),
        )

    return current_dt


def get_parser():
    parser_generate_features = generate_features.get_parser()
    parser_inference = inference.get_parser_minimal()

    parser = argparse.ArgumentParser(
        parents=[parser_generate_features, parser_inference]
    )
    parser.add_argument(
        "--path-dataset",
        type=str,
        default=None,
        help="path (from base scope directory or fully qualified) to parquet, hdf5 or csv file containing specific sources",
    )
    parser.add_argument(
        "--cone-radius-arcsec",
        type=float,
        default=2.0,
        help="radius of cone search query for ZTF lightcurve IDs, if inputting ra/dec",
    )
    parser.add_argument(
        "--save-sources-filepath",
        type=str,
        default="sources.parquet",
        help="path to parquet, hdf5 or csv file to save specific sources",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs='+',
        help="ML algorithms to run (currently dnn/xgb)",
    )
    parser.add_argument(
        "--group-names",
        type=str,
        nargs='+',
        help="group names of trained models (with order corresponding to --algorithms input)",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    run_scope_local(**vars(args))
