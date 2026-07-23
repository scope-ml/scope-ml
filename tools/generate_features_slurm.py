#!/usr/bin/env python
import argparse
import pathlib
import os
from penquins import Kowalski
import numpy as np
import json
from scope.utils import parse_load_config
from tools.generate_features import get_parser


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
alerts_catalog = config['kowalski']['collections']['alerts']
gaia_catalog = config['kowalski']['collections']['gaia']
ext_catalog_info = config['feature_generation']['external_catalog_features']
cesium_feature_list = config['feature_generation']['cesium_features']
path_to_features = config['feature_generation']['path_to_features']

if path_to_features is not None:
    BASE_DIR = pathlib.Path(path_to_features)


def check_quads_for_sources(
    fields: list = np.arange(0, 2000),
    catalog: str = source_catalog,
    count_sources: bool = False,
    minobs: int = 0,
    save: bool = False,
    filename: str = 'catalog_completeness',
):
    """
    Check ZTF field/ccd/quadrant combos for any sources. By default, lists any quadrants that have at least one source.

    :param fields: list of integer field numbers to query (list)
    :param catalog: name of source catalog to query (str)
    :param count_sources: if set, count number of sources per quad and return (bool)
    :param minobs: minimum number of observations needed to count a source (int)
    :param save: if set, save results dictionary in json format (bool)
    :param filename: filename of saved results (str)

    :return field_dct: dictionary containing quadrants having at least one source (and optionally, number of sources per quad)
    :return has_sources: boolean stating whether each field in fields has sources
    :return missing_ccd_quad: boolean stating whether each field in fields has no sources in at least one ccd/quad
    """

    parser = get_check_quads_parser()
    args, _ = parser.parse_known_args()

    fields = args.fields
    catalog = args.catalog
    count_sources = args.count_sources
    minobs = args.minobs
    save = args.save
    filename = args.filename

    running_total_sources = 0
    has_sources = np.zeros(len(fields), dtype=bool)
    missing_ccd_quad = np.zeros(len(fields), dtype=bool)
    field_dct = {}

    if save and (os.path.isfile(BASE_DIR / f'{filename}.json')):
        with open(BASE_DIR / f'{filename}.json', 'r') as f:
            field_dct = json.load(f)

    for idx, field in enumerate(fields):
        print('Running field %d' % int(field))
        except_count = 0
        # Run minimal query to determine if sources exist in field
        q = {
            "query_type": "find",
            "query": {
                "catalog": catalog,
                "filter": {
                    'field': {'$eq': int(field)},
                },
                "projection": {"_id": 1},
            },
            "kwargs": {"limit": 1},
        }
        responses = kowalski_instances.query(q)

        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    data = response.get("data")

        if len(data) > 0:
            has_sources[idx] = True
        else:
            continue

        if has_sources[idx]:
            print(f'Field {field} has sources...')
            field_dct[str(field)] = {}
            for ccd in range(1, 17):
                if count_sources:
                    quads = {}
                else:
                    quads = []
                for quadrant in range(1, 5):
                    fltr = {
                        'field': {'$eq': int(field)},
                        'ccd': {'$eq': int(ccd)},
                        'quad': {'$eq': int(quadrant)},
                    }
                    if minobs > 0:
                        fltr.update({'nobs': {'$gte': int(minobs)}})

                    # Another minimal query for each ccd/quad combo
                    q = {
                        "query_type": 'count_documents',
                        "query": {
                            "catalog": catalog,
                            "filter": fltr,
                        },
                    }
                    responses = kowalski_instances.query(q)

                    for name in responses.keys():
                        if len(responses[name]) > 0:
                            response = responses[name]
                            if response.get("status", "error") == "success":
                                data = response.get("data")

                    if data > 0:
                        if count_sources:
                            quads[str(quadrant)] = data
                            running_total_sources += data
                        else:
                            quads += [quadrant]

                    else:
                        except_count += 1

                if len(quads) > 0:
                    field_dct[str(field)].update({str(ccd): quads})

        print(f"{64 - except_count} ccd/quad combos")
        if except_count > 0:
            missing_ccd_quad[idx] = True

        if save:
            with open(BASE_DIR / f'{filename}.json', 'w') as f:
                json.dump(field_dct, f)

    print(f"Sources found in {np.sum(has_sources)} fields.")
    if count_sources:
        print(f"Found {running_total_sources} sources.")

    return field_dct, has_sources, missing_ccd_quad


def get_check_quads_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields",
        type=int,
        nargs="+",
        default=np.arange(0, 2000),
        help="list of integer field numbers to query",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=source_catalog,
        help="name of source catalog to query",
    )
    parser.add_argument(
        "--count-sources",
        action="store_true",
        help="if set, count number of sources per quad and return",
    )
    parser.add_argument(
        "--minobs",
        type=int,
        default=0,
        help="minimum number of observations needed to count a source",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="if set, save results dictionary in json format",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="catalog_completeness",
        help="filename of saved results",
    )

    return parser


def get_slurm_parser():

    fg_parser = get_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[fg_parser])
    parser.add_argument(
        "--job-name",
        type=str,
        default='ztf_fg',
        help="job name",
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default='Expanse',
        help="Name of HPC cluster",
    )
    parser.add_argument(
        "--partition-type",
        type=str,
        default='gpu-shared',
        help="Partition name to request for computing",
    )
    parser.add_argument(
        "--submit-partition-type",
        type=str,
        default='shared',
        help="Partition name to request for job submission",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to request",
    )
    parser.add_argument(
        "--memory-GB",
        type=int,
        default=180,
        help="Memory allocation to request for computing",
    )
    parser.add_argument(
        "--submit-memory-GB",
        type=int,
        default=16,
        help="Memory allocation to request for job submission",
    )
    parser.add_argument(
        "--time",
        type=str,
        default='48:00:00',
        help="Walltime for instance",
    )
    parser.add_argument(
        "--mail-user",
        type=str,
        default='healyb@umn.edu',
        help="contact email address",
    )
    parser.add_argument(
        "--account-name",
        type=str,
        default='umn131',
        help="Name of account with current HPC allocation",
    )
    parser.add_argument(
        "--python-env-name",
        type=str,
        default='scope-env',
        help="Name of python environment to activate",
    )
    parser.add_argument(
        "--generateQuadrantFile",
        action='store_true',
        default=False,
        help="if set, generate a list of fields/ccd/quads and job numbers, save to slurm.dat",
    )
    parser.add_argument(
        "--field-list",
        type=int,
        nargs='+',
        default=None,
        help="space-separated list of fields for which to generate quadrant file. If None, all populated fields included.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=20,
        help="Max number of instances to run in parallel",
    )
    parser.add_argument(
        "--wait-time-minutes",
        type=float,
        default=5.0,
        help="Time to wait between job status checks",
    )
    parser.add_argument(
        "--doSubmitLoop",
        action="store_true",
        default=False,
        help="If set, loop to initiate instances until out of jobs (hard on Kowalski)",
    )
    parser.add_argument(
        "--runParallel",
        action="store_true",
        default=False,
        help="If set, run jobs in parallel using slurm. Otherwise, run in series on a single instance.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="bhealy",
        help="HPC username",
    )
    parser.add_argument(
        "--submit-interval-minutes",
        type=float,
        default=1.0,
        help="Time to wait between job submissions, minutes",
    )

    return parser


def main():
    parser = get_slurm_parser()
    args, _ = parser.parse_known_args()

    if not (args.doCPU or args.doGPU):
        print("--doCPU or --doGPU required")
        exit(0)

    if args.doQuadrantFile and args.doSpecificIDs:
        print("Choose one of --doQuadrantFile or --doSpecificIDs")
        exit(0)

    source_catalog = args.source_catalog
    alerts_catalog = args.alerts_catalog
    gaia_catalog = args.gaia_catalog
    bright_star_query_radius_arcsec = args.bright_star_query_radius_arcsec
    xmatch_radius_arcsec = args.xmatch_radius_arcsec
    query_size_limit = args.query_size_limit
    period_batch_size = args.period_batch_size
    doCPU = args.doCPU
    doGPU = args.doGPU
    samples_per_peak = args.samples_per_peak
    doScaleMinPeriod = args.doScaleMinPeriod
    doRemoveTerrestrial = args.doRemoveTerrestrial
    Ncore = args.Ncore
    field = args.field
    ccd = args.ccd
    quad = args.quad
    min_n_lc_points = args.min_n_lc_points
    min_cadence_minutes = args.min_cadence_minutes
    dirname = args.dirname
    filename = args.filename
    doCesium = args.doCesium
    doFlareDetection = args.doFlareDetection
    doDRW = args.doDRW
    doNotSave = args.doNotSave
    stop_early = args.stop_early
    doSpecificIDs = args.doSpecificIDs
    skipCloseSources = args.skipCloseSources
    top_n_periods = args.top_n_periods
    max_freq = args.max_freq
    max_timestamp_hjd = args.max_timestamp_hjd

    if doCPU:
        cpu_gpu_flag = "--doCPU"
    elif doGPU:
        cpu_gpu_flag = "--doGPU"

    extra_flags = []
    if doScaleMinPeriod:
        extra_flags.append("--doScaleMinPeriod")
    if doRemoveTerrestrial:
        extra_flags.append("--doRemoveTerrestrial")
    if doCesium:
        extra_flags.append("--doCesium")
    if doFlareDetection:
        extra_flags.append("--doFlareDetection")
    if doDRW:
        extra_flags.append("--doDRW")
    if doNotSave:
        extra_flags.append("--doNotSave")
    if stop_early:
        extra_flags.append("--stop-early")
    if doSpecificIDs:
        extra_flags.append("--doSpecificIDs")
    if skipCloseSources:
        extra_flags.append("--skipCloseSources")
    extra_flags = " ".join(extra_flags)

    dirpath = BASE_DIR / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, 'slurm')
    if not os.path.isdir(slurmDir):
        os.makedirs(slurmDir)

    logsDir = os.path.join(dirpath, 'logs')
    if not os.path.isdir(logsDir):
        os.makedirs(logsDir)

    quadrantfile = os.path.join(slurmDir, args.quadrant_file)
    if args.generateQuadrantFile:
        if args.field_list is None:
            field_dct, _, _ = check_quads_for_sources(
                catalog=source_catalog,
            )
        else:
            field_dct, _, _ = check_quads_for_sources(
                fields=args.field_list,
                catalog=source_catalog,
            )

        job_number = 0

        fid = open(quadrantfile, 'w')
        for field in field_dct.keys():
            for ccd in field_dct[field].keys():
                for quad in field_dct[field][ccd]:
                    fid.write(
                        '%d %d %d %d\n'
                        % (int(job_number), int(field), int(ccd), int(quad))
                    )
                    job_number += 1
        fid.close()

    # Main script to run feature generation
    fid = open(os.path.join(slurmDir, 'slurm.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={args.job_name}.job\n')
    fid.write(f'#SBATCH --output={dirname}/logs/{args.job_name}_%A_%a.out\n')
    fid.write(f'#SBATCH --error={dirname}/logs/{args.job_name}_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.partition_type}\n')
    fid.write(f'#SBATCH --nodes {args.nodes}\n')
    fid.write(f'#SBATCH --ntasks-per-node {args.Ncore}\n')
    fid.write(f'#SBATCH --gpus {args.gpus}\n')
    fid.write(f'#SBATCH --mem {args.memory_GB}G\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        fid.write('module add gpu/0.15.4\n')
        fid.write('module add cuda\n')
        fid.write(f'source activate {args.python_env_name}\n')
    elif args.cluster_name in ['Delta', 'delta', 'DELTA']:
        fid.write('module purge\n')
        fid.write('module add anaconda3_gpu\n')
        fid.write('module add cuda\n')
        fid.write('module add gcc-runtime\n')
        fid.write(f'source activate {args.python_env_name}\n')

    if args.doQuadrantFile:
        qid = '$QID'
        if args.quadrant_index is not None:
            qid = args.quadrant_index
        fid.write(
            'generate-features --source-catalog %s --alerts-catalog %s --gaia-catalog %s --bright-star-query-radius-arcsec %s --xmatch-radius-arcsec %s --query-size-limit %s --period-batch-size %s --samples-per-peak %s --Ncore %s --min-n-lc-points %s --min-cadence-minutes %s --dirname %s --filename %s --top-n-periods %s --max-freq %s --max-timestamp-hjd %s --doQuadrantFile --quadrant-file %s --quadrant-index %s %s %s\n'
            % (
                source_catalog,
                alerts_catalog,
                gaia_catalog,
                bright_star_query_radius_arcsec,
                xmatch_radius_arcsec,
                query_size_limit,
                period_batch_size,
                samples_per_peak,
                Ncore,
                min_n_lc_points,
                min_cadence_minutes,
                dirname,
                filename,
                top_n_periods,
                max_freq,
                max_timestamp_hjd,
                args.quadrant_file,
                qid,
                cpu_gpu_flag,
                extra_flags,
            )
        )
    else:
        fid.write(
            'generate-features --source-catalog %s --alerts-catalog %s --gaia-catalog %s --bright-star-query-radius-arcsec %s --xmatch-radius-arcsec %s --query-size-limit %s --period-batch-size %s --samples-per-peak %s --Ncore %s --field %s --ccd %s --quad %s --min-n-lc-points %s --min-cadence-minutes %s --dirname %s --filename %s --top-n-periods %s --max-freq %s --max-timestamp-hjd %s %s %s\n'
            % (
                source_catalog,
                alerts_catalog,
                gaia_catalog,
                bright_star_query_radius_arcsec,
                xmatch_radius_arcsec,
                query_size_limit,
                period_batch_size,
                samples_per_peak,
                Ncore,
                field,
                ccd,
                quad,
                min_n_lc_points,
                min_cadence_minutes,
                dirname,
                filename,
                top_n_periods,
                max_freq,
                max_timestamp_hjd,
                cpu_gpu_flag,
                extra_flags,
            )
        )
    fid.close()

    # Secondary script to manage job submission using generate_features_job_submission.py
    # (Python code can also be run interactively)
    fid = open(os.path.join(slurmDir, 'slurm_submission.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write('#SBATCH --job-name=submit_jobs.job\n')
    fid.write(f'#SBATCH --output={dirname}/logs/{args.job_name}_submit_%A_%a.out\n')
    fid.write(f'#SBATCH --error={dirname}/logs/{args.job_name}_submit_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.submit_partition_type}\n')
    fid.write(f'#SBATCH --mem {args.submit_memory_GB}G\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        fid.write('module add slurm\n')
        fid.write(f'source activate {args.python_env_name}\n')
    elif args.cluster_name in ['Delta', 'delta', 'DELTA']:
        fid.write('module purge\n')
        fid.write('module add anaconda3_cpu\n')
        fid.write(f'source activate {args.python_env_name}\n')

    if not args.doSubmitLoop:
        if args.runParallel:
            fid.write(
                'generate-features-job-submission --dirname %s --filename %s --doSubmit --runParallel --max-instances %s --wait-time-minutes %s --user %s --submit-interval-minutes %s\n'
                % (
                    dirpath,
                    filename,
                    args.max_instances,
                    args.wait_time_minutes,
                    args.user,
                    args.submit_interval_minutes,
                )
            )
        else:
            fid.write(
                'generate-features-job-submission --dirname %s --filename %s --doSubmit --max-instances %s --wait-time-minutes %s --user %s --submit-interval-minutes %s\n'
                % (
                    dirpath,
                    filename,
                    args.max_instances,
                    args.wait_time_minutes,
                    args.user,
                    args.submit_interval_minutes,
                )
            )
    else:
        if args.runParallel:
            fid.write(
                'generate-features-job-submission --dirname %s --filename %s --doSubmitLoop --runParallel\n'
                % (
                    dirpath,
                    filename,
                )
            )
        else:
            fid.write(
                'generate-features-job-submission --dirname %s --filename %s --doSubmitLoop\n'
                % (
                    dirpath,
                    filename,
                )
            )
    fid.close()
