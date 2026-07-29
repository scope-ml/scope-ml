#!/usr/bin/env python
"""
Download a Rubin Visit table via TAP and cache it as parquet.

The DP2/EDP2 ``ForcedSourceOnDiaObject`` export carries no timestamp, only
a ``visit`` ID, so ``RubinLocalDP2Client`` needs a local copy of the visit
table to convert visits into ``expMidptMJD``. This script fetches it.

Example
-------
    python tools/fetch_rubin_visits.py --release dp2 \
        --output /fred/oz480/mcoughli/EDP2/Visit.parquet
"""

import argparse
import os
import pathlib

from scope.utils import parse_load_config

BASE_DIR = pathlib.Path.cwd()

# Columns worth keeping: expMidptMJD is required, the rest are cheap and
# useful for cadence / airmass diagnostics.
VISIT_COLUMNS = [
    "visit",
    "band",
    "physical_filter",
    "ra",
    "dec",
    "expTime",
    "expMidptMJD",
    "obsStartMJD",
    "airmass",
]


def fetch_visits(release="dp2", columns=None, config=None, timeout=None):
    """
    Query ``<release>.Visit`` over TAP and return it as a DataFrame.

    Parameters
    ----------
    release : str, optional
        Data release prefix, e.g. ``'dp1'`` or ``'dp2'`` (default ``'dp2'``).
    columns : list of str, optional
        Columns to select (default :data:`VISIT_COLUMNS`).
    config : dict, optional
        Rubin config section. Defaults to the loaded ``config['rubin']``.
    timeout : int, optional
        Async job timeout in seconds.

    Returns
    -------
    pandas.DataFrame
    """
    import pyvo
    import requests

    if config is None:
        config = parse_load_config().get("rubin", {})

    token = os.environ.get("RUBIN_TAP_TOKEN") or config.get("token")
    if not token:
        raise ValueError(
            "A Rubin TAP token is required. Set rubin.token in config.yaml "
            "or the RUBIN_TAP_TOKEN environment variable."
        )

    columns = columns or VISIT_COLUMNS
    timeout = timeout or config.get("timeout", 300)
    tap_url = config.get("tap_url", "https://data.lsst.cloud/api/tap")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    service = pyvo.dal.TAPService(tap_url, session=session)

    query = f"SELECT {', '.join(columns)} FROM {release}.Visit"
    print(f"Submitting: {query}")

    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=timeout)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"TAP job ended in phase {job.phase}")

    df = job.fetch_result().to_table().to_pandas()
    print(f"Fetched {len(df)} visits.")
    return df


def get_parser():
    parser = argparse.ArgumentParser(
        description="Download a Rubin Visit table via TAP and save it as parquet."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="dp2",
        help="Data release prefix for the TAP table (default dp2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (default: Visit.parquet in current directory)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Async TAP job timeout in seconds",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    df = fetch_visits(release=args.release, timeout=args.timeout)

    output_path = args.output or str(BASE_DIR / "Visit.parquet")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(
        f"Saved {len(df)} visits to {output_path} "
        f"(MJD {df['expMidptMJD'].min():.3f} - {df['expMidptMJD'].max():.3f})"
    )


if __name__ == "__main__":
    main()
