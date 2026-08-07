#!/usr/bin/env python
"""
Source discovery tool for Rubin LSST data.

Parallel to tools/get_quad_ids.py, this script finds Rubin object IDs
via cone search or reads them from a CSV file.
"""

import argparse
import os
import pathlib
import pandas as pd
from scope.utils import parse_load_config
from scope.surveys.rubin import make_rubin_client

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

# Rubin TAP token from environment variable
rubin_token_env = os.environ.get("RUBIN_TAP_TOKEN")
if rubin_token_env is not None:
    config["rubin"]["token"] = rubin_token_env


def _get_rubin_client():
    """Create a Rubin client from the current config (local or TAP)."""
    rubin_config = config.get("rubin", {})
    return make_rubin_client(config=rubin_config)


def get_rubin_objects_by_cone(ra, dec, radius_arcsec, limit=10000, client=None):
    """
    Cone search on the Rubin dp1.Object table.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    radius_arcsec : float
        Search radius in arcseconds.
    limit : int, optional
        Maximum number of objects to return.
    client : RubinTAPClient, optional
        Pre-initialized client. If None, one will be created.

    Returns
    -------
    dict
        {objectId: {'coord_ra': ra, 'coord_dec': dec}, ...}
    """
    if client is None:
        client = _get_rubin_client()

    print(
        f"Searching for Rubin objects within {radius_arcsec}\" of "
        f"(RA={ra}, Dec={dec})..."
    )
    objects = client.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
    print(f"Found {len(objects)} objects.")
    return objects


def get_rubin_objects_from_file(filepath):
    """
    Read Rubin object IDs from a CSV file.

    The CSV file must have an 'objectId' column. Optionally it can have
    'coord_ra' and 'coord_dec' columns.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    dict
        {objectId: {'coord_ra': ra, 'coord_dec': dec}, ...}
        If coordinates are not in the file, they are set to None.
    """
    df = pd.read_csv(filepath)

    if "objectId" not in df.columns:
        raise ValueError("CSV file must have an 'objectId' column.")

    has_coords = "coord_ra" in df.columns and "coord_dec" in df.columns

    # Use column arrays directly instead of iterrows() to avoid
    # int64 -> float64 precision loss on 18-digit Rubin objectIds.
    oid_arr = df["objectId"].values
    if has_coords:
        ra_arr = df["coord_ra"].values
        dec_arr = df["coord_dec"].values

    objects = {}
    for i in range(len(df)):
        oid = int(oid_arr[i])
        if has_coords:
            objects[oid] = {
                "coord_ra": float(ra_arr[i]),
                "coord_dec": float(dec_arr[i]),
            }
        else:
            objects[oid] = {"coord_ra": None, "coord_dec": None}

    print(f"Loaded {len(objects)} object IDs from {filepath}.")
    return objects


def get_parser():
    parser = argparse.ArgumentParser(
        description="Discover Rubin LSST object IDs via cone search or CSV file."
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Right ascension in degrees (for cone search)",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Declination in degrees (for cone search)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=30.0,
        help="Search radius in arcseconds (default 30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum number of objects to return (default 10000)",
    )
    parser.add_argument(
        "--objectid-file",
        type=str,
        default=None,
        help="Path to CSV file with objectId column",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: rubin_ids.csv in current directory)",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if args.objectid_file is not None:
        # Read from file
        objects = get_rubin_objects_from_file(args.objectid_file)
    elif args.ra is not None and args.dec is not None:
        # Cone search
        objects = get_rubin_objects_by_cone(
            args.ra, args.dec, args.radius, limit=args.limit
        )
    else:
        parser.error("Must specify either --ra/--dec or --objectid-file")
        return

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(BASE_DIR / "rubin_ids.csv")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = []
    for oid, info in objects.items():
        rows.append(
            {
                "objectId": oid,
                "coord_ra": info.get("coord_ra"),
                "coord_dec": info.get("coord_dec"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} object IDs to {output_path}")


if __name__ == "__main__":
    main()
