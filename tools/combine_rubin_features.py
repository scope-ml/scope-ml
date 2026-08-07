#!/usr/bin/env python
"""
Merge per-chunk Rubin feature-generation outputs into a single file.

Each SLURM array task from generate-features-rubin-slurm writes one
parquet file (gen_features_rubin_<TASK_ID>.parquet); this concatenates
all of them into a single combined parquet.

Usage:
    combine-rubin-features \
        --input-dir generated_features_rubin \
        --output generated_features_rubin/dp1_features_combined.parquet
"""

import argparse
import glob
import os
import pandas as pd


def combine_features(input_dir, output, pattern="gen_features_rubin_*.parquet"):
    """
    Concatenate per-chunk Rubin feature parquet files into one output file.

    Parameters
    ----------
    input_dir : str
        Directory containing the per-chunk parquet files.
    output : str
        Path to write the combined parquet file.
    pattern : str, optional
        Glob pattern (relative to input_dir) matching per-chunk files
        (default 'gen_features_rubin_*.parquet').

    Returns
    -------
    pandas.DataFrame
        The combined features.
    """
    chunk_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if len(chunk_files) == 0:
        raise ValueError(f"No files matching {pattern} found in {input_dir}.")

    print(f"Found {len(chunk_files)} chunk file(s) in {input_dir}")

    dfs = []
    for cf in chunk_files:
        df = pd.read_parquet(cf)
        if len(df) > 0:
            dfs.append(df)
        else:
            print(f"  Skipping empty file: {cf}")

    if len(dfs) == 0:
        print("No data to combine.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    combined.to_parquet(output)

    print(f"Combined {len(combined):,} sources from {len(dfs)} file(s) into {output}")

    return combined


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge per-chunk Rubin feature-generation outputs into one file."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the per-chunk parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the combined parquet file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="gen_features_rubin_*.parquet",
        help="Glob pattern (relative to --input-dir) matching per-chunk files",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    combine_features(
        input_dir=args.input_dir,
        output=args.output,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
