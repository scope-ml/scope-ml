#!/usr/bin/env python
"""
EVT calibration of DRW anomaly scores.

Population-level step of the AGN/anomaly feature set: pools the drw_max_z
values produced by generate_features.py (--doDRW) across feature files, fits
a generalized Pareto distribution (GPD) to the tail via peaks-over-threshold
(POT), and converts each source's drw_max_z into

    anomaly_score  -log10 of the exceedance probability P(max_z > z)
    anomaly_flag   1 if drw_max_z exceeds the detection threshold derived
                   from the target false-alarm rate, else 0

Calibration is performed per filter (per ZTF band) by default and saved to a
JSON file so that later runs can score new sources against the same frozen
threshold. By default the tool only reports; pass --apply to write the
anomaly_score/anomaly_flag columns back into the feature files.

Usage:
    python tools/evt_calibration.py --features-dir generated_features
    python tools/evt_calibration.py --features-dir generated_features --apply
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import genpareto

DEFAULT_POT_PERCENTILE = 95.0
DEFAULT_P_TARGET = 0.01

N_QUANTILE_GRID = 201


def _load_evt_config_defaults():
    """Read feature_generation.evt from the SCoPe config, if available."""
    try:
        from scope.utils import parse_load_config

        config = parse_load_config()
        return config.get('feature_generation', {}).get('evt', {}) or {}
    except Exception:
        return {}


def fit_gpd(max_z, pot_percentile=DEFAULT_POT_PERCENTILE):
    """Fit a GPD to the POT exceedances of a pooled max_z sample.

    Returns a calibration dict with the GPD parameters, threshold u, sample
    counts, and an empirical quantile grid used to score sub-threshold values.
    """
    vals = np.asarray(max_z, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 20:
        raise ValueError(f"Need at least 20 finite max_z values, got {n}")

    u = float(np.percentile(vals, pot_percentile))
    excesses = vals[vals > u] - u
    nu = len(excesses)
    if nu < 5:
        raise ValueError(f"Only {nu} exceedances above threshold {u:.3f}")

    # Location fixed at zero: excesses start at zero by definition
    shape, _, scale = genpareto.fit(excesses, floc=0)

    levels = np.linspace(0.0, 1.0, N_QUANTILE_GRID)
    quantiles = np.quantile(vals, levels)

    return {
        'pot_percentile': float(pot_percentile),
        'u': u,
        'shape': float(shape),
        'scale': float(scale),
        'n': int(n),
        'nu': int(nu),
        'quantile_levels': levels.tolist(),
        'quantile_values': np.asarray(quantiles, dtype=float).tolist(),
    }


def detection_threshold(calib, p_target=DEFAULT_P_TARGET):
    """Invert the GPD exceedance probability for a target false-alarm rate."""
    u, shape, scale = calib['u'], calib['shape'], calib['scale']
    n, nu = calib['n'], calib['nu']
    if nu == 0:
        return np.inf
    ratio = (n * p_target) / nu
    if abs(shape) < 1e-12:
        return u + scale * np.log(1.0 / ratio)
    return u + (scale / shape) * (ratio ** (-shape) - 1.0)


def exceedance_prob(z, calib):
    """P(max_z > z): GPD tail above u, empirical quantile grid below."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    p = np.full(z.shape, np.nan)

    u, shape, scale = calib['u'], calib['shape'], calib['scale']
    frac_u = calib['nu'] / calib['n']
    levels = np.asarray(calib['quantile_levels'])
    quantiles = np.asarray(calib['quantile_values'])

    finite = np.isfinite(z)

    below = finite & (z <= u)
    cdf = np.interp(z[below], quantiles, levels)
    p[below] = 1.0 - cdf

    above = finite & (z > u)
    zz = z[above] - u
    if abs(shape) < 1e-12:
        tail = np.exp(-zz / scale)
    else:
        arg = 1.0 + shape * zz / scale
        tail = np.where(arg > 0, arg ** (-1.0 / shape), 0.0)
    p[above] = frac_u * tail

    return np.clip(p, 1e-300, 1.0)


def anomaly_score(z, calib):
    """-log10 exceedance probability; higher = more anomalous. NaN-safe."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    score = np.full(z.shape, np.nan)
    finite = np.isfinite(z)
    score[finite] = -np.log10(exceedance_prob(z[finite], calib))
    return score


def collect_feature_files(features_dir, pattern):
    paths = sorted(pathlib.Path(features_dir).rglob(pattern))
    usable = []
    for path in paths:
        try:
            columns = pd.read_parquet(path, columns=['drw_max_z']).columns
        except Exception:
            continue
        if 'drw_max_z' in columns:
            usable.append(path)
    return usable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="directory containing generated feature files",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.parquet",
        help="glob pattern for feature files (searched recursively)",
    )
    parser.add_argument(
        "--pot-percentile",
        type=float,
        default=None,
        help="peaks-over-threshold percentile of the pooled max_z distribution "
        "(default: feature_generation.evt in config, else 95)",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=None,
        help="target false-alarm rate for the detection threshold "
        "(default: feature_generation.evt in config, else 0.01)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="path to config file providing feature_generation.evt defaults",
    )
    parser.add_argument(
        "--no-per-filter",
        action="store_true",
        default=False,
        help="pool all filters into a single calibration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path for the calibration JSON (default: <features-dir>/evt_calibration.json)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="write anomaly_score/anomaly_flag columns back into the feature files",
    )
    args = parser.parse_args()

    config_defaults = _load_evt_config_defaults()
    pot_percentile = (
        args.pot_percentile
        if args.pot_percentile is not None
        else config_defaults.get('pot_percentile', DEFAULT_POT_PERCENTILE)
    )
    p_target = (
        args.p_target
        if args.p_target is not None
        else config_defaults.get('p_target', DEFAULT_P_TARGET)
    )

    features_dir = pathlib.Path(args.features_dir)
    paths = collect_feature_files(features_dir, args.file_pattern)
    if not paths:
        print(
            f"No feature files with a drw_max_z column found under {features_dir} "
            f"(pattern {args.file_pattern}). Run generate-features with --doDRW first."
        )
        return

    print(f"Found {len(paths)} feature file(s) with drw_max_z.")

    frames = [pd.read_parquet(path) for path in paths]
    pooled = pd.concat(frames, ignore_index=True)

    per_filter = (not args.no_per_filter) and ('filter' in pooled.columns)
    if per_filter:
        groups = {
            int(flt): grp['drw_max_z'].values
            for flt, grp in pooled.groupby('filter')
        }
    else:
        groups = {'all': pooled['drw_max_z'].values}

    calibration = {
        'p_target': p_target,
        'per_filter': per_filter,
        'groups': {},
    }

    for key, values in groups.items():
        try:
            calib = fit_gpd(values, pot_percentile=pot_percentile)
        except ValueError as e:
            print(f"Group {key}: calibration failed ({e}); skipping.")
            continue
        threshold = float(detection_threshold(calib, p_target))
        calib['detection_threshold'] = threshold
        calibration['groups'][str(key)] = calib

        n_flagged = int(np.nansum(values > threshold))
        print(
            f"Group {key}: n={calib['n']} u={calib['u']:.3f} "
            f"shape={calib['shape']:.3f} scale={calib['scale']:.3f} "
            f"threshold={threshold:.3f} flagged={n_flagged} "
            f"({100.0 * n_flagged / max(calib['n'], 1):.2f}%)"
        )

    if not calibration['groups']:
        print("No group could be calibrated; nothing written.")
        return

    output = (
        pathlib.Path(args.output)
        if args.output is not None
        else features_dir / 'evt_calibration.json'
    )
    with open(output, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"Calibration saved to {output}")

    if not args.apply:
        print("Dry run: pass --apply to write anomaly_score/anomaly_flag columns.")
        return

    for path, df in zip(paths, frames):
        score = np.full(len(df), np.nan)
        flag = np.full(len(df), np.nan)

        if per_filter:
            keys = df['filter'].astype('Int64')
        else:
            keys = pd.Series(['all'] * len(df))

        for key, calib in calibration['groups'].items():
            if per_filter:
                sel = (keys.astype(str) == key).values
            else:
                sel = np.ones(len(df), dtype=bool)
            if not sel.any():
                continue
            z = df.loc[sel, 'drw_max_z'].values
            score[sel] = anomaly_score(z, calib)
            finite = np.isfinite(z)
            group_flag = np.full(z.shape, np.nan)
            group_flag[finite] = (
                z[finite] > calib['detection_threshold']
            ).astype(float)
            flag[sel] = group_flag

        df['anomaly_score'] = score
        df['anomaly_flag'] = flag
        df.to_parquet(path, index=False)
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
