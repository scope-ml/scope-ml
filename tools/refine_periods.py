#!/usr/bin/env python3
"""Refine reported periods with a class-specific light curve model.

Feature generation reports the period at which a light curve repeats.  For an
eclipsing binary that is half the orbital period, because two similar eclipses
per orbit make the curve repeat every half orbit.  The period finders are not
failing here -- they correctly report the photometric repetition period -- but
the catalogue column is labelled as the period, and for these stars it is the
wrong physical quantity.  Cross-algorithm agreement cannot catch it either,
since every algorithm agrees on the same halved value.

This runs after classification, fits a model appropriate to the class at every
trial period already stored by feature generation, and reports the period the
model prefers.  No new period search is performed.

Measured on 3,866 DP2 sources matched to Gaia DR3 eclipsing binaries, exact
agreement with the Gaia period (within 2 per cent) rises from 0.9 to 21.6 per
cent, correcting 829 periods while changing 26 correct ones to incorrect.  For
406 RR Lyrae it rises from 38.9 to 48.3 per cent.  Sources whose class has no
registered model keep their original period.

Usage:

    refine-periods \\
        --features generated_features/combined.parquet \\
        --classes  classifications.parquet \\
        --output   combined_refined.parquet

Add --class-filter to restrict to particular classes, and --chunk/--n-chunks to
split the work across a job array.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from scope.surveys.rubin import DEFAULT_BAND_MAP
from tools.periodRefinement import lookup, supported
from tools.periodRefinement.preprocess import prepare_all

ALGORITHMS = ("LS", "CE", "AOV", "FPW", "MHF")
#: filter id -> band name.  Both surveys hand back integer filter ids in the
#: Kowalski format, and integers make poor feature names: eclipse depth ratios
#: are per band and only mean something with the band attached.
ZTF_BANDS = {1: "g", 2: "r", 3: "i"}
RUBIN_BANDS = {v: k for k, v in DEFAULT_BAND_MAP.items()}
DEDUP_TOLERANCE = 0.01
DEFAULT_N_PERIODS = 20


def load_lightcurves(identifiers, survey, config=None, **cleaning):
    """Fetch and clean light curves for a list of ids.

    Cleaning matters: the period search ran on catflag-filtered, cadence-thinned
    and sigma-clipped photometry, so refining on anything else would be choosing
    among candidate periods derived from data the fit never sees.
    """
    if survey == "rubin":
        from scope.surveys.rubin import make_rubin_client

        client = make_rubin_client(
            config=config or {},
            use_dia=True,
            release=os.environ.get("RUBIN_RELEASE", "dp2"),
        )
        return prepare_all(
            client.get_lightcurves(identifiers),
            band_names=RUBIN_BANDS,
            **cleaning,
        )
    if survey == "ztf":
        from scope.surveys.fritz import get_lightcurves_via_ids

        return prepare_all(
            get_lightcurves_via_ids(identifiers), band_names=ZTF_BANDS, **cleaning
        )
    raise ValueError(f"unknown survey {survey!r}")


def collect_trial_periods(
    row, n_per_algorithm, pmin, pmax, extra=(), algorithms=ALGORITHMS
):
    """Stored candidate periods for one source, deduplicated and in range.

    Feature generation keeps the top N peaks from each algorithm.  Offering all
    of them is not free: on the validation set, 57 per cent of the sources that
    end up with a wrong period are given an alias that genuinely fits better
    than the truth, and more candidates means more chances to meet one.  Going
    from 50 to 20 per algorithm left overall accuracy unchanged while costing
    2.5 times less, so 20 is the default.
    """
    values = []
    for i in range(1, n_per_algorithm + 1):
        for algorithm in algorithms:
            value = row.get(f"period_{i}_{algorithm}", np.nan)
            if value is not None and np.isfinite(value) and pmin <= value <= pmax:
                values.append(float(value))
    for value in extra:
        if value is not None and np.isfinite(value) and pmin <= value <= pmax:
            values.append(float(value))

    kept = []
    for value in values:
        if not any(abs(value - k) <= DEDUP_TOLERANCE * k for k in kept):
            kept.append(value)
    return kept


def refine_one(model, times, mags, bands, trial_periods):
    """Fit every geometry at every trial period and select, as Gaia do.

    Candidates from all trial periods are pooled before selection.  That
    pooling is not incidental: it is what allows a two-eclipse model at 2P to
    win over a one-eclipse model at P, which is how the half-period case is
    resolved.  Selecting the best model at each period first and comparing the
    winners afterwards loses it.
    """
    pooled = []
    for period in trial_periods:
        candidates = model.fit_period_models(times, mags, bands, period)
        if candidates:
            pooled.extend(candidates)
    if not pooled:
        return None
    return model.select(pooled)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Refine periods with a class-specific light curve model."
    )
    parser.add_argument(
        "--features", required=True, help="feature table written by generate-features"
    )
    parser.add_argument(
        "--classes",
        help="table with an id column and a class column; may "
        "be omitted if --features already carries one",
    )
    parser.add_argument(
        "--output", required=True, help="output parquet; the input is never modified"
    )
    parser.add_argument("--id-column", default="_id")
    parser.add_argument("--class-column", default="class")
    parser.add_argument(
        "--period-column",
        default="best_agree_period",
        help="the period currently reported (default: " "%(default)s)",
    )
    parser.add_argument(
        "--class-filter",
        default=None,
        help="comma separated class labels to refine; the "
        "default is every class with a registered model "
        f"({', '.join(supported())})",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=DEFAULT_N_PERIODS,
        help="trial periods to take from each algorithm " "(default: %(default)s)",
    )
    parser.add_argument(
        "--min-cadence-minutes",
        type=float,
        default=5.0,
        help="points closer together than this are thinned, keeping the first. "
        "This MUST match what feature generation used, or the fit sees "
        "different photometry from the period search that produced the "
        "candidates; the Rubin DIA path uses 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=60,
        help="skip sources with fewer usable points",
    )
    parser.add_argument(
        "--no-characterise",
        action="store_true",
        help="skip the class-specific parameter stage",
    )
    parser.add_argument(
        "--survey",
        default="rubin",
        choices=("rubin", "ztf"),
        help="where to fetch light curves from " "(default: %(default)s)",
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()

    wanted = None
    if args.class_filter:
        wanted = {c.strip().upper() for c in args.class_filter.split(",")}
        unknown = {c for c in wanted if lookup(c) is None}
        if unknown:
            sys.exit(
                f"no model registered for {sorted(unknown)}; "
                f"available: {', '.join(supported())}"
            )

    features = pd.read_parquet(args.features)
    if args.classes:
        classes = pd.read_parquet(args.classes)
        features = features.merge(classes, on=args.id_column, how="left")
    if args.class_column not in features.columns:
        sys.exit(
            f"no {args.class_column!r} column; pass --classes or " "--class-column"
        )

    if args.n_chunks > 1:
        features = features.iloc[args.chunk :: args.n_chunks]
    features = features.reset_index(drop=True)
    print(f"{len(features)} sources", flush=True)

    def _blank():
        return np.full(len(features), np.nan)

    refined = np.full(len(features), np.nan)
    was_refined = np.zeros(len(features), bool)
    quality = np.full(len(features), np.nan)
    model_name = np.array([""] * len(features), dtype=object)
    extra_columns = {}

    identifiers = features[args.id_column].to_numpy()
    original = features[args.period_column].to_numpy(float)
    labels = features[args.class_column].to_numpy()

    lightcurves = load_lightcurves(
        [int(x) for x in identifiers],
        args.survey,
        min_cadence_minutes=args.min_cadence_minutes,
    )
    print(f"light curves for {len(lightcurves)} of them", flush=True)

    started = time.time()
    for i in range(len(features)):
        refined[i] = original[i]
        label = labels[i]
        if wanted is not None and str(label).strip().upper() not in wanted:
            continue
        model = lookup(label)
        if model is None:
            continue

        curve = lightcurves.get(int(identifiers[i]))
        if curve is None:
            continue
        times, mags, errors, bands = curve
        usable = np.isfinite(times) & np.isfinite(mags)
        if usable.sum() < args.min_points:
            continue
        times, mags, bands = times[usable], mags[usable], bands[usable]

        # per-band offsets are not the signal; removing them stops the fit
        # spending its freedom on the difference between filters
        mags = mags.copy()
        for band in np.unique(bands):
            in_band = bands == band
            if in_band.sum() >= 3:
                mags[in_band] -= np.median(mags[in_band])

        pmin, pmax = model.period_range
        current = original[i]
        trials = collect_trial_periods(
            features.iloc[i],
            args.n_periods,
            pmin,
            pmax,
            extra=(current, 2.0 * current, 0.5 * current),
        )
        if not trials:
            continue

        chosen = refine_one(model, times, mags, bands, trials)
        if chosen is None:
            continue

        period = float(chosen["period"])
        if model.characterise is not None and not args.no_characterise:
            try:
                derived = model.characterise(
                    times, mags, bands, period, int(chosen.get("order", 3))
                )
                if derived:
                    period = float(derived.get("period_refined", period))
                    for key, value in derived.items():
                        extra_columns.setdefault(
                            key, np.full(len(features), np.nan, dtype=object)
                        )
                        extra_columns[key][i] = value
            except Exception as exc:  # pragma: no cover
                print(
                    f"  characterisation failed for {identifiers[i]}: {exc}", flush=True
                )

        # the shape parameters the fit already derived.  eclipse width as a
        # fraction of the period and the depth ratio are the axes the eclipsing
        # subtypes are defined on, and a secondary away from phase 0.5 means an
        # eccentric orbit, so these are worth carrying into classification.
        info = chosen.get("info") or {}
        centres = info.get("centres") or []
        widths = info.get("widths") or []
        depths = info.get("depths") or {}
        if widths:
            extra_columns.setdefault("eclipse_width", _blank())
            extra_columns["eclipse_width"][i] = float(np.mean(widths))
        if len(centres) == 2:
            extra_columns.setdefault("eclipse_separation", _blank())
            extra_columns["eclipse_separation"][i] = float(
                info.get("separation", np.nan)
            )
        if depths:
            primary = np.array([d[0] for d in depths.values() if len(d) >= 1], float)
            secondary = np.array([d[1] for d in depths.values() if len(d) >= 2], float)
            if primary.size:
                extra_columns.setdefault("eclipse_depth", _blank())
                extra_columns["eclipse_depth"][i] = float(np.nanmean(primary))
            if secondary.size and primary.size:
                extra_columns.setdefault("eclipse_depth_ratio", _blank())
                denom = np.nanmean(primary)
                if abs(denom) > 1e-9:
                    extra_columns["eclipse_depth_ratio"][i] = float(
                        np.nanmean(secondary) / denom
                    )
                # per-band depth ratios constrain the temperature ratio of the
                # two stars; a single band cannot measure this at all
                for band, values in depths.items():
                    if len(values) >= 2 and abs(values[0]) > 1e-9:
                        key = f"eclipse_depth_ratio_{band}"
                        extra_columns.setdefault(key, _blank())
                        extra_columns[key][i] = float(values[1] / values[0])

        refined[i] = period
        was_refined[i] = True
        model_name[i] = chosen.get("name", "")
        fvu = chosen.get("fvu", np.nan)
        if np.isfinite(fvu):
            quality[i] = (
                model.select.__globals__["global_ranking"](fvu)
                if "global_ranking" in model.select.__globals__
                else np.nan
            )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(features)}", flush=True)

    elapsed = time.time() - started
    print(
        f"refined {int(was_refined.sum())} of {len(features)} " f"in {elapsed:.0f}s",
        flush=True,
    )

    out = features.copy()
    out["period_refined"] = refined
    out["period_was_refined"] = was_refined
    out["period_model"] = model_name
    out["period_model_rank"] = quality
    for key, values in extra_columns.items():
        if key != "period_refined":
            out[f"refined_{key}"] = values

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
