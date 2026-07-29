# Rubin EDP2/DP2 Feature Generation

The pipeline also reads Rubin Early Data Preview 2 (EDP2, `dp2.*` in TAP) DIA
forced photometry. The DP2 export has a different layout from
[DP1](rubin-dp1.md), so it uses a separate backend
(`scope.surveys.rubin.RubinLocalDP2Client`) selected with `--release dp2`.

## Prerequisites

DP1 arrives as one parquet file per table, joined at read time. DP2 arrives as a
**single pre-joined file** — `ForcedSourceOnDiaObject` with `ra`/`dec` from
`DiaObject` and `expMidptMJD` from `Visit` already attached, pre-filtered to
variability candidates:

```
/path/to/edp2_data/VarCands_DP2.parquet
```

That one file is everything the pipeline needs. The reader checks the schema up
front and tells you which columns are missing if the export was built without
one of those joins.

Two properties of the export are worth knowing, since both silently corrupt
period searches if an export regresses:

- **One row per `(diaObjectId, visit, detector)`.** An export built by joining
  against `DiaSource` without `DISTINCT` repeats every epoch once per matching
  source, inflating epoch counts several-fold. The reader de-duplicates on that
  key as a cheap guard; pass `dedupe=False` to see the raw rows.
- **`expMidptMJD` must come from the `Visit` join.** Do not try to recover times
  from the visit ID — it encodes only the observing night, not the exposure time.

## Configuration

`dp2_data_path` is separate from `data_path` so one config can describe both
releases:

```yaml
rubin:
  release: dp1                                             # default when --release is omitted
  data_path: /path/to/dp1_data/                            # DP1: a directory
  dp2_data_path: /path/to/edp2_data/VarCands_DP2.parquet   # DP2: a file (or its directory)
  flux_column: psfFlux                                     # or psfDiffFlux
```

The equivalent environment variables are `RUBIN_RELEASE` and
`RUBIN_DP2_DATA_PATH`. No TAP token is needed once the file is local.

### Which flux?

`flux_column: psfFlux` (the default) is the forced PSF flux measured on the
direct image, which converts to a real AB magnitude and is what you want for
period searching. `psfDiffFlux` is the difference-image flux; it is negative for
a large fraction of epochs and those are dropped by the magnitude conversion, so
use it only if you specifically want difference photometry.

## Feature Generation

```bash
# Every object in the export (it is already a variability-candidate selection)
generate-features-rubin --release dp2 --all-objects --doCPU

# A subset, from a CSV with an objectId column
generate-features-rubin --release dp2 --objectid-file my_objects.csv --doGPU

# Cone search within the export
generate-features-rubin --release dp2 --ra 58.46 --dec -50.72 --radius 60 --doCPU
```

`--release dp2` implies DIA photometry, so the default high-cadence filter
(`--min-cadence-minutes`) is switched off exactly as `--use-dia` does for DP1.

The chunked SLURM workflow documented for [DP1](rubin-dp1.md) works the same
way; pass `--release dp2` to the per-chunk `generate-features-rubin` calls.

## Python API

```python
from scope.surveys.rubin import RubinLocalDP2Client

client = RubinLocalDP2Client("/path/to/edp2_data/VarCands_DP2.parquet")

objects = client.get_all_objects()                     # {objectId: {coord_ra, coord_dec}}
lcs = client.get_lightcurves(list(objects)[:10], bands=["r"])

# One row per visit, for cadence / observing-condition checks
visits = client.get_visit_positions(columns=["expTime", "airmass"])
```

Lightcurves come back in the same Kowalski format as the DP1 clients — one entry
per `(objectId, band)`, with `hjd` (MJD), `mag`, `magerr` and `catflags`.
`catflags` sums the same five pixel flags as DP1 plus the DP2 `invalidPsfFlag`
and the flux flag for the selected `flux_column`; the feature pipeline keeps only
`catflags == 0` epochs.

Loading the reference export (2.55M epochs, 8,380 objects) costs about 0.9 GB of
RAM; `get_all_objects()` and `get_lightcurves()` load lazily and cache.
