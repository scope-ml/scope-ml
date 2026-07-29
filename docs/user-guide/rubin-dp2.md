# Rubin EDP2/DP2 Feature Generation

The pipeline also reads Rubin Early Data Preview 2 (EDP2, `dp2.*` in TAP) DIA
forced photometry. The DP2 export is shaped differently from
[DP1](rubin-dp1.md), so it uses a separate backend
(`scope.surveys.rubin.RubinLocalDP2Client`) selected with `--release dp2`.

## How the DP2 export differs from DP1

DP1 is stored as one parquet file per table and joined at read time. The DP2
data we work from is a **single pre-joined table**, produced by a TAP query
like:

```sql
SELECT
    f.*,
    d.ra,
    d.dec
FROM dp2.ForcedSourceOnDiaObject AS f,
     dp2.DiaObject AS d,
     dp2.DiaSource AS s
WHERE f.diaObjectId = d.diaObjectId
  AND d.diaObjectId = s.diaObjectId
  AND d.r_psfFluxNdata > 10
  AND s.snr > 5
  AND s.band = 'r'
  AND s.isDipole = 0
  AND s.shape_flag = 0
  AND s.extendedness < 0.1
  AND s.pixelFlags = 0
  AND s.reliability > 0.9
```

Two consequences are handled by the reader:

1. **Duplicated rows.** The three-way join has no `DISTINCT`, so every forced
   source is repeated once per `DiaSource` row that passed the cuts. In the
   reference export that inflates 2.55M real epochs to 22.35M rows. The reader
   de-duplicates on `(diaObjectId, visit, detector)`, the primary key of
   `ForcedSourceOnDiaObject`. Pass `dedupe=False` to keep the raw rows.
2. **No timestamps.** `ForcedSourceOnDiaObject` carries only `visit`, not an
   MJD. A separate `dp2.Visit` export is required to convert visits to
   `expMidptMJD` (see below). Do not try to decode the visit ID: it encodes
   only the observing night, not the exposure time.

## Prerequisites

Two files:

```
/path/to/edp2_data/
  VarCands_DP2.parquet   # the pre-joined forced-source export
  Visit.parquet          # visit -> expMidptMJD
```

Fetch the visit table from the Rubin Science Platform (needs a TAP token in
`config.yaml` or `RUBIN_TAP_TOKEN`):

```bash
fetch-rubin-visits --release dp2 --output /path/to/edp2_data/Visit.parquet
```

The reader looks for `Visit.parquet` next to the data file, then falls back to
`$RUBIN_VISIT_PATH` and the `rubin.visit_path` config key.

## Configuration

`dp2_data_path` is separate from `data_path` so one config can describe both
releases:

```yaml
rubin:
  release: dp1                                       # default when --release is omitted
  data_path: /path/to/dp1_data/                      # DP1: a directory
  dp2_data_path: /path/to/edp2_data/VarCands_DP2.parquet   # DP2: a file (or its directory)
  visit_path:                                        # blank = look next to dp2_data_path
  flux_column: psfFlux                               # or psfDiffFlux
```

The equivalent environment variables are `RUBIN_RELEASE`,
`RUBIN_DP2_DATA_PATH` and `RUBIN_VISIT_PATH`. No token is needed once the
files are local.

### Which flux?

`flux_column: psfFlux` (the default) is the forced PSF flux measured on the
direct image, which converts to a real AB magnitude and is what you want for
period searching. `psfDiffFlux` is the difference-image flux; it is negative
for roughly half of all epochs and those epochs are dropped by the magnitude
conversion, so use it only if you specifically want difference photometry.

## Feature Generation

```bash
# Every object in the export (it is already a variability-candidate selection)
generate-features-rubin --release dp2 --all-objects --doCPU

# A subset, from a CSV with an objectId column
generate-features-rubin --release dp2 --objectid-file my_objects.csv --doCPU

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

client = RubinLocalDP2Client(
    data_path="/path/to/edp2_data/VarCands_DP2.parquet",
    visit_path="/path/to/edp2_data/Visit.parquet",
)

objects = client.get_all_objects()                     # {objectId: {coord_ra, coord_dec}}
lcs = client.get_lightcurves(list(objects)[:10], bands=["r"])
```

Lightcurves come back in the same Kowalski format as the DP1 clients — one
entry per `(objectId, band)`, with `hjd` (MJD), `mag`, `magerr` and
`catflags`. `catflags` sums the same five pixel flags as DP1 plus the DP2
`invalidPsfFlag` and the flux flag for the selected `flux_column`; the feature
pipeline keeps only `catflags == 0` epochs.

Loading the reference export costs about 2.7 GB of RAM and roughly 15 s;
`get_all_objects()` and `get_lightcurves()` load lazily and cache.
