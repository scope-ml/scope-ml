# SCoPe: ZTF Source Classification Project

[![PyPI version](https://badge.fury.io/py/scope-ml.svg)](https://badge.fury.io/py/scope-ml)
[![arXiv](https://img.shields.io/badge/arXiv-2102.11304-blue)](https://arxiv.org/abs/2102.11304)
[![arXiv](https://img.shields.io/badge/arXiv-2009.14071-blue)](https://arxiv.org/abs/2009.14071)
[![arXiv](https://img.shields.io/badge/arXiv-2312.00143-blue)](https://arxiv.org/abs/2312.00143)

`scope-ml` uses machine learning to classify light curves from the Zwicky Transient Facility ([ZTF](https://www.ztf.caltech.edu)) and the Vera C. Rubin Observatory ([LSST](https://rubinobs.org)). The documentation is hosted at [https://zwickytransientfacility.github.io/scope-ml/](https://zwickytransientfacility.github.io/scope-ml/). For local preview, install `mkdocs-material` and run `mkdocs serve`.

Feature generation includes period-finding (Conditional Entropy, Analysis of Variance, Lomb-Scargle, FPW) and Fourier decomposition via the [periodfind](https://github.com/ZwickyTransientFacility/periodfind) library. Fourier features are computed using a batched weighted linear least-squares solver with BIC model selection, replacing the previous per-source `scipy.optimize.curve_fit` loop.

## Rubin DP1 Feature Generation (Local Parquet Files)

The pipeline supports running against Rubin Data Preview 1 (DP1) data stored as local parquet files, bypassing the TAP API entirely. This is the recommended approach for large-scale feature generation.

### Prerequisites

You need three gzip-compressed parquet files downloaded from the Rubin Science Platform and placed in a single directory:

```
/path/to/dp1_data/
  Object.parquet.gzip
  ForcedSource.parquet.gzip
  Visit.parquet.gzip
```

### Configuration

Tell scope-ml to use local files instead of the TAP API by setting `data_path` in `config.yaml`:

```yaml
rubin:
  data_path: /path/to/dp1_data/
```

Or set the environment variable:

```bash
export RUBIN_DATA_PATH=/path/to/dp1_data/
```

When `data_path` (or `RUBIN_DATA_PATH`) is set, all Rubin commands automatically use the local parquet backend (`RubinLocalClient`) instead of the TAP client. No token is needed for local mode.

### Single-run feature generation

For a small number of sources (e.g. a cone search or a short object list):

```bash
# Cone search
generate-features-rubin --ra 62.0 --dec -37.0 --radius 60 --doCPU

# From a CSV file with an objectId column
generate-features-rubin --objectid-file my_objects.csv --doCPU
```

Output is written to `generated_features_rubin/gen_features_rubin.parquet` by default.

### Large-scale processing with SLURM

For processing the full DP1 catalog, use the chunked SLURM workflow:

**1. Prepare chunks** -- scan the local parquet files, filter to objects with enough detections, and split into chunk CSVs:

```bash
prepare-rubin-chunks \
  --data-path /path/to/dp1_data/ \
  --chunk-size 5000 \
  --min-n-lc-points 50 \
  --output-dir rubin_chunks
```

This writes `rubin_chunks/chunk_000.csv`, `rubin_chunks/chunk_001.csv`, etc., plus a master list `rubin_chunks/all_eligible_objectids.csv`.

**2. Generate the SLURM array script:**

```bash
generate-features-rubin-slurm \
  --chunk-dir rubin_chunks \
  --output-dir rubin_slurm \
  --venv /path/to/your/.venv \
  --cpus-per-task 8 \
  --top-n-periods 50
```

This writes `rubin_slurm/run_rubin_features.sh`. Edit the script to adjust partition, account, memory, and module loads for your cluster.

**3. Submit the array job:**

```bash
sbatch rubin_slurm/run_rubin_features.sh
```

Each array task processes one chunk and writes `generated_features_rubin/gen_features_rubin_<TASK_ID>.parquet`.

**4. Combine results** after all jobs finish:

```bash
combine-rubin-features \
  --input-dir generated_features_rubin \
  --output generated_features_rubin/dp1_features_combined.parquet
```

### Single-node chunked runner (no SLURM)

If you don't have a SLURM cluster, you can run chunks sequentially (or resume after interruption) with:

```bash
python tools/run_rubin_chunked.py \
  --objectid-file rubin_chunks/all_eligible_objectids.csv \
  --doCPU \
  --chunk-size 5000 \
  --top-n-periods 50
```

Completed chunks are saved to `generated_features_rubin/chunks/` and skipped on restart, so the job is resumable.

### CLI reference

| Command | Description |
|---------|-------------|
| `get-rubin-ids` | Discover object IDs via cone search or read from CSV |
| `generate-features-rubin` | Generate features for a set of Rubin sources |
| `prepare-rubin-chunks` | Scan local parquet files and split eligible objects into chunk CSVs |
| `generate-features-rubin-slurm` | Generate a SLURM array job script from chunk files |
| `combine-rubin-features` | Merge per-chunk parquet outputs into a single file |

## Installing with periodfind

To install SCoPe with periodfind, please complete periodfind GPU installation with Rust.
Once that is done, you might encounter an issue in numpy version discrepancy between what SCoPe requires and what periodfind requires. SCoPe-ml 0.9.5 requires numpy<1.24,>=1.23 and periodfind might have installed a different version.
You can make it work by installing SCoPe witout dependencies and then installing the dependencies one-by-one while skipping numpy.

```bash
pip install scope-ml --no-deps
pip install deepdiff gsutil matplotlib questionary scikit-learn wandb h5py astropy fast-histogram healpy "jinja2<=3.1" pandas penquins pyyaml tdtax pyarrow "numba>=0.56.4" cesium xgboost seaborn pydot notebook cython "tables>=3.7,<3.9.2" pyvo
```

## Funding
 We gratefully acknowledge previous and current support from the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for <a href="https://a3d3.ai">Accelerated AI Algorithms for Data-Driven Discovery (A3D3)</a> under Cooperative Agreement No. <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997">PHY-2117997</a>.

 <p align="center">
 <img src="https://github.com/ZwickyTransientFacility/scope/raw/main/assets/a3d3.png" alt="A3D3" width="200"/>
 <img src="https://github.com/ZwickyTransientFacility/scope/raw/main/assets/nsf.png" alt="NSF" width="200"/>
