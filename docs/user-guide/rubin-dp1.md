# Rubin DP1 Feature Generation

The pipeline supports running against Rubin Data Preview 1 (DP1) data stored as local parquet files, bypassing the TAP API entirely. This is the recommended approach for large-scale feature generation.

For Early Data Preview 2, whose export has a different layout, see [Rubin EDP2/DP2](rubin-dp2.md).

## Prerequisites

You need three gzip-compressed parquet files downloaded from the Rubin Science Platform and placed in a single directory:

```
/path/to/dp1_data/
  Object.parquet.gzip
  ForcedSource.parquet.gzip
  Visit.parquet.gzip
```

## Configuration

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

## Single-Run Feature Generation

For a small number of sources (e.g. a cone search or a short object list):

```bash
# Cone search
generate-features-rubin --ra 62.0 --dec -37.0 --radius 60 --doCPU

# From a CSV file with an objectId column
generate-features-rubin --objectid-file my_objects.csv --doCPU
```

Output is written to `/fred/oz480/mcoughli/generated_features_rubin/gen_features_rubin.parquet` by default.

## Large-Scale Processing with SLURM

For processing the full DP1 catalog, use the chunked SLURM workflow:

### 1. Prepare Chunks

Scan the local parquet files, filter to objects with enough detections, and split into chunk CSVs:

```bash
prepare-rubin-chunks \
  --data-path /path/to/dp1_data/ \
  --chunk-size 5000 \
  --min-n-lc-points 50 \
  --output-dir /fred/oz480/mcoughli/rubin_chunks
```

This writes `/fred/oz480/mcoughli/rubin_chunks/chunk_000.csv`, `/fred/oz480/mcoughli/rubin_chunks/chunk_001.csv`, etc., plus a master list `/fred/oz480/mcoughli/rubin_chunks/all_eligible_objectids.csv`.

### 2. Generate the SLURM Array Script

```bash
generate-features-rubin-slurm \
  --chunk-dir /fred/oz480/mcoughli/rubin_chunks \
  --output-dir /fred/oz480/mcoughli/rubin_slurm \
  --venv /path/to/your/.venv \
  --cpus-per-task 8 \
  --top-n-periods 50
```

This writes `/fred/oz480/mcoughli/rubin_slurm/run_rubin_features.sh`. Edit the script to adjust partition, account, memory, and module loads for your cluster.

### 3. Submit the Array Job

```bash
sbatch /fred/oz480/mcoughli/rubin_slurm/run_rubin_features.sh
```

Each array task processes one chunk and writes `/fred/oz480/mcoughli/generated_features_rubin/gen_features_rubin_<TASK_ID>.parquet`.

### 4. Combine Results

After all jobs finish:

```bash
combine-rubin-features \
  --input-dir /fred/oz480/mcoughli/generated_features_rubin \
  --output /fred/oz480/mcoughli/generated_features_rubin/dp1_features_combined.parquet
```

## Single-Node Chunked Runner (No SLURM)

If you don't have a SLURM cluster, you can run chunks sequentially (or resume after interruption) with:

```bash
python tools/run_rubin_chunked.py \
  --objectid-file /fred/oz480/mcoughli/rubin_chunks/all_eligible_objectids.csv \
  --doCPU \
  --chunk-size 5000 \
  --top-n-periods 50
```

Completed chunks are saved to `/fred/oz480/mcoughli/generated_features_rubin/chunks/` and skipped on restart, so the job is resumable.

## CLI Reference

| Command | Description |
|---------|-------------|
| `get-rubin-ids` | Discover object IDs via cone search or read from CSV |
| `generate-features-rubin` | Generate features for a set of Rubin sources |
| `prepare-rubin-chunks` | Scan local parquet files and split eligible objects into chunk CSVs |
| `generate-features-rubin-slurm` | Generate a SLURM array job script from chunk files |
| `combine-rubin-features` | Merge per-chunk parquet outputs into a single file |
