# CLI Reference

SCoPe provides a set of command-line tools installed via the `scope-ml` package. Below is a categorized reference of all available commands.

## Core commands

| Command | Entry point | Description |
|---------|-------------|-------------|
| `scope-initialize` | `scope.__init__:initialize` | Initialize required directories and copy necessary files to run SCoPe |
| `scope-develop` | `scope._instantiate:develop` | Set up a development installation of SCoPe |
| `scope-lint` | `scope.scope_class:Scope.lint` | Run linting checks on the codebase |
| `scope-doc` | `scope._instantiate:doc` | Generate imagery and build the documentation |
| `scope-train` | `scope._instantiate:train` | Train a binary classifier (DNN or XGBoost) for a given classification tag |
| `scope-test` | `scope._instantiate:test` | Run the full test suite (requires Kowalski access) |
| `scope-test-limited` | `scope._instantiate:test_limited` | Run a limited test suite (no Kowalski access required) |

## Training

| Command | Entry point | Description |
|---------|-------------|-------------|
| `create-training-script` | `scope._instantiate:create_training_script` | Generate a shell script containing multiple `scope-train` calls |
| `assemble-training-stats` | `scope._instantiate:assemble_training_stats` | Assemble and summarize training statistics across classifiers |
| `train-algorithm-slurm` | `tools.train_algorithm_slurm:main` | Generate SLURM scripts to train all classifiers in parallel on HPC |
| `train-algorithm-job-submission` | `tools.train_algorithm_job_submission:main` | Submit SLURM training jobs to the cluster |

## Feature generation

| Command | Entry point | Description |
|---------|-------------|-------------|
| `generate-features` | `tools.generate_features:main` | Generate features for ZTF light curves |
| `generate-features-slurm` | `tools.generate_features_slurm:main` | Generate SLURM scripts for parallelized feature generation |
| `generate-features-job-submission` | `tools.generate_features_job_submission:main` | Submit SLURM feature generation jobs to the cluster |
| `check-quads-for-sources` | `tools.generate_features_slurm:check_quads_for_sources` | Check which quadrants contain sources of interest |

## Inference

| Command | Entry point | Description |
|---------|-------------|-------------|
| `run-inference` | `tools.inference:main` | Run trained classifiers on a field of generated features |
| `create-inference-script` | `scope._instantiate:create_inference_script` | Generate a shell script for running inference across all classifiers |
| `run-inference-slurm` | `tools.run_inference_slurm:main` | Generate SLURM scripts for parallelized inference |
| `run-inference-job-submission` | `tools.run_inference_job_submission:main` | Submit SLURM inference jobs to the cluster |
| `combine-preds` | `tools.combine_preds:main` | Combine DNN and XGB prediction files for a field |
| `combine-preds-slurm` | `tools.combine_preds_slurm:main` | Generate SLURM scripts for parallelized prediction combining |

## Data management

| Command | Entry point | Description |
|---------|-------------|-------------|
| `scope-download-classification` | `tools.scope_download_classification:main` | Download classifications from Fritz/SkyPortal |
| `scope-upload-classification` | `tools.scope_upload_classification:main` | Upload classifications to Fritz/SkyPortal |
| `scope-manage-annotation` | `tools.scope_manage_annotation:main` | Manage annotations on Fritz/SkyPortal |
| `post-taxonomy` | `tools.taxonomy:main` | Post the SCoPe taxonomy to Fritz/SkyPortal |
| `select-fritz-sample` | `scope._instantiate:select_fritz_sample` | Select a sample of sources from Fritz for labeling or analysis |
| `get-quad-ids` | `tools.get_quad_ids:main` | Retrieve quadrant IDs for ZTF fields |

## Rubin-specific

| Command | Entry point | Description |
|---------|-------------|-------------|
| `get-rubin-ids` | `tools.get_rubin_ids:main` | Discover Rubin object IDs via cone search or read from CSV |
| `generate-features-rubin` | `tools.generate_features_rubin:main` | Generate SCoPe features for a set of Rubin sources |
| `prepare-rubin-chunks` | *(see README)* | Scan local parquet files and split eligible objects into chunk CSVs |
| `generate-features-rubin-slurm` | *(see README)* | Generate a SLURM array job script from chunk files |
| `combine-rubin-features` | *(see README)* | Merge per-chunk parquet outputs into a single combined file |

## Utilities

| Command | Entry point | Description |
|---------|-------------|-------------|
| `run-scope-local` | `tools.run_scope_local:main` | Run the SCoPe classification pipeline locally on a set of sources |
| `analyze-logs` | `tools.analyze_logs:main` | Analyze log files from SCoPe pipeline runs |
