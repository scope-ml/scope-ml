#!/usr/bin/env python
"""
Generate a SLURM array job script for chunked Rubin feature generation.

Reads the chunk CSVs written by ``prepare-rubin-chunks`` and writes a
single SLURM array script where each array task runs one
``generate-features-rubin`` call over one chunk. Works for both DP1 and
DP2: pass ``--release dp2`` (and, if needed, ``--flux-column``) to have
those flags forwarded to every per-chunk call.

Usage:
    generate-features-rubin-slurm \
        --chunk-dir rubin_chunks \
        --output-dir rubin_slurm \
        --venv /path/to/your/.venv \
        --cpus-per-task 8 \
        --top-n-periods 50

    generate-features-rubin-slurm \
        --chunk-dir rubin_chunks \
        --output-dir rubin_slurm \
        --venv /path/to/your/.venv \
        --release dp2 \
        --doCPU

This writes ``<output-dir>/run_rubin_features.sh``. Edit the script to
adjust partition, account, memory, and module loads for your cluster
before submitting with ``sbatch``.
"""

import argparse
import glob
import os
import pathlib

BASE_DIR = pathlib.Path.cwd()


def generate_slurm_script(
    chunk_dir,
    output_dir="rubin_slurm",
    features_output_dir="generated_features_rubin",
    venv=None,
    job_name="rubin_fg",
    partition="shared",
    account=None,
    time="24:00:00",
    mem_gb=32,
    cpus_per_task=8,
    max_concurrent=None,
    modules=None,
    doCPU=True,
    doGPU=False,
    release=None,
    use_dia=False,
    flux_column=None,
    bands=None,
    min_n_lc_points=50,
    min_cadence_minutes=None,
    max_freq=None,
    top_n_periods=50,
    Ncore=None,
    extra_args=None,
    script_name="run_rubin_features.sh",
):
    """
    Build the SLURM array script and write it to disk.

    Parameters
    ----------
    chunk_dir : str
        Directory containing chunk_*.csv files from prepare-rubin-chunks.
    output_dir : str
        Directory to write the SLURM script (and its logs/ subdirectory).
    features_output_dir : str
        Directory each array task passes to generate-features-rubin
        --dirname; per-chunk parquet files land here.
    venv : str, optional
        Path to a virtualenv to activate (runs
        ``source <venv>/bin/activate``). Required on most clusters.
    job_name : str
        SLURM job name.
    partition : str
        SLURM partition to request.
    account : str, optional
        SLURM account/allocation to charge.
    time : str
        Walltime per array task, HH:MM:SS.
    mem_gb : int
        Memory per array task, in GB.
    cpus_per_task : int
        CPUs per array task.
    max_concurrent : int, optional
        Cap on simultaneously running array tasks (SLURM `%N` suffix).
    modules : list of str, optional
        Modules to `module load` before activating the venv.
    doCPU, doGPU : bool
        Which generate-features-rubin period-search backend to request.
    release : str, optional
        Rubin data release: 'dp1' or 'dp2'. Forwarded to each call.
    use_dia : bool
        DP1 only: forward --use-dia.
    flux_column : str, optional
        DP2 only: forward --flux-column.
    bands : list of str, optional
        Forward --bands.
    min_n_lc_points : int
        Forward --min-n-lc-points.
    min_cadence_minutes : float, optional
        Forward --min-cadence-minutes.
    max_freq : float, optional
        Forward --max-freq.
    top_n_periods : int
        Forward --top-n-periods.
    Ncore : int, optional
        Forward --Ncore. Defaults to cpus_per_task.
    extra_args : str, optional
        Additional raw arguments appended verbatim to each
        generate-features-rubin call.
    script_name : str
        Filename for the generated script (default 'run_rubin_features.sh').

    Returns
    -------
    str
        Path to the written SLURM script.
    """
    if doCPU and doGPU:
        raise ValueError("Choose only one of doCPU or doGPU.")
    if not doCPU and not doGPU:
        doCPU = True

    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.csv")))
    n_chunks = len(chunk_files)
    if n_chunks == 0:
        raise ValueError(
            f"No chunk_*.csv files found in {chunk_dir}. "
            f"Run prepare-rubin-chunks first."
        )

    output_path = pathlib.Path(output_dir)
    logs_dir = output_path / "logs"
    os.makedirs(logs_dir, exist_ok=True)

    array_range = f"0-{n_chunks - 1}"
    if max_concurrent is not None:
        array_range += f"%{max_concurrent}"

    fg_flags = ["--doCPU" if doCPU else "--doGPU"]
    if release is not None:
        fg_flags.append(f"--release {release}")
    if use_dia:
        fg_flags.append("--use-dia")
    if flux_column is not None:
        fg_flags.append(f"--flux-column {flux_column}")
    if bands is not None and len(bands) > 0:
        fg_flags.append(f"--bands {' '.join(bands)}")
    fg_flags.append(f"--min-n-lc-points {min_n_lc_points}")
    if min_cadence_minutes is not None:
        fg_flags.append(f"--min-cadence-minutes {min_cadence_minutes}")
    if max_freq is not None:
        fg_flags.append(f"--max-freq {max_freq}")
    fg_flags.append(f"--top-n-periods {top_n_periods}")
    fg_flags.append(f"--Ncore {Ncore if Ncore is not None else cpus_per_task}")
    if extra_args:
        fg_flags.append(extra_args)
    fg_flags_str = " \\\n    ".join(fg_flags)

    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --output={logs_dir}/%x_%A_%a.out")
    lines.append(f"#SBATCH --error={logs_dir}/%x_%A_%a.err")
    lines.append(f"#SBATCH --partition={partition}")
    if account is not None:
        lines.append(f"#SBATCH --account={account}")
    lines.append("#SBATCH --nodes=1")
    lines.append("#SBATCH --ntasks=1")
    lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    lines.append(f"#SBATCH --mem={mem_gb}G")
    lines.append(f"#SBATCH --time={time}")
    lines.append(f"#SBATCH --array={array_range}")
    lines.append("")
    lines.append("# Edit the lines above (partition/account/mem/module loads) for your cluster.")
    lines.append("module purge")
    for module in modules or []:
        lines.append(f"module load {module}")
    if venv is not None:
        lines.append(f"source {venv}/bin/activate")
    lines.append("")
    lines.append(f'TASK_ID=$(printf "%03d" ${{SLURM_ARRAY_TASK_ID}})')
    lines.append(f'CHUNK_FILE="{os.path.abspath(chunk_dir)}/chunk_${{TASK_ID}}.csv"')
    lines.append("")
    lines.append(
        "generate-features-rubin \\\n"
        "    --objectid-file ${CHUNK_FILE} \\\n"
        f"    --dirname {features_output_dir} \\\n"
        '    --filename gen_features_rubin_${SLURM_ARRAY_TASK_ID} \\\n'
        f"    {fg_flags_str}"
    )
    script = "\n".join(lines) + "\n"

    script_path = output_path / script_name
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Found {n_chunks} chunk file(s) in {chunk_dir}")
    print(f"Wrote SLURM array script to {script_path}")
    print(f"Submit with: sbatch {script_path}")

    return str(script_path)


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a SLURM array job script from Rubin feature-generation "
            "chunk files."
        )
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        required=True,
        help="Directory containing chunk_*.csv files from prepare-rubin-chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rubin_slurm",
        help="Directory to write the SLURM script (default 'rubin_slurm')",
    )
    parser.add_argument(
        "--features-output-dir",
        type=str,
        default="generated_features_rubin",
        help="Directory each array task writes its parquet output to",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help="Path to a virtualenv to activate before running",
    )
    parser.add_argument("--job-name", type=str, default="rubin_fg")
    parser.add_argument("--partition", type=str, default="shared")
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--time", type=str, default="24:00:00")
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Cap on simultaneously running array tasks",
    )
    parser.add_argument(
        "--module",
        dest="modules",
        action="append",
        default=None,
        help="Module to 'module load' before activating the venv "
        "(may be given multiple times)",
    )
    parser.add_argument("--doCPU", action="store_true", default=False)
    parser.add_argument("--doGPU", action="store_true", default=False)
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        choices=["dp1", "dp2"],
        help="Rubin data release forwarded to each generate-features-rubin call",
    )
    parser.add_argument(
        "--use-dia",
        action="store_true",
        default=False,
        help="DP1 only: forward --use-dia to each call",
    )
    parser.add_argument(
        "--flux-column",
        type=str,
        default=None,
        choices=["psfFlux", "psfDiffFlux"],
        help="DP2 only: forward --flux-column to each call",
    )
    parser.add_argument("--bands", type=str, nargs="+", default=None)
    parser.add_argument("--min-n-lc-points", type=int, default=50)
    parser.add_argument("--min-cadence-minutes", type=float, default=None)
    parser.add_argument("--max-freq", type=float, default=None)
    parser.add_argument("--top-n-periods", type=int, default=50)
    parser.add_argument(
        "--Ncore",
        type=int,
        default=None,
        help="Forwarded --Ncore (defaults to --cpus-per-task)",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default=None,
        help="Additional raw arguments appended to each generate-features-rubin call",
    )
    parser.add_argument(
        "--script-name",
        type=str,
        default="run_rubin_features.sh",
        help="Filename for the generated script",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    generate_slurm_script(
        chunk_dir=args.chunk_dir,
        output_dir=args.output_dir,
        features_output_dir=args.features_output_dir,
        venv=args.venv,
        job_name=args.job_name,
        partition=args.partition,
        account=args.account,
        time=args.time,
        mem_gb=args.mem_gb,
        cpus_per_task=args.cpus_per_task,
        max_concurrent=args.max_concurrent,
        modules=args.modules,
        doCPU=args.doCPU,
        doGPU=args.doGPU,
        release=args.release,
        use_dia=args.use_dia,
        flux_column=args.flux_column,
        bands=args.bands,
        min_n_lc_points=args.min_n_lc_points,
        min_cadence_minutes=args.min_cadence_minutes,
        max_freq=args.max_freq,
        top_n_periods=args.top_n_periods,
        Ncore=args.Ncore,
        extra_args=args.extra_args,
        script_name=args.script_name,
    )


if __name__ == "__main__":
    main()
