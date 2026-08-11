#!/bin/bash
#SBATCH --job-name=fig07_08_sobol_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128       # requested explicitly (does NOT match "seed_repetitions"=64 in
                                   # constants/base_params_SA.json, unlike every other job here) --
                                   # get_num_workers() reads SLURM_CPUS_PER_TASK regardless of AV_reps.
#SBATCH --mem=128G                 # adjust after checking actual usage on a first run (seff <jobid>)
#SBATCH --time=24:00:00           # N_samples=256, calc_second_order=False, D=10 params:
                                   # runs = N_samples x (D+2) x seed_repetitions = 256 x 12 x 64 = 196,608
                                   # -> 1536 sequential 128-wide batches, ~8.5h at ~20s/run measured
                                   # elsewhere in this repo. Pad generously on the first run, then trim
                                   # from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Figures 7 (first-order Sobol index) and 8 (total-order): global sensitivity
# of six model outputs to the same ten parameters Figure 6 varies locally.
# calc_second_order=False (vs. the underlying gen script's own default of
# True) since the paper's Figures 7/8 only show first/total order, not
# parameter-interaction terms -- see fig07_08_sobol_gen.py's docstring for
# the run-count arithmetic and package/supplementary_runs/README.md for how
# these numbers were chosen and how they compare to Figure 6's cost.

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

set -euo pipefail

module purge
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

uv sync

# Prints one results/sensitivity_analysis_<timestamp> folder near the top of
# this job's log. Figures land under its Prints/ subfolder:
# 10_256_First_multi_output_sensitivity_plot_2row.png is Figure 7,
# 10_256_Total_multi_output_sensitivity_plot_2row.png is Figure 8
# (both also saved as matching .eps files).
uv run python -m package.supplementary_runs.fig07_08_sobol_gen

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig07_08_sobol_gen
