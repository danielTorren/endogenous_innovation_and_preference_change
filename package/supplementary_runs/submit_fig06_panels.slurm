#!/bin/bash
#SBATCH --job-name=fig06_panels
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # match "seed_repetitions" in constants/base_params_calibration.json
#SBATCH --mem=128G                # 2G per worker at 64 workers. MEASURED: one run of this config
                                   # (456 steps, 3000 individuals) peaks at ~0.82 GiB RSS, so 64
                                   # workers need ~52 GiB for run data alone -- 81% of the old 64G,
                                   # leaving nothing for the parent or the per-worker interpreter.
#SBATCH --time=12:00:00           # 43 value-settings x 64 seeds = 2,752 runs at 456 steps
                                   # (9 parameters x 4 values + b_chi's 7) -> 43 sequential
                                   # 64-wide batches, panel a included since it is re-run
                                   # below. Pad generously, then trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Supplementary Figure 6: the ten local-sensitivity sweeps, one per parameter,
# plus the combined 10-panel figure.
#
# Same work as submit_fig06_local_sensitivity_gen.slurm, but resumable: it runs
# package.supplementary_runs.fig06_panels, which skips any parameter that
# already has a results/single_param_vary_* folder with the right value grid and
# keeps going when one sweep fails instead of losing the rest. Re-submitting
# after a failure or a --time kill therefore only runs what is still missing.
#
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

mkdir -p slurm_logs
uv sync

# What is already on disk, so the log opens with the starting point.
uv run python -m package.supplementary_runs.fig06_panels list

# Panel a (alpha) is the one survivor of an earlier, interrupted Figure 6 run
# (results/single_param_vary_14_11_40__19_08_2026), so re-run it here rather
# than trusting it: --force ignores the existing folder and makes a new one, and
# resolve_panels then picks the newest match, so run-all below reuses the fresh
# panel a and the old folder is simply left unused. 4 values x 64 seeds = 256
# runs, a couple of minutes -- cheap enough to redo on every resubmit.
uv run python -m package.supplementary_runs.fig06_panels run a --force

# Run every missing panel, then write
# results/single_param_vary_<first panel's timestamp>_fig6_combined/
#   Plots/fig6_local_sensitivity_combined.png
# and print the folder list to paste into RUNS["local_sensitivity"] in
# package/paper_figures/build_figures.py.
uv run python -m package.supplementary_runs.fig06_panels run-all

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig06_panels run-all
