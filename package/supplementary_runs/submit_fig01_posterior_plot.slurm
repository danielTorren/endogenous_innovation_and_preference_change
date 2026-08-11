#!/bin/bash
#SBATCH --job-name=fig01_posterior_plot
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # pure plotting from an already-saved samples.pkl, no simulation
#SBATCH --mem=8G                   # samples.pkl is 500,000 x 4 floats -- trivially small
#SBATCH --time=00:15:00
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# This job ONLY plots. It does NOT run the SBI calibration -- that is a
# separate, much longer job that already exists at
# package/calibration/submit_sbi_single_seed.slurm. Submit that first (or
# reuse an existing results/sbi_single_seed_* folder produced against the
# current base_params_calibration.json) and point RESULTS_FOLDER below, and
# fig01_posterior_plot.py's own __main__ block, at it.

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

# RESULTS_FOLDER lets you point this at a fresh sbi_single_seed run without
# editing the script; falls back to the folder hardcoded in the script's
# __main__ block otherwise.
#   sbatch --export=ALL,RESULTS_FOLDER=results/sbi_single_seed_XX_XX_XX__XX_XX_XXXX \
#       package/supplementary_runs/submit_fig01_posterior_plot.slurm
if [ -n "${RESULTS_FOLDER:-}" ]; then
    uv run python -c "
from package.supplementary_runs.fig01_posterior_plot import plot_fig1_posterior
plot_fig1_posterior(results_folder='${RESULTS_FOLDER}')
"
else
    uv run python -m package.supplementary_runs.fig01_posterior_plot
fi

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.supplementary_runs.fig01_posterior_plot
