#!/bin/bash
#SBATCH --job-name=b_chi_grid_search_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # get_num_workers() reads the SLURM allocation, so this is also joblib's
                                   # n_jobs and therefore the number of model instances alive at once --
                                   # the only lever on peak memory. 5 x 64 = 320 runs, so 5 batches.
                                   # Raising this to 128 buys ~2 batches of wall-clock and DOUBLES peak
                                   # memory: take --mem to 256G with it, do not raise it alone.
#SBATCH --mem=128G                 # 2G per worker. base_params_vary_single.json sets
                                   # save_timeseries_data_state=1, so each worker still records the full
                                   # per-step history for 366 steps (90 + 276, duration_future=0) over
                                   # 3000 individuals -- this job returns only history_prop_EV, but that
                                   # trims what crosses the joblib boundary, not what a live worker holds.
                                   # Trim from `seff <jobid>`.
#SBATCH --time=01:00:00           # 320 runs at 366 steps, ~17 s each -> 5 sequential 64-wide batches,
                                   # ~2 min of real work. Padded against a busy node; trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Calibration grid search over parameters_social_network.b_chi.
# Sweeps the values in constants/vary_single_b_innov.json against
# constants/base_params_vary_single.json, scores each one by RMSE on the
# California EV stock share (April of each year 2010-2023), and writes the
# ranked table to the job log.
#
# Both JSON paths are hardcoded in the gen script's __main__ block -- edit the
# JSONs, or that block, not this file. To widen or refine the grid, edit
# property_list in vary_single_b_innov.json; --time and --mem above are sized per
# 320 runs, so scale --time roughly linearly with the number of grid points.
#
# No separate plotting step: the gen script builds its own single-axes figure,
# Plots/b_chi_grid_search.png, one line per b_chi value against the observed
# series. MPLBACKEND=Agg below is what lets it do that on a headless node.

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

uv run python -m package.generating_data.b_chi_grid_search_gen

# main() prints the folder it made as its last line, but resolve it here too so
# the path is easy to grep out of the log -- unambiguous because the run above
# just created it.
RESULTS_FOLDER="$(ls -td results/b_chi_grid_search_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"
echo "Figure: $RESULTS_FOLDER/Plots/b_chi_grid_search.png"
echo "Scores: $RESULTS_FOLDER/Data/rmse_of_mean.pkl (ranked table is above in this log)"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.b_chi_grid_search_gen
