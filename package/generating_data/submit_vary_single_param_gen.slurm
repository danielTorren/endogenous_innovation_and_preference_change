#!/bin/bash
#SBATCH --job-name=vary_single_param_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # get_num_workers() reads the SLURM allocation, so this is also joblib's
                                   # n_jobs and therefore the number of model instances alive at once --
                                   # the only lever on peak memory. 320 runs / 32 workers = 10 batches.
#SBATCH --mem=64G                  # 2G per worker. base_params_vary_single.json sets
                                   # save_timeseries_data_state=1, so each worker holds full per-step
                                   # history for 546 steps (180+90+276, duration_future=0) over 3000
                                   # individuals. Unlike the used-car job the second-hand fleet is at its
                                   # default size here, so this is far cheaper. Trim from `seff <jobid>`.
#SBATCH --time=01:30:00           # 5 b_chi values x 64 seeds = 320 runs at 546 steps, ~25 s each ->
                                   # 10 sequential 32-wide batches, ~5 min of real work. Padded against a
                                   # busy node; trim from seff.
#SBATCH --partition=generic        # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Sweeps parameters_social_network.b_chi over [3, 3.125, 3.25, 3.375, 3.5]
# (constants/vary_single_b_innov.json) against
# constants/base_params_vary_single.json. Both paths are hardcoded in the gen
# script's __main__ block -- edit the JSONs, or that block, not this file.
#
# Outputs one fresh results/single_param_vary_<timestamp> folder holding
# Data/data_array_ev_prop, Data/data_array_price, Data/data_array_margins,
# Data/base_params and Data/vary_single.

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

uv run python -m package.generating_data.vary_single_param_gen

# Plotting is NOT run here: package/plotting_data/vary_single_param_plot.py takes
# no argv, its __main__ hardcodes a results folder, and it ends in plt.show().
# Resolve the folder this run made and plot it interactively afterwards.
RESULTS_FOLDER="$(ls -td results/single_param_vary_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"
echo "To plot: edit the main(...) call in package/plotting_data/vary_single_param_plot.py"
echo "to \"$RESULTS_FOLDER\", then run: python -m package.plotting_data.vary_single_param_plot"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.vary_single_param_gen
