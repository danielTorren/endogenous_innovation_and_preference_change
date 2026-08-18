#!/bin/bash
#SBATCH --job-name=kappa_sweep_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # get_num_workers() reads the SLURM allocation, so this is also joblib's
                                   # n_jobs. The script runs the 10 kappa values SEQUENTIALLY and fans out
                                   # over the 16 seeds inside each, so 16 is exactly one joblib batch per
                                   # kappa. Raising this above SEEDS buys nothing; raise --seeds instead.
#SBATCH --mem=32G                  # 2G per worker. base_params_calibration.json sets
                                   # save_timeseries_data_state=1, which the HHI series REQUIRES, so each
                                   # worker holds full per-step history for 456 steps (180+276,
                                   # duration_future=0) over 3000 individuals. Only 16 live at once.
                                   # Trim from `seff <jobid>`.
#SBATCH --time=03:00:00            # 10 kappa x 16 seeds = 160 runs at 456 steps, ~25-60 s each, as 10
                                   # sequential 16-wide batches -> ~10-20 min of real work. Padded
                                   # against a busy node; trim from seff.
#SBATCH --partition=generic         # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Sweeps parameters_vehicle_user.kappa over 10 values in [1e-4, 3e-4] against
# package/constants/base_params_calibration.json (calibrated value 2.25e-4), and
# records new-car HHI and whole-fleet mean car age. The kappa grid and the seed
# count are constants at the top of the gen script, not arguments here.
#
# Outputs one fresh results/kappa_sweep_<timestamp> folder holding
# Data/{data_array_hhi,data_array_age_fleet,data_array_ev_prop,kappa_list,
# base_params,summary} plus Plots/kappa_sweep_timeseries.png and
# Plots/kappa_sweep_response.png. The per-kappa 2023 means are also printed as a
# table in this job's log, with the target bands on the row above.

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

# Both figures are written by the gen script's __main__ block, under Agg above,
# so nothing needs re-plotting by hand afterwards.
uv run python -m package.generating_data.kappa_sweep_gen

RESULTS_FOLDER="$(ls -td results/kappa_sweep_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"
echo "Re-plot with: python -m package.plotting_data.kappa_sweep_plot $RESULTS_FOLDER"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.kappa_sweep_gen
