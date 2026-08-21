#!/bin/bash
#SBATCH --job-name=delta_sweep_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # get_num_workers() reads SLURM_CPUS_PER_TASK, so this is ALSO joblib's
                                   # n_jobs and therefore the number of model instances alive at once --
                                   # the only lever on peak memory. Deliberately BELOW seed_repetitions:
                                   # 64 seeds then run as 2 sequential 32-wide batches per delta, which
                                   # halves peak memory for a few extra minutes of wall clock. Raising
                                   # this to match the seed count REQUIRES raising --mem with it.
#SBATCH --mem=64G                  # 2G per worker. base_params_calibration.json sets
                                   # save_timeseries_data_state=1, which the HHI series REQUIRES, so each
                                   # worker holds full per-step history for 456 steps (180+276,
                                   # duration_future=0) over 3000 individuals. Measured floor is ~1G per
                                   # worker: submit_fig05_calibration_cars_gen.slurm runs 64 of them on
                                   # the same base params in 64G. 64 workers in 32G was an OOM kill on
                                   # the kappa version of this job. Trim from `seff <jobid>`.
#SBATCH --time=04:00:00            # 10 delta x 64 seeds = 640 runs at 456 steps, ~25-60 s each, as 20
                                   # sequential 32-wide batches -> ~30-60 min of real work. Padded
                                   # against a busy node; trim from seff.
#SBATCH --partition=generic         # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Sweeps parameters_ICE.delta (monthly efficiency depreciation) over 10 values in
# [0.001, 0.0033] -- the calibration prior range -- against
# package/constants/base_params_calibration.json (calibrated value 0.00175), and
# records new-car HHI and whole-fleet mean car age. The grid and the seed count are
# constants at the top of the gen script, not arguments here. Seeds and workers are
# independent: the seed count sets how many runs happen, --cpus-per-task above sets
# how many of them are resident at once.
#
# TWO CAVEATS, both spelled out in the gen script's docstring and worth knowing
# before reading the log:
#   * delta is copied from parameters_ICE to parameters_EV by the controller, so
#     this moves BOTH technologies. It is not an ICE-only sweep.
#   * delta also scales every agent's emissions WTP, down to 0.33x the calibrated
#     value at the top of this grid. Read the EV 2023 column with that in mind;
#     HHI and fleet age are far less affected, which is why they are the targets
#     here. The job prints this factor before the first run.
# A feasibility precheck runs first and aborts in one second, not 640 runs, if the
# grid ever reaches delta >= r/(1+r) = 0.0040576, where Controller.gen_gamma raises.
#
# Outputs one fresh results/delta_sweep_<timestamp> folder holding
# Data/{data_array_hhi,data_array_age_fleet,data_array_ev_prop,param_list,
# param_name,param_label,base_params,summary} plus
# Plots/delta_sweep_timeseries.png and Plots/delta_sweep_response.png. The
# per-delta 2023 means are also printed as a table in this job's log, with the
# target bands on the row above, and the values landing inside BOTH bands are
# listed at the end.

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

# Both figures are written by the gen script under Agg above, so nothing needs
# re-plotting by hand afterwards.
uv run python -m package.generating_data.delta_sweep_gen

RESULTS_FOLDER="$(ls -td results/delta_sweep_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"
echo "Re-plot with: python -m package.plotting_data.sweep_hhi_age_plot $RESULTS_FOLDER"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.delta_sweep_gen
