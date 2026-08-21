#!/bin/bash
#SBATCH --job-name=a_chi_sweep_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # get_num_workers() reads SLURM_CPUS_PER_TASK, so this is ALSO joblib's
                                   # n_jobs and therefore the number of model instances alive at once --
                                   # the only lever on peak memory. Deliberately BELOW seed_repetitions:
                                   # 64 seeds then run as 2 sequential 32-wide batches per a_chi, which
                                   # halves peak memory for a few extra minutes of wall clock. Raising
                                   # this to match the seed count REQUIRES raising --mem with it.
#SBATCH --mem=64G                  # 2G per worker. base_params_calibration.json sets
                                   # save_timeseries_data_state=1, which the HHI series REQUIRES, so each
                                   # worker holds full per-step history for 456 steps (180+276,
                                   # duration_future=0) over 3000 individuals. Measured floor is ~1G per
                                   # worker: submit_fig05_calibration_cars_gen.slurm runs 64 of them on
                                   # the same base params in 64G. 64 workers in 32G was an OOM kill on
                                   # the kappa version of this job. Trim from `seff <jobid>`.
#SBATCH --time=04:00:00            # 10 a_chi x 64 seeds = 640 runs at 456 steps, ~25-60 s each, as 20
                                   # sequential 32-wide batches -> ~30-60 min of real work. Padded
                                   # against a busy node; trim from seff.
#SBATCH --partition=generic         # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Sweeps parameters_social_network.a_chi (first Beta shape of the innovativeness
# distribution chi) over 10 values in [0.8, 2] -- the SBI calibration prior range,
# see Data/var_dict of any sbi_seed_av run -- against
# package/constants/base_params_calibration.json, and records new-car HHI and
# whole-fleet mean car age. The grid and the seed count are constants at the top of
# the gen script, not arguments here. Seeds and workers are independent: the seed
# count sets how many runs happen, --cpus-per-task above sets how many of them are
# resident at once.
#
# ONE BIG CAVEAT, spelled out at length in the gen script's docstring and worth
# knowing before reading the log. Unlike the delta and kappa sweeps next to it,
# this one is confounded by construction. Controller.gen_chi draws chi with
# RandomState.beta, a rejection sampler, so the number of draws it takes off the
# pinned seed_inputs stream depends on a_chi. Every later draw on that stream
# therefore moves with a_chi: the incomes, the emissions WTP, the network
# permutation, the firm placement, and both the ICE and EV NK landscapes. Each row
# of this sweep is a different a_chi AND a different population AND a different
# technology landscape, and 64 seeds do NOT average that away, because the seeds
# vary `seed` while the landscape is pinned to seed_inputs.
#   * Read a non-monotonic row-to-row jump as the landscape moving, not as
#     structure in a_chi.
#   * The job quantifies the confound before the first run and prints it above the
#     results table: the sorted chi distribution barely moves, the agent-by-agent
#     chi correlation collapses to ~0, and all 10 rows leave the stream at a
#     different position. Check that table first.
#   * a_chi is the ONLY parameter in this sweep family with this problem. delta and
#     kappa are not drawn on that stream, so those sweeps are clean.
# A feasibility precheck also aborts in one second, not 640 runs, if the grid ever
# reaches a_chi <= 0, where the Beta distribution is undefined.
#
# Outputs one fresh results/a_chi_sweep_<timestamp> folder holding
# Data/{data_array_hhi,data_array_age_fleet,data_array_ev_prop,param_list,
# param_name,param_label,base_params,summary} plus
# Plots/a_chi_sweep_timeseries.png and Plots/a_chi_sweep_response.png. The
# per-a_chi 2023 means are also printed as a table in this job's log, with the
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
uv run python -m package.generating_data.a_chi_sweep_gen

RESULTS_FOLDER="$(ls -td results/a_chi_sweep_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"
echo "Re-plot with: python -m package.plotting_data.sweep_hhi_age_plot $RESULTS_FOLDER"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.a_chi_sweep_gen
