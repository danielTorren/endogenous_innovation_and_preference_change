#!/bin/bash
#SBATCH --job-name=seed_inputs_sweep_gen
#SBATCH --output=slurm_logs/%x_%j.log     # slurm_logs/ must exist in the dir sbatch is run from -- see .gitkeep there
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # get_num_workers() reads SLURM_CPUS_PER_TASK, so this is ALSO joblib's
                                   # n_jobs and therefore the number of model instances alive at once --
                                   # the only lever on peak memory. Deliberately BELOW seed_repetitions:
                                   # 64 seeds then run as 2 sequential 32-wide batches per seed_inputs,
                                   # which halves peak memory for a few extra minutes of wall clock.
                                   # Raising this to match the seed count REQUIRES raising --mem with it.
#SBATCH --mem=64G                  # 2G per worker. base_params_calibration.json sets
                                   # save_timeseries_data_state=1, which the HHI series REQUIRES, so each
                                   # worker holds full per-step history for 456 steps (180+276,
                                   # duration_future=0) over 3000 individuals. Measured floor is ~1G per
                                   # worker: submit_fig05_calibration_cars_gen.slurm runs 64 of them on
                                   # the same base params in 64G. 64 workers in 32G was an OOM kill on
                                   # the kappa sweep. Trim from `seff <jobid>`.
#SBATCH --time=12:00:00            # 32 seed_inputs x 64 seeds = 2048 runs at 456 steps, ~25-60 s each, as
                                   # 64 sequential 32-wide batches -> ~2-4 h of real work. This job is
                                   # 3.2x the delta/kappa sweeps, hence the longer wall clock. Padded
                                   # against a busy node; trim from seff.
#SBATCH --partition=generic         # default partition on jupyter-torrenperaire (see `sinfo`); CPU-only, 10-day limit

# Single-node: joblib fans out across CPUs on this one machine, there is no
# multi-node/MPI parallelism anywhere in this package.

# Scores every candidate NK landscape / population draw against the calibration
# targets, at the parameters exactly as package/constants/base_params_calibration.json
# has them. Only seed_inputs moves. Sweeps seed_inputs = 1..32 by default; the grid
# and the seed count are constants at the top of the gen script, and --values LO HI
# / --seeds N override them without editing it.
#
# WHY THIS IS WORTH 2048 RUNS. seed_inputs is the only seed the technology
# landscapes, the agent population and the firms' starting designs are drawn from,
# and it is held FIXED across every replicate of every run in this project --
# seed_repetitions varies `seed`, not seed_inputs. One draw of it is therefore baked
# into every result, and the 2023 EV share moves over most of the unit interval
# across that draw alone at unchanged parameters. The value pinned in the JSON is a
# modelling choice on the same footing as a calibrated parameter, so it should be
# measured rather than inherited.
#
# HOW TO READ THE LOG. One row per seed_inputs, printed as it finishes:
#   ev_rmse   RMSE of the EV stock trajectory 2010-2023 against the observed
#             California series, in LOGIT units, sampled at April of each year with
#             the logit taken per seed before averaging -- the same convention as
#             the SBI/NN calibration, so the numbers are comparable with those runs.
#   EV 2023   final-year mean EV stock share, against the observed 0.038.
#   HHI       new-car market concentration, final-year mean, band 0.11-0.18.
#   age yr    whole-fleet mean car age in years, final-year mean, band 10-12.
#   bands     'both ok', or which band the row misses.
# The job then ranks the rows that pass BOTH bands by ev_rmse, prints the best five
# of those and the best five near-misses, reports the ev_rmse SPREAD across the
# grid, and names a PICK.
#
# TWO THINGS NOT TO DO WITH THE RESULT.
#   * Do not treat a passing seed_inputs as a calibration. The parameters were not
#     refitted; only the landscape moved. If the printed spread of ev_rmse across
#     the grid is large, that IS the finding: the result depends more on the
#     landscape draw than on the parameters, and it should be reported, not tuned
#     away.
#   * If nothing passes both bands, do not widen the grid until something does. A
#     value picked from a long enough list fits by chance and will not survive a
#     change of any other parameter. Refit the parameters on this code path instead.
#
# Outputs one fresh results/seed_inputs_sweep_<timestamp> folder holding
# Data/{data_array_ev_prop,data_array_hhi,data_array_age_fleet,param_list,
# param_name,param_label,base_params,observed_ev_prop,summary}. Data/summary is the
# whole table as a list of dicts, so the ranking can be redone downstream without
# rerunning anything.

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

uv run python -m package.generating_data.seed_inputs_sweep_gen

RESULTS_FOLDER="$(ls -td results/seed_inputs_sweep_* | head -n 1)"
echo "Results folder: $RESULTS_FOLDER"

# Otherwise, activate a plain venv instead of the uv lines above:
# source .venv/bin/activate
# python -m package.generating_data.seed_inputs_sweep_gen
